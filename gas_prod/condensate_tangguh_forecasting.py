# %%
import logging
import configparser
import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import pmdarima as pm
import psycopg2
import seaborn as sns

from tokenize import Ignore
from datetime import datetime
from tracemalloc import start
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('fivethirtyeight')
from connection import config, retrieve_data

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pmdarima import model_selection 
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
import mlflow

pd.options.plotting.backend = "plotly"
from adtk.detector import ThresholdAD
from adtk.visualization import plot
from adtk.data import validate_series

def stationarity_check(ts):
            
    # Calculate rolling statistics
    roll_mean = ts.rolling(window=8, center=False).mean()
    roll_std = ts.rolling(window=8, center=False).std()

    # Perform the Dickey Fuller test
    dftest = adfuller(ts) 
    
    # Plot rolling statistics:
    fig = plt.figure(figsize=(12,6))
    orig = plt.plot(ts, color='blue',label='Original')
    mean = plt.plot(roll_mean, color='red', label='Rolling Mean')
    std = plt.plot(roll_std, color='green', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Print Dickey-Fuller test results

    print('\nResults of Dickey-Fuller Test: \n')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', 
                                             '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

def decomposition_plot(ts):
# Apply seasonal_decompose 
    decomposition = seasonal_decompose(np.log(ts))
    
# Get trend, seasonality, and residuals
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

# Plotting
    plt.figure(figsize=(12,8))
    plt.subplot(411)
    plt.plot(np.log(ts), label='Original', color='blue')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend', color='blue')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality', color='blue')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals', color='blue')
    plt.legend(loc='best')
    plt.tight_layout()

def plot_acf_pacf(ts, figsize=(10,8),lags=24):
    
    fig,ax = plt.subplots(nrows=3, figsize=figsize)
    
    # Plot ts
    ts.plot(ax=ax[0])
    
    # Plot acf, pavf
    plot_acf(ts, ax=ax[1], lags=lags)
    plot_pacf(ts, ax=ax[2], lags=lags) 
    fig.tight_layout()
    
    for a in ax[1:]:
        a.xaxis.set_major_locator(mpl.ticker.MaxNLocator(min_n_ticks=lags, integer=True))
        a.xaxis.grid()
    return fig,ax

#%%
##### CLEANING CONDENSATE DATA #####
#Load Data from Database
query_1 = open("condensate_tangguh_data_query.sql", mode="rt").read()
real_data = retrieve_data(query_1)
real_data['date'] = pd.DatetimeIndex(real_data['date'], freq='D')
ds = 'date'
real_data = real_data.set_index(ds)
real_data['unplanned_shutdown'] = real_data['unplanned_shutdown'].astype('int')
s = validate_series(real_data)
real_data.head(60)

#%%
# Calculate standar deviation
condensate_std = real_data['condensate'].std()
condensate_mean = real_data['condensate'].mean()

# Create anomaly detection model
high_limit1 = condensate_mean+3*condensate_std
low_limit1 = condensate_mean-3*condensate_std
high_limit2 = condensate_mean+condensate_std
low_limit2 = condensate_mean-condensate_std

#Detect Anomaly Values
threshold_ad = ThresholdAD(real_data['unplanned_shutdown']==0)
anomalies = threshold_ad.detect(s)

anomalies = anomalies.drop('condensate', axis=1)
anomalies = anomalies.drop('wpnb_minyak', axis=1)
anomalies = anomalies.drop('planned_shutdown', axis=1)
#anomalies.head(60)

# Create anomaly detection model
#threshold_ad = ThresholdAD(high=high_limit2, low=low_limit1)
#anomalies =  threshold_ad.detect(s)

# Copy data frame of anomalies
copy_anomalies =  anomalies.copy()
# Rename columns
copy_anomalies.rename(columns={'unplanned_shutdown':'anomaly'}, inplace=True)
# Merge original dataframe with anomalies
new_s = pd.concat([s, copy_anomalies], axis=1)

# Get only anomalies data
anomalies_data = new_s[new_s['anomaly'] == False]
#anomalies_data.tail(100)

#%%
# Plot data and its anomalies
from cProfile import label
from imaplib import Time2Internaldate

fig = px.line(new_s, y='condensate')
# Add horizontal line for 3 sigma
fig.add_hline(y=high_limit2, line_color='red', line_dash="dot",
              annotation_text="Mean + std", 
              annotation_position="top right")
fig.add_hline(y=low_limit1, line_color='red', line_dash="dot",
              annotation_text="Mean - 3*std", 
              annotation_position="bottom right")
fig.add_scatter(x=anomalies_data.index, y=anomalies_data['condensate'], mode='markers', marker=dict(color='red'), name="Unplanned Shutdown", showlegend=True)
fig.update_layout(title_text='Condensate Tangguh', title_font_size=24)
fig.show()
#plt.close()

#%%
#Replace Anomaly Values
from datetime import date, datetime, timedelta

def get_first_date_of_current_month(year, month):
    """Return the first date of the month.

    Args:
        year (int): Year
        month (int): Month

    Returns:
        date (datetime): First date of the current month
    """
    first_date = datetime(year, month, 1)
    return first_date.strftime("%Y-%m-%d")

def get_last_date_of_month(year, month):
    """Return the last date of the month.
    
    Args:
        year (int): Year, i.e. 2022
        month (int): Month, i.e. 1 for January

    Returns:
        date (datetime): Last date of the current month
    """
    
    if month == 12:
        last_date = datetime(year, month, 31)
    else:
        last_date = datetime(year, month + 1, 1) + timedelta(days=-1)
    
    return last_date.strftime("%Y-%m-%d")

from dateutil.relativedelta import *

def get_first_date_of_prev_month(year, month, step=-1):
    """Return the first date of the month.

    Args:
        year (int): Year
        month (int): Month

    Returns:
        date (datetime): First date of the current month
    """
    first_date = datetime(year, month, 1)
    first_date = first_date + relativedelta(months=step)
    return first_date.strftime("%Y-%m-%d")

def get_last_date_of_prev_month(year, month, step=-1):
    """Return the last date of the month.
    
    Args:
        year (int): Year, i.e. 2022
        month (int): Month, i.e. 1 for January

    Returns:
        date (datetime): Last date of the current month
    """
    
    if month == 12:
        last_date = datetime(year, month, 31)
    else:
        last_date = datetime(year, month + 1, 1) + timedelta(days=-1)
        
    last_date = last_date + relativedelta(months=step)
    
    return last_date.strftime("%Y-%m-%d")

for index, row in anomalies_data.iterrows():
    yr = index.year
    mt = index.month
    
    # Get start month and end month
    #start_month = str(get_first_date_of_current_month(yr, mt))
    #end_month = str(get_last_date_of_month(yr, mt))
    
    # Get last year start date month
    start_month = get_first_date_of_prev_month(yr,mt,step=-12)
    
    # Get last month last date
    end_month = get_last_date_of_prev_month(yr,mt,step=-1)
    
    # Get mean fead gas data for the month
    sql = "date>='"+start_month+ "' & "+ "date<='" +end_month+"'"
    mean_month=new_s['condensate'].reset_index().query(sql).mean().values[0]
    
    # update value at specific location
    new_s.at[index,'condensate'] = mean_month
    
    print(sql), print(mean_month)

# Check if updated
new_s[new_s['anomaly'] == False]    

#Update data
anomaly_upd = new_s[new_s['anomaly'] == False]

#%%
#Display Cleaned Data
# Plot data and its anomalies
from cProfile import label
from imaplib import Time2Internaldate


fig = px.line(new_s, y='condensate')

# Add horizontal line for 3 sigma
fig.add_hline(y=high_limit2, line_color='red', line_dash="dot",
              annotation_text="Mean + std", 
              annotation_position="top right")
fig.add_hline(y=high_limit1, line_color='red', line_dash="dot",
              annotation_text="Mean + 3*std", 
              annotation_position="top right")
fig.add_hline(y=low_limit1, line_color='red', line_dash="dot",
              annotation_text="Mean - 3*std", 
              annotation_position="bottom right")
fig.add_hline(y=low_limit2, line_color='red', line_dash="dot",
              annotation_text="Mean - std", 
              annotation_position="bottom right")
fig.add_scatter(x=anomalies_data.index, y=anomalies_data['condensate'], mode='markers', marker=dict(color='red'), name="Unplanned Shutdown", showlegend=True)
fig.add_scatter(x=anomaly_upd.index, y=anomaly_upd['condensate'], mode='markers', marker=dict(color='green'), name="Unplanned Cleaned", showlegend=True)
fig.update_layout(title_text='Condensate BP Tangguh', title_font_size=24)

#fig.show()
#plt.close()

# %%
# Prepare data
data = new_s[['condensate', 'wpnb_minyak', 'planned_shutdown']].copy()
data = data.reset_index()

ds = 'date'
y = 'condensate' 
df = data[[ds,y]]
df = df.set_index(ds)
df.index = pd.DatetimeIndex(df.index, freq='D')

#Select column target
train_df = df['condensate']
train_df

#%%
#stationarity_check(train_df)

#%%
#decomposition_plot(train_df)

#%%
#plot_acf_pacf(train_df)

#%%
from chart_studio.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(train_df.values, model="additive", period=365)
fig = result.plot()
plt.show()

#%%
from statsmodels.tsa.stattools import adfuller
def ad_test(dataset):
     dftest = adfuller(df, autolag = 'AIC')
     print("1. ADF : ",dftest[0])
     print("2. P-Value : ", dftest[1])
     print("3. Num Of Lags : ", dftest[2])
     print("4. Num Of Observations Used For ADF Regression:", dftest[3])
     print("5. Critical Values :")
     for key, val in dftest[4].items():
         print("\t",key, ": ", val)
ad_test(train_df)

#%%
#Create Exogenous Features for Training
df['planned_shutdown'] = data['planned_shutdown'].values
df['wpnb_minyak'] = data['wpnb_minyak'].values
df['month'] = [i.month for i in df.index]
df['day'] = [i.day for i in df.index]
train_exog = df.iloc[:,1:]
train_exog

#%%
from sktime.forecasting.base import ForecastingHorizon
time_predict = pd.period_range('2022-09-14', periods=109, freq='D')

#%%
#Load Data from Database
query_2 = open("condensate_tangguh_exog_query.sql", mode="rt").read()
data_exog = retrieve_data(query_2)
data_exog['date'] = pd.DatetimeIndex(data_exog['date'], freq='D')
data_exog.sort_index(inplace=True)
data_exog = data_exog.reset_index()
#data_exog

#%%
ds_exog = 'date'
x_exog = 'planned_shutdown'
y_exog = 'wpnb_minyak'

future_exog = data_exog[[ds_exog,x_exog,y_exog]]
future_exog = future_exog.set_index(ds_exog)
future_exog.index = pd.DatetimeIndex(future_exog.index, freq='D')

#Create exogenous date index
future_exog['month'] = [i.month for i in future_exog.index]
future_exog['day'] = [i.day for i in future_exog.index]
future_exog['wpnb_minyak'] = future_exog['wpnb_minyak'].astype(np.float32)
future_exog[['wpnb_minyak']].applymap('{:.2f}'.format)
future_exog

#%%
fh = ForecastingHorizon(future_exog.index, is_relative=False)
fh

#%%
#Plotting for illustration
fig1, ax = plt.subplots(figsize=(20,8))
ax.plot(train_df, label='train')
ax.set_ylabel("Condensate")
ax.set_xlabel("Datestamp")
ax.legend(loc='best')
#plt.close()


#%%
##### FORECASTING #####
##### ARIMAX MODEL #####
from pmdarima.arima.utils import ndiffs, nsdiffs
import statsmodels.api as sm
from sktime.forecasting.arima import AutoARIMA

#Set parameters
arimax_differencing = 0
arimax_error_action = "ignore"
arimax_suppress_warnings = True

# Create ARIMA Model
arimax_model = AutoARIMA(d=arimax_differencing, error_action=arimax_error_action, suppress_warnings=arimax_suppress_warnings)
arimax_model.fit(train_df, X=train_exog)
arimax_forecast = arimax_model.predict(fh, X=future_exog)
y_pred_arimax = pd.DataFrame(arimax_forecast).applymap('{:.2f}'.format)
y_pred_arimax['day_num'] = [i.day for i in arimax_forecast.index]
y_pred_arimax['month_num'] = [i.month for i in arimax_forecast.index]
y_pred_arimax['year_num'] = [i.year for i in arimax_forecast.index]
y_pred_arimax['date'] = y_pred_arimax['year_num'].astype(str) + '-' + y_pred_arimax['month_num'].astype(str) + '-' + y_pred_arimax['day_num'].astype(str)
y_pred_arimax['date'] = pd.DatetimeIndex(y_pred_arimax['date'], freq='D')
#Rename colum 0
y_pred_arimax.rename(columns={0:'forecast_a'}, inplace=True)

#%%
##### SARIMAX MODEL #####
from pmdarima.arima import auto_arima, ARIMA

#Set parameters
sarimax_differencing = 0
sarimax_seasonal_differencing = 1
sarimax_sp = 12
sarimax_stationary = False
sarimax_seasonal = True
sarimax_trace = True
sarimax_error_action = "ignore"
sarimax_suppress_warnings = True

# Create SARIMAX Model
#sarimax_model = AutoARIMA(d=sarimax_differencing, D=sarimax_seasonal_differencing, sp=sarimax_sp, stationary=sarimax_stationary,
#                  seasonal=sarimax_seasonal, trace=sarimax_trace, error_action=sarimax_error_action, suppress_warnings=sarimax_suppress_warnings)
sarimax_model = ARIMA(order=(2,0,2), seasonal_order=(2,1,0,12),  suppress_warnings=sarimax_suppress_warnings)
sarimax_model.fit(train_df, X=train_exog)
sarimax_forecast = sarimax_model.predict(fh, X=future_exog)
y_pred_sarimax = pd.DataFrame(sarimax_forecast).applymap('{:.2f}'.format)
y_pred_sarimax['day_num'] = [i.day for i in sarimax_forecast.index]
y_pred_sarimax['month_num'] = [i.month for i in sarimax_forecast.index]
y_pred_sarimax['year_num'] = [i.year for i in sarimax_forecast.index]
y_pred_sarimax['date'] = y_pred_sarimax['year_num'].astype(str) + '-' + y_pred_sarimax['month_num'].astype(str) + '-' + y_pred_sarimax['day_num'].astype(str)
y_pred_sarimax['date'] = pd.DatetimeIndex(y_pred_sarimax['date'], freq='D')
#Rename colum 0
y_pred_sarimax.rename(columns={0:'forecast_b'}, inplace=True)


##### PROPHET MODEL #####
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.compose import make_reduction

#Set parameters
prophet_seasonality_mode = 'multiplicative'
prophet_n_changepoints = 40
prophet_seasonality_prior_scale = 0.2
prophet_changepoint_prior_scale = 0.1
prophet_holidays_prior_scale = 8
prophet_daily_seasonality = 5
prophet_weekly_seasonality = 1
prophet_yearly_seasonality = 10

#Create regressor object
prophet_forecaster = Prophet(
        seasonality_mode=prophet_seasonality_mode,
        n_changepoints=prophet_n_changepoints,
        seasonality_prior_scale=prophet_seasonality_prior_scale, #Flexibility of the seasonality (0.01,10)
        changepoint_prior_scale=prophet_changepoint_prior_scale, #Flexibility of the trend (0.001,0.5)
        holidays_prior_scale=prophet_holidays_prior_scale, #Flexibility of the holiday effects (0.01,10)
        #changepoint_range=0.8, #proportion of the history in which the trend is allowed to change
        daily_seasonality=prophet_daily_seasonality,
        weekly_seasonality=prophet_weekly_seasonality,
        yearly_seasonality=prophet_yearly_seasonality)

prophet_forecaster.fit(train_df, train_exog) #, X_train
prophet_forecast = prophet_forecaster.predict(fh, X=future_exog) #, X=X_test
y_pred_prophet = pd.DataFrame(prophet_forecast).applymap('{:.2f}'.format)
y_pred_prophet['day_num'] = [i.day for i in prophet_forecast.index]
y_pred_prophet['month_num'] = [i.month for i in prophet_forecast.index]
y_pred_prophet['year_num'] = [i.year for i in prophet_forecast.index]
y_pred_prophet['date'] = y_pred_prophet['year_num'].astype(str) + '-' + y_pred_prophet['month_num'].astype(str) + '-' + y_pred_prophet['day_num'].astype(str)
y_pred_prophet['date'] = pd.DatetimeIndex(y_pred_prophet['date'], freq='D')
#Rename colum 0
y_pred_prophet.rename(columns={0:'forecast_c'}, inplace=True)


##### RANDOM FOREST MODEL #####
from sklearn.ensemble import RandomForestRegressor

#Set parameters
ranfor_n_estimators = 150
ranfor_random_state = 0
ranfor_criterion =  "squared_error"
ranfor_lags = 53
ranfor_strategy = "recursive"

#Create regressor object
ranfor_regressor = RandomForestRegressor(n_estimators = ranfor_n_estimators, random_state=ranfor_random_state, criterion=ranfor_criterion)
ranfor_forecaster = make_reduction(ranfor_regressor, window_length=ranfor_lags, strategy=ranfor_strategy) #30, nonexog=30

ranfor_forecaster.fit(train_df, train_exog) #, X_train
ranfor_forecast = ranfor_forecaster.predict(fh, X=future_exog) #, X=X_test
y_pred_ranfor = pd.DataFrame(ranfor_forecast).applymap('{:.2f}'.format)
y_pred_ranfor['day_num'] = [i.day for i in ranfor_forecast.index]
y_pred_ranfor['month_num'] = [i.month for i in ranfor_forecast.index]
y_pred_ranfor['year_num'] = [i.year for i in ranfor_forecast.index]
y_pred_ranfor['date'] = y_pred_ranfor['year_num'].astype(str) + '-' + y_pred_ranfor['month_num'].astype(str) + '-' + y_pred_ranfor['day_num'].astype(str)
y_pred_ranfor['date'] = pd.DatetimeIndex(y_pred_ranfor['date'], freq='D')
#Rename colum 0
y_pred_ranfor.rename(columns={0:'forecast_d'}, inplace=True)


##### XGBOOST MODEL #####
from xgboost import XGBRegressor

#Set parameters
xgb_objective = 'reg:squarederror'
xgb_lags = 25
xgb_strategy = "recursive"

#Create regressor object
xgb_regressor = XGBRegressor(objective=xgb_objective)
xgb_forecaster = make_reduction(xgb_regressor, window_length=xgb_lags, strategy=xgb_strategy)
xgb_forecaster.fit(train_df, train_exog) #, X_train
xgb_forecast = xgb_forecaster.predict(fh, X=future_exog) #, X=X_test
y_pred_xgb = pd.DataFrame(xgb_forecast).applymap('{:.2f}'.format)
y_pred_xgb['day_num'] = [i.day for i in xgb_forecast.index]
y_pred_xgb['month_num'] = [i.month for i in xgb_forecast.index]
y_pred_xgb['year_num'] = [i.year for i in xgb_forecast.index]
y_pred_xgb['date'] = y_pred_xgb['year_num'].astype(str) + '-' + y_pred_xgb['month_num'].astype(str) + '-' + y_pred_xgb['day_num'].astype(str)
y_pred_xgb['date'] = pd.DatetimeIndex(y_pred_xgb['date'], freq='D')
#Rename colum 0
y_pred_xgb.rename(columns={0:'forecast_e'}, inplace=True)


##### LINEAR REGRESSION MODEL #####
from sklearn.linear_model import LinearRegression

#Set parameters
linreg_normalize = True
linreg_lags = 22
linreg_strategy = "recursive"

# Create regressor object
linreg_regressor = LinearRegression(normalize=linreg_normalize)
linreg_forecaster = make_reduction(linreg_regressor, window_length=linreg_lags, strategy=linreg_strategy)
linreg_forecaster.fit(train_df, X=train_exog)
linreg_forecast = linreg_forecaster.predict(fh, X=future_exog) #, X=X_test
y_pred_linreg = pd.DataFrame(linreg_forecast).applymap('{:.2f}'.format)
y_pred_linreg['day_num'] = [i.day for i in linreg_forecast.index]
y_pred_linreg['month_num'] = [i.month for i in linreg_forecast.index]
y_pred_linreg['year_num'] = [i.year for i in linreg_forecast.index]
y_pred_linreg['date'] = y_pred_linreg['year_num'].astype(str) + '-' + y_pred_linreg['month_num'].astype(str) + '-' + y_pred_linreg['day_num'].astype(str)
y_pred_linreg['date'] = pd.DatetimeIndex(y_pred_linreg['date'], freq='D')
#Rename colum 0
y_pred_linreg.rename(columns={0:'forecast_f'}, inplace=True)


##### POLYNOMIAL REGRESSION DEGREE=2 MODEL #####
from polyfit import PolynomRegressor, Constraints

#Set parameters
poly2_regularization = None
poly2_interactions = False
poly2_lags = 5
poly2_strategy = "recursive"

# Create regressor object
poly2_regressor = PolynomRegressor(deg=2, regularization=poly2_regularization, interactions=poly2_interactions)
poly2_forecaster = make_reduction(poly2_regressor, window_length=poly2_lags, strategy=poly2_strategy)
poly2_forecaster.fit(train_df, X=train_exog) #, X=X_train
poly2_forecast = poly2_forecaster.predict(fh, X=future_exog) #, X=X_test
y_pred_poly2 = pd.DataFrame(poly2_forecast).applymap('{:.2f}'.format)
y_pred_poly2['day_num'] = [i.day for i in poly2_forecast.index]
y_pred_poly2['month_num'] = [i.month for i in poly2_forecast.index]
y_pred_poly2['year_num'] = [i.year for i in poly2_forecast.index]
y_pred_poly2['date'] = y_pred_poly2['year_num'].astype(str) + '-' + y_pred_poly2['month_num'].astype(str) + '-' + y_pred_poly2['day_num'].astype(str)
y_pred_poly2['date'] = pd.DatetimeIndex(y_pred_poly2['date'], freq='D')
#Rename colum 0
y_pred_poly2.rename(columns={0:'forecast_g'}, inplace=True)


##### POLYNOMIAL REGRESSION DEGREE=3 MODEL #####
from polyfit import PolynomRegressor, Constraints

#Set parameters
poly3_regularization = None
poly3_interactions = False
poly3_lags = 0.55
poly3_strategy = "recursive"

# Create regressor object
poly3_regressor = PolynomRegressor(deg=2, regularization=poly3_regularization, interactions=poly3_interactions)
poly3_forecaster = make_reduction(poly3_regressor, window_length=poly3_lags, strategy=poly3_strategy)
poly3_forecaster.fit(train_df, X=train_exog) #, X=X_train
poly3_forecast = poly3_forecaster.predict(fh, X=future_exog) #, X=X_test
y_pred_poly3 = pd.DataFrame(poly3_forecast).applymap('{:.2f}'.format)
y_pred_poly3['day_num'] = [i.day for i in poly3_forecast.index]
y_pred_poly3['month_num'] = [i.month for i in poly3_forecast.index]
y_pred_poly3['year_num'] = [i.year for i in poly3_forecast.index]
y_pred_poly3['date'] = y_pred_poly3['year_num'].astype(str) + '-' + y_pred_poly3['month_num'].astype(str) + '-' + y_pred_poly3['day_num'].astype(str)
y_pred_poly3['date'] = pd.DatetimeIndex(y_pred_poly3['date'], freq='D')
#Rename colum 0
y_pred_poly3.rename(columns={0:'forecast_h'}, inplace=True)

#%%
# Plot prediction
fig, ax = plt.subplots(figsize=(20,8))
ax.plot(train_df, label='train')
ax.plot(arimax_forecast, label='arimax_pred')
ax.plot(sarimax_forecast, label='sarimax_pred')
ax.plot(prophet_forecast, label='prophet_pred')
ax.plot(ranfor_forecast, label='ranfor_pred')
ax.plot(xgb_forecast, label='xgb_pred')
ax.plot(linreg_forecast, label='linreg_pred')
ax.plot(poly2_forecast, label='poly2_pred')
ax.plot(poly3_forecast, label='poly3_pred')
title = 'Condensate BP Tangguh Forecasting with Exogenous Variable WPNB Oil, Planned Shuwdown, Day & Month)'
ax.set_title(title)
ax.set_ylabel("LNG Production")
ax.set_xlabel("Datestamp")
ax.legend(loc='best')
#plt.savefig("Condensate BP Tangguh Forecasting" + ".jpg")
#plt.show()
plt.close()

# %%
##### JOIN PREDICTION RESULT TO DATAFRAME #####
y_all_pred = pd.concat([y_pred_arimax[['forecast_a']],
                            y_pred_sarimax[['forecast_b']],
                            y_pred_prophet[['forecast_c']],
                            y_pred_ranfor[['forecast_d']],
                            y_pred_xgb[['forecast_e']],
                            y_pred_linreg[['forecast_f']],
                            y_pred_poly2[['forecast_g']],
                            y_pred_poly3[['forecast_h']]], axis=1)
y_all_pred['date'] = future_exog.index.values
#y_all_pred

conn = psycopg2.connect(
            host = '188.166.239.112',
            dbname = 'skk_da_lng_analytics',
            user = 'postgres',
            password = 'rahasia',
            port = 5432)

def insert_forecast(conn, y_all_pred):
    total_updated_rows = 0
    for index, row in y_all_pred.iterrows():
        year_num = str(index) #row['year_num']
        forecast_a, forecast_b, forecast_c, forecast_d, forecast_e, forecast_f, forecast_g, forecast_h = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]
        
        #sql = f'UPDATE trir_monthly_test SET forecast_a = {} WHERE year_num = {} AND month_num = {}'.format(forecast, year_num, month_num)
        updated_rows = update_value(conn, forecast_a, forecast_b, forecast_c, forecast_d, forecast_e, forecast_f, forecast_g, forecast_h, date)
        total_updated_rows = total_updated_rows + updated_rows 
        
    return total_updated_rows

def update_value(conn, forecast_a, forecast_b, forecast_c, forecast_d, forecast_e, forecast_f, forecast_g, forecast_h, prod_date):
    """ update table name based on the table id """
    sql = """ UPDATE hse_analytics_trir_yearly
                SET forecast_a = %s,
                SET forecast_b = %s,
                SET forecast_c = %s,
                SET forecast_d = %s,
                SET forecast_e = %s,
                SET forecast_f = %s,
                SET forecast_g = %s,
                SET forecast_h = %s,
                WHERE prod_date = %s"""
    conn = None
    updated_rows = 0
    try:
        # create a new cursor
        cur = conn.cursor()
        # execute the UPDATE  statement
        cur.execute(sql, (forecast_a, forecast_b, forecast_c, forecast_d, forecast_e, forecast_f, forecast_g, forecast_h, prod_date))
        # get the number of updated rows
        updated_rows = cur.rowcount
        # Commit the changes to the database
        conn.commit()
        # Close communication with the PostgreSQL database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

    return updated_rows

# %%
total_updated_rows = insert_forecast(conn, y_all_pred)
print("Done")
# %%
