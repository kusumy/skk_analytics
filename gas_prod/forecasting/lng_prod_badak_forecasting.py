# %%
import logging
import configparser
import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import psycopg2
import seaborn as sns
import time

from tracemalloc import start
from pmdarima.arima.auto import auto_arima
import matplotlib as mpl
import matplotlib.pyplot as plt
from connection import config, retrieve_data, create_db_connection, get_sql_data
#from utils import configLogging, logMessage, ad_test

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.style.use('fivethirtyeight')

import pmdarima as pm
from pmdarima import model_selection 
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

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

# %%
conn = create_db_connection(section='postgresql_ml_lng_skk')
query_1 = open("lng_prod_badak_data_query.sql", mode="rt").read()
data = get_sql_data(query_1, conn)
data['date'] = pd.DatetimeIndex(data['date'], freq='D')
data = data.reset_index()
data

#%%
ds = 'date'
y = 'lng_production' 

df = data[[ds,y]]
df = df.set_index(ds)
df.index = pd.DatetimeIndex(df.index, freq='D')

#Create column target
train_df = df['lng_production']

#%%
#stationarity_check(train_df)

#%%
#decomposition_plot(train_df)

#%%
#plot_acf_pacf(train_df)

#%%
from chart_studio.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df.lng_production.values, model="multiplicative", period=365)
fig = result.plot()
plt.close()

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
ad_test(df['lng_production'])

#%%
#CREATE EXOGENOUS VARIABLES
df['fg_exog'] = data['fg_exog'].values
df['month'] = [i.month for i in df.index]
df['day'] = [i.day for i in df.index]
train_exog = df.iloc[:,1:]

from sktime.forecasting.base import ForecastingHorizon
time_predict = pd.period_range('2022-11-11', periods=51, freq='D')

#%%
#query_exog = os.path.join('gas_prod/sql','lng_prod_tangguh_exog_query.sql')
query_2 = open("lng_prod_badak_exog_query.sql", mode="rt").read()
data_exog = get_sql_data(query_2, conn)
data_exog['date'] = pd.DatetimeIndex(data_exog['date'], freq='D')
data_exog.sort_index(inplace=True)
data_exog = data_exog.reset_index()

ds_exog = 'date'
y_exog = 'fg_exog'

test_exog = data_exog[[ds_exog,y_exog]]
test_exog = test_exog.set_index(ds_exog)
test_exog.index = pd.DatetimeIndex(test_exog.index, freq='D')

#Create exogenous date index
test_exog['fg_exog'] = test_exog['fg_exog']
test_exog['month'] = [i.month for i in test_exog.index]
test_exog['day'] = [i.day for i in test_exog.index]

#Create Forecasting Horizon
fh = ForecastingHorizon(test_exog.index, is_relative=False)

# plotting for illustration
fig1, ax = plt.subplots(figsize=(20,8))
ax.plot(train_df, label='train')
ax.set_ylabel("LNG Production")
ax.set_xlabel("Datestamp")
ax.legend(loc='best')
plt.close()

# %%
from pmdarima.arima.utils import ndiffs, nsdiffs
import statsmodels.api as sm
from sktime.forecasting.arima import AutoARIMA

##### FORECASTING #####
#%%

##### ARIMAX MODEL #####
from pmdarima.arima.utils import ndiffs, nsdiffs
import statsmodels.api as sm
from sktime.forecasting.arima import AutoARIMA

#Set parameters
arimax_differencing = 1
arimax_stationary = False
arimax_trace = True
arimax_error_action = "ignore"
arimax_suppress_warnings = True

# Create ARIMA Model
#ARIMA(1,1,3)(0,0,0)[0]
arimax_model = AutoARIMA(d=arimax_differencing, stationary=arimax_stationary, trace=arimax_trace, error_action=arimax_error_action, suppress_warnings=arimax_suppress_warnings)
arimax_model.fit(train_df, X=train_exog)
arimax_forecast = arimax_model.predict(fh, X=test_exog)
y_pred_arimax = pd.DataFrame(arimax_forecast).applymap('{:.2f}'.format)
y_pred_arimax['day_num'] = [i.day for i in arimax_forecast.index]
y_pred_arimax['month_num'] = [i.month for i in arimax_forecast.index]
y_pred_arimax['year_num'] = [i.year for i in arimax_forecast.index]
y_pred_arimax['date'] = y_pred_arimax['year_num'].astype(str) + '-' + y_pred_arimax['month_num'].astype(str) + '-' + y_pred_arimax['day_num'].astype(str)
y_pred_arimax['date'] = pd.DatetimeIndex(y_pred_arimax['date'], freq='D')


#%%
##### SARIMAX MODEL #####

#Set parameters
sarimax_differencing = 1
sarimax_seasonal_differencing = 1
sarimax_sp = 12
sarimax_stationary = False
sarimax_seasonal = True
sarimax_trace = True
sarimax_error_action = "ignore"
sarimax_suppress_warnings = True

# Create SARIMA Model
sarimax_model = AutoARIMA(d=sarimax_differencing, D=sarimax_seasonal_differencing, sp=sarimax_sp, stationary=sarimax_stationary,
                  seasonal=sarimax_seasonal, trace=sarimax_trace, error_action=sarimax_error_action, suppress_warnings=sarimax_suppress_warnings)
sarimax_model.fit(train_df, X=train_exog)
#ARIMA(2,1,3)(0,0,0)[12] 
sarimax_forecast = sarimax_model.predict(fh, X=test_exog)
y_pred_sarimax = pd.DataFrame(sarimax_forecast).applymap('{:.2f}'.format)
y_pred_sarimax['day_num'] = [i.day for i in sarimax_forecast.index]
y_pred_sarimax['month_num'] = [i.month for i in sarimax_forecast.index]
y_pred_sarimax['year_num'] = [i.year for i in sarimax_forecast.index]
y_pred_sarimax['date'] = y_pred_sarimax['year_num'].astype(str) + '-' + y_pred_sarimax['month_num'].astype(str) + '-' + y_pred_sarimax['day_num'].astype(str)
y_pred_sarimax['date'] = pd.DatetimeIndex(y_pred_sarimax['date'], freq='D')
sarimax_model.summary()

#%%
##### PROPHET MODEL #####
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.compose import make_reduction

#Set parameters
prophet_seasonality_mode = 'additive'
prophet_n_changepoints = 2
prophet_seasonality_prior_scale = 0.05
prophet_changepoint_prior_scale = 0.4
prophet_holidays_prior_scale = 8
prophet_daily_seasonality = 7
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
prophet_forecast = prophet_forecaster.predict(fh, X=test_exog) #, X=X_test
y_pred_prophet = pd.DataFrame(prophet_forecast).applymap('{:.2f}'.format)
y_pred_prophet['day_num'] = [i.day for i in prophet_forecast.index]
y_pred_prophet['month_num'] = [i.month for i in prophet_forecast.index]
y_pred_prophet['year_num'] = [i.year for i in prophet_forecast.index]
y_pred_prophet['date'] = y_pred_prophet['year_num'].astype(str) + '-' + y_pred_prophet['month_num'].astype(str) + '-' + y_pred_prophet['day_num'].astype(str)
y_pred_prophet['date'] = pd.DatetimeIndex(y_pred_prophet['date'], freq='D')


##### RANDOM FOREST MODEL #####
from sklearn.ensemble import RandomForestRegressor

#Set parameters
ranfor_n_estimators = 100
ranfor_random_state = 0
ranfor_criterion =  "squared_error"
ranfor_lags = 32
ranfor_strategy = "recursive"

#Create regressor object
ranfor_regressor = RandomForestRegressor(n_estimators = ranfor_n_estimators, random_state=ranfor_random_state, criterion=ranfor_criterion)
ranfor_forecaster = make_reduction(ranfor_regressor, window_length=ranfor_lags, strategy=ranfor_strategy) #30, nonexog=30

ranfor_forecaster.fit(train_df, train_exog) #, X_train
ranfor_forecast = ranfor_forecaster.predict(fh, X=test_exog) #, X=X_test
y_pred_ranfor = pd.DataFrame(ranfor_forecast).applymap('{:.2f}'.format)
y_pred_ranfor['day_num'] = [i.day for i in ranfor_forecast.index]
y_pred_ranfor['month_num'] = [i.month for i in ranfor_forecast.index]
y_pred_ranfor['year_num'] = [i.year for i in ranfor_forecast.index]
y_pred_ranfor['date'] = y_pred_ranfor['year_num'].astype(str) + '-' + y_pred_ranfor['month_num'].astype(str) + '-' + y_pred_ranfor['day_num'].astype(str)
y_pred_ranfor['date'] = pd.DatetimeIndex(y_pred_ranfor['date'], freq='D')


##### XGBOOST MODEL #####
from xgboost import XGBRegressor

#Set parameters
xgb_objective = 'reg:squarederror'
xgb_lags = 42
xgb_strategy = "recursive"

#Create regressor object
xgb_regressor = XGBRegressor(objective=xgb_objective)
xgb_forecaster = make_reduction(xgb_regressor, window_length=xgb_lags, strategy=xgb_strategy)
xgb_forecaster.fit(train_df, train_exog) #, X_train
xgb_forecast = xgb_forecaster.predict(fh, X=test_exog) #, X=X_test
y_pred_xgb = pd.DataFrame(xgb_forecast).applymap('{:.2f}'.format)
y_pred_xgb['day_num'] = [i.day for i in xgb_forecast.index]
y_pred_xgb['month_num'] = [i.month for i in xgb_forecast.index]
y_pred_xgb['year_num'] = [i.year for i in xgb_forecast.index]
y_pred_xgb['date'] = y_pred_xgb['year_num'].astype(str) + '-' + y_pred_xgb['month_num'].astype(str) + '-' + y_pred_xgb['day_num'].astype(str)
y_pred_xgb['date'] = pd.DatetimeIndex(y_pred_xgb['date'], freq='D')


##### LINEAR REGRESSION MODEL #####
from sklearn.linear_model import LinearRegression

#Set parameters
linreg_normalize = True
linreg_lags = 44
linreg_strategy = "recursive"

# Create regressor object
linreg_regressor = LinearRegression(normalize=linreg_normalize)
linreg_forecaster = make_reduction(linreg_regressor, window_length=linreg_lags, strategy=linreg_strategy)
linreg_forecaster.fit(train_df, X=train_exog)
linreg_forecast = linreg_forecaster.predict(fh, X=test_exog) #, X=X_test
y_pred_linreg = pd.DataFrame(linreg_forecast).applymap('{:.2f}'.format)
y_pred_linreg['day_num'] = [i.day for i in linreg_forecast.index]
y_pred_linreg['month_num'] = [i.month for i in linreg_forecast.index]
y_pred_linreg['year_num'] = [i.year for i in linreg_forecast.index]
y_pred_linreg['date'] = y_pred_linreg['year_num'].astype(str) + '-' + y_pred_linreg['month_num'].astype(str) + '-' + y_pred_linreg['day_num'].astype(str)
y_pred_linreg['date'] = pd.DatetimeIndex(y_pred_linreg['date'], freq='D')


##### POLYNOMIAL REGRESSION DEGREE=2 MODEL #####
from polyfit import PolynomRegressor, Constraints

#Set parameters
poly2_regularization = None
poly2_interactions = False
poly2_lags = 7
poly2_strategy = "recursive"

# Create regressor object
poly2_regressor = PolynomRegressor(deg=2, regularization=poly2_regularization, interactions=poly2_interactions)
poly2_forecaster = make_reduction(poly2_regressor, window_length=poly2_lags, strategy=poly2_strategy)
poly2_forecaster.fit(train_df, X=train_exog) #, X=X_train
poly2_forecast = poly2_forecaster.predict(fh, X=test_exog) #, X=X_test
y_pred_poly2 = pd.DataFrame(poly2_forecast).applymap('{:.2f}'.format)
y_pred_poly2['day_num'] = [i.day for i in poly2_forecast.index]
y_pred_poly2['month_num'] = [i.month for i in poly2_forecast.index]
y_pred_poly2['year_num'] = [i.year for i in poly2_forecast.index]
y_pred_poly2['date'] = y_pred_poly2['year_num'].astype(str) + '-' + y_pred_poly2['month_num'].astype(str) + '-' + y_pred_poly2['day_num'].astype(str)
y_pred_poly2['date'] = pd.DatetimeIndex(y_pred_poly2['date'], freq='D')


##### POLYNOMIAL REGRESSION DEGREE=3 MODEL #####
from polyfit import PolynomRegressor, Constraints

#Set parameters
poly3_regularization = None
poly3_interactions = False
poly3_lags = 2
poly3_strategy = "recursive"

# Create regressor object
poly3_regressor = PolynomRegressor(deg=2, regularization=poly3_regularization, interactions=poly3_interactions)
poly3_forecaster = make_reduction(poly3_regressor, window_length=poly3_lags, strategy=poly3_strategy)
poly3_forecaster.fit(train_df, X=train_exog) #, X=X_train
poly3_forecast = poly3_forecaster.predict(fh, X=test_exog) #, X=X_test
y_pred_poly3 = pd.DataFrame(poly3_forecast).applymap('{:.2f}'.format)
y_pred_poly3['day_num'] = [i.day for i in poly3_forecast.index]
y_pred_poly3['month_num'] = [i.month for i in poly3_forecast.index]
y_pred_poly3['year_num'] = [i.year for i in poly3_forecast.index]
y_pred_poly3['date'] = y_pred_poly3['year_num'].astype(str) + '-' + y_pred_poly3['month_num'].astype(str) + '-' + y_pred_poly3['day_num'].astype(str)
y_pred_poly3['date'] = pd.DatetimeIndex(y_pred_poly3['date'], freq='D')

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
title = 'LNG Production PT Badak with Exogenous Variable (Feed Gas, Day & Month)'
ax.set_title(title)
ax.set_ylabel("LNG Production")
ax.set_xlabel("Datestamp")
ax.legend(loc='best')
#plt.savefig("LNG Production PT Badak Arimax Model with Exogenous Variables (Feed Gas + Day-Month)" + ".jpg")
plt.show()

# %%
import psycopg2

def update_trir(forecast, prod_date):
    """ insert forecasting result after last row in table """
    sql = """ UPDATE lng_production_daily
                SET forecast_a = %s
                WHERE prod_date = %s
                AND lng_plant = 'PT Badak'"""
    conn = None
    updated_rows = 0
    try:
        # connect to the PostgreSQL database
        conn = psycopg2.connect(
            host = '188.166.239.112',
            dbname = 'skk_da_lng_analytics',
            user = 'postgres',
            password = 'rahasia',
            port = 5432)
        # create a new cursor
        cur = conn.cursor()
        # execute the UPDATE  statement
        cur.execute(sql, (forecast, prod_date))
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

#%%
for index, row in y_pred.iterrows():
    prod_date = row['date']
    forecast = row[0]
  
    #sql = f'UPDATE trir_monthly_test SET forecast_a = {} WHERE year_num = {} AND month_num = {}'.format(forecast, year_num, month_num)
    update_trir(forecast, prod_date)

# %%
