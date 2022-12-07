# %%
from tokenize import Ignore
from tracemalloc import start
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('fivethirtyeight')
from connection import config, retrieve_data, get_sql_data, create_db_connection

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import pmdarima as pm
from pmdarima import model_selection 
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
import mlflow
from adtk.detector import ThresholdAD
from adtk.visualization import plot
from adtk.data import validate_series
pd.options.plotting.backend = "plotly"
from dateutil.relativedelta import *

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
# Prepare data
conn = create_db_connection(section='postgresql_ml_lng_skk')
query_1 = open(os.path.join('lng_prod_tangguh_data_query.sql'), mode="rt").read()
data = get_sql_data(query_1, conn)

data['date'] = pd.DatetimeIndex(data['date'], freq='D')
data = data.reset_index()
data.head()

#%%
ds = 'date'
y = 'lng_production' 

df = data[[ds,y]]
df = df.set_index(ds)
df.index = pd.DatetimeIndex(df.index, freq='D')
df

#%%
data_cleaning = data[['date', 'lng_production', 'unplanned_shutdown', 'planned_shutdown']].copy()
ds_cleaning = 'date'
data_cleaning = data_cleaning.set_index(ds_cleaning)
data_cleaning['unplanned_shutdown'] = data_cleaning['unplanned_shutdown'].astype('int')
s = validate_series(data_cleaning)

# plotting for illustration
plt.style.use('fivethirtyeight')

fig1, ax = plt.subplots(figsize=(20,8))
ax.plot(data_cleaning['lng_production'], label='LNG Production')
ax.set_ylabel("LNG Production")
ax.set_xlabel("Datestamp")
ax.legend(loc='best')
plt.show()

#%%
# Calculate standar deviation
lng_prod_std = data_cleaning['lng_production'].std()
lng_prod_mean = data_cleaning['lng_production'].mean()

#Detect Anomaly Values
# Create anomaly detection model
high_limit1 = lng_prod_mean+3*lng_prod_std
low_limit1 = lng_prod_mean-3*lng_prod_std
high_limit2 = lng_prod_mean+lng_prod_std
low_limit2 = lng_prod_mean-lng_prod_std

threshold_ad = ThresholdAD(data_cleaning['unplanned_shutdown']==0)
anomalies = threshold_ad.detect(s)

anomalies = anomalies.drop('lng_production', axis=1)
anomalies = anomalies.drop('planned_shutdown', axis=1)
#anomalies

#%%
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

fig = px.line(new_s, y='lng_production')

# Add horizontal line for 3 sigma
fig.add_hline(y=high_limit2, line_color='red', line_dash="dot",
              annotation_text="Mean + std", 
              annotation_position="top right")
fig.add_hline(y=low_limit1, line_color='red', line_dash="dot",
              annotation_text="Mean - 3*std", 
              annotation_position="bottom right")
fig.add_scatter(x=anomalies_data.index, y=anomalies_data['lng_production'], mode='markers', marker=dict(color='red'), name="Unplanned Shutdown", showlegend=True)
fig.update_layout(title_text='LNG Production Tangguh', title_font_size=24)

fig.show()
#plt.close()

#%%
#REPLACE ANOMALY VALUES
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
    mean_month=new_s['lng_production'].reset_index().query(sql).mean().values[0]
    
    # update value at specific location
    new_s.at[index,'lng_production'] = mean_month
    
    print(sql), print(mean_month)

# Check if updated
new_s[new_s['anomaly'] == False]

anomaly_upd = new_s[new_s['anomaly'] == False]

#%%
# Plot data and its anomalies
from cProfile import label
from imaplib import Time2Internaldate


fig = px.line(new_s, y='lng_production')

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
fig.add_scatter(x=anomalies_data.index, y=anomalies_data['lng_production'], mode='markers', marker=dict(color='red'), name="Unplanned Shutdown", showlegend=True)
fig.add_scatter(x=anomaly_upd.index, y=anomaly_upd['lng_production'], mode='markers', marker=dict(color='green'), name="Unplanned Cleaned", showlegend=True)
fig.update_layout(title_text='LNG Production BP Tangguh', title_font_size=24)

fig.show()
#plt.close()


#%%
data_cleaned = new_s[['lng_production', 'planned_shutdown']].copy()
data_cleaned = data_cleaned.reset_index()

ds_cleaned = 'date'
y_cleaned = 'lng_production'
df_cleaned = data_cleaned[[ds_cleaned, y_cleaned]]
df_cleaned = df_cleaned.set_index(ds_cleaned)
df_cleaned.index = pd.DatetimeIndex(df_cleaned.index, freq='D')

#Select column target
train_df = df_cleaned['lng_production']

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
#plt.show()
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
ad_test(train_df)

#%%
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon

# Test size
test_size = 0.2
# Split data (original data)
y_train, y_test = temporal_train_test_split(df, test_size=test_size)
# Split data (original data)
y_train_cleaned, y_test_cleaned = temporal_train_test_split(df_cleaned, test_size=test_size)
# Horizon
fh = ForecastingHorizon(y_test.index, is_relative=False)

#%%
## Create Exogenous Variable
df_cleaned['month'] = [i.month for i in df_cleaned.index]
df_cleaned['planned_shutdown'] = data['planned_shutdown'].values
df_cleaned['day'] = [i.day for i in df_cleaned.index]
#df['day_of_year'] = [i.dayofyear for i in df.index]
#df['week_of_year'] = [i.weekofyear for i in df.index]
#df.tail(20)

#%%
# Split into train and test
X_train, X_test = temporal_train_test_split(df_cleaned.iloc[:,1:], test_size=test_size)

exogenous_features = ["month", "day", "planned_shutdown"]

#%%
# plotting for illustration
fig1, ax = plt.subplots(figsize=(20,8))
ax.plot(y_train_cleaned, label='train')
ax.plot(y_test_cleaned, label='test')
ax.set_ylabel("LNG Production")
ax.set_xlabel("Datestamp")
ax.legend(loc='best')
#plt.show()


##### ARIMAX MODEL (forecast_a) #####
#ARIMA(1,1,1)
# %%
from pmdarima.arima.utils import ndiffs, nsdiffs
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
from pmdarima import auto_arima

# Create ARIMAX (forecast_a) Model
#arimax_model = AutoARIMA(d=1, suppress_warnings=True, error_action='ignore', information_criterion='hqic') #If using SKTime AutoArima
arimax_model = auto_arima(y_train_cleaned, X_train[exogenous_features], d=1, trace=True, error_action="ignore", suppress_warnings=True)
arimax_model.fit(y_train_cleaned, X=X_train[exogenous_features])

arimax_forecast = arimax_model.predict(len(fh), X=X_test[exogenous_features]) #If using sktime (fh), if using pmdarima (len(fh))
y_pred_arimax = pd.DataFrame(arimax_forecast).applymap('{:.2f}'.format)
y_pred_arimax['day_num'] = [i.day for i in arimax_forecast.index]
y_pred_arimax['month_num'] = [i.month for i in arimax_forecast.index]
y_pred_arimax['year_num'] = [i.year for i in arimax_forecast.index]
y_pred_arimax['date'] = y_pred_arimax['year_num'].astype(str) + '-' + y_pred_arimax['month_num'].astype(str) + '-' + y_pred_arimax['day_num'].astype(str)
y_pred_arimax['date'] = pd.DatetimeIndex(y_pred_arimax['date'], freq='D')
# Rename column to forecast_a
y_pred_arimax.rename(columns={0:'forecast_a'}, inplace=True)

#Create MAPE
arimax_mape = mean_absolute_percentage_error(y_test.lng_production, arimax_forecast)
arimax_mape_str = str('MAPE: %.4f' % arimax_mape)


#%%
##### SARIMAX MODEL (forecast_b) #####
##### ARIMA(1,0,2)(0,1,1)[4] #####
#%%
from pmdarima.arima import auto_arima
sarimax_model = auto_arima(y=y_train_cleaned.lng_production, X=X_train[exogenous_features], d=0, D=1, seasonal=True, m=4, trace=True, error_action="ignore", suppress_warnings=True)
sarimax_model.fit(y_train_cleaned.lng_production, X=X_train[exogenous_features])
sarimax_model.summary()

sarimax_forecast = sarimax_model.predict(len(fh), X=X_test[exogenous_features])
y_pred_sarimax = pd.DataFrame(sarimax_forecast).applymap('{:.2f}'.format)
y_pred_sarimax['day_num'] = [i.day for i in sarimax_forecast.index]
y_pred_sarimax['month_num'] = [i.month for i in sarimax_forecast.index]
y_pred_sarimax['year_num'] = [i.year for i in sarimax_forecast.index]
y_pred_sarimax['date'] = y_pred_sarimax['year_num'].astype(str) + '-' + y_pred_sarimax['month_num'].astype(str) + '-' + y_pred_sarimax['day_num'].astype(str)
y_pred_sarimax['date'] = pd.DatetimeIndex(y_pred_sarimax['date'], freq='D')
# Rename column to forecast_a
y_pred_sarimax.rename(columns={0:'forecast_b'}, inplace=True)

#Create MAPE
sarimax_mape = mean_absolute_percentage_error(y_test.lng_production, sarimax_forecast)
sarimax_mape_str = str('MAPE: %.4f' % sarimax_mape)


##### PROPHET MODEL (forecast_c) #####
#%%
# Create model
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.compose import make_reduction

#Set Parameters
seasonality_mode = 'additive'
n_changepoints = 26
seasonality_prior_scale = 0.05
changepoint_prior_scale = 0.1
holidays_prior_scale = 8
daily_seasonality = 8
weekly_seasonality = 1
yearly_seasonality = 10

# create regressor object
prophet_forecaster = Prophet(
    seasonality_mode=seasonality_mode,
    n_changepoints=n_changepoints,
    seasonality_prior_scale=seasonality_prior_scale, #Flexibility of the seasonality (0.01,10)
    changepoint_prior_scale=changepoint_prior_scale, #Flexibility of the trend (0.001,0.5)
    holidays_prior_scale=holidays_prior_scale, #Flexibility of the holiday effects (0.01,10)
    #changepoint_range=0.8, #proportion of the history in which the trend is allowed to change
    daily_seasonality=daily_seasonality,
    weekly_seasonality=weekly_seasonality,
    yearly_seasonality=yearly_seasonality)

prophet_forecaster.fit(y_train_cleaned, X=X_train) #, X_train
prophet_forecast = prophet_forecaster.predict(fh, X=X_test) #, X=X_test
y_pred_prophet = pd.DataFrame(prophet_forecast).applymap('{:.2f}'.format)
y_pred_prophet['day_num'] = [i.day for i in prophet_forecast.index]
y_pred_prophet['month_num'] = [i.month for i in prophet_forecast.index]
y_pred_prophet['year_num'] = [i.year for i in prophet_forecast.index]
y_pred_prophet['date'] = y_pred_prophet['year_num'].astype(str) + '-' + y_pred_prophet['month_num'].astype(str) + '-' + y_pred_prophet['day_num'].astype(str)
y_pred_prophet['date'] = pd.DatetimeIndex(y_pred_prophet['date'], freq='D')
# Rename column to forecast_c
y_pred_prophet.rename(columns={'lng_production':'forecast_c'}, inplace=True)

#Create MAPE
prophet_mape = mean_absolute_percentage_error(y_test['lng_production'], prophet_forecast)
prophet_mape_str = str('MAPE: %.4f' % prophet_mape)
prophet_mape_str


##### RANDOM FOREST MODEL (forecast_d) #####
#%%
from sklearn.ensemble import RandomForestRegressor

#Set Parameters
ranfor_n_estimators = 157
ranfor_lags = 75
ranfor_random_state = 0
ranfor_criterion = "squared_error"
ranfor_strategy = "recursive"

# create regressor object
ranfor_regressor = RandomForestRegressor(n_estimators = ranfor_n_estimators, random_state = ranfor_random_state, criterion = ranfor_criterion)
ranfor_forecaster = make_reduction(ranfor_regressor, window_length = ranfor_lags, strategy = ranfor_strategy)

ranfor_forecaster.fit(y_train_cleaned, X_train) #, X_train
ranfor_forecast = ranfor_forecaster.predict(fh, X=X_test) #, X=X_test
y_pred_ranfor = pd.DataFrame(ranfor_forecast).applymap('{:.2f}'.format)
y_pred_ranfor['day_num'] = [i.day for i in ranfor_forecast.index]
y_pred_ranfor['month_num'] = [i.month for i in ranfor_forecast.index]
y_pred_ranfor['year_num'] = [i.year for i in ranfor_forecast.index]
y_pred_ranfor['date'] = y_pred_ranfor['year_num'].astype(str) + '-' + y_pred_ranfor['month_num'].astype(str) + '-' + y_pred_ranfor['day_num'].astype(str)
y_pred_ranfor['date'] = pd.DatetimeIndex(y_pred_ranfor['date'], freq='D')
# Rename column to forecast_d
y_pred_prophet.rename(columns={'lng_production':'forecast_d'}, inplace=True)

#Create MAPE
ranfor_mape = mean_absolute_percentage_error(y_test['lng_production'], ranfor_forecast)
ranfor_mape_str = str('MAPE: %.4f' % ranfor_mape)
ranfor_mape_str


##### XGBOOST MODEL (forecast_e) #####
#%%
# Create model
from xgboost import XGBRegressor

#Set Parameters
xgb_objective = 'reg:squarederror'
xgb_lags = 16
xgb_strategy = "recursive"

xgb_regressor = XGBRegressor(objective=xgb_objective)
xgb_forecaster = make_reduction(xgb_regressor, window_length=xgb_lags, strategy=xgb_strategy)

xgb_forecaster.fit(y_train_cleaned, X=X_train) #, X_train
xgb_forecast = xgb_forecaster.predict(fh, X=X_test) #, X=X_test
y_pred_xgb = pd.DataFrame(xgb_forecast).applymap('{:.2f}'.format)
y_pred_xgb['day_num'] = [i.day for i in xgb_forecast.index]
y_pred_xgb['month_num'] = [i.month for i in xgb_forecast.index]
y_pred_xgb['year_num'] = [i.year for i in xgb_forecast.index]
y_pred_xgb['date'] = y_pred_xgb['year_num'].astype(str) + '-' + y_pred_xgb['month_num'].astype(str) + '-' + y_pred_xgb['day_num'].astype(str)
y_pred_xgb['date'] = pd.DatetimeIndex(y_pred_xgb['date'], freq='D')
# Rename column to forecast_e
y_pred_xgb.rename(columns={'lng_production':'forecast_e'}, inplace=True)

#Create MAPE
xgb_mape = mean_absolute_percentage_error(y_test['lng_production'], xgb_forecast)
xgb_mape_str = str('MAPE: %.4f' % xgb_mape)


##### LINEAR REGRESSION MODEL (forecast_f) #####
#%%
# Create model
from sklearn.linear_model import LinearRegression

#Set Parameters
linreg_lags = 10
linreg_strategy = "recursive"

linreg_regressor = LinearRegression(normalize=True)
linreg_forecaster = make_reduction(linreg_regressor, window_length=linreg_lags, strategy=linreg_strategy)

linreg_forecaster.fit(y_train_cleaned, X=X_train) #, X=X_train
linreg_forecast = linreg_forecaster.predict(fh, X=X_test) #, X=X_test
y_pred_linreg = pd.DataFrame(linreg_forecast).applymap('{:.2f}'.format)
y_pred_linreg['day_num'] = [i.day for i in linreg_forecast.index]
y_pred_linreg['month_num'] = [i.month for i in linreg_forecast.index]
y_pred_linreg['year_num'] = [i.year for i in linreg_forecast.index]
y_pred_linreg['date'] = y_pred_linreg['year_num'].astype(str) + '-' + y_pred_linreg['month_num'].astype(str) + '-' + y_pred_linreg['day_num'].astype(str)
y_pred_linreg['date'] = pd.DatetimeIndex(y_pred_linreg['date'], freq='D')
# Rename column to forecast_f
y_pred_xgb.rename(columns={'lng_production':'forecast_f'}, inplace=True)

#Create MAPE
linreg_mape = mean_absolute_percentage_error(y_test['lng_production'], linreg_forecast)
linreg_mape_str = str('MAPE: %.4f' % linreg_mape)
linreg_mape_str


##### POLYNOMIAL REGRESSION DEGREE=2 MODEL (forecast_g) #####
#%%
#Create model
from polyfit import PolynomRegressor, Constraints

#Set Parameters
poly2_lags = 3
poly2_regularization = None
poly2_interactions = False
poly2_strategy = "recursive"

poly2_regressor = PolynomRegressor(deg=2, regularization=poly2_regularization, interactions=poly2_interactions)
poly2_forecaster = make_reduction(poly2_regressor, window_length=poly2_lags, strategy=poly2_strategy)

poly2_forecaster.fit(y_train_cleaned, X=X_train) #, X=X_train
poly2_forecast = poly2_forecaster.predict(fh, X=X_test) #, X=X_test
y_pred_poly2 = pd.DataFrame(poly2_forecast).applymap('{:.2f}'.format)
y_pred_poly2['day_num'] = [i.day for i in poly2_forecast.index]
y_pred_poly2['month_num'] = [i.month for i in poly2_forecast.index]
y_pred_poly2['year_num'] = [i.year for i in poly2_forecast.index]
y_pred_poly2['date'] = y_pred_poly2['year_num'].astype(str) + '-' + y_pred_poly2['month_num'].astype(str) + '-' + y_pred_poly2['day_num'].astype(str)
y_pred_poly2['date'] = pd.DatetimeIndex(y_pred_poly2['date'], freq='D')
# Rename column to forecast_g
y_pred_xgb.rename(columns={'lng_production':'forecast_g'}, inplace=True)

#Create MAPE
poly2_mape = mean_absolute_percentage_error(y_test['lng_production'], poly2_forecast)
poly2_mape_str = str('MAPE: %.4f' % poly2_mape)


##### POLYNOMIAL REGRESSION DEGREE=3 MODEL (forecast_h) #####
#%%
#Create model
from polyfit import PolynomRegressor, Constraints

#Set Parameters
poly3_lags = 0.59
poly3_regularization = None
poly3_interactions = False
poly3_strategy = "recursive"

poly3_regressor = PolynomRegressor(deg=3, regularization=poly3_regularization, interactions=poly3_interactions)
poly3_forecaster = make_reduction(poly3_regressor, window_length=poly3_lags, strategy=poly3_strategy)

poly3_forecaster.fit(y_train_cleaned, X=X_train) #, X=X_train
poly3_forecast = poly3_forecaster.predict(fh, X=X_test) #, X=X_test
y_pred_poly3 = pd.DataFrame(poly3_forecast).applymap('{:.2f}'.format)
y_pred_poly3['day_num'] = [i.day for i in poly3_forecast.index]
y_pred_poly3['month_num'] = [i.month for i in poly3_forecast.index]
y_pred_poly3['year_num'] = [i.year for i in poly3_forecast.index]
y_pred_poly3['date'] = y_pred_poly3['year_num'].astype(str) + '-' + y_pred_poly3['month_num'].astype(str) + '-' + y_pred_poly3['day_num'].astype(str)
y_pred_poly3['date'] = pd.DatetimeIndex(y_pred_poly3['date'], freq='D')
# Rename column to forecast_h
y_pred_xgb.rename(columns={'lng_production':'forecast_h'}, inplace=True)

#Create MAPE
poly3_mape = mean_absolute_percentage_error(y_test['lng_production'], poly3_forecast)
poly3_mape_str = str('MAPE: %.4f' % poly3_mape)

#%%
##### PLOT PREDICTION #####
fig, ax = plt.subplots(figsize=(20,8))
ax.plot(y_train, label='train')
ax.plot(y_test, label='test', color='green')
ax.plot(arimax_forecast, label='pred_arimax')
ax.plot(sarimax_forecast, label='pred_sarimax')
ax.plot(prophet_forecast, label='pred_prophet')
ax.plot(ranfor_forecast, label='pred_ranfor')
ax.plot(xgb_forecast, label='pred_xgb')
ax.plot(linreg_forecast, label='pred_linreg')
ax.plot(poly2_forecast, label='pred_poly2')
ax.plot(poly3_forecast, label='pred_poly3')
title = 'Condensate BP Tangguh Forecasting with Exogenous Variable and Cleaning Data'
ax.set_title(title)
ax.set_ylabel("Condensate")
ax.set_xlabel("Datestamp")
ax.legend(loc='best')
plt.show()

#%%
##### JOIN PREDICTION RESULT TO DATAFRAME #####
y_all_pred = pd.concat([y_pred_arimax[['forecast_a']],
                            y_pred_sarimax[['forecast_b']],
                            y_pred_prophet[['forecast_c']],
                            y_pred_ranfor[['forecast_d']],
                            y_pred_xgb[['forecast_e']],
                            y_pred_linreg[['forecast_f']],
                            y_pred_poly2[['forecast_g']],
                            y_pred_poly3[['forecast_h']]], axis=1)
y_all_pred['date'] = y_test.index.values

#%%
#CREATE MAPE TO DATAFRAME
all_mape_pred =  {'mape_forecast_a': [arimax_mape],
                'mape_forecast_b': [sarimax_mape],
                'mape_forecast_c': [prophet_mape],
                'mape_forecast_d': [ranfor_mape],
                'mape_forecast_e': [xgb_mape],
                'mape_forecast_f': [linreg_mape],
                'mape_forecast_g': [poly2_mape],
                'mape_forecast_h': [poly3_mape],
                'lng_plant' : 'BP Tangguh',
                'product' : 'LNG Production',
                'running_date' : '2022-12-07'}

all_mape_pred = pd.DataFrame(all_mape_pred)

# %%
import psycopg2

def update_trir(forecast, prod_date):
    """ insert forecasting result after last row in table """
    sql = """ UPDATE lng_condensate_daily
                SET forecast_a = %s
                WHERE prod_date = %s
                AND lng_plant = 'BP Tangguh'"""
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
