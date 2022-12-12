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

from humanfriendly import format_timespan
from tokenize import Ignore
from datetime import datetime
from tracemalloc import start
from pmdarima.arima.auto import auto_arima
import matplotlib.pyplot as plt
import matplotlib as mpl

from connection import config, retrieve_data, create_db_connection, get_sql_data
from utils import configLogging, logMessage, ad_test

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import pmdarima as pm
from pmdarima import model_selection 
from pmdarima.arima import auto_arima
#import mlflow
plt.style.use('fivethirtyeight')

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
def main():
    # Configure logging
    configLogging("feed_gas_tangguh.log")
    
    # Connect to database
    # Exit program if not connected to database
    logMessage("Connecting to database ...")
    conn = create_db_connection(section='postgresql_ml_lng_skk')
    if conn == None:
        exit()
        
#Load data from database
query_1 = open("c3_badak_data_query.sql", mode="rt").read()
data = get_sql_data(query_1, conn)
data['date'] = pd.DatetimeIndex(data['date'], freq='D')
data = data.reset_index()

#%%
ds = 'date'
y = 'lpg_c3' #Choose the column target
df = data[[ds,y]]
df = df.set_index(ds)
df.index = pd.DatetimeIndex(df.index, freq='D')

#Select column target
train_df = df['lpg_c3']

#%%
import chart_studio.plotly
import cufflinks as cf

from plotly.offline import iplot
cf.go_offline()
cf.set_config_file(offline = False, world_readable = True)

#df.iplot(title="LPG C3 PT Badak")

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
# Split data
y_train, y_test = temporal_train_test_split(df, test_size=test_size)
# Horizon
fh = ForecastingHorizon(y_test.index, is_relative=False)

#%%
# create features (exog) from date
df['month'] = [i.month for i in df.index]
df['day'] = [i.day for i in df.index]
#df['day_of_year'] = [i.dayofyear for i in df.index]
#df['week_of_year'] = [i.weekofyear for i in df.index]
df.tail(20)

#%%
# Split into train and test
X_train, X_test = temporal_train_test_split(df.iloc[:,1:], test_size=test_size)
X_train

#%%
exogenous_features = ["month", "day"]
exogenous_features

# %%
##### FORECASTING #####

##### ARIMAX MODEL #####
from pmdarima.arima.utils import ndiffs, nsdiffs
import statsmodels.api as sm
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.arima import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score

#Set parameters
arimax_differencing = 0
arimax_trace = True
arimax_error_action = "ignore"
arimax_suppress_warnings = True
arimax_stepwise = True
arimax_parallel = True
arimax_n_fits = 50

# Create ARIMAX Model
#arimax_model = auto_arima(train_df, exogenous=future_exog, d=arimax_differencing, trace=arimax_trace, error_action=arimax_error_action, suppress_warnings=arimax_suppress_warnings)
arimax_model = AutoARIMA(d=arimax_differencing, trace=arimax_trace, n_fits=arimax_n_fits, stepwise=arimax_stepwise, error_action=arimax_error_action, suppress_warnings=arimax_suppress_warnings)
arimax_model.fit(y_train.lpg_c3, X=X_train) #exogenous=train_exog
arimax_model.summary()

arimax_forecast = arimax_model.predict(fh, X=X_test)
y_pred_arimax = pd.DataFrame(arimax_forecast).applymap('{:.2f}'.format)
y_pred_arimax['day_num'] = [i.day for i in arimax_forecast.index]
y_pred_arimax['month_num'] = [i.month for i in arimax_forecast.index]
y_pred_arimax['year_num'] = [i.year for i in arimax_forecast.index]
y_pred_arimax['date'] = y_pred_arimax['year_num'].astype(str) + '-' + y_pred_arimax['month_num'].astype(str) + '-' + y_pred_arimax['day_num'].astype(str)
y_pred_arimax['date'] = pd.DatetimeIndex(y_pred_arimax['date'], freq='D')
#Rename colum 0
y_pred_arimax.rename(columns={'lpg_c3':'forecast_a'}, inplace=True)

# Calculate model performance
arimax_r2 = r2_score(y_test.lpg_c3, arimax_forecast)
arimax_r2 = str('MAPE: %.4f' % arimax_r2)

#Get parameter
arimax_param = str(arimax_model.get_fitted_params()['order'])


#%%
##### SARIMAX MODEL #####

#Set parameters
sarimax_differencing = 0
sarimax_seasonal_differencing = 1
sarimax_seasonal = True
sarimax_m = 12
sarimax_trace = True
sarimax_error_action = "ignore"
sarimax_suppress_warnings = True
sarimax_stepwise = True
sarimax_parallel = True
sarimax_n_fits = 50

# Create SARIMA Model
#sarimax_model = auto_arima(train_df, exogenous=future_exog, d=sarimax_differencing, D=sarimax_seasonal_differencing, seasonal=sarimax_seasonal, m=sarimax_m, trace=sarimax_trace, error_action=sarimax_error_action, suppress_warnings=sarimax_suppress_warnings)
sarimax_model = AutoARIMA(d=sarimax_differencing, D=sarimax_seasonal_differencing, sp=sarimax_m, trace=sarimax_trace, n_fits=sarimax_n_fits, stepwise=sarimax_stepwise, error_action=sarimax_error_action, suppress_warnings=sarimax_suppress_warnings)
sarimax_model.fit(y_train.lpg_c3, X=X_train) #exogenous=train_exog
sarimax_model.summary()

sarimax_forecast = sarimax_model.predict(fh, X=X_test)
y_pred_sarimax = pd.DataFrame(sarimax_forecast).applymap('{:.2f}'.format)
y_pred_sarimax['day_num'] = [i.day for i in sarimax_forecast.index]
y_pred_sarimax['month_num'] = [i.month for i in sarimax_forecast.index]
y_pred_sarimax['year_num'] = [i.year for i in sarimax_forecast.index]
y_pred_sarimax['date'] = y_pred_sarimax['year_num'].astype(str) + '-' + y_pred_sarimax['month_num'].astype(str) + '-' + y_pred_sarimax['day_num'].astype(str)
y_pred_sarimax['date'] = pd.DatetimeIndex(y_pred_sarimax['date'], freq='D')
#Rename colum 0
y_pred_sarimax.rename(columns={'lpg_c3':'forecast_b'}, inplace=True)

#%%
# Calculate model performance
sarimax_r2 = r2_score(y_test.lpg_c3, sarimax_forecast)
sarimax_r2 = str('R2: %.4f' % sarimax_r2)

#Get parameters
sarimax_param_order = str(sarimax_model.get_fitted_params()['order'])
sarimax_param_order_seasonal = str(sarimax_model.get_fitted_params()['seasonal_order'])
sarimax_param = sarimax_param_order + sarimax_param_order_seasonal
    

#%%
##### PROPHET MODEL #####
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.compose import make_reduction

#Set parameters
prophet_seasonality_mode = 'multiplicative'
prophet_n_changepoints = 1
prophet_seasonality_prior_scale = 10
prophet_changepoint_prior_scale = 0.001
prophet_holidays_prior_scale = 2
prophet_daily_seasonality = 3
prophet_weekly_seasonality = 10
prophet_yearly_seasonality = 7

#Create Forecaster
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

prophet_forecaster.fit(y_train.lpg_c3, X=X_train) #, X_train
prophet_forecast = prophet_forecaster.predict(fh, X=X_test) #, X=X_test
y_pred_prophet = pd.DataFrame(prophet_forecast).applymap('{:.2f}'.format)
y_pred_prophet['day_num'] = [i.day for i in prophet_forecast.index]
y_pred_prophet['month_num'] = [i.month for i in prophet_forecast.index]
y_pred_prophet['year_num'] = [i.year for i in prophet_forecast.index]
y_pred_prophet['date'] = y_pred_prophet['year_num'].astype(str) + '-' + y_pred_prophet['month_num'].astype(str) + '-' + y_pred_prophet['day_num'].astype(str)
y_pred_prophet['date'] = pd.DatetimeIndex(y_pred_prophet['date'], freq='D')
#Rename colum 0
y_pred_prophet.rename(columns={'lpg_c3':'forecast_c'}, inplace=True)

# Calculate model performance
prophet_r2 = r2_score(y_test.lpg_c3, prophet_forecast)
prophet_r2 = str('R2: %.4f' % prophet_r2)

#Get parameters
prophet_param_seasonality_mode = str(prophet_forecaster.get_params()['seasonality_mode'])
prophet_param_n_changepoints = str(prophet_forecaster.get_params()['n_changepoints'])
prophet_param_seasonality_prior_scale = str(prophet_forecaster.get_params()['seasonality_prior_scale'])
prophet_param_changepoint_prior_scale = str(prophet_forecaster.get_params()['changepoint_prior_scale'])
prophet_param_holidays_prior_scale = str(prophet_forecaster.get_params()['holidays_prior_scale'])
prophet_param_daily_seasonality = str(prophet_forecaster.get_params()['daily_seasonality'])
prophet_param_weekly_seasonality = str(prophet_forecaster.get_params()['weekly_seasonality'])
prophet_param_yearly_seasonality = str(prophet_forecaster.get_params()['yearly_seasonality'])
prophet_param = prophet_param_seasonality_mode + ', ' + prophet_param_n_changepoints + ', ' + prophet_param_seasonality_prior_scale + ', ' + prophet_param_changepoint_prior_scale + ', ' + prophet_param_holidays_prior_scale + ', ' + prophet_param_daily_seasonality + ', ' + prophet_param_weekly_seasonality + ', ' + prophet_param_yearly_seasonality


#%%
##### RANDOM FOREST MODEL #####
from sklearn.ensemble import RandomForestRegressor

#Set parameters
ranfor_lags = 0.33
ranfor_n_estimators = 100
ranfor_random_state = 0
ranfor_criterion = "squared_error"
ranfor_strategy = "recursive"

# create regressor object
ranfor_regressor = RandomForestRegressor(n_estimators = ranfor_n_estimators, random_state=ranfor_random_state, criterion=ranfor_criterion)
ranfor_forecaster = make_reduction(ranfor_regressor, window_length=ranfor_lags, strategy=ranfor_strategy)

ranfor_forecaster.fit(y_train.lpg_c3, X=X_train) #, X_train
ranfor_forecast = ranfor_forecaster.predict(fh, X=X_test) #, X=X_test
y_pred_ranfor = pd.DataFrame(ranfor_forecast).applymap('{:.2f}'.format)
y_pred_ranfor['day_num'] = [i.day for i in ranfor_forecast.index]
y_pred_ranfor['month_num'] = [i.month for i in ranfor_forecast.index]
y_pred_ranfor['year_num'] = [i.year for i in ranfor_forecast.index]
y_pred_ranfor['date'] = y_pred_ranfor['year_num'].astype(str) + '-' + y_pred_ranfor['month_num'].astype(str) + '-' + y_pred_ranfor['day_num'].astype(str)
y_pred_ranfor['date'] = pd.DatetimeIndex(y_pred_ranfor['date'], freq='D')
#Rename colum 0
y_pred_ranfor.rename(columns={'lpg_c3':'forecast_d'}, inplace=True)

# Calculate model performance
ranfor_r2 = r2_score(y_test.lpg_c3, ranfor_forecast)
ranfor_r2 = str('R2: %.4f' % ranfor_r2)

#Get Parameters
ranfor_param_estimator = str(ranfor_forecaster.get_fitted_params()['estimator'])
ranfor_param_lags = str(ranfor_forecaster.get_fitted_params()['window_length'])
ranfor_param = ranfor_param_estimator + ', ' + ranfor_param_lags


#%%
##### XGBOOST MODEL #####
from xgboost import XGBRegressor

#Set parameters
xgb_lags = 0.4
xgb_objective = 'reg:squarederror'
xgb_strategy = "recursive"

# Create regressor object
xgb_regressor = XGBRegressor(objective=xgb_objective)
xgb_forecaster = make_reduction(xgb_regressor, window_length=xgb_lags, strategy=xgb_strategy)

xgb_forecaster.fit(y_train.lpg_c3, X=X_train) #, X_train
xgb_forecast = xgb_forecaster.predict(fh, X=X_test) #, X=X_test
y_pred_xgb = pd.DataFrame(xgb_forecast).applymap('{:.2f}'.format)
y_pred_xgb['day_num'] = [i.day for i in xgb_forecast.index]
y_pred_xgb['month_num'] = [i.month for i in xgb_forecast.index]
y_pred_xgb['year_num'] = [i.year for i in xgb_forecast.index]
y_pred_xgb['date'] = y_pred_xgb['year_num'].astype(str) + '-' + y_pred_xgb['month_num'].astype(str) + '-' + y_pred_xgb['day_num'].astype(str)
y_pred_xgb['date'] = pd.DatetimeIndex(y_pred_xgb['date'], freq='D')

#Rename colum 0
y_pred_xgb.rename(columns={'lpg_c3':'forecast_e'}, inplace=True)

# Calculate model performance
xgb_r2 = r2_score(y_test.lpg_c3, xgb_forecast)
xgb_r2 = str('R2: %.4f' % xgb_r2)

#Get Parameters
xgb_param_lags = str(xgb_forecaster.get_params()['window_length'])
xgb_param_objective = str(xgb_forecaster.get_params()['estimator__objective'])
xgb_param = xgb_param_lags + ', ' + xgb_param_objective


#%%
##### LINEAR REGRESSION MODEL #####
from sklearn.linear_model import LinearRegression

#Set parameters
linreg_lags = 50
linreg_normalize = True
linreg_strategy = "recursive"

# Create regressor object
linreg_regressor = LinearRegression(normalize=linreg_normalize)
linreg_forecaster = make_reduction(linreg_regressor, window_length=linreg_lags, strategy=linreg_strategy)
linreg_forecaster.fit(y_train.lpg_c3, X=X_train)

# Create forecasting
linreg_forecast = linreg_forecaster.predict(fh, X=X_test) #, X=X_test
y_pred_linreg = pd.DataFrame(linreg_forecast).applymap('{:.2f}'.format)
y_pred_linreg['day_num'] = [i.day for i in linreg_forecast.index]
y_pred_linreg['month_num'] = [i.month for i in linreg_forecast.index]
y_pred_linreg['year_num'] = [i.year for i in linreg_forecast.index]
y_pred_linreg['date'] = y_pred_linreg['year_num'].astype(str) + '-' + y_pred_linreg['month_num'].astype(str) + '-' + y_pred_linreg['day_num'].astype(str)
y_pred_linreg['date'] = pd.DatetimeIndex(y_pred_linreg['date'], freq='D')

#Rename colum 0
y_pred_linreg.rename(columns={'lpg_c3':'forecast_f'}, inplace=True)

# Calculate model performance
linreg_r2 = r2_score(y_test.lpg_c3, linreg_forecast)
linreg_r2 = str('R2: %.4f' % linreg_r2)

#Get parameters
linreg_param_estimator = str(linreg_forecaster.get_fitted_params()['estimator'])
linreg_param_lags = str(linreg_forecaster.get_fitted_params()['window_length'])
linreg_param = linreg_param_estimator + ', ' + linreg_param_lags


#%%
##### POLYNOMIAL REGRESSION DEGREE=2 #####
from polyfit import PolynomRegressor, Constraints

#Set parameters
poly2_lags = 63
poly2_regularization = None
poly2_interactions = False
poly2_strategy = "recursive"

# Create regressor object
poly2_regressor = PolynomRegressor(deg=2, regularization=poly2_regularization, interactions=poly2_interactions)
poly2_forecaster = make_reduction(poly2_regressor, window_length=poly2_lags, strategy=poly2_strategy) #WL=0.9 (degree 2), WL=0.7 (degree 3)
poly2_forecaster.fit(y_train.lpg_c3, X=X_train) #, X=X_train

# Create forecasting
poly2_forecast = poly2_forecaster.predict(fh, X=X_test) #, X=X_test
y_pred_poly2 = pd.DataFrame(poly2_forecast).applymap('{:.2f}'.format)
y_pred_poly2['day_num'] = [i.day for i in poly2_forecast.index]
y_pred_poly2['month_num'] = [i.month for i in poly2_forecast.index]
y_pred_poly2['year_num'] = [i.year for i in poly2_forecast.index]
y_pred_poly2['date'] = y_pred_poly2['year_num'].astype(str) + '-' + y_pred_poly2['month_num'].astype(str) + '-' + y_pred_poly2['day_num'].astype(str)
y_pred_poly2['date'] = pd.DatetimeIndex(y_pred_poly2['date'], freq='D')

#Rename colum 0
y_pred_poly2.rename(columns={'lpg_c3':'forecast_g'}, inplace=True)

# Calculate model performance
poly2_r2 = r2_score(y_test.lpg_c3, poly2_forecast)
poly2_r2 = str('R2: %.4f' % poly2_r2)

#Get parameters
poly2_param_estimator = str(poly2_forecaster.get_fitted_params()['estimator'])
poly2_param_lags = str(poly2_forecaster.get_fitted_params()['window_length'])
poly2_param = poly2_param_estimator + ', ' + poly2_param_lags


#%%
##### POLYNOMIAL REGRESSION DEGREE=3 #####
from polyfit import PolynomRegressor, Constraints

#Set parameters
poly3_lags = 32
poly3_regularization = None
poly3_interactions = False
poly3_strategy = "recursive"

# Create regressor object
poly3_regressor = PolynomRegressor(deg=3, regularization=poly3_regularization, interactions=poly3_interactions)
poly3_forecaster = make_reduction(poly3_regressor, window_length=poly3_lags, strategy=poly3_strategy) #WL=0.9 (degree 2), WL=0.7 (degree 3)
poly3_forecaster.fit(y_train.lpg_c3, X=X_train) #, X=X_train

# Create forecasting
poly3_forecast = poly3_forecaster.predict(fh, X=X_test) #, X=X_test
y_pred_poly3 = pd.DataFrame(poly3_forecast).applymap('{:.2f}'.format)
y_pred_poly3['day_num'] = [i.day for i in poly3_forecast.index]
y_pred_poly3['month_num'] = [i.month for i in poly3_forecast.index]
y_pred_poly3['year_num'] = [i.year for i in poly3_forecast.index]
y_pred_poly3['date'] = y_pred_poly3['year_num'].astype(str) + '-' + y_pred_poly3['month_num'].astype(str) + '-' + y_pred_poly3['day_num'].astype(str)
y_pred_poly3['date'] = pd.DatetimeIndex(y_pred_poly3['date'], freq='D')

#Rename colum 0
y_pred_poly3.rename(columns={'lpg_c3':'forecast_h'}, inplace=True)

# Calculate model performance
poly3_r2 = r2_score(y_test.lpg_c3, poly3_forecast)
poly3_r2 = str('R2: %.4f' % poly3_r2)

#Get parameters
poly3_param_estimator = str(poly3_forecaster.get_fitted_params()['estimator'])
poly3_param_lags = str(poly3_forecaster.get_fitted_params()['window_length'])
poly3_param = poly3_param_estimator + ', ' + poly3_param_lags
poly3_r2

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
title = 'LPG C3 PT Badak Forecasting (Insample) with Exogenous Date-Month'
ax.set_title(title)
ax.set_ylabel("Feed Gas")
ax.set_xlabel("Datestamp")
ax.legend(loc='best')
plt.close()
    
    ##### JOIN PREDICTION RESULT TO DATAFRAME #####
#    logMessage("Creating all model prediction result data frame ...")
y_all_pred = pd.concat([y_pred_arimax[['forecast_a']],
                            y_pred_sarimax[['forecast_b']],
                            y_pred_prophet[['forecast_c']],
                            y_pred_ranfor[['forecast_d']],
                            y_pred_xgb[['forecast_e']],
                            y_pred_linreg[['forecast_f']],
                            y_pred_poly2[['forecast_g']],
                            y_pred_poly3[['forecast_h']]], axis=1)
y_all_pred['date'] = y_test.index.values

    #CREATE DATAFRAME MAPE
#    logMessage("Creating all model mape result data frame ...")
all_r2_pred =  {'mape_forecast_a': [arimax_r2],
                'mape_forecast_b': [sarimax_r2],
                'mape_forecast_c': [prophet_r2],
                'mape_forecast_d': [ranfor_r2],
                'mape_forecast_e': [xgb_r2],
                'mape_forecast_f': [linreg_r2],
                'mape_forecast_g': [poly2_r2],
                'mape_forecast_h': [poly3_r2],
                'lng_plant' : 'PT Badak',
                'product' : 'LPG C3'}

all_r2_pred = pd.DataFrame(all_r2_pred)
    
    #CREATE MAPE TO DATAFRAME
#    logMessage("Creating all model params result data frame ...")
all_model_param =  {'model_param_a': [arimax_param],
                    'model_param_b': [sarimax_param],
                    'model_param_c': [prophet_param],
                    'model_param_d': [ranfor_param],
                    'model_param_e': [xgb_param],
                    'model_param_f': [linreg_param],
                    'model_param_g': [poly2_param],
                    'model_param_h': [poly3_param],
                    'lng_plant' : 'PT Badak',
                    'product' : 'LPG C3'}

all_model_param = pd.DataFrame(all_model_param)
#%%    
    # Save forecast result to database
#    logMessage("Updating forecast result to database ...")
#    total_updated_rows = insert_forecast(conn, y_all_pred)
#    logMessage("Updated rows: {}".format(total_updated_rows))
    
    # Save mape result to database
#    logMessage("Updating MAPE result to database ...")
#    total_updated_rows = insert_mape(conn, all_mape_pred)
#    logMessage("Updated rows: {}".format(total_updated_rows))
    
    # Save mape result to database
#    logMessage("Updating Model Parameter result to database ...")
#    total_updated_rows = insert_param(conn, all_model_param)
#    logMessage("Updated rows: {}".format(total_updated_rows))
    
#    print("Done")

# %%
def insert_forecast(conn, y_pred):
    total_updated_rows = 0
    for index, row in y_pred.iterrows():
        prod_date = str(index) #row['date']
        forecast_a, forecast_b, forecast_c, forecast_d, forecast_e, forecast_f, forecast_g, forecast_h = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]
        
        #sql = f'UPDATE trir_monthly_test SET forecast_a = {} WHERE year_num = {} AND month_num = {}'.format(forecast, year_num, month_num)
        updated_rows = update_value(conn, forecast_a, forecast_b, forecast_c, forecast_d, forecast_e, forecast_f, forecast_g, forecast_h, prod_date)
        total_updated_rows = total_updated_rows + updated_rows 
        
    return total_updated_rows

def insert_mape(conn, all_mape_pred):
    total_updated_rows = 0
    for index, row in all_mape_pred.iterrows():
        lng_plant = row['lng_plant']
        product = row['product']
        mape_forecast_a, mape_forecast_b, mape_forecast_c, mape_forecast_d, mape_forecast_e, mape_forecast_f, mape_forecast_g, mape_forecast_h = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]
        
        #sql = f'UPDATE trir_monthly_test SET forecast_a = {} WHERE year_num = {} AND month_num = {}'.format(forecast, year_num, month_num)
        updated_rows = update_mape_value(conn, mape_forecast_a, mape_forecast_b, mape_forecast_c, mape_forecast_d, mape_forecast_e, mape_forecast_f, mape_forecast_g, mape_forecast_h , lng_plant, product)
        total_updated_rows = total_updated_rows + updated_rows 
        
    return total_updated_rows

def insert_param(conn, all_model_param):
    total_updated_rows = 0
    for index, row in all_model_param.iterrows():
        lng_plant = row['lng_plant']
        product = row['product']
        model_param_a, model_param_b, model_param_c, model_param_d, model_param_e, model_param_f, model_param_g, model_param_h = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]
        
        #sql = f'UPDATE trir_monthly_test SET forecast_a = {} WHERE year_num = {} AND month_num = {}'.format(forecast, year_num, month_num)
        updated_rows = update_param_value(conn, model_param_a, model_param_b, model_param_c, model_param_d, model_param_e, model_param_f, model_param_g, model_param_h , lng_plant, product)
        total_updated_rows = total_updated_rows + updated_rows 
        
    return total_updated_rows

def update_value(conn, forecast_a, forecast_b, forecast_c, 
                        forecast_d, forecast_e, forecast_f, forecast_g, forecast_h, prod_date):
    
    date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    created_by = 'PYTHON'
    
    """ insert forecasting result after last row in table """
    sql = """ UPDATE lpg_c3_daily
                SET forecast_a = %s, 
                    forecast_b = %s, 
                    forecast_c = %s, 
                    forecast_d = %s, 
                    forecast_e = %s, 
                    forecast_f = %s, 
                    forecast_g = %s, 
                    forecast_h = %s, 
                    updated_at = %s, 
                    updated_by = %s
                WHERE prod_date = %s"""
    #conn = None
    updated_rows = 0
    try:
        # create a new cursor
        cur = conn.cursor()
        # execute the UPDATE  statement
        cur.execute(sql, (forecast_a, forecast_b, forecast_c, forecast_d, forecast_e, forecast_f, forecast_g, forecast_h, date_now, created_by, prod_date))
        # get the number of updated rows
        updated_rows = cur.rowcount
        # Commit the changes to the database
        conn.commit()
        # Close cursor
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(error)

    return updated_rows

def update_mape_value(conn, mape_forecast_a, mape_forecast_b, mape_forecast_c, 
                        mape_forecast_d, mape_forecast_e, mape_forecast_f, mape_forecast_g, mape_forecast_h,
                        lng_plant, product):
    
    date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    created_by = 'PYTHON'
    
    """ insert mape result after last row in table """
    sql = """ UPDATE lng_analytics_mape
                SET mape_forecast_a = %s, 
                    mape_forecast_b = %s, 
                    mape_forecast_c = %s, 
                    mape_forecast_d = %s, 
                    mape_forecast_e = %s, 
                    mape_forecast_f = %s,
                    mape_forecast_g = %s,
                    mape_forecast_h = %s,
                    updated_at = %s, 
                    updated_by = %s
                WHERE lng_plant = %s
                AND product = %s"""
    #conn = None
    updated_rows = 0
    try:
        # create a new cursor
        cur = conn.cursor()
        # execute the UPDATE  statement
        cur.execute(sql, (mape_forecast_a, mape_forecast_b, mape_forecast_c, mape_forecast_d, mape_forecast_e, mape_forecast_f, mape_forecast_g, mape_forecast_h,
                          date_now, created_by, lng_plant, product))
        # get the number of updated rows
        updated_rows = cur.rowcount
        # Commit the changes to the database
        conn.commit()
        # Close cursor
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        logMessage(error)

    return updated_rows

def update_param_value(conn, model_param_a, model_param_b, model_param_c, 
                        model_param_d, model_param_e, model_param_f, model_param_g, model_param_h,
                        lng_plant, product):
    
    date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    created_by = 'PYTHON'
    
    """ insert mape result after last row in table """
    sql = """ INSERT INTO lng_analytics_model_param
                    (lng_plant,
                    product,
                    running_date,
                    model_param_a,
                    model_param_b,
                    model_param_c,
                    model_param_d,
                    model_param_e,
                    model_param_f,
                    model_param_g,
                    model_param_h,
                    created_by)
                    VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
          """
    
    #conn = None
    updated_rows = 0
    try:
        # create a new cursor
        cur = conn.cursor()
        # execute the UPDATE  statement
        cur.execute(sql, (lng_plant, product, date_now, model_param_a, model_param_b, model_param_c, model_param_d, model_param_e, model_param_f, model_param_g, model_param_h,
                          created_by))
        # get the number of updated rows
        updated_rows = cur.rowcount
        # Commit the changes to the database
        conn.commit()
        # Close cursor
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        logMessage(error)

    return updated_rows