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
import plotly.express as px
from pmdarima.arima.auto import auto_arima
import seaborn as sns
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
from adtk.detector import ThresholdAD
from adtk.visualization import plot
from adtk.data import validate_series
pd.options.plotting.backend = "plotly"
from dateutil.relativedelta import *
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score

from cProfile import label
from imaplib import Time2Internaldate
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from pmdarima.arima.utils import ndiffs, nsdiffs
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
from pmdarima.arima import auto_arima, ARIMA
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.compose import make_reduction
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from polyfit import PolynomRegressor, Constraints

from sktime.forecasting.model_selection import ForecastingGridSearchCV, ForecastingRandomizedSearchCV, SlidingWindowSplitter, ExpandingWindowSplitter, SingleWindowSplitter
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError, MeanSquaredError
from sklearn.model_selection import GridSearchCV

# Model scoring for Cross Validation
mape = MeanAbsolutePercentageError(symmetric=False)
mse = MeanSquaredError()


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
def main():
    # Configure logging
    #configLogging("lpg_c4_badak.log")
    
    # Connect to database
    # Exit program if not connected to database
    logMessage("Connecting to database ...")
    conn = create_db_connection(section='postgresql_ml_lng_skk')
    if conn == None:
        exit()

    #Load data from database
    query_1 = open(os.path.join('gas_prod/sql', 'c4_badak_data_query.sql'), mode="rt").read()
    data = get_sql_data(query_1, conn)
    data['date'] = pd.DatetimeIndex(data['date'], freq='D')
    data = data.reset_index()

    data_null_cleaning = data[['date', 'lpg_c4']].copy()
    data_null_cleaning['lpg_c4_copy'] = data[['lpg_c4']].copy()
    ds_null_cleaning = 'date'
    data_null_cleaning = data_null_cleaning.set_index(ds_null_cleaning)
    s = validate_series(data_null_cleaning)

    #%%
    threshold_ad = ThresholdAD(data_null_cleaning['lpg_c4_copy'].isnull())
    anomalies = threshold_ad.detect(s)

    anomalies = anomalies.drop('lpg_c4', axis=1)

    # Create anomaly detection model
    #threshold_ad = ThresholdAD(high=high_limit2, low=low_limit1)
    #anomalies =  threshold_ad.detect(s)

    # Copy data frame of anomalies
    copy_anomalies =  anomalies.copy()
    # Rename columns
    copy_anomalies.rename(columns={'lpg_c4_copy':'anomaly'}, inplace=True)
    # Merge original dataframe with anomalies
    new_s = pd.concat([s, copy_anomalies], axis=1)

    # Get only anomalies data
    anomalies_data = new_s[new_s['anomaly'].isnull()]
    #anomalies_data.tail(100)

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
        mean_month=new_s['lpg_c4'].reset_index().query(sql).mean().values[0]
            
        # update value at specific location
        new_s.at[index,'lpg_c4'] = mean_month
            
        print(sql), print(mean_month)

    # Check if updated
    anomaly_upd = new_s[new_s['anomaly'].isnull()]

    #%%
    df_cleaned = new_s[['lpg_c4']].copy()
    #df_cleaned = df_cleaned.reset_index()
    #df_cleaned['date'] = pd.to_datetime(df_cleaned.date)
    #df_cleaned.index = pd.DatetimeIndex(df_cleaned.date)

    #ds_cleaned = 'date'
    #y_cleaned = 'lpg_c4'
    #df_cleaned = data_cleaned[[ds_cleaned, y_cleaned]]
    #df_cleaned = df_cleaned.set_index(ds_cleaned)
    #df_cleaned.index = pd.DatetimeIndex(df_cleaned.index, freq='D')
    #df_cleaned
    #Select column target
    #train_df = data_cleaned['lpg_c4']

    #%%
    #import chart_studio.plotly
    #import cufflinks as cf

    #from plotly.offline import iplot
    #cf.go_offline()
    #cf.set_config_file(offline = False, world_readable = True)

    #df.iplot(title="LPG C3 PT Badak")

    #%%
    #stationarity_check(train_df)

    #%%
    #decomposition_plot(train_df)

    #%%
    #plot_acf_pacf(train_df)

    #%%
    #from chart_studio.plotly import plot_mpl
    #from statsmodels.tsa.seasonal import seasonal_decompose
    #result = seasonal_decompose(df_cleaned['lpg_c4'], model="additive", period=365)
    #fig = result.plot()
    #plt.close()

    #%%
    # Ad-Fuller Test
    ad_test(df_cleaned['lpg_c4'])

    #%%
    # Test size
    test_size = 0.2
    # Split data
    y_train, y_test = temporal_train_test_split(df_cleaned, test_size=test_size)
    # Horizon
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    fh_int = np.arange(1, len(fh))

    #%%
    # create features (exog) from date
    #df_cleaned['month'] = [i.month for i in df_cleaned.index]
    df_cleaned['day'] = [i.day for i in df_cleaned.index]

    #%%
    # Split into train and test
    X_train, X_test = temporal_train_test_split(df_cleaned.iloc[:,1:], test_size=test_size)
    exogenous_features = ["day"]

    # %%
    ##### FORECASTING #####

    ##### ARIMAX MODEL #####
    #Set parameters
    arimax_differencing = 1
    arimax_trace = True
    arimax_error_action = "ignore"
    arimax_suppress_warnings = True
    arimax_stepwise = True
    arimax_parallel = True
    arimax_n_fits = 50

    # Create ARIMAX Model
    arimax_model = AutoARIMA(d=arimax_differencing, trace=arimax_trace, n_fits=arimax_n_fits, stepwise=arimax_stepwise, error_action=arimax_error_action, suppress_warnings=arimax_suppress_warnings)
    
    logMessage("Creating ARIMAX Model ...")
    arimax_model.fit(y_train.lpg_c4, X=X_train) #, X=X_train
    logMessage("ARIMAX Model Summary")
    logMessage(arimax_model.summary())

    logMessage("ARIMAX Model Prediction ..")
    arimax_forecast = arimax_model.predict(fh, X=X_test) #, X=X_test
    y_pred_arimax = pd.DataFrame(arimax_forecast).applymap('{:.2f}'.format)

    # Calculate model performance
    arimax_mape = mean_absolute_percentage_error(y_test.lpg_c4, arimax_forecast)
    arimax_mape_str = str('MAPE: %.4f' % arimax_mape)
    logMessage("ARIMAX Model "+arimax_mape_str)

    #Get parameter
    arimax_param = str(arimax_model.get_fitted_params()['order'])
    logMessage("Arimax Model Parameters "+arimax_param)


    #%%
    ##### SARIMAX MODEL #####

    #Set parameters
    sarimax_differencing = 1
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
    sarimax_model = AutoARIMA(d=sarimax_differencing, D=sarimax_seasonal_differencing, seasonal=sarimax_seasonal, sp=sarimax_m, trace=sarimax_trace, n_fits=sarimax_n_fits, stepwise=sarimax_stepwise, error_action=sarimax_error_action, suppress_warnings=sarimax_suppress_warnings)
    logMessage("Creating SARIMAX Model ...")
    sarimax_model.fit(y_train.lpg_c4, X=X_train) #, X=X_train
    logMessage("SARIMAX Model Summary")
    logMessage(sarimax_model.summary())

    logMessage("SARIMAX Model Prediction ..")
    sarimax_forecast = sarimax_model.predict(fh, X=X_test) #, X=X_test
    y_pred_sarimax = pd.DataFrame(sarimax_forecast).applymap('{:.2f}'.format)

    #%%
    # Calculate model performance
    sarimax_mape = mean_absolute_percentage_error(y_test.lpg_c4, sarimax_forecast)
    sarimax_mape_str = str('MAPE: %.4f' % sarimax_mape)
    logMessage("SARIMAX Model "+sarimax_mape_str)

    #Get parameters
    sarimax_param_order = str(sarimax_model.get_fitted_params()['order'])
    sarimax_param_order_seasonal = str(sarimax_model.get_fitted_params()['seasonal_order'])
    sarimax_param = sarimax_param_order + sarimax_param_order_seasonal
    logMessage("Sarimax Model Parameters "+sarimax_param)

    #%%
    ##### PROPHET MODEL #####
    # Create Prophet Parameter Grid
    prophet_param_grid = {'seasonality_mode':['additive','multiplicative']
                        ,'n_changepoints':[3, 5, 7, 11]
                        ,'seasonality_prior_scale':[1, 8] #Flexibility of the seasonality (0.01,10)
                        ,'changepoint_prior_scale':[0.002, 0.1] #Flexibility of the trend (0.001,0.5)
                        ,'daily_seasonality':[3,10]
                        ,'weekly_seasonality':[5,10]
                        ,'yearly_seasonality':[7,13]
                        }

    #Create Forecaster
    logMessage("Creating Prophet Regressor Object ....") 
    # create regressor object
    prophet_forecaster = Prophet()

    logMessage("Creating Window Splitter Prophet Model ....")   
    cv_prophet = SingleWindowSplitter(fh=fh_int)
    gscv_prophet = ForecastingGridSearchCV(prophet_forecaster, cv=cv_prophet, param_grid=prophet_param_grid, n_jobs=-1, scoring=mse)

    logMessage("Creating Prophet Model ...")
    gscv_prophet.fit(y_train.lpg_c4, X_train) #, X_train

    # Show top 10 best models based on scoring function
    gscv_prophet.cv_results_.sort_values(by='rank_test_MeanSquaredError', ascending=True)

    # Show best model parameters
    logMessage("Show Best Prophet Models ...")
    prophet_best_params = gscv_prophet.best_params_
    prophet_best_params_str = str(prophet_best_params)
    logMessage("Best Prophet Models "+prophet_best_params_str)

    logMessage("Prophet Model Prediction ...")
    prophet_forecast = gscv_prophet.best_forecaster_.predict(fh, X=X_test)#, X=X_test
    y_pred_prophet = pd.DataFrame(prophet_forecast).applymap('{:.2f}'.format)

    #Create MAPE
    prophet_mape = mean_absolute_percentage_error(y_test.lpg_c4, prophet_forecast)
    prophet_mape_str = str('MAPE: %.4f' % prophet_mape)
    logMessage("Prophet Model "+prophet_mape_str)


    #%%
    ##### RANDOM FOREST MODEL #####
    # Create Random Forest Parameter Grid
    ranfor_random_state = 0
    ranfor_criterion = "squared_error"
    ranfor_strategy = "recursive"

    ranfor_forecaster_param_grid = {"window_length": [3, 5, 7, 11], 
                                    "estimator__n_estimators": [100,200]}

    # create regressor object
    ranfor_regressor = RandomForestRegressor(random_state = ranfor_random_state, criterion = ranfor_criterion, n_jobs=-1)
    ranfor_forecaster = make_reduction(ranfor_regressor, strategy = ranfor_strategy)

    logMessage("Creating Window Splitter Random Forest Model ....")   
    cv_ranfor = SingleWindowSplitter(fh=fh_int)
    gscv_ranfor = ForecastingGridSearchCV(ranfor_forecaster, cv=cv_ranfor, param_grid=ranfor_forecaster_param_grid, n_jobs=-1, scoring=mse)

    logMessage("Creating Random Forest Model ...")
    gscv_ranfor.fit(y_train.lpg_c4, X_train) #, X_train

    # Show top 10 best models based on scoring function
    gscv_ranfor.cv_results_.sort_values(by='rank_test_MeanSquaredError', ascending=True)

    # Show best model parameters
    logMessage("Show Best Random Forest Models ...")
    ranfor_best_params = gscv_ranfor.best_params_
    ranfor_best_params_str = str(ranfor_best_params)
    logMessage("Best Random Forest Models "+ranfor_best_params_str)
    
    logMessage("Random Forest Model Prediction ...")
    ranfor_forecast = gscv_ranfor.best_forecaster_.predict(fh, X=X_test) #, X=X_test
    y_pred_ranfor = pd.DataFrame(ranfor_forecast).applymap('{:.2f}'.format)

    #Create MAPE
    ranfor_mape = mean_absolute_percentage_error(y_test['lpg_c4'], y_pred_ranfor)
    ranfor_mape_str = str('MAPE: %.4f' % ranfor_mape)
    logMessage("Random Forest Model "+ranfor_mape_str)
    

    #%%
    ##### XGBOOST MODEL #####
    # Create XGBoost Parameter Grid
    xgb_objective = 'reg:squarederror'
    xgb_strategy = "recursive"

    # Create regressor object
    xgb_forecaster_param_grid = {"window_length": [3, 5, 7, 11]
                                ,"estimator__n_estimators": [100, 200]
                                }

    xgb_regressor = XGBRegressor(objective=xgb_objective, n_jobs=-1, seed = 42)
    xgb_forecaster = make_reduction(xgb_regressor, strategy=xgb_strategy)

    cv_xgb = SingleWindowSplitter(fh=fh_int)
    gscv_xgb = ForecastingGridSearchCV(xgb_forecaster, cv=cv_xgb, param_grid=xgb_forecaster_param_grid, n_jobs=-1, scoring=mse)

    logMessage("Creating XGBoost Model ....")
    gscv_xgb.fit(y_train.lpg_c4, X=X_train) #, X_train

    # Show top 10 best models based on scoring function
    gscv_xgb.cv_results_.sort_values(by='rank_test_MeanSquaredError', ascending=True)

    # Show best model parameters
    logMessage("Show Best XGBoost Models ...")
    xgb_best_params = gscv_xgb.best_params_
    xgb_best_params_str = str(xgb_best_params)
    logMessage("Best XGBoost Models "+xgb_best_params_str)
    
    logMessage("XGBoost Model Prediction ...")
    xgb_forecast = gscv_xgb.best_forecaster_.predict(fh, X=X_test) #, X=X_test
    y_pred_xgb = pd.DataFrame(xgb_forecast).applymap('{:.2f}'.format)

    #Create MAPE
    xgb_mape = mean_absolute_percentage_error(y_test['lpg_c4'], y_pred_xgb)
    xgb_mape_str = str('MAPE: %.4f' % xgb_mape)
    logMessage("XGBoost Model "+xgb_mape_str)


    #%%
    ##### LINEAR REGRESSION MODEL #####
    # Create Linear Regression Parameter Grid
    linreg_strategy = "recursive"

    linreg_forecaster_param_grid = {"window_length": [3, 5, 7, 11]}

    # Create regressor object
    linreg_regressor = LinearRegression(normalize=True, n_jobs=-1)
    linreg_forecaster = make_reduction(linreg_regressor, strategy=linreg_strategy)

    cv_linreg = SingleWindowSplitter(fh=fh_int)
    gscv_linreg = ForecastingGridSearchCV(linreg_forecaster, cv=cv_linreg, param_grid=linreg_forecaster_param_grid, n_jobs=-1, scoring=mse)

    logMessage("Creating Linear Regression Model ...")
    gscv_linreg.fit(y_train.lpg_c4, X=X_train) #, X=X_train
    
    # Show top 10 best models based on scoring function
    gscv_linreg.cv_results_.sort_values(by='rank_test_MeanSquaredError', ascending=True)

    # Show best model parameters
    logMessage("Show Best Linear Regression Models ...")
    linreg_best_params = gscv_linreg.best_params_
    linreg_best_params_str = str(linreg_best_params)
    logMessage("Best Linear Regression Models "+linreg_best_params_str)
    
    logMessage("Linear Regression Model Prediction ...")
    linreg_forecast = gscv_linreg.best_forecaster_.predict(fh, X=X_test) #, X=X_test
    y_pred_linreg = pd.DataFrame(linreg_forecast).applymap('{:.2f}'.format)

    #Create MAPE
    linreg_mape = mean_absolute_percentage_error(y_test['lpg_c4'], y_pred_linreg)
    linreg_mape_str = str('MAPE: %.4f' % linreg_mape)
    logMessage("Linear Regression Model "+linreg_mape_str)

    #%%
    ##### POLYNOMIAL REGRESSION DEGREE=2 #####
    # Create Polynomial Regression Degree=2 Parameter Grid
    poly2_regularization = None
    poly2_interactions = False
    poly2_strategy = "recursive"

    poly2_forecaster_param_grid = {"window_length": [0.8, 1, 2, 3, 4]}

    # Create regressor object
    poly2_regressor = PolynomRegressor(deg=2, regularization=poly2_regularization, interactions=poly2_interactions)
    poly2_forecaster = make_reduction(poly2_regressor, strategy=poly2_strategy)

    cv_poly2 = SingleWindowSplitter(fh=fh_int)
    gscv_poly2 = ForecastingGridSearchCV(poly2_forecaster, cv=cv_poly2, param_grid=poly2_forecaster_param_grid, scoring=mse, error_score='raise')

    logMessage("Creating Polynomial Regression Orde 2 Model ...")
    gscv_poly2.fit(y_train.lpg_c4, X=X_train) #, X=X_train
    
    # Show top 10 best models based on scoring function
    gscv_poly2.cv_results_.sort_values(by='rank_test_MeanSquaredError', ascending=True)

    # Show best model parameters
    logMessage("Show Best Polynomial Regression Degree=2 Models ...")
    poly2_best_params = gscv_poly2.best_params_
    poly2_best_params_str = str(poly2_best_params)
    logMessage("Best Polynomial Regression Degree=2 Models "+poly2_best_params_str)
    
    logMessage("Polynomial Regression Degree=2 Model Prediction ...")
    poly2_forecast = gscv_poly2.best_forecaster_.predict(fh, X=X_test) #, X=X_test
    y_pred_poly2 = pd.DataFrame(poly2_forecast).applymap('{:.2f}'.format)

    #Create MAPE
    poly2_mape = mean_absolute_percentage_error(y_test['lpg_c4'], y_pred_poly2)
    poly2_mape_str = str('MAPE: %.4f' % poly2_mape)
    logMessage("Polynomial Regression Degree=2 Model "+poly2_mape_str)

    #%%
    ##### POLYNOMIAL REGRESSION DEGREE=3 #####
    # Create Polynomial Regression Degree=3 Parameter Grid
    poly3_regularization = None
    poly3_interactions = False
    poly3_strategy = "recursive"

    poly3_forecaster_param_grid = {"window_length": [0.8, 1, 2, 3, 4]}

    # Create regressor object
    poly3_regressor = PolynomRegressor(deg=3, regularization=poly3_regularization, interactions=poly3_interactions)
    poly3_forecaster = make_reduction(poly3_regressor, strategy=poly3_strategy)

    cv_poly3 = SingleWindowSplitter(fh=fh_int)
    gscv_poly3 = ForecastingGridSearchCV(poly3_forecaster, cv=cv_poly3, param_grid=poly3_forecaster_param_grid, n_jobs=-1, scoring=mse)

    logMessage("Creating Polynomial Regression Orde 3 Model ...")
    gscv_poly3.fit(y_train.lpg_c4) #, X=X_train
    
    # Show top 10 best models based on scoring function
    gscv_poly3.cv_results_.sort_values(by='rank_test_MeanSquaredError', ascending=True)

    # Show best model parameters
    logMessage("Show Best Polynomial Regression Degree=3 Models ...")
    poly3_best_params = gscv_poly3.best_params_
    poly3_best_params_str = str(poly3_best_params)
    logMessage("Best Polynomial Regression Degree=3 Models "+poly3_best_params_str)
    
    logMessage("Polynomial Regression Degree=3 Model Prediction ...")
    poly3_forecast = gscv_poly3.best_forecaster_.predict(fh) #, X=X_test
    y_pred_poly3 = pd.DataFrame(poly3_forecast).applymap('{:.2f}'.format)

    #Create MAPE
    poly3_mape = mean_absolute_percentage_error(y_test['lpg_c4'], y_pred_poly3)
    poly3_mape_str = str('MAPE: %.4f' % poly3_mape)
    logMessage("Polynomial Regression Degree=3 Model "+poly3_mape_str)

    #%%
    #CREATE DATAFRAME MAPE
    logMessage("Creating all model mape result data frame ...")
    all_mape_pred =  {'mape_forecast_a': [arimax_mape],
                    'mape_forecast_b': [sarimax_mape],
                    'mape_forecast_c': [prophet_mape],
                    'mape_forecast_d': [ranfor_mape],
                    'mape_forecast_e': [xgb_mape],
                    'mape_forecast_f': [linreg_mape],
                    'mape_forecast_g': [poly2_mape],
                    'mape_forecast_h': [poly3_mape],
                    'lng_plant' : 'PT Badak',
                    'product' : 'LPG C4'}

    all_mape_pred = pd.DataFrame(all_mape_pred)

    #%%    
    #CREATE PARAMETERS TO DATAFRAME
    logMessage("Creating all model params result data frame ...")
    all_model_param =  {'model_param_a': [arimax_param],
                        'model_param_b': [sarimax_param],
                        'model_param_c': [prophet_best_params_str],
                        'model_param_d': [ranfor_best_params_str],
                        'model_param_e': [xgb_best_params_str],
                        'model_param_f': [linreg_best_params_str],
                        'model_param_g': [poly2_best_params_str],
                        'model_param_h': [poly3_best_params_str],
                        'lng_plant' : 'PT Badak',
                        'product' : 'LPG C4'}

    all_model_param = pd.DataFrame(all_model_param)

#%%    
    # Save mape result to database
    logMessage("Updating MAPE result to database ...")
    total_updated_rows = insert_mape(conn, all_mape_pred)
    logMessage("Updated rows: {}".format(total_updated_rows))
    
    # Save param result to database
    logMessage("Updating Model Parameter result to database ...")
    total_updated_rows = insert_param(conn, all_model_param)
    logMessage("Updated rows: {}".format(total_updated_rows))
    
    print("Done")

# %%
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

def update_mape_value(conn, mape_forecast_a, mape_forecast_b, mape_forecast_c, 
                        mape_forecast_d, mape_forecast_e, mape_forecast_f, mape_forecast_g, mape_forecast_h,
                        lng_plant, product):
    
    date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    created_by = 'PYTHON'
    
    """ insert mape result after last row in table """
    sql = """ INSERT INTO lng_analytics_mape
                    (lng_plant,
                    product,
                    running_date,
                    mape_forecast_a,
                    mape_forecast_b,
                    mape_forecast_c,
                    mape_forecast_d,
                    mape_forecast_e,
                    mape_forecast_f,
                    mape_forecast_g,
                    mape_forecast_h,
                    created_by)
                    VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
          """
                
    #conn = None
    updated_rows = 0
    try:
        # create a new cursor
        cur = conn.cursor()
        # execute the UPDATE  statement
        cur.execute(sql, (lng_plant, product, date_now, mape_forecast_a, mape_forecast_b, mape_forecast_c, mape_forecast_d, mape_forecast_e, mape_forecast_f, mape_forecast_g, mape_forecast_h,
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