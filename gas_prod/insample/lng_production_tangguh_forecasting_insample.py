###### LNG PRODUCTION BP TANGGUH FORECASTING INSAMPLE ######
# This python script is used to perform forecasting on testing data from each method.
# Data source from the SKK Migas (10.6.7.74) database in the lng_production_daily table with lng_plant = 'BP Tangguh'.

##### METHODS FOR TIME SERIES FORECASTING #####
# There are many methods that we can use for this forecasting, such as ARIMAX, SARIMAX, PROPHET, RANDOM FOREST, XGBOOST, LINEAR REGRESSION, POLYNOMIAL REGRESSION DEGREE 2, POLYNOMIAL REGRESSION DEGREE 3.

##### FLOW PROCESS OF THIS ALGORITHM #####
# 1. Import the required packages and defining functions that maybe used.
# 2. Import data from database.
# 3. EDA process (Search null values in column wpnb_gas and lng_production, Stationary Check, Decomposition Plot, ACF-PACF Plot).
# 4. Data Preprocessing (Replace null values in column lng_production with mean 1 year before, replace lng_production values with unplanned shutdown case using mean 1 year before).
# 5. Split data after cleaning process to train and test.
# 6. Define the Forecasting Horizon. In this case, length for horizon is 365 data or 365 days.
# 7. Create exogenous variables to support the forecasting process. In this case, we use the data of month and day index and planned_shutdown data.
# 8. Split exogenous data to train and test. Train test proportion is same with train test data.
# 9. Forecasting process using 8 methods (Arimax, Sarimax, Prophet, Random Forest, XGBoost, Linear Regression, Polynomial Regression Degree=2, Polynomial Regression Degree=3).
# 10. For each methods, there are several steps :
#    10.1 Arimax - Sarimax
#         - For Arimax and Sarimax use Auto Arima Algorithm to find best order and seasonal order.
#         - Fitting best order or seasonal order. We can add exogenous variables to this fitting process.
#         - See the summary of best model.
#         - Predict the future data using parameter of forecasting horizon and exogenous testing data.
#         - Calculate error between testing data and prediction data using Mean Absolute Percentage Error.
#         - Save the parameters model
#    10.2 Prophet, Random Forest, XGBoost, Linear Regression, Polynomial Regression Degree=2, Polynomial Regression Degree=2
#         - Define some parameter options to find the best parameter.
#         - Run each method regressors.
#         - Create and run the SingleWindowSplitter and ForecastingGridSearchCV to find the best parameters for each methods.
#         - Fitting best parameter model in training data.
#         - Show top 10 best parameter model (optional) and show best parameter model.
#         - Predict the future data using best parameter with forecasting and exogenous testing data.
#         - Calculate error between testing data and prediction data using Mean Absolute Percentage Error.
# 11. Create all model parameters to dataframe.
# 12. Create all model mape result to dataframe.
# 13. Define function and query to save the model parameter and mape value to database.

##### SCRIPT OUTPUT #####
# The output for this script is best parameter and error value of each forecasting method. Which is best parameter and error value will be save in database table.

##### HOW TO USE THIS SCRIPT #####
# We can run this script using command prompt (directory same with this python script). But in this case, we can run this script using main_lng_insample.py.
# For example : We will run this script only, we can comment (#) script main_lng_insample.py on other script .py (example: lng_production_tangguh_forecasting_insample.py etc.)

# %%
import logging
import os
import sys
import numpy as np
import pandas as pd
import psycopg2
from configparser import ConfigParser
import gc

from humanfriendly import format_timespan
from tokenize import Ignore
from datetime import datetime
from tracemalloc import start
import matplotlib.pyplot as plt
from adtk.data import validate_series
from adtk.detector import ThresholdAD
from adtk.visualization import plot
from dateutil.relativedelta import *
from sklearn.metrics import (mean_absolute_percentage_error)

plt.style.use('fivethirtyeight')
pd.options.plotting.backend = "plotly"
from statsmodels.tsa.stattools import adfuller

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import GridSearchCV
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.model_selection import (ForecastingGridSearchCV,
                                                SingleWindowSplitter,
                                                temporal_train_test_split)
#from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
from sktime.performance_metrics.forecasting import (MeanAbsolutePercentageError, MeanSquaredError)
from xgboost import XGBRegressor

# Model scoring for Cross Validation
mape = MeanAbsolutePercentageError(symmetric=False)
mse = MeanSquaredError()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, message="Non-invertible starting MA parameters found.")
warnings.filterwarnings('ignore', 'y_pred and y_true do not have the same column index')
warnings.filterwarnings('ignore', 'Maximum Likelihood optimization failed to converge')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# %%
def main():
    from connection import create_db_connection, get_sql_data
    from utils import (logMessage, get_first_date_of_prev_month, get_last_date_of_prev_month, configLogging,
                       get_last_date_of_current_year, end_day_forecast_april, get_first_date_of_november)
    from polyfit import PolynomRegressor
    import datetime

    # Logs Directory
    logs_file_path = os.path.join('./logs', 'lng_production_tangguh_insample.log')

    # Configure logging
    configLogging(logs_file_path)

    # Connect to configuration file
    config = ConfigParser()
    config.read('config_lng.ini')
    
    # Accessing sections
    section_1 = config['config_tangguh']
    
    # Get values from configuration
    USE_DEFAULT_DATE = section_1.getboolean('use_default_date')

    TRAIN_START_YEAR= section_1.getint('train_start_year')
    TRAIN_START_MONTH = section_1.getint('train_start_month')
    TRAIN_START_DAY = section_1.getint('train_start_day')

    TRAIN_END_YEAR= section_1.getint('train_end_year')
    TRAIN_END_MONTH = section_1.getint('train_end_month')
    TRAIN_END_DAY = section_1.getint('train_end_day')

    TRAIN_START_DATE = (datetime.date(TRAIN_START_YEAR, TRAIN_START_MONTH, TRAIN_START_DAY)).strftime("%Y-%m-%d")
    TRAIN_END_DATE = (datetime.date(TRAIN_END_YEAR, TRAIN_END_MONTH, TRAIN_END_DAY)).strftime("%Y-%m-%d")
    
    # Accessing sections
    section_2 = config['config_sarimax']
    
    # Get values from sarimax configuration
    start_p = section_2.getint('START_P')
    max_p = section_2.getint('MAX_P')
    
    start_q = section_2.getint('START_Q')
    max_q= section_2.getint('MAX_Q')
    
    start_P = section_2.getint('START_P_SEASONAL')
    max_P = section_2.getint('MAX_P_SEASONAL')
    
    start_Q = section_2.getint('START_Q_SEASONAL')
    max_Q = section_2.getint('MAX_Q_SEASONAL')
    
    # Configure logging
    #configLogging("lng_production_tangguh.log")
    logMessage("Creating LNG Production Tangguh Model ...")
    
    # Connect to database
    # Exit program if not connected to database
    logMessage("Connecting to database ...")
    conn = create_db_connection(section='postgresql_ml_lng_skk')
    if conn == None:
        exit()

    logMessage("Cleaning data ...")
    ##### CLEANING LNG PRODUCTION DATA #####
    #Load Data from Database
    from datetime import datetime
    end_date = get_last_date_of_current_year()
    end_date_april = end_day_forecast_april()
    first_date_nov = get_first_date_of_november()
    current_date = datetime.now()
    date_nov = datetime.strptime(first_date_nov, "%Y-%m-%d")
    
    query = os.path.join('./sql','lng_prod_tangguh_data_query.sql')
    query_1 = open(query, mode="rt").read()
    sql = ''
    if USE_DEFAULT_DATE == True:
        if current_date < date_nov:
            sql = query_1.format('2016-01-01', end_date)
        else :
            sql = query_1.format('2016-01-01', end_date_april)
    else :
        sql = query_1.format(TRAIN_START_DATE, TRAIN_END_DATE)
   
    data = get_sql_data(sql, conn)
    data['date'] = pd.DatetimeIndex(data['date'], freq='D')
    data = data.reset_index()

    #%%
    ds = 'date'
    y = 'lng_production' 

    df = data[[ds,y]]
    df = df.set_index(ds)
    df.index = pd.DatetimeIndex(df.index, freq='D')

    logMessage("Null Value Cleaning ...")
    data_null_cleaning = data[['date', 'lng_production', 'unplanned_shutdown', 'planned_shutdown']].copy()
    data_null_cleaning['lng_production_copy'] = data[['lng_production']].copy()
    ds_null_cleaning = 'date'
    data_null_cleaning = data_null_cleaning.set_index(ds_null_cleaning)
    data_null_cleaning['unplanned_shutdown'] = data_null_cleaning['unplanned_shutdown'].astype('int')
    s = validate_series(data_null_cleaning)

    #%%
    # Detect Anomaly Values
    threshold_ad = ThresholdAD(data_null_cleaning['lng_production_copy'].isnull())
    anomalies = threshold_ad.detect(s)

    anomalies = anomalies.drop('lng_production', axis=1)
    anomalies = anomalies.drop('unplanned_shutdown', axis=1)
    anomalies = anomalies.drop('planned_shutdown', axis=1)

    #%%
    # Copy data frame of anomalies
    copy_anomalies =  anomalies.copy()
    # Rename columns
    copy_anomalies.rename(columns={'lng_production_copy':'anomaly'}, inplace=True)
    # Merge original dataframe with anomalies
    new_s = pd.concat([s, copy_anomalies], axis=1)

    # Get only anomalies data
    anomalies_data = new_s[new_s['anomaly'].isnull()]

    #%%
    #REPLACE ANOMALY VALUES
    for index, row in anomalies_data.iterrows():
        yr = index.year
        mt = index.month
       
        # Get last year start date month
        start_month = get_first_date_of_prev_month(yr,mt,step=-12)
        
        # Get last month last date
        end_month = get_last_date_of_prev_month(yr,mt,step=-1)
        
        # Get mean fead gas data for the month
        sql = "date>='"+start_month+ "' & "+ "date<='" +end_month+"'"
        mean_month=new_s['lng_production'].reset_index().query(sql).mean().values[0]
        
        # update value at specific location
        new_s.at[index,'lng_production'] = mean_month
        
    #%%
    logMessage("Unplanned Shutdown Cleaning ...")
    # Detect Unplanned Shutdown Value
    data2 = new_s[['lng_production', 'unplanned_shutdown', 'planned_shutdown']].copy()
    s2 = validate_series(data2)

    threshold_ad2 = ThresholdAD(data2['unplanned_shutdown']==0)
    anomalies2 = threshold_ad2.detect(s2)

    anomalies2 = anomalies2.drop('lng_production', axis=1)
    anomalies2 = anomalies2.drop('planned_shutdown', axis=1)
    
    # Copy data frame of anomalies
    copy_anomalies2 =  anomalies2.copy()
    # Rename columns
    copy_anomalies2.rename(columns={'unplanned_shutdown':'anomaly'}, inplace=True)
    # Merge original dataframe with anomalies
    new_s2 = pd.concat([s2, copy_anomalies2], axis=1)

    # Get only anomalies data
    anomalies_data2 = new_s2[new_s2['anomaly'] == False]
    
    #%%
    import datetime
    yesterday_date = anomalies_data2.head(1).index - datetime.timedelta(days=1)
    prev_date_year = yesterday_date - datetime.timedelta(days=364)

    yesterday_date = str(yesterday_date[0])
    prev_date_year = str(prev_date_year[0])   

    #%%
    for index, row in anomalies_data2.iterrows():
        yr = index.year
        mt = index.month
               
        # Get mean fead gas data for the month
        yesterday_date = index - datetime.timedelta(days=1)
        prev_date_year = yesterday_date - datetime.timedelta(days=364)
        
        yesterday_date = yesterday_date.strftime("%Y-%m-%d")
        prev_date_year = prev_date_year.strftime("%Y-%m-%d")
        sql = "date>='"+prev_date_year+ "' & "+ "date<='" +yesterday_date+"'"
        mean_month=new_s2['lng_production'].reset_index().query(sql).mean(skipna = True).values[0] 
        
        # update value at specific location
        new_s2.at[index,'lng_production'] = mean_month


    #%%
    logMessage("Final Data Prepare ...")
    data_cleaned = new_s2[['lng_production', 'planned_shutdown']].copy()
    data_cleaned = data_cleaned.reset_index()

    ds_cleaned = 'date'
    y_cleaned = 'lng_production'
    df_cleaned = data_cleaned[[ds_cleaned, y_cleaned]]
    df_cleaned = df_cleaned.set_index(ds_cleaned)
    df_cleaned.index = pd.DatetimeIndex(df_cleaned.index, freq='D')

    #Select column target
    #train_df = df_cleaned['lng_production']
   
    #%%
    logMessage("AD Fuller Test ...")
    ad_fuller = adfuller(df_cleaned)
    num_lags = ad_fuller[2]

    #%%
    # Test size
    test_size = 365
    # Split data (original data)
    y_train, y_test = temporal_train_test_split(df, test_size=test_size)
    # Split data (original data)
    y_train_cleaned, y_test_cleaned = temporal_train_test_split(df_cleaned, test_size=test_size)
    # Horizon
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    fh_int = np.arange(1, len(fh))

    #%%
    ## Create Exogenous Variable
    df_cleaned['month'] = [i.month for i in df_cleaned.index]
    df_cleaned['planned_shutdown'] = data['planned_shutdown'].values
    df_cleaned['day'] = [i.day for i in df_cleaned.index]

    #%%
    # Split into train and test
    X_train, X_test = temporal_train_test_split(df_cleaned.iloc[:,1:], test_size=test_size)

    exogenous_features = ["month", "day", "planned_shutdown"]
    
    # Delete variabel that not used
    del data
    del data_null_cleaning
    del y_train
    del new_s
    del new_s2
    del anomalies
    del anomalies2
    del anomalies_data
    del anomalies_data2
    gc.collect()

    ###### FORECASTING ######
    #%%
    ##### SARIMAX MODEL (forecast_b) #####
    logMessage("Creating Sarimax Model Forecasting Insample LNG Production BP Tangguh ...")
    #%%
    #Set parameters
    sarimax_differencing = 0
    sarimax_seasonal_differencing = 1
    sarimax_sp = 12
    sarimax_seasonal = True
    sarimax_trace = True
    sarimax_error_action = "ignore"
    sarimax_suppress_warnings = True
    sarimax_n_fits = 50
    sarimax_stepwise = True
    
    #sarimax_model = auto_arima(y=y_train_cleaned.lng_production, X=X_train[exogenous_features], d=0, D=1, seasonal=True, m=4, trace=True, error_action="ignore", suppress_warnings=True)
    sarimax_model = AutoARIMA(start_p = start_p, max_p = max_p, start_q = start_q, max_q = max_q, d=sarimax_differencing, 
                              start_P = start_P, max_P = max_P, start_Q = start_Q, max_Q = max_Q, D=sarimax_seasonal_differencing, 
                              seasonal=sarimax_seasonal, sp=sarimax_sp, trace=sarimax_trace, n_fits=sarimax_n_fits, stepwise=sarimax_stepwise, error_action=sarimax_error_action, suppress_warnings=sarimax_suppress_warnings)
    logMessage("Creating SARIMAX Model ...") 
    sarimax_fit = sarimax_model.fit(y_train_cleaned.lng_production, X=X_train[exogenous_features])
    logMessage("SARIMAX Model Summary")
    logMessage(sarimax_fit.summary())
    
    logMessage("SARIMAX Model Prediction ..")
    sarimax_forecast = sarimax_fit.predict(fh, X=X_test[exogenous_features])

    #Create MAPE
    sarimax_mape = mean_absolute_percentage_error(y_test_cleaned.lng_production, sarimax_forecast)
    sarimax_mape_str = str('MAPE: %.4f' % sarimax_mape)
    logMessage("SARIMAX Model "+sarimax_mape_str)
    
    #Get parameters
    sarimax_param_order = sarimax_fit.get_fitted_params()['order']
    sarimax_param_order_seasonal = sarimax_fit.get_fitted_params()['seasonal_order']
    sarimax_param = str({'sarimax_order': sarimax_param_order, 'sarimax_seasonal_order': sarimax_param_order_seasonal})
    logMessage("Sarimax Model Parameters "+sarimax_param)

    # Create Adjusment Value for Sarimax
    df_adjustment_sarimax = df_cleaned.loc[df_cleaned.index[-365:]].copy()
    df_adjustment_sarimax['sarimax_forecast'] = sarimax_forecast.copy()
    #df_adjustment_sarimax = df_cleaned.drop(['month', 'day', 'wpnb_gas'], axis=1)
    df_adjustment_sarimax = df_adjustment_sarimax.drop(df_adjustment_sarimax[df_adjustment_sarimax['planned_shutdown'] == 1].index)

    # Calculate mean lng_production data testing
    test_mean = df_adjustment_sarimax['lng_production'].mean()

    # Calculate mean sarimax_forecast
    sarimax_forecast_mean = df_adjustment_sarimax['sarimax_forecast'].mean()

    # Calculate difference mean value between testing and forecast data
    adj_forecast_b = test_mean - sarimax_forecast_mean

    # add diff_value to each value in column 'sarimax_forecast' using the + operator
    df_adjustment_sarimax['sarimax_forecast_add'] = df_adjustment_sarimax['sarimax_forecast'] + adj_forecast_b

    # Calculate MAPE Value after adjustment value
    sarimax_mape_adj = mean_absolute_percentage_error(df_adjustment_sarimax['lng_production'], df_adjustment_sarimax['sarimax_forecast_add'])
    
    # Empty the SARIMAX memory
    del sarimax_model
    del sarimax_forecast
    del sarimax_param_order
    del sarimax_param_order_seasonal
    del sarimax_fit
    del df_adjustment_sarimax
    gc.collect()


    ##### ARIMAX MODEL (forecast_a) #####
    logMessage("Creating Arimax Model Forecasting Insample LNG Production BP Tangguh ...")
    # %%
    # Create ARIMAX (forecast_a) Model
    arimax_model = AutoARIMA(d=1, suppress_warnings=True, error_action='ignore', trace=True) #If using SKTime AutoArima
    #arimax_model = auto_arima(y_train_cleaned, X_train[exogenous_features], d=1, trace=True, error_action="ignore", suppress_warnings=True)
    logMessage("Creating ARIMAX Model ...")
    arimax_fit = arimax_model.fit(y_train_cleaned, X=X_train[exogenous_features])
    logMessage("ARIMAX Model Summary")
    logMessage(arimax_fit.summary())
    
    logMessage("ARIMAX Model Prediction ..")
    arimax_forecast = arimax_fit.predict(fh, X=X_test[exogenous_features]) #If using sktime (fh), if using pmdarima (len(fh))

    #Create MAPE
    arimax_mape = mean_absolute_percentage_error(y_test_cleaned.lng_production, arimax_forecast)
    arimax_mape_str = str('MAPE: %.4f' % arimax_mape)
    logMessage("ARIMAX Model "+arimax_mape_str)
    
    #Get parameter
    arimax_param = str(arimax_fit.get_fitted_params()['order'])
    logMessage("Arimax Model Parameters "+arimax_param)

    # Create Adjusment Value for Arimax
    df_adjustment_arimax = df_cleaned.loc[df_cleaned.index[-365:]].copy()
    df_adjustment_arimax['arimax_forecast'] = arimax_forecast.copy()
    #df_adjustment_arimax = df_cleaned.drop(['month', 'day', 'wpnb_gas'], axis=1)
    df_adjustment_arimax = df_adjustment_arimax.drop(df_adjustment_arimax[df_adjustment_arimax['planned_shutdown'] == 1].index)

    # Calculate mean arimax_forecast
    arimax_forecast_mean = df_adjustment_arimax['arimax_forecast'].mean()

    # Calculate difference mean value between testing and forecast data
    adj_forecast_a = test_mean - arimax_forecast_mean

    # add diff_value to each value in column 'arimax_forecast' using the + operator
    df_adjustment_arimax['arimax_forecast_add'] = df_adjustment_arimax['arimax_forecast'] + adj_forecast_a

    # Calculate MAPE Value after adjustment value
    arimax_mape_adj = mean_absolute_percentage_error(df_adjustment_arimax['lng_production'], df_adjustment_arimax['arimax_forecast_add'])
    
    # Empty the ARIMAX memory
    del arimax_model
    del arimax_forecast
    del arimax_fit
    del df_adjustment_arimax
    gc.collect()


    ##### PROPHET MODEL (forecast_c) #####
    logMessage("Creating Prophet Model Forecasting Insample LNG Production BP Tangguh ...")
    #%%
    #Create Prophet Parameter Grid
    prophet_param_grid = {'seasonality_mode':['additive','multiplicative']
                        ,'n_changepoints':[num_lags, 6]
                        ,'seasonality_prior_scale':[0.05, 0.1] #Flexibility of the seasonality (0.01,10)
                        ,'changepoint_prior_scale':[0.1, 0.5] #Flexibility of the trend (0.001,0.5)
                        ,'daily_seasonality':[8,10]
                        ,'weekly_seasonality':[1,5]
                        ,'yearly_seasonality':[8,10]
                        }

    logMessage("Creating Prophet Regressor Object ....") 
    # create regressor object
    prophet_forecaster = Prophet()

    logMessage("Creating Window Splitter Prophet Model ....")   
    cv_prophet = SingleWindowSplitter(fh=fh_int)
    gscv_prophet = ForecastingGridSearchCV(prophet_forecaster, cv=cv_prophet, param_grid=prophet_param_grid, scoring=mape, error_score='raise')

    logMessage("Creating Prophet Model ...")
    prophet_fit = gscv_prophet.fit(y_train_cleaned, X=X_train) #, X_train

    # Show best model parameters
    logMessage("Show Best Prophet Models ...")
    prophet_best_params = prophet_fit.best_params_
    prophet_best_params_str = str(prophet_best_params)
    logMessage("Best Prophet Models "+prophet_best_params_str)

    logMessage("Prophet Model Prediction ...")
    prophet_forecast = prophet_fit.best_forecaster_.predict(fh, X=X_test)#, X=X_test
 
    #Create MAPE
    prophet_mape = mean_absolute_percentage_error(y_test_cleaned['lng_production'], prophet_forecast)
    prophet_mape_str = str('MAPE: %.4f' % prophet_mape)
    logMessage("Prophet Model "+prophet_mape_str)

    # Create Adjusment Value for Prophet
    df_adjustment_prophet = df_cleaned.loc[df_cleaned.index[-365:]].copy()
    df_adjustment_prophet['prophet_forecast'] = prophet_forecast.copy()
    #df_adjustment_prophet = df_cleaned.drop(['month', 'day', 'wpnb_gas'], axis=1)
    df_adjustment_prophet = df_adjustment_prophet.drop(df_adjustment_prophet[df_adjustment_prophet['planned_shutdown'] == 1].index)

    # Calculate mean prophet_forecast
    prophet_forecast_mean = df_adjustment_prophet['prophet_forecast'].mean()

    # Calculate difference mean value between testing and forecast data
    adj_forecast_c = test_mean - prophet_forecast_mean

    # add diff_value to each value in column 'prophet_forecast' using the + operator
    df_adjustment_prophet['prophet_forecast_add'] = df_adjustment_prophet['prophet_forecast'] + adj_forecast_c

    # Calculate MAPE Value after adjustment value
    prophet_mape_adj = mean_absolute_percentage_error(df_adjustment_prophet['lng_production'], df_adjustment_prophet['prophet_forecast_add'])
    
    # Empty the Prophet memory
    del prophet_param_grid
    del cv_prophet
    del gscv_prophet
    del prophet_forecast
    del prophet_fit
    del prophet_best_params
    del prophet_mape_str
    del df_adjustment_prophet
    gc.collect()


    ##### RANDOM FOREST MODEL (forecast_d) #####
    logMessage("Creating Random Forest Model Forecasting Insample LNG Production BP Tangguh ...")
    #%%
    #Create Random Forest Parameter Grid
    ranfor_random_state = 0
    ranfor_criterion = "squared_error"
    ranfor_strategy = "recursive"

    ranfor_forecaster_param_grid = {"window_length": [1, 6, 11, num_lags], 
                                    "estimator__n_estimators": [150, 200]}

    # create regressor object
    ranfor_regressor = RandomForestRegressor(random_state = ranfor_random_state, criterion = ranfor_criterion)
    ranfor_forecaster = make_reduction(ranfor_regressor, strategy = ranfor_strategy)

    logMessage("Creating Window Splitter Random Forest Model ....")   
    cv_ranfor = SingleWindowSplitter(fh=fh_int)
    gscv_ranfor = ForecastingGridSearchCV(ranfor_forecaster, cv=cv_ranfor, param_grid=ranfor_forecaster_param_grid, scoring=mape)

    logMessage("Creating Random Forest Model ...")
    ranfor_fit = gscv_ranfor.fit(y_train_cleaned, X=X_train) #, X_train

    # Show best model parameters
    logMessage("Show Best Random Forest Models ...")
    ranfor_best_params = ranfor_fit.best_params_
    ranfor_best_params_str = str(ranfor_best_params)
    logMessage("Best Random Forest Models "+ranfor_best_params_str)
    
    logMessage("Random Forest Model Prediction ...")
    ranfor_forecast = ranfor_fit.best_forecaster_.predict(fh, X=X_test) #, X=X_test

    #Create MAPE
    ranfor_mape = mean_absolute_percentage_error(y_test_cleaned['lng_production'], ranfor_forecast)
    ranfor_mape_str = str('MAPE: %.4f' % ranfor_mape)
    logMessage("Random Forest Model "+ranfor_mape_str)

    # Create Adjusment Value for Random Forest
    df_adjustment_ranfor = df_cleaned.loc[df_cleaned.index[-365:]].copy()
    df_adjustment_ranfor['ranfor_forecast'] = ranfor_forecast.copy()
    #df_adjustment_ranfor = df_cleaned.drop(['month', 'day', 'wpnb_gas'], axis=1)
    df_adjustment_ranfor = df_adjustment_ranfor.drop(df_adjustment_ranfor[df_adjustment_ranfor['planned_shutdown'] == 1].index)

    # Calculate mean ranfor_forecast
    ranfor_forecast_mean = df_adjustment_ranfor['ranfor_forecast'].mean()

    # Calculate difference mean value between testing and forecast data
    adj_forecast_d = test_mean - ranfor_forecast_mean

    # add diff_value to each value in column 'ranfor_forecast' using the + operator
    df_adjustment_ranfor['ranfor_forecast_add'] = df_adjustment_ranfor['ranfor_forecast'] + adj_forecast_d

    # Calculate MAPE Value after adjustment value
    ranfor_mape_adj = mean_absolute_percentage_error(df_adjustment_ranfor['lng_production'], df_adjustment_ranfor['ranfor_forecast_add'])
    
    # Empty Random Forest Memory
    del ranfor_forecaster_param_grid
    del ranfor_regressor
    del ranfor_forecaster
    del cv_ranfor
    del gscv_ranfor
    del ranfor_forecast
    del ranfor_fit
    del ranfor_best_params
    del ranfor_mape_str
    del df_adjustment_ranfor
    gc.collect()


    ##### XGBOOST MODEL (forecast_e) #####
    logMessage("Creating XGBoost Model Forecasting Insample LNG Production BP Tangguh ...")
    #%%
    #Create XGBoost Parameter Grid
    xgb_objective = 'reg:squarederror'
    xgb_strategy = "recursive"

    xgb_forecaster_param_grid = {"window_length": [2, 6, 7, num_lags]
                                ,"estimator__n_estimators": [100, 200]
                                }

    xgb_regressor = XGBRegressor(objective=xgb_objective, seed = 42)
    xgb_forecaster = make_reduction(xgb_regressor, strategy=xgb_strategy)

    cv_xgb = SingleWindowSplitter(fh=fh_int)
    gscv_xgb = ForecastingGridSearchCV(xgb_forecaster, cv=cv_xgb, param_grid=xgb_forecaster_param_grid, scoring=mape)

    logMessage("Creating XGBoost Model ....")
    xgb_fit = gscv_xgb.fit(y_train_cleaned, X=X_train) #, X_train

    # Show best model parameters
    logMessage("Show Best XGBoost Models ...")
    xgb_best_params = xgb_fit.best_params_
    xgb_best_params_str = str(xgb_best_params)
    logMessage("Best XGBoost Models "+xgb_best_params_str)
    
    logMessage("XGBoost Model Prediction ...")
    xgb_forecast = xgb_fit.best_forecaster_.predict(fh, X=X_test) #, X=X_test

    #Create MAPE
    xgb_mape = mean_absolute_percentage_error(y_test_cleaned['lng_production'], xgb_forecast)
    xgb_mape_str = str('MAPE: %.4f' % xgb_mape)
    logMessage("XGBoost Model "+xgb_mape_str)

    # Create Adjusment Value for XGBoost
    df_adjustment_xgb = df_cleaned.loc[df_cleaned.index[-365:]].copy()
    df_adjustment_xgb['xgb_forecast'] = xgb_forecast.copy()
    #df_adjustment_xgb = df_cleaned.drop(['month', 'day', 'wpnb_gas'], axis=1)
    df_adjustment_xgb = df_adjustment_xgb.drop(df_adjustment_xgb[df_adjustment_xgb['planned_shutdown'] == 1].index)

    # Calculate mean xgb_forecast
    xgb_forecast_mean = df_adjustment_xgb['xgb_forecast'].mean()

    # Calculate difference mean value between testing and forecast data
    adj_forecast_e = test_mean - xgb_forecast_mean

    # add diff_value to each value in column 'xgb_forecast' using the + operator
    df_adjustment_xgb['xgb_forecast_add'] = df_adjustment_xgb['xgb_forecast'] + adj_forecast_e

    # Calculate MAPE Value after adjustment value
    xgb_mape_adj = mean_absolute_percentage_error(df_adjustment_xgb['lng_production'], df_adjustment_xgb['xgb_forecast_add'])
    
    # Empty Random Forest Memory
    del xgb_forecaster_param_grid
    del xgb_regressor
    del xgb_forecaster
    del cv_xgb
    del gscv_xgb
    del xgb_forecast
    del xgb_fit
    del xgb_best_params
    del xgb_mape_str
    del df_adjustment_xgb
    gc.collect()


    ##### LINEAR REGRESSION MODEL (forecast_f) #####
    logMessage("Creating Linear Regression Model Forecasting Insample LNG Production BP Tangguh ...")
    #%%
    # Create Linear Regression Parameter Grid
    linreg_strategy = "recursive"
    linreg_forecaster_param_grid = {"window_length": [2, 6, 7, num_lags]}

    linreg_regressor = LinearRegression()
    linreg_forecaster = make_reduction(linreg_regressor, strategy=linreg_strategy)

    cv_linreg = SingleWindowSplitter(fh=fh_int)
    gscv_linreg = ForecastingGridSearchCV(linreg_forecaster, cv=cv_linreg, param_grid=linreg_forecaster_param_grid, scoring=mape)

    logMessage("Creating Linear Regression Model ...")
    linreg_fit = gscv_linreg.fit(y_train_cleaned, X=X_train) #, X=X_train

    # Show best model parameters
    logMessage("Show Best Linear Regression Models ...")
    linreg_best_params = linreg_fit.best_params_
    linreg_best_params_str = str(linreg_best_params)
    logMessage("Best Linear Regression Models "+linreg_best_params_str)
    
    logMessage("Linear Regression Model Prediction ...")
    linreg_forecast = linreg_fit.best_forecaster_.predict(fh, X=X_test) #, X=X_test

    #Create MAPE
    linreg_mape = mean_absolute_percentage_error(y_test_cleaned['lng_production'], linreg_forecast)
    linreg_mape_str = str('MAPE: %.4f' % linreg_mape)
    logMessage("Linear Regression Model "+linreg_mape_str)

    # Create Adjusment Value for Linear Regression
    df_adjustment_linreg = df_cleaned.loc[df_cleaned.index[-365:]].copy()
    df_adjustment_linreg['linreg_forecast'] = linreg_forecast.copy()
    #df_adjustment_linreg = df_cleaned.drop(['month', 'day', 'wpnb_gas'], axis=1)
    df_adjustment_linreg = df_adjustment_linreg.drop(df_adjustment_linreg[df_adjustment_linreg['planned_shutdown'] == 1].index)

    # Calculate mean linreg_forecast
    linreg_forecast_mean = df_adjustment_linreg['linreg_forecast'].mean()

    # Calculate difference mean value between testing and forecast data
    adj_forecast_f = test_mean - linreg_forecast_mean

    # add diff_value to each value in column 'linreg_forecast' using the + operator
    df_adjustment_linreg['linreg_forecast_add'] = df_adjustment_linreg['linreg_forecast'] + adj_forecast_f

    # Calculate MAPE Value after adjustment value
    linreg_mape_adj = mean_absolute_percentage_error(df_adjustment_linreg['lng_production'], df_adjustment_linreg['linreg_forecast_add'])
    
    # Empty Linear Regression Memory
    del linreg_forecaster_param_grid
    del linreg_regressor
    del linreg_forecaster
    del cv_linreg
    del gscv_linreg
    del linreg_forecast
    del linreg_fit
    del linreg_best_params
    del linreg_mape_str
    del df_adjustment_linreg
    gc.collect()

    ##### POLYNOMIAL REGRESSION DEGREE=2 MODEL (forecast_g) #####
    logMessage("Creating Polynomial Regression Degree=2 Model Forecasting Insample LNG Production BP Tangguh ...")
    #%%
    # Create Polynomial Regression Degree=2 Parameter Grid
    poly2_regularization = None
    poly2_interactions = False
    poly2_strategy = "recursive"

    poly2_forecaster_param_grid = {"window_length": [1]}

    poly2_regressor = PolynomRegressor(deg=2, regularization=poly2_regularization, interactions=poly2_interactions)
    poly2_forecaster = make_reduction(poly2_regressor, strategy=poly2_strategy)

    cv_poly2 = SingleWindowSplitter(fh=fh_int)
    gscv_poly2 = ForecastingGridSearchCV(poly2_forecaster, cv=cv_poly2, param_grid=poly2_forecaster_param_grid, scoring=mape, error_score='raise')

    logMessage("Creating Polynomial Regression Orde 2 Model ...")
    poly2_fit = gscv_poly2.fit(y_train_cleaned, X=X_train) #, X=X_train

    # Show best model parameters
    logMessage("Show Best Polynomial Regression Degree=2 Models ...")
    poly2_best_params = poly2_fit.best_params_
    poly2_best_params_str = str(poly2_best_params)
    logMessage("Best Polynomial Regression Degree=3 Models "+poly2_best_params_str)
    
    logMessage("Polynomial Regression Degree=2 Model Prediction ...")
    poly2_forecast = poly2_fit.best_forecaster_.predict(fh, X=X_test) #, X=X_test

    #Create MAPE
    poly2_mape = mean_absolute_percentage_error(y_test_cleaned['lng_production'], poly2_forecast)
    poly2_mape_str = str('MAPE: %.4f' % poly2_mape)
    logMessage("Polynomial Regression Degree=2 Model "+poly2_mape_str)

    # Create Adjusment Value for Polynomial Regression Degree=2
    df_adjustment_poly2 = df_cleaned.loc[df_cleaned.index[-365:]].copy()
    df_adjustment_poly2['poly2_forecast'] = poly2_forecast.copy()
    #df_adjustment_poly2 = df_cleaned.drop(['month', 'day', 'wpnb_gas'], axis=1)
    df_adjustment_poly2 = df_adjustment_poly2.drop(df_adjustment_poly2[df_adjustment_poly2['planned_shutdown'] == 1].index)

    # Calculate mean poly2_forecast
    poly2_forecast_mean = df_adjustment_poly2['poly2_forecast'].mean()

    # Calculate difference mean value between testing and forecast data
    adj_forecast_g = test_mean - poly2_forecast_mean

    # add diff_value to each value in column 'poly2_forecast' using the + operator
    df_adjustment_poly2['poly2_forecast_add'] = df_adjustment_poly2['poly2_forecast'] + adj_forecast_g

    # Calculate MAPE Value after adjustment value
    poly2_mape_adj = mean_absolute_percentage_error(df_adjustment_poly2['lng_production'], df_adjustment_poly2['poly2_forecast_add'])
    
    # Empty Polynomial Regression Degree=2 Memory
    del poly2_forecaster_param_grid
    del poly2_regressor
    del poly2_forecaster
    del cv_poly2
    del gscv_poly2
    del poly2_forecast
    del poly2_fit
    del poly2_best_params
    del poly2_mape_str
    del df_adjustment_poly2
    gc.collect() 


    ##### POLYNOMIAL REGRESSION DEGREE=3 MODEL (forecast_h) #####
    logMessage("Creating Polynomial Regression Degree=3 Model Forecasting Insample LNG Production BP Tangguh ...")
    #%%
    # Create Polynomial Regression Degree=3 Parameter Grid
    poly3_regularization = None
    poly3_interactions = False
    poly3_strategy = "recursive"

    poly3_forecaster_param_grid = {"window_length": [0.8]}

    poly3_regressor = PolynomRegressor(deg=3, regularization=poly3_regularization, interactions=poly3_interactions)
    poly3_forecaster = make_reduction(poly3_regressor, strategy=poly3_strategy)

    cv_poly3 = SingleWindowSplitter(fh=fh_int)
    gscv_poly3 = ForecastingGridSearchCV(poly3_forecaster, cv=cv_poly3, param_grid=poly3_forecaster_param_grid, scoring=mape, error_score='raise')

    logMessage("Creating Polynomial Regression Orde 3 Model ...")
    poly3_fit = gscv_poly3.fit(y_train_cleaned) #, X=X_train

    # Show best model parameters
    logMessage("Show Best Polynomial Regression Degree=3 Models ...")
    poly3_best_params = poly3_fit.best_params_
    poly3_best_params_str = str(poly3_best_params)
    logMessage("Best Polynomial Regression Degree=3 Models "+poly3_best_params_str)
    
    logMessage("Polynomial Regression Degree=3 Model Prediction ...")
    poly3_forecast = poly3_fit.best_forecaster_.predict(fh) #, X=X_test

    #Create MAPE
    poly3_mape = mean_absolute_percentage_error(y_test_cleaned['lng_production'], poly3_forecast)
    poly3_mape_str = str('MAPE: %.4f' % poly3_mape)
    logMessage("Polynomial Regression Degree=3 Model "+poly3_mape_str)

    # Create Adjusment Value for Polynomial Regression Degree=3
    df_adjustment_poly3 = df_cleaned.loc[df_cleaned.index[-365:]].copy()
    df_adjustment_poly3['poly3_forecast'] = poly3_forecast.copy()
    #df_adjustment_poly3 = df_cleaned.drop(['month', 'day', 'wpnb_gas'], axis=1)
    df_adjustment_poly3 = df_adjustment_poly3.drop(df_adjustment_poly3[df_adjustment_poly3['planned_shutdown'] == 1].index)

    # Calculate mean poly3_forecast
    poly3_forecast_mean = df_adjustment_poly3['poly3_forecast'].mean()

    # Calculate difference mean value between testing and forecast data
    adj_forecast_h = test_mean - poly3_forecast_mean

    # add diff_value to each value in column 'poly3_forecast' using the + operator
    df_adjustment_poly3['poly3_forecast_add'] = df_adjustment_poly3['poly3_forecast'] + adj_forecast_h

    # Calculate MAPE Value after adjustment value
    poly3_mape_adj = mean_absolute_percentage_error(df_adjustment_poly3['lng_production'], df_adjustment_poly3['poly3_forecast_add'])
    
    # Empty Polynomial Regression Degree=2 Memory
    del poly3_forecaster_param_grid
    del poly3_regressor
    del poly3_forecaster
    del cv_poly3
    del gscv_poly3
    del poly3_forecast
    del poly3_fit
    del poly3_best_params
    del poly3_mape_str
    del df_adjustment_poly3
    gc.collect()

    # CREATE BEST MODEL CONFIG TO DATAFRAME
    logMessage("Creating best model config dataframe ...")
    best_model = [{'lng_plant': 'BP Tangguh', 'lng_parameter': 'LNG Production', 'model_choosen': 'a', 'mape': arimax_mape_adj},
                {'lng_plant': 'BP Tangguh', 'lng_parameter': 'LNG Production', 'model_choosen': 'b', 'mape': sarimax_mape_adj},
                {'lng_plant': 'BP Tangguh', 'lng_parameter': 'LNG Production', 'model_choosen': 'c', 'mape': prophet_mape_adj},
                {'lng_plant': 'BP Tangguh', 'lng_parameter': 'LNG Production', 'model_choosen': 'd', 'mape': ranfor_mape_adj},
                {'lng_plant': 'BP Tangguh', 'lng_parameter': 'LNG Production', 'model_choosen': 'e', 'mape': xgb_mape_adj},
                {'lng_plant': 'BP Tangguh', 'lng_parameter': 'LNG Production', 'model_choosen': 'f', 'mape': linreg_mape_adj},
                {'lng_plant': 'BP Tangguh', 'lng_parameter': 'LNG Production', 'model_choosen': 'g', 'mape': poly2_mape_adj},
                {'lng_plant': 'BP Tangguh', 'lng_parameter': 'LNG Production', 'model_choosen': 'h', 'mape': poly3_mape_adj}]
    best_model = pd.DataFrame(best_model)

    # get the row with the smallest value in column mape
    best_model = best_model.loc[best_model['mape'].idxmin()]
    best_model_df = {'lng_plant': 'BP Tangguh',
                    'lng_parameter': 'LNG Production',
                    'model_choosen': [best_model['model_choosen']]}
    best_model_df = pd.DataFrame(best_model_df)

    del best_model
    gc.collect()

    #%%
    #CREATE MAPE TO DATAFRAME
    logMessage("Creating all model mape result data frame ...")
    all_mape_pred =  {'mape_forecast_a': [arimax_mape_adj],
                    'mape_forecast_b': [sarimax_mape_adj],
                    'mape_forecast_c': [prophet_mape_adj],
                    'mape_forecast_d': [ranfor_mape_adj],
                    'mape_forecast_e': [xgb_mape_adj],
                    'mape_forecast_f': [linreg_mape_adj],
                    'mape_forecast_g': [poly2_mape_adj],
                    'mape_forecast_h': [poly3_mape_adj],
                    'mape_fc_a_before_adj': [arimax_mape],
                    'mape_fc_b_before_adj': [sarimax_mape],
                    'mape_fc_c_before_adj': [prophet_mape],
                    'mape_fc_d_before_adj': [ranfor_mape],
                    'mape_fc_e_before_adj': [xgb_mape],
                    'mape_fc_f_before_adj': [linreg_mape],
                    'mape_fc_g_before_adj': [poly2_mape],
                    'mape_fc_h_before_adj': [poly3_mape],
                    'lng_plant' : 'BP Tangguh',
                    'product' : 'LNG Production'}

    all_mape_pred = pd.DataFrame(all_mape_pred)

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
                        'lng_plant' : 'BP Tangguh',
                        'product' : 'LNG Production'}

    all_model_param = pd.DataFrame(all_model_param)

    # CREATE ADJUSTMENT VALUE TO DATAFRAME
    logMessage("Creating all adjustment value dataframe ...")
    all_adj_value =  {'adj_forecast_a': [adj_forecast_a],
                        'adj_forecast_b': [adj_forecast_b],
                        'adj_forecast_c': [adj_forecast_c],
                        'adj_forecast_d': [adj_forecast_d],
                        'adj_forecast_e': [adj_forecast_e],
                        'adj_forecast_f': [adj_forecast_f],
                        'adj_forecast_g': [adj_forecast_g],
                        'adj_forecast_h': [adj_forecast_h],
                        'lng_plant' : 'BP Tangguh',
                        'product' : 'LNG Production'}

    all_adj_value = pd.DataFrame(all_adj_value)
    
    # Save mape result to database
    logMessage("Updating MAPE result to database ...")
    total_updated_rows = insert_mape(conn, all_mape_pred)
    logMessage("Updated rows: {}".format(total_updated_rows))

    del all_mape_pred
    gc.collect()
    
    # Save param result to database
    logMessage("Updating Model Parameter result to database ...")
    total_updated_rows = insert_param(conn, all_model_param)
    logMessage("Updated rows: {}".format(total_updated_rows))

    del all_model_param
    gc.collect()

    # Save adjustment value result to database
    logMessage("Updating Adjustment Value result to database ...")
    total_updated_rows = insert_adj_value(conn, all_adj_value)
    logMessage("Updated rows: {}".format(total_updated_rows))

    del all_adj_value
    gc.collect()

    # Save model config to database
    logMessage("Updating Model Config to database ...")
    total_updated_rows = insert_model_config(conn, best_model_df)
    logMessage("Updated rows: {}".format(total_updated_rows))

    del best_model_df
    gc.collect()
    
    print("Done")
    
# %%
def insert_mape(conn, all_mape_pred):
    total_updated_rows = 0
    for index, row in all_mape_pred.iterrows():
        lng_plant = row['lng_plant']
        product = row['product']
        mape_forecast_a, mape_forecast_b, mape_forecast_c, mape_forecast_d, mape_forecast_e, mape_forecast_f, mape_forecast_g, mape_forecast_h, mape_fc_a_before_adj, mape_fc_b_before_adj, mape_fc_c_before_adj, mape_fc_d_before_adj, mape_fc_e_before_adj, mape_fc_f_before_adj, mape_fc_g_before_adj, mape_fc_h_before_adj = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15]
        
        #sql = f'UPDATE trir_monthly_test SET forecast_a = {} WHERE year_num = {} AND month_num = {}'.format(forecast, year_num, month_num)
        updated_rows = update_mape_value(conn, mape_forecast_a, mape_forecast_b, mape_forecast_c, mape_forecast_d, mape_forecast_e, mape_forecast_f, mape_forecast_g, mape_forecast_h, mape_fc_a_before_adj, mape_fc_b_before_adj, mape_fc_c_before_adj, mape_fc_d_before_adj, mape_fc_e_before_adj, mape_fc_f_before_adj, mape_fc_g_before_adj, mape_fc_h_before_adj, lng_plant, product)
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

def insert_adj_value(conn, all_adj_value):
    total_updated_rows = 0
    for index, row in all_adj_value.iterrows():
        lng_plant = row['lng_plant']
        product = row['product']
        adj_forecast_a, adj_forecast_b, adj_forecast_c, adj_forecast_d, adj_forecast_e, adj_forecast_f, adj_forecast_g, adj_forecast_h = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]
        
        #sql = f'UPDATE trir_monthly_test SET forecast_a = {} WHERE year_num = {} AND month_num = {}'.format(forecast, year_num, month_num)
        updated_rows = update_adj_value(conn, adj_forecast_a, adj_forecast_b, adj_forecast_c, adj_forecast_d, adj_forecast_e, adj_forecast_f, adj_forecast_g, adj_forecast_h , lng_plant, product)
        total_updated_rows = total_updated_rows + updated_rows 
        
    return total_updated_rows

def insert_model_config(conn, best_model_df):
    total_updated_rows = 0
    for index, row in best_model_df.iterrows():
        lng_plant = row['lng_plant']
        lng_parameter = row['lng_parameter']
        model_choosen = row[2]
        
        updated_rows = update_model_config(conn, model_choosen, lng_plant, lng_parameter)
        total_updated_rows = total_updated_rows + updated_rows
        
    return total_updated_rows

def update_mape_value(conn, mape_forecast_a, mape_forecast_b, mape_forecast_c, mape_forecast_d, mape_forecast_e, mape_forecast_f, mape_forecast_g, mape_forecast_h, mape_fc_a_before_adj,
                      mape_fc_b_before_adj, mape_fc_c_before_adj, mape_fc_d_before_adj, mape_fc_e_before_adj, mape_fc_f_before_adj, mape_fc_g_before_adj, mape_fc_h_before_adj, lng_plant, product):
    
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
                    mape_fc_a_before_adj,
                    mape_fc_b_before_adj,
                    mape_fc_c_before_adj,
                    mape_fc_d_before_adj,
                    mape_fc_e_before_adj,
                    mape_fc_f_before_adj,
                    mape_fc_g_before_adj,
                    mape_fc_h_before_adj,
                    created_by)
                    VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
          """
                
    #conn = None
    updated_rows = 0
    try:
        # create a new cursor
        cur = conn.cursor()
        # execute the UPDATE  statement
        cur.execute(sql, (lng_plant, product, date_now, mape_forecast_a, mape_forecast_b, mape_forecast_c, mape_forecast_d, mape_forecast_e, mape_forecast_f, mape_forecast_g, mape_forecast_h,
                          mape_fc_a_before_adj, mape_fc_b_before_adj, mape_fc_c_before_adj, mape_fc_d_before_adj, mape_fc_e_before_adj, mape_fc_f_before_adj, mape_fc_g_before_adj, mape_fc_h_before_adj, created_by))
        # get the number of updated rows
        updated_rows = cur.rowcount
        # Commit the changes to the database
        conn.commit()
        # Close cursor
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(error)

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
        logging.error(error)

    return updated_rows

def update_adj_value(conn, adj_forecast_a, adj_forecast_b, adj_forecast_c, 
                        adj_forecast_d, adj_forecast_e, adj_forecast_f, adj_forecast_g, adj_forecast_h,
                        lng_plant, product):
    
    date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    updated_by = 'PYTHON'
    
    """ insert mape result after last row in table """
    sql = """ INSERT INTO lng_analytics_adjustment
                    (lng_plant,
                    product,
                    running_date,
                    adj_forecast_a,
                    adj_forecast_b,
                    adj_forecast_c,
                    adj_forecast_d,
                    adj_forecast_e,
                    adj_forecast_f,
                    adj_forecast_g,
                    adj_forecast_h,
                    updated_by)
                    VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
          """
    
    #conn = None
    updated_rows = 0
    try:
        # create a new cursor
        cur = conn.cursor()
        # execute the UPDATE  statement
        cur.execute(sql, (lng_plant, product, date_now, adj_forecast_a, adj_forecast_b, adj_forecast_c, adj_forecast_d, adj_forecast_e, adj_forecast_f, adj_forecast_g, adj_forecast_h,
                          updated_by))
        # get the number of updated rows
        updated_rows = cur.rowcount
        # Commit the changes to the database
        conn.commit()
        # Close cursor
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(error)

    return updated_rows

def update_model_config(conn, model_choosen, lng_plant, lng_parameter):
    
    date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    updated_by = 'PYTHON'
      
    sql = """   UPDATE model_config_daily
                SET model_choosen = %s,
                    updated_at = %s,
                    updated_by = %s
                WHERE lng_plant = 'BP Tangguh' AND lng_parameter = 'LNG'
          """
    
    #conn = None
    updated_rows = 0
    try:
        # create a new cursor
        cur = conn.cursor()
        # execute the UPDATE  statement
        cur.execute(sql, (model_choosen, date_now, updated_by))
        # get the number of updated rows
        updated_rows = cur.rowcount
        # Commit the changes to the database
        conn.commit()
        # Close cursor
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(error)

    return updated_rows

if __name__ == "__main__":
    # getting the name of the directory
    # where the this file is present.
    current = os.path.dirname(os.path.abspath("__file__"))

    # Getting the parent directory name
    # where the current directory is present.
    parent = os.path.dirname(current)

    # Getting the parent directory name
    gr_parent = os.path.dirname(parent)

    # adding the parent directory to
    # the sys.path.
    sys.path.append(current)

    main()