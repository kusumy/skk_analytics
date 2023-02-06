###### FEED GAS BP TANGGUH FORECASTING INSAMPLE ######
# This python script is used to perform forecasting on testing data from each method.
# Data source from the SKK Migas (10.6.7.74) database in the lng_feed_gas_daily table with lng_plant = 'BP Tangguh'.

##### METHODS FOR TIME SERIES FORECASTING #####
# There are many methods that we can use for this forecasting, such as ARIMAX, SARIMAX, PROPHET, RANDOM FOREST, XGBOOST, LINEAR REGRESSION, POLYNOMIAL REGRESSION DEGREE 2, POLYNOMIAL REGRESSION DEGREE 3.

##### FLOW PROCESS OF THIS ALGORITHM #####
# 1. Import the required packages and defining functions that maybe used.
# 2. Import data from database.
# 3. EDA process (Search null values in column wpnb_gas and feed_gas, Stationary Check, Decomposition Plot, ACF-PACF Plot).
# 4. Data Preprocessing (Replace null values in column feed_gas with mean 1 year before, replace feed_gas values with unplanned shutdown case using mean 1 year before).
# 5. Split data after cleaning process to train and test.
# 6. Define the Forecasting Horizon. In this case, length for horizon is 365 data or 365 days.
# 7. Create exogenous variables to support the forecasting process. In this case, we use the data of month and day index, planned_shutdown and wpnb_gas data.
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
# For example : We will run this script only, we can comment (#) script main_lng_insample.py on other script .py (example: feed_gas_tangguh_forecasting_insample.py etc.)


# %%
import logging
import os
import sys
import numpy as np
import pandas as pd
import psycopg2
import time
from configparser import ConfigParser
import ast

from humanfriendly import format_timespan
from tokenize import Ignore
from datetime import datetime
from tracemalloc import start
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('fivethirtyeight')

import matplotlib as mpl
import numpy as np
import pandas as pd
import psycopg2

from adtk.data import validate_series
from adtk.detector import ThresholdAD
from adtk.visualization import plot
from sklearn.metrics import mean_absolute_percentage_error

pd.options.plotting.backend = "plotly"
from cProfile import label
from imaplib import Time2Internaldate

import statsmodels.api as sm
from dateutil.relativedelta import *
from pmdarima.arima.utils import ndiffs, nsdiffs
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.model_selection import (ForecastingGridSearchCV,
                                                SingleWindowSplitter,
                                                temporal_train_test_split)
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
from sktime.performance_metrics.forecasting import (MeanAbsolutePercentageError, MeanSquaredError)
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, message="Non-invertible starting MA parameters found.")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Model scoring for Cross Validation
mape = MeanAbsolutePercentageError(symmetric=False)
mse = MeanSquaredError()

# %%
def main():
    from connection import create_db_connection, get_sql_data
    from utils import (logMessage, ad_test, get_first_date_of_prev_month, get_last_date_of_prev_month,
                       get_last_date_of_current_year, end_day_forecast_april, get_first_date_of_november)
    from polyfit import PolynomRegressor
    import datetime
    
    # Connect to configuration file
    config = ConfigParser()
    config.read('config_lng.ini')
    
    # Accessing sections
    section_1 = config['config']
    
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
    logMessage("Creating Feed Gas BP Tangguh Model ...")
    
    # Connect to database
    # Exit program if not connected to database
    logMessage("Connecting to database ...")
    conn = create_db_connection(section='postgresql_ml_lng_skk')
    if conn == None:
        exit()
        
    logMessage("Cleaning data ...")
    ##### CLEANING FEED GAS DATA #####
    #Load Data from Database
    from datetime import datetime
    end_date = get_last_date_of_current_year()
    end_date_april = end_day_forecast_april()
    first_date_nov = get_first_date_of_november()
    current_date = datetime.now()
    date_nov = datetime.strptime(first_date_nov, "%Y-%m-%d")
    
    query = os.path.join('gas_prod/sql','fg_tangguh_data_query.sql')
    query_1 = open(query, mode="rt").read()
    sql = ''
    if USE_DEFAULT_DATE == True:
        if current_date < date_nov:
            sql = query_1.format('2016-01-01', end_date)
        else :
            sql = query_1.format('2016-01-01', end_date_april)
    else :
        sql = query_1.format(TRAIN_START_DATE, TRAIN_END_DATE)

    #print(sql)
    
    data = get_sql_data(sql, conn)
    data['date'] = pd.DatetimeIndex(data['date'], freq='D')
    data = data.reset_index()
    data['wpnb_gas'].fillna(method='ffill', inplace=True)

    #%%
    ds = 'date'
    y = 'feed_gas' 

    df = data[[ds,y]]
    df = df.set_index(ds)
    df.index = pd.DatetimeIndex(df.index, freq='D')

    #%%
    logMessage("Null Value Cleaning ...")
    data_null_cleaning = data[['date', 'feed_gas', 'wpnb_gas', 'unplanned_shutdown', 'planned_shutdown']].copy()
    data_null_cleaning['feed_gas_copy'] = data[['feed_gas']].copy()
    ds_null_cleaning = 'date'
    data_null_cleaning = data_null_cleaning.set_index(ds_null_cleaning)
    data_null_cleaning['unplanned_shutdown'] = data_null_cleaning['unplanned_shutdown'].astype('int')
    s = validate_series(data_null_cleaning)

    #%%
    # Detect Anomaly Values
    threshold_ad = ThresholdAD(data_null_cleaning['feed_gas_copy'].isnull())
    anomalies = threshold_ad.detect(s)

    anomalies = anomalies.drop('feed_gas', axis=1)
    anomalies = anomalies.drop('wpnb_gas', axis=1)
    anomalies = anomalies.drop('unplanned_shutdown', axis=1)
    anomalies = anomalies.drop('planned_shutdown', axis=1)

    #%%
    # Copy data frame of anomalies
    copy_anomalies =  anomalies.copy()
    # Rename columns
    copy_anomalies.rename(columns={'feed_gas_copy':'anomaly'}, inplace=True)
    # Merge original dataframe with anomalies
    new_s = pd.concat([s, copy_anomalies], axis=1)

    # Get only anomalies data
    anomalies_data = new_s[new_s['anomaly'].isnull()]

    #%%
    #REPLACE ANOMALY VALUES
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
        mean_month=new_s['feed_gas'].reset_index().query(sql).mean().values[0]
        
        # update value at specific location
        new_s.at[index,'feed_gas'] = mean_month


    #%%
    logMessage("Unplanned Shutdown Cleaning ...")
    # Detect Unplanned Shutdown Value
    data2 = new_s[['feed_gas', 'wpnb_gas', 'unplanned_shutdown', 'planned_shutdown']].copy()
    #data2 = data2.reset_index()
    #data2['date'] = pd.DatetimeIndex(data2['date'], freq='D')
    s2 = validate_series(data2)

    threshold_ad2 = ThresholdAD(data2['unplanned_shutdown']==0)
    anomalies2 = threshold_ad2.detect(s2)

    anomalies2 = anomalies2.drop('feed_gas', axis=1)
    anomalies2 = anomalies2.drop('wpnb_gas', axis=1)
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
        #sql = "date>='"+start_month+ "' & "+ "date<='" +end_month+"'"
        
        sql = "date>='"+prev_date_year+ "' & "+ "date<='" +yesterday_date+"'"
        mean_month=new_s2['feed_gas'].reset_index().query(sql).mean(skipna = True).values[0] 
        
        # update value at specific location
        new_s2.at[index,'feed_gas'] = mean_month
        
        #print(index), print(sql), print(mean_month)

    # Check if updated
    new_s2[new_s2['anomaly'] == False]

    #%%
    logMessage("Final Data Prepare ...")
    data_cleaned = new_s2[['feed_gas', 'wpnb_gas', 'planned_shutdown']].copy()
    data_cleaned = data_cleaned.reset_index()

    ds_cleaned = 'date'
    y_cleaned = 'feed_gas'
    df_cleaned = data_cleaned[[ds_cleaned, y_cleaned]]
    df_cleaned = df_cleaned.set_index(ds_cleaned)
    df_cleaned.index = pd.DatetimeIndex(df_cleaned.index, freq='D')

    #Select column target
    train_df = df_cleaned['feed_gas']

    #%%
    #stationarity_check(train_df)

    #%%
    #decomposition_plot(train_df)

    #%%
    #plot_acf_pacf(df_cleaned['feed_gas'])

    #%%
    logMessage("AD Fuller Test ...")
    ad_test(train_df)

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
    # create features (exog) from date
    df_cleaned['month'] = [i.month for i in df_cleaned.index]
    df_cleaned['planned_shutdown'] = data['planned_shutdown'].values
    df_cleaned['day'] = [i.day for i in df_cleaned.index]
    df_cleaned['wpnb_gas'] = data['wpnb_gas'].values
    df_cleaned['wpnb_gas'].fillna(method='ffill', inplace=True)

    #%%
    # Split into train and test
    X_train, X_test = temporal_train_test_split(df_cleaned.iloc[:,1:], test_size=test_size)
    #X_train

    #%%
    exogenous_features = ["month", "planned_shutdown", "day", "wpnb_gas"]
    
    # Empty y_train and data
    y_train = None
    data = None
    
    
    ###### FORECASTING ######
    ##### SARIMAX MODEL (forecast_b) #####
    logMessage("Creating Sarimax Model Forecasting Insample Feed Gas BP Tangguh ...")
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
    
    #sarimax_model = auto_arima(y=y_train_cleaned.feed_gas, X=X_train[exogenous_features], d=0, D=1, seasonal=True, m=12, trace=True, error_action="ignore", suppress_warnings=True)
    sarimax_model = AutoARIMA(start_p = start_p, max_p = max_p, start_q = start_q, max_q = max_q, d=sarimax_differencing, 
                              start_P = start_P, max_P = max_P, start_Q = start_Q, max_Q = max_Q, D=sarimax_seasonal_differencing, seasonal=sarimax_seasonal, sp=sarimax_sp,
                              trace=sarimax_trace, n_fits=sarimax_n_fits, stepwise=sarimax_stepwise, error_action=sarimax_error_action, suppress_warnings=sarimax_suppress_warnings)
    logMessage("Creating SARIMAX Model ...") 
    sarimax_fit = sarimax_model.fit(y_train_cleaned.feed_gas, X=X_train[exogenous_features])
    logMessage("SARIMAX Model Summary")
    logMessage(sarimax_fit.summary())
    
    logMessage("SARIMAX Model Prediction ..")
    sarimax_forecast = sarimax_fit.predict(fh, X=X_test[exogenous_features]) #len(fh)

    #Create MAPE
    sarimax_mape = mean_absolute_percentage_error(y_test_cleaned.feed_gas, sarimax_forecast)
    sarimax_mape_str = str('MAPE: %.4f' % sarimax_mape)
    logMessage("SARIMAX Model "+sarimax_mape_str)
    
    #Get parameters
    sarimax_param_order = sarimax_fit.get_fitted_params()['order']
    sarimax_param_order_seasonal = sarimax_fit.get_fitted_params()['seasonal_order']
    sarimax_param = str({'sarimax_order': sarimax_param_order, 'sarimax_seasonal_order': sarimax_param_order_seasonal})
    logMessage("Sarimax Model Parameters "+sarimax_param)
    
    # Empty the SARIMAX memory
    sarimax_model = None
    sarimax_forecast = None
    sarimax_param_order = None
    sarimax_param_order_seasonal = None
    sarimax_fit = None
    
    ##### ARIMAX MODEL (forecast_a) #####
    logMessage("Creating Arimax Model Forecasting Insample Feed Gas BP Tangguh ...")
    # %%
    # Create ARIMAX (forecast_a) Model
    arimax_model = AutoARIMA(d=0, suppress_warnings=True, error_action='ignore')
    logMessage("Creating ARIMAX Model ...")
    arimax_fit = arimax_model.fit(y_train_cleaned.feed_gas, X=X_train[exogenous_features])
    logMessage("ARIMAX Model Summary")
    logMessage(arimax_fit.summary())
    
    logMessage("ARIMAX Model Prediction ..")
    arimax_forecast = arimax_fit.predict(fh, X=X_test[exogenous_features])

    #Create MAPE
    arimax_mape = mean_absolute_percentage_error(y_test_cleaned.feed_gas, arimax_forecast)
    arimax_mape_str = str('MAPE: %.4f' % arimax_mape)
    logMessage("ARIMAX Model "+arimax_mape_str)
    
    #Get parameter
    arimax_param = str(arimax_fit.get_fitted_params()['order'])
    logMessage("Arimax Model Parameters "+arimax_param)
    
    # Empty the SARIMAX memory
    arimax_model = None
    arimax_forecast = None
    arimax_fit = None
    

    ##### PROPHET MODEL (forecast_c) #####
    logMessage("Creating Prophet Model Forecasting Insample Feed Gas BP Tangguh ...")
    #%%
    #Set Parameters
    prophet_param_grid = {'seasonality_mode':['additive','multiplicative']
                        ,'n_changepoints':[2,11,19]
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
    gscv_prophet = ForecastingGridSearchCV(prophet_forecaster, cv=cv_prophet, param_grid=prophet_param_grid, scoring=mape)

    logMessage("Creating Prophet Model ...")
    prophet_fit = gscv_prophet.fit(y_train_cleaned, X_train) #, X_train

    # Show best model parameters
    logMessage("Show Best Prophet Models ...")
    prophet_best_params = prophet_fit.best_params_
    prophet_best_params_str = str(prophet_best_params)
    logMessage("Best Prophet Models "+prophet_best_params_str)

    logMessage("Prophet Model Prediction ...")
    prophet_forecast = prophet_fit.best_forecaster_.predict(fh, X=X_test)#, X=X_test

    #Create MAPE
    prophet_mape = mean_absolute_percentage_error(y_test_cleaned['feed_gas'], prophet_forecast)
    prophet_mape_str = str('MAPE: %.4f' % prophet_mape)
    logMessage("Prophet Model "+prophet_mape_str)

    # Empty the Prophet memory
    prophet_param_grid = None
    cv_prophet = None
    gscv_prophet = None
    prophet_forecast = None
    prophet_fit = None
    

    ##### RANDOM FOREST MODEL (forecast_d) #####
    logMessage("Creating Random Forest Model Forecasting Insample Feed Gas BP Tangguh ...")
    #%%
    #Set Parameters
    ranfor_random_state = 0
    ranfor_criterion = "squared_error"
    ranfor_strategy = "recursive"

    ranfor_forecaster_param_grid = {"window_length": [2, 11, 19, 21, 29], 
                                    "estimator__n_estimators": [150, 200]}

    # create regressor object
    ranfor_regressor = RandomForestRegressor(random_state = ranfor_random_state, criterion = ranfor_criterion)
    ranfor_forecaster = make_reduction(ranfor_regressor, strategy = ranfor_strategy)

    logMessage("Creating Window Splitter Random Forest Model ....")   
    cv_ranfor = SingleWindowSplitter(fh=fh_int)
    gscv_ranfor = ForecastingGridSearchCV(ranfor_forecaster, cv=cv_ranfor, param_grid=ranfor_forecaster_param_grid, scoring=mape)

    logMessage("Creating Random Forest Model ...")
    ranfor_fit = gscv_ranfor.fit(y_train_cleaned, X_train) #, X_train

    # Show best model parameters
    logMessage("Show Best Random Forest Models ...")
    ranfor_best_params = ranfor_fit.best_params_
    ranfor_best_params_str = str(ranfor_best_params)
    logMessage("Best Random Forest Models "+ranfor_best_params_str)
    
    logMessage("Random Forest Model Prediction ...")
    ranfor_forecast = ranfor_fit.best_forecaster_.predict(fh, X=X_test) #, X=X_test

    #Create MAPE
    ranfor_mape = mean_absolute_percentage_error(y_test_cleaned['feed_gas'], ranfor_forecast)
    ranfor_mape_str = str('MAPE: %.4f' % ranfor_mape)
    logMessage("Random Forest Model "+ranfor_mape_str)
    
    # Empty Random Forest Memory
    ranfor_forecaster_param_grid = None
    ranfor_regressor = None
    ranfor_forecaster = None
    cv_ranfor = None
    gscv_ranfor = None
    ranfor_forecast = None
    ranfor_fit = None


    ##### XGBOOST MODEL (forecast_e) #####
    logMessage("Creating XGBoost Model Forecasting Insample Feed Gas BP Tangguh ...")
    #%%
    #Set Parameters
    xgb_objective = 'reg:squarederror'
    xgb_strategy = "recursive"

    xgb_forecaster_param_grid = {"window_length": [2, 6, 7, 11, 19, 27]
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
    xgb_mape = mean_absolute_percentage_error(y_test_cleaned['feed_gas'], xgb_forecast)
    xgb_mape_str = str('MAPE: %.4f' % xgb_mape)
    logMessage("XGBoost Model "+xgb_mape_str)
    
    # Empty Random Forest Memory
    xgb_forecaster_param_grid = None
    xgb_regressor = None
    xgb_forecaster = None
    cv_xgb = None
    gscv_xgb = None
    xgb_forecast = None
    xgb_fit = None


    ##### LINEAR REGRESSION MODEL (forecast_f) #####
    logMessage("Creating Linear Regression Model Forecasting Insample Feed Gas BP Tangguh ...")
    #%%
    #Set Parameters
    linreg_strategy = "recursive"
    linreg_forecaster_param_grid = {"window_length": [2, 6, 7, 11, 19, 27]}

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
    linreg_mape = mean_absolute_percentage_error(y_test_cleaned['feed_gas'], linreg_forecast)
    linreg_mape_str = str('MAPE: %.4f' % linreg_mape)
    logMessage("Linear Regression Model "+linreg_mape_str)
    
    # Empty Linear Regression Memory
    linreg_forecaster_param_grid = None
    linreg_regressor = None
    linreg_forecaster = None
    cv_linreg = None
    gscv_linreg = None
    linreg_forecast = None
    linreg_fit = None
    

    ##### POLYNOMIAL REGRESSION DEGREE=2 MODEL (forecast_g) #####
    logMessage("Creating Polynomial Regression Degree=2 Model Forecasting Insample Feed Gas BP Tangguh ...")
    #%%
    #Set Parameters
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
    poly2_mape = mean_absolute_percentage_error(y_test_cleaned['feed_gas'], poly2_forecast)
    poly2_mape_str = str('MAPE: %.4f' % poly2_mape)
    logMessage("Polynomial Regression Degree=2 Model "+poly2_mape_str)
    
    # Empty Polynomial Regression Degree=2 Memory
    poly2_forecaster_param_grid = None
    poly2_regressor = None
    poly2_forecaster = None
    cv_poly2 = None
    gscv_poly2 = None
    poly2_forecast = None
    poly2_fit = None  


    ##### POLYNOMIAL REGRESSION DEGREE=3 MODEL (forecast_h) #####
    logMessage("Creating Polynomial Regression Degree=3 Model Forecasting Insample Feed Gas BP Tangguh ...")
    #%%
    #Set Parameters
    poly3_regularization = None
    poly3_interactions = False
    poly3_strategy = "recursive"

    poly3_forecaster_param_grid = {"window_length": [0.8]}
    poly3_regressor = PolynomRegressor(deg=3, regularization=poly3_regularization, interactions=poly3_interactions)
    poly3_forecaster = make_reduction(poly3_regressor, strategy=poly3_strategy)

    cv_poly3 = SingleWindowSplitter(fh=fh_int)
    gscv_poly3 = ForecastingGridSearchCV(poly3_forecaster, cv=cv_poly3, param_grid=poly3_forecaster_param_grid, scoring=mape, error_score='raise')

    logMessage("Creating Polynomial Regression Orde 3 Model ...")
    poly3_fit = gscv_poly3.fit(y_train_cleaned, X=X_train) #, X=X_train

    # Show best model parameters
    logMessage("Show Best Polynomial Regression Degree=3 Models ...")
    poly3_best_params = poly3_fit.best_params_
    poly3_best_params_str = str(poly3_best_params)
    logMessage("Best Polynomial Regression Degree=3 Models "+poly3_best_params_str)
    
    logMessage("Polynomial Regression Degree=3 Model Prediction ...")
    poly3_forecast = poly3_fit.best_forecaster_.predict(fh, X=X_test) #, X=X_test

    #Create MAPE
    poly3_mape = mean_absolute_percentage_error(y_test_cleaned['feed_gas'], poly3_forecast)
    poly3_mape_str = str('MAPE: %.4f' % poly3_mape)
    logMessage("Polynomial Regression Degree=3 Model "+poly3_mape_str)
    
    # Empty Polynomial Regression Degree=2 Memory
    poly3_forecaster_param_grid = None
    poly3_regressor = None
    poly3_forecaster = None
    cv_poly3 = None
    gscv_poly3 = None
    poly3_forecast = None   
    poly3_fit = None
    

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
                'lng_plant' : 'BP Tangguh',
                'product' : 'Feed Gas'}

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
                        'product' : 'Feed Gas'}

    all_model_param = pd.DataFrame(all_model_param)
      
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