###### CONDENSATE PT BADAK FORECASTING INSAMPLE ######
# This python script is used to perform forecasting on testing data from each method.
# Data source from the SKK Migas (188.166.239.112) database in the lng_condensate_daily table with lng_plant = 'BP Tangguh'.

##### METHODS FOR TIME SERIES FORECASTING #####
# There are many methods that we can use for this forecasting, such as ARIMAX, SARIMAX, PROPHET, RANDOM FOREST, XGBOOST, LINEAR REGRESSION, POLYNOMIAL REGRESSION DEGREE 2, POLYNOMIAL REGRESSION DEGREE 3.

##### FLOW PROCESS OF THIS ALGORITHM #####
# 1. Import the required packages and defining functions that maybe used.
# 2. Import data from database.
# 3. EDA process (Search null values in column condensate, Stationary Check.
# 4. Data Preprocessing (Replace null values in condensate with value before null.
# 5. Split data after cleaning process to train and test.
# 6. Define the Forecasting Horizon. In this case, length for horizon is 0.2 from total data.
# 7. Create exogenous variables to support the forecasting process. In this case, we use the data of day index.
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
# 12. Create all mape result to dataframe.
# 13. Define function and query to save the model parameter and mape value to database.

##### SCRIPT OUTPUT #####
# The output for this script is best parameter and error value of each forecasting method. Which is best parameter and error value will be save in database table.

##### HOW TO USE THIS SCRIPT #####
# We can run this script using command prompt (directory same with this python script). But in this case, we can run this script using main_lng.py.
# For example : We will run this script only, we can comment (#) script main_lng_insample.py on other script .py (example: feed_gas_tangguh_forecasting_insample.py etc.)

# %%
import logging
import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import pmdarima as pm
import psycopg2
import seaborn as sns
import time
from configparser import ConfigParser
import ast

from tokenize import Ignore
from datetime import datetime
from tracemalloc import start
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('fivethirtyeight')

from adtk.data import validate_series
from adtk.detector import ThresholdAD
from adtk.visualization import plot
from pmdarima import model_selection
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)

pd.options.plotting.backend = "plotly"
from cProfile import label
from imaplib import Time2Internaldate

import statsmodels.api as sm
from dateutil.relativedelta import *
from pmdarima.arima import ARIMA, auto_arima
from pmdarima.arima.utils import ndiffs, nsdiffs
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.model_selection import (ExpandingWindowSplitter,
                                                ForecastingGridSearchCV,
                                                ForecastingRandomizedSearchCV,
                                                SingleWindowSplitter,
                                                SlidingWindowSplitter,
                                                temporal_train_test_split)
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
from sktime.performance_metrics.forecasting import (
    MeanAbsolutePercentageError, MeanSquaredError)
from xgboost import XGBRegressor

# Model scoring for Cross Validation
mape = MeanAbsolutePercentageError(symmetric=False)
mse = MeanSquaredError()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# %%
def main():
    from connection import config, retrieve_data, create_db_connection, get_sql_data
    from utils import logMessage, ad_test, get_first_date_of_prev_month, get_last_date_of_prev_month, get_last_date_of_current_year, end_day_forecast_april, get_first_date_of_november
    from polyfit import PolynomRegressor
    import datetime
    
    config = ConfigParser()
    config.read('config_lng.ini')
    section = config['config']

    USE_DEFAULT_DATE = section.getboolean('use_default_date')

    TRAIN_START_YEAR= section.getint('train_start_year')
    TRAIN_START_MONTH = section.getint('train_start_month')
    TRAIN_START_DAY = section.getint('train_start_day')

    TRAIN_END_YEAR= section.getint('train_end_year')
    TRAIN_END_MONTH = section.getint('train_end_month')
    TRAIN_END_DAY = section.getint('train_end_day')

    FORECAST_START_YEAR= section.getint('forecast_start_year')
    FORECAST_START_MONTH = section.getint('forecast_start_month')
    FORECAST_START_DAY = section.getint('forecast_start_day')

    FORECAST_END_YEAR= section.getint('forecast_end_year')
    FORECAST_END_MONTH = section.getint('forecast_end_month')
    FORECAST_END_DAY = section.getint('forecast_end_day')

    TRAIN_START_DATE = (datetime.date(TRAIN_START_YEAR, TRAIN_START_MONTH, TRAIN_START_DAY)).strftime("%Y-%m-%d")
    TRAIN_END_DATE = (datetime.date(TRAIN_END_YEAR, TRAIN_END_MONTH, TRAIN_END_DAY)).strftime("%Y-%m-%d")
    FORECAST_START_DATE = (datetime.date(FORECAST_START_YEAR, FORECAST_START_MONTH, FORECAST_START_DAY)).strftime("%Y-%m-%d")
    FORECAST_END_DATE = (datetime.date(FORECAST_END_YEAR, FORECAST_END_MONTH, FORECAST_END_DAY)).strftime("%Y-%m-%d")

    # Configure logging
    #configLogging("condensate_tangguh.log")
    logMessage("Creating Condensate Tangguh Model ...")
    
    # Connect to database
    # Exit program if not connected to database
    logMessage("Connecting to database ...")
    conn = create_db_connection(section='postgresql_ml_lng_skk')
    if conn == None:
        exit()
        
    logMessage("Cleaning data ...")
    ##### CLEANING CONDENSATE DATA #####
    #Load Data from Database
    from datetime import datetime
    end_date = get_last_date_of_current_year()
    end_date_april = end_day_forecast_april()
    first_date_nov = get_first_date_of_november()
    current_date = datetime.now()
    date_nov = datetime.strptime(first_date_nov, "%Y-%m-%d")
    
    query = os.path.join('gas_prod/sql','condensate_tangguh_data_query.sql')
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
    data['wpnb_oil'].fillna(method='ffill', inplace=True)
    data = data.reset_index()

    #%%
    ds = 'date'
    y = 'condensate' 

    df = data[[ds,y]]
    df = df.set_index(ds)
    df.index = pd.DatetimeIndex(df.index, freq='D')

    #%%
    logMessage("Null Value Cleaning ...")
    data_null_cleaning = data[['date', 'condensate', 'wpnb_oil', 'unplanned_shutdown', 'planned_shutdown']].copy()
    data_null_cleaning['condensate_copy'] = data[['condensate']].copy()
    ds_null_cleaning = 'date'
    data_null_cleaning = data_null_cleaning.set_index(ds_null_cleaning)
    data_null_cleaning['unplanned_shutdown'] = data_null_cleaning['unplanned_shutdown'].astype('int')
    s = validate_series(data_null_cleaning)

    #%%
    # Calculate standar deviation
    fg_std = data_null_cleaning['condensate'].std()
    fg_mean = data_null_cleaning['condensate'].mean()

    #Detect Anomaly Values
    # Create anomaly detection model
    high_limit1 = fg_mean+3*fg_std
    low_limit1 = fg_mean-3*fg_std
    high_limit2 = fg_mean+fg_std
    low_limit2 = fg_mean-fg_std

    threshold_ad = ThresholdAD(data_null_cleaning['condensate_copy'].isnull())
    anomalies = threshold_ad.detect(s)

    anomalies = anomalies.drop('condensate', axis=1)
    anomalies = anomalies.drop('wpnb_oil', axis=1)
    anomalies = anomalies.drop('unplanned_shutdown', axis=1)
    anomalies = anomalies.drop('planned_shutdown', axis=1)
    
    #%%
    # Create anomaly detection model
    #threshold_ad = ThresholdAD(high=high_limit2, low=low_limit1)
    #anomalies =  threshold_ad.detect(s)

    # Copy data frame of anomalies
    copy_anomalies =  anomalies.copy()
    # Rename columns
    copy_anomalies.rename(columns={'condensate_copy':'anomaly'}, inplace=True)
    # Merge original dataframe with anomalies
    new_s = pd.concat([s, copy_anomalies], axis=1)

    # Get only anomalies data
    anomalies_data = new_s[new_s['anomaly'].isnull()]
    #anomalies_data.tail(100)

    #%%
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
        
        #print(sql), print(mean_month)

    # Check if updated
    anomaly_upd = new_s[new_s['anomaly'].isnull()]

    #%%
    # Plot data and its anomalies
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
    plt.close()
    
    #%%
    logMessage("Unplanned Shutdown Cleaning ...")
    # Detect Unplanned Shutdown Value
    data2 = new_s[['condensate', 'wpnb_oil', 'unplanned_shutdown', 'planned_shutdown']].copy()
    #data2 = data2.reset_index()
    #data2['date'] = pd.DatetimeIndex(data2['date'], freq='D')
    s2 = validate_series(data2)

    threshold_ad2 = ThresholdAD(data2['unplanned_shutdown']==0)
    anomalies2 = threshold_ad2.detect(s2)

    anomalies2 = anomalies2.drop('condensate', axis=1)
    anomalies2 = anomalies2.drop('wpnb_oil', axis=1)
    anomalies2 = anomalies2.drop('planned_shutdown', axis=1)
    
    # Create anomaly detection model
    #threshold_ad2 = ThresholdAD(high=high_limit2, low=low_limit1)
    #anomalies2 =  threshold_ad2.detect(s2)

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

    print(yesterday_date)
    print(prev_date_year)

       
    #%%
    for index, row in anomalies_data2.iterrows():
        yr = index.year
        mt = index.month
        
        # Get start month and end month
        #start_month = str(get_first_date_of_current_month(yr, mt))
        #end_month = str(get_last_date_of_month(yr, mt))
        
        # Get last year start date month
        #start_month = get_first_date_of_prev_month(yr,mt,step=-12)
        
        # Get last month last date
        #end_month = get_last_date_of_prev_month(yr,mt,step=-1)
        
        # Get mean fead gas data for the month
        #sql = "date>='"+start_month+ "' & "+ "date<='" +end_month+"'"
        yesterday_date = index - datetime.timedelta(days=1)
        prev_date_year = yesterday_date - datetime.timedelta(days=364)
        
        yesterday_date = yesterday_date.strftime("%Y-%m-%d")
        prev_date_year = prev_date_year.strftime("%Y-%m-%d")
        sql = "date>='"+prev_date_year+ "' & "+ "date<='" +yesterday_date+"'"
        mean_month=new_s2['condensate'].reset_index().query(sql).mean(skipna = True).values[0] 
        
        # update value at specific location
        new_s2.at[index,'condensate'] = mean_month
        
        #print(index), print(sql), print(mean_month)

    # Check if updated
    new_s2[new_s2['anomaly'] == False]

    anomaly_upd2 = new_s2[new_s2['anomaly'] == False]

    #%%
    # Plot data and its anomalies
    fig = px.line(new_s2, y='condensate')

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
    fig.add_scatter(x=anomalies_data2.index, y=anomalies_data2['condensate'], mode='markers', marker=dict(color='red'), name="Unplanned Shutdown", showlegend=True)
    fig.add_scatter(x=anomaly_upd2.index, y=anomaly_upd2['condensate'], mode='markers', marker=dict(color='green'), name="Unplanned Cleaned", showlegend=True)
    fig.update_layout(title_text='Condensate BP Tangguh', title_font_size=24)

    #fig.show()
    plt.close()

    #%%
    data_cleaned = new_s2[['condensate', 'wpnb_oil', 'planned_shutdown']].copy()
    data_cleaned = data_cleaned.reset_index()

    ds_cleaned = 'date'
    y_cleaned = 'condensate'
    df_cleaned = data_cleaned[[ds_cleaned, y_cleaned]]
    df_cleaned = df_cleaned.set_index(ds_cleaned)
    df_cleaned.index = pd.DatetimeIndex(df_cleaned.index, freq='D')
    df_cleaned['condensate'].fillna(method='ffill', inplace=True)

    #Select column target
    train_df = df_cleaned['condensate']

    #%%
    #stationarity_check(train_df)

    #%%
    #decomposition_plot(train_df)

    #%%
    #plot_acf_pacf(df_cleaned['condensate'])

    #%%
    # from chart_studio.plotly import plot_mpl
    # from statsmodels.tsa.seasonal import seasonal_decompose
    # result = seasonal_decompose(train_df.values, model="additive", period=365)
    # fig = result.plot()
    # plt.close()

    #%%
    # Ad-Fuller Testing
    ad_test(train_df)

    #%%
    # Test size
    test_size = 0.2 #365
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
    df_cleaned['wpnb_oil'] = data['wpnb_oil'].values
    df_cleaned['wpnb_oil'].fillna(method='ffill', inplace=True)

    #%%
    # Split into train and test
    X_train, X_test = temporal_train_test_split(df_cleaned.iloc[:,1:], test_size=test_size)
    
    #%%
    exogenous_features = ["month", "planned_shutdown", "day", "wpnb_oil"]

    #%%
    # plotting for illustration
    fig1, ax = plt.subplots(figsize=(20,8))
    ax.plot(y_train, label='train')
    ax.plot(y_test, label='test')
    ax.set_ylabel("Condensate")
    ax.set_xlabel("Datestamp")
    ax.legend(loc='best')
    plt.close()


    ##### FORECASTING METHODS #####
    # %%
    logMessage("Creating Arimax Model Forecasting Insample Condensate BP Tangguh ...")
    # Create ARIMAX (forecast_a) Model
    arimax_model = AutoARIMA(d=0, suppress_warnings=True, error_action='ignore')
    logMessage("Creating ARIMAX Model ...")
    arimax_model.fit(y_train_cleaned, X=X_train)
    logMessage("ARIMAX Model Summary")
    logMessage(arimax_model.summary())

    logMessage("ARIMAX Model Prediction ..")
    arimax_forecast = arimax_model.predict(fh, X=X_test)
    y_pred_arimax = pd.DataFrame(arimax_forecast).applymap('{:.2f}'.format)

    #Create MAPE
    arimax_mape = mean_absolute_percentage_error(y_test_cleaned.condensate, arimax_forecast)
    arimax_mape_str = str('MAPE: %.4f' % arimax_mape)
    logMessage("ARIMAX Model "+arimax_mape_str)
    
    #Get parameter
    arimax_param = str(arimax_model.get_fitted_params()['order'])
    logMessage("Arimax Model Parameters "+arimax_param)

    ##### SARIMAX MODEL (forecast_b) #####
    #%%
    logMessage("Creating Sarimax Model Forecasting Insample Condensate BP Tangguh ...")
    #Set parameters
    sarimax_differencing = 0
    sarimax_seasonal_differencing = 1
    sarimax_sp = 12
    sarimax_stationary = False
    sarimax_seasonal = True
    sarimax_trace = True
    sarimax_error_action = "ignore"
    sarimax_suppress_warnings = True
    sarimax_n_fits = 50
    sarimax_stepwise = True
    
    sarimax_model = AutoARIMA(start_p = 0, max_p = 3, d=sarimax_differencing, max_q = 2, max_P = 2, max_Q = 2, D=sarimax_seasonal_differencing, seasonal=sarimax_seasonal, sp=sarimax_sp, trace=sarimax_trace, n_fits=sarimax_n_fits, stepwise=sarimax_stepwise, error_action=sarimax_error_action, suppress_warnings=sarimax_suppress_warnings)
    logMessage("Creating SARIMAX Model ...")
    sarimax_model.fit(y_train_cleaned, X=X_train)
    logMessage("SARIMAX Model Summary")
    logMessage(sarimax_model.summary())

    logMessage("SARIMAX Model Prediction ..")
    sarimax_forecast = sarimax_model.predict(fh, X=X_test)
    y_pred_sarimax = pd.DataFrame(sarimax_forecast).applymap('{:.2f}'.format)

    #Create MAPE
    sarimax_mape = mean_absolute_percentage_error(y_test_cleaned.condensate, sarimax_forecast)
    sarimax_mape_str = str('MAPE: %.4f' % sarimax_mape)
    logMessage("SARIMAX Model "+sarimax_mape_str)
    
    #Get parameters
    sarimax_param_order = sarimax_model.get_fitted_params()['order']
    sarimax_param_order_seasonal = sarimax_model.get_fitted_params()['seasonal_order']
    sarimax_param = str({'sarimax_order': sarimax_param_order, 'sarimax_seasonal_order': sarimax_param_order_seasonal})
    logMessage("Sarimax Model Parameters "+sarimax_param)

    ##### PROPHET MODEL (forecast_c) #####
    #%%
    logMessage("Creating Prophet Model Forecasting Insample Condensate BP Tangguh ...")
    # Create Prophet Parameter Grid
    prophet_param_grid = {'seasonality_mode':['additive','multiplicative']
                        ,'n_changepoints':[3, 6, 9, 18, 27]
                        ,'seasonality_prior_scale':[0.2, 0.1] #Flexibility of the seasonality (0.01,10)
                        ,'changepoint_prior_scale':[0.1, 0.5] #Flexibility of the trend (0.001,0.5)
                        ,'daily_seasonality':[5,10]
                        ,'weekly_seasonality':[1,5]
                        ,'yearly_seasonality':[8,10]
                        }

    logMessage("Creating Prophet Regressor Object ....") 
    # create regressor object
    prophet_forecaster = Prophet()

    logMessage("Creating Window Splitter Prophet Model ....")   
    cv_prophet = SingleWindowSplitter(fh=fh_int)
    gscv_prophet = ForecastingGridSearchCV(prophet_forecaster, cv=cv_prophet, param_grid=prophet_param_grid, n_jobs=-1, scoring=mape)

    logMessage("Creating Prophet Model ...")
    gscv_prophet.fit(y_train_cleaned, X_train) #, X_train

    # Show top 10 best models based on scoring function
    gscv_prophet.cv_results_.sort_values(by='rank_test_MeanAbsolutePercentageError', ascending=True)

    # Show best model parameters
    logMessage("Show Best Prophet Models ...")
    prophet_best_params = gscv_prophet.best_params_
    prophet_best_params_str = str(prophet_best_params)
    logMessage("Best Prophet Models "+prophet_best_params_str)

    logMessage("Prophet Model Prediction ...")
    prophet_forecast = gscv_prophet.best_forecaster_.predict(fh, X=X_test)#, X=X_test
    y_pred_prophet = pd.DataFrame(prophet_forecast).applymap('{:.2f}'.format)

    #Create MAPE
    prophet_mape = mean_absolute_percentage_error(y_test_cleaned['condensate'], prophet_forecast)
    prophet_mape_str = str('MAPE: %.4f' % prophet_mape)
    logMessage("Prophet Model "+prophet_mape_str)
    
    ##### RANDOM FOREST MODEL (forecast_d) #####
    #%%
    logMessage("Creating Random Forest Model Forecasting Insample Condensate BP Tangguh ...")
    # Create Random Forest Parameter Grid
    ranfor_random_state = 0
    ranfor_criterion = "squared_error"
    ranfor_strategy = "recursive"

    # create regressor object
    ranfor_forecaster_param_grid = {"window_length": [3, 6, 9, 18, 27], 
                                    "estimator__n_estimators": [100, 150]}

    # create regressor object
    ranfor_regressor = RandomForestRegressor(random_state = ranfor_random_state, criterion = ranfor_criterion, n_jobs=-1)
    ranfor_forecaster = make_reduction(ranfor_regressor, strategy = ranfor_strategy)

    logMessage("Creating Window Splitter Random Forest Model ....")   
    cv_ranfor = SingleWindowSplitter(fh=fh_int)
    gscv_ranfor = ForecastingGridSearchCV(ranfor_forecaster, cv=cv_ranfor, param_grid=ranfor_forecaster_param_grid, n_jobs=-1, scoring=mape)

    logMessage("Creating Random Forest Model ...")
    gscv_ranfor.fit(y_train_cleaned, X_train) #, X_train

    # Show top 10 best models based on scoring function
    gscv_ranfor.cv_results_.sort_values(by='rank_test_MeanAbsolutePercentageError', ascending=True)

    # Show best model parameters
    logMessage("Show Best Random Forest Models ...")
    ranfor_best_params = gscv_ranfor.best_params_
    ranfor_best_params_str = str(ranfor_best_params)
    logMessage("Best Random Forest Models "+ranfor_best_params_str)
    
    logMessage("Random Forest Model Prediction ...")
    ranfor_forecast = gscv_ranfor.best_forecaster_.predict(fh, X=X_test) #, X=X_test
    y_pred_ranfor = pd.DataFrame(ranfor_forecast).applymap('{:.2f}'.format)

    #Create MAPE
    ranfor_mape = mean_absolute_percentage_error(y_test_cleaned['condensate'], y_pred_ranfor)
    ranfor_mape_str = str('MAPE: %.4f' % ranfor_mape)
    logMessage("Random Forest Model "+ranfor_mape_str)

    ##### XGBOOST MODEL (forecast_e) #####
    #%%
    logMessage("Creating XGBoost Model Forecasting Insample Condensate BP Tangguh ...")
    # Create XGBoost Parameter Grid
    xgb_objective = 'reg:squarederror'
    xgb_strategy = "recursive"

    xgb_forecaster_param_grid = {"window_length": [3, 6, 9, 18, 27]
                                ,"estimator__n_estimators": [100, 200, 300]
                                }

    xgb_regressor = XGBRegressor(objective=xgb_objective, n_jobs=-1, seed = 42)
    xgb_forecaster = make_reduction(xgb_regressor, strategy=xgb_strategy)

    cv_xgb = SingleWindowSplitter(fh=fh_int)
    gscv_xgb = ForecastingGridSearchCV(xgb_forecaster, cv=cv_xgb, param_grid=xgb_forecaster_param_grid, n_jobs=-1, scoring=mape)

    logMessage("Creating XGBoost Model ....")
    gscv_xgb.fit(y_train_cleaned, X=X_train) #, X_train

    # Show top 10 best models based on scoring function
    gscv_xgb.cv_results_.sort_values(by='rank_test_MeanAbsolutePercentageError', ascending=True)

    # Show best model parameters
    logMessage("Show Best XGBoost Models ...")
    xgb_best_params = gscv_xgb.best_params_
    xgb_best_params_str = str(xgb_best_params)
    logMessage("Best XGBoost Models "+xgb_best_params_str)
    
    logMessage("XGBoost Model Prediction ...")
    xgb_forecast = gscv_xgb.best_forecaster_.predict(fh, X=X_test) #, X=X_test
    y_pred_xgb = pd.DataFrame(xgb_forecast).applymap('{:.2f}'.format)

    #Create MAPE
    xgb_mape = mean_absolute_percentage_error(y_test_cleaned['condensate'], y_pred_xgb)
    xgb_mape_str = str('MAPE: %.4f' % xgb_mape)
    logMessage("XGBoost Model "+xgb_mape_str)
    

    ##### LINEAR REGRESSION MODEL (forecast_f) #####
    #%%
    logMessage("Creating Linear Regression Model Forecasting Insample Condensate BP Tangguh ...")
    # Create Linear Regression Parameter Grid
    linreg_strategy = "recursive"
    linreg_forecaster_param_grid = {"window_length": [3, 6, 9, 18, 27]}

    linreg_regressor = LinearRegression(n_jobs=-1)
    linreg_forecaster = make_reduction(linreg_regressor, strategy=linreg_strategy)

    cv_linreg = SingleWindowSplitter(fh=fh_int)
    gscv_linreg = ForecastingGridSearchCV(linreg_forecaster, cv=cv_linreg, param_grid=linreg_forecaster_param_grid, n_jobs=-1, scoring=mape)

    logMessage("Creating Linear Regression Model ...")
    gscv_linreg.fit(y_train_cleaned, X=X_train) #, X=X_train
    
    # Show top 10 best models based on scoring function
    gscv_linreg.cv_results_.sort_values(by='rank_test_MeanAbsolutePercentageError', ascending=True)

    # Show best model parameters
    logMessage("Show Best Linear Regression Models ...")
    linreg_best_params = gscv_linreg.best_params_
    linreg_best_params_str = str(linreg_best_params)
    logMessage("Best Linear Regression Models "+linreg_best_params_str)
    
    logMessage("Linear Regression Model Prediction ...")
    linreg_forecast = gscv_linreg.best_forecaster_.predict(fh, X=X_test) #, X=X_test
    y_pred_linreg = pd.DataFrame(linreg_forecast).applymap('{:.2f}'.format)

    #Create MAPE
    linreg_mape = mean_absolute_percentage_error(y_test_cleaned['condensate'], y_pred_linreg)
    linreg_mape_str = str('MAPE: %.4f' % linreg_mape)
    logMessage("Linear Regression Model "+linreg_mape_str)
    

    ##### POLYNOMIAL REGRESSION DEGREE=2 MODEL (forecast_g) #####
    #%%
    logMessage("Creating Polynomial Regression Degree=2 Model Forecasting Insample Condensate BP Tangguh ...")
    # Create Polynomial Regression Degree=2 Parameter Grid
    poly2_lags = 3 #3, 9, 18
    poly2_regularization = None
    poly2_interactions = False
    poly2_strategy = "recursive"

    poly2_forecaster_param_grid = {"window_length": [1]}

    poly2_regressor = PolynomRegressor(deg=2, regularization=poly2_regularization, interactions=poly2_interactions)
    poly2_forecaster = make_reduction(poly2_regressor, strategy=poly2_strategy)

    cv_poly2 = SingleWindowSplitter(fh=fh_int)
    gscv_poly2 = ForecastingGridSearchCV(poly2_forecaster, cv=cv_poly2, param_grid=poly2_forecaster_param_grid, scoring=mape, error_score='raise')

    logMessage("Creating Polynomial Regression Orde 2 Model ...")
    gscv_poly2.fit(y_train_cleaned, X=X_train) #, X=X_train
    
    # Show top 10 best models based on scoring function
    gscv_poly2.cv_results_.sort_values(by='rank_test_MeanAbsolutePercentageError', ascending=True)

    # Show best model parameters
    logMessage("Show Best Polynomial Regression Degree=2 Models ...")
    poly2_best_params = gscv_poly2.best_params_
    poly2_best_params_str = str(poly2_best_params)
    logMessage("Best Polynomial Regression Degree=3 Models "+poly2_best_params_str)
    
    logMessage("Polynomial Regression Degree=2 Model Prediction ...")
    poly2_forecast = gscv_poly2.best_forecaster_.predict(fh, X=X_test) #, X=X_test
    y_pred_poly2 = pd.DataFrame(poly2_forecast).applymap('{:.2f}'.format)

    #Create MAPE
    poly2_mape = mean_absolute_percentage_error(y_test_cleaned['condensate'], y_pred_poly2)
    poly2_mape_str = str('MAPE: %.4f' % poly2_mape)
    logMessage("Polynomial Regression Degree=2 Model "+poly2_mape_str)
    
    

    ##### POLYNOMIAL REGRESSION DEGREE=3 MODEL (forecast_h) #####
    #%%
    logMessage("Creating Polynomial Regression Degree=3 Model Forecasting Insample Condensate BP Tangguh ...")
    # Create Polynomial Regression Degree=3 Parameter Grid
    poly3_regularization = None
    poly3_interactions = False
    poly3_strategy = "recursive"

    poly3_forecaster_param_grid = {"window_length": [1]}

    poly3_regressor = PolynomRegressor(deg=3, regularization=poly3_regularization, interactions=poly3_interactions)
    poly3_forecaster = make_reduction(poly3_regressor, strategy=poly3_strategy)

    cv_poly3 = SingleWindowSplitter(fh=fh_int)
    gscv_poly3 = ForecastingGridSearchCV(poly3_forecaster, cv=cv_poly3, param_grid=poly3_forecaster_param_grid, n_jobs=-1, scoring=mape, error_score='raise')

    logMessage("Creating Polynomial Regression Orde 3 Model ...")
    gscv_poly3.fit(y_train_cleaned) #, X=X_train
    
    # Show top 10 best models based on scoring function
    gscv_poly3.cv_results_.sort_values(by='rank_test_MeanAbsolutePercentageError', ascending=True)

    # Show best model parameters
    logMessage("Show Best Polynomial Regression Degree=3 Models ...")
    poly3_best_params = gscv_poly3.best_params_
    poly3_best_params_str = str(poly3_best_params)
    logMessage("Best Polynomial Regression Degree=3 Models "+poly3_best_params_str)
    
    logMessage("Polynomial Regression Degree=3 Model Prediction ...")
    poly3_forecast = gscv_poly3.best_forecaster_.predict(fh) #, X=X_test
    y_pred_poly3 = pd.DataFrame(poly3_forecast).applymap('{:.2f}'.format)

    #Create MAPE
    poly3_mape = mean_absolute_percentage_error(y_test_cleaned['condensate'], y_pred_poly3)
    poly3_mape_str = str('MAPE: %.4f' % poly3_mape)
    logMessage("Polynomial Regression Degree=3 Model "+poly3_mape_str)
    
    # %%  
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
                'product' : 'Condensate'}

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
                        'product' : 'Condensate'}

    all_model_param = pd.DataFrame(all_model_param)

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
    plt.close()
    #plt.show()
    
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