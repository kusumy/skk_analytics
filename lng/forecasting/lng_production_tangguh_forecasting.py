# %%
import logging
import os
import sys
import pandas as pd
import psycopg2
import ast
from configparser import ConfigParser
from pathlib import Path

from humanfriendly import format_timespan
from tokenize import Ignore
from datetime import datetime
from tracemalloc import start
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from adtk.detector import ThresholdAD
from adtk.visualization import plot
from adtk.data import validate_series
pd.options.plotting.backend = "plotly"
from dateutil.relativedelta import *

from pmdarima.arima.utils import ndiffs, nsdiffs
import statsmodels.api as sm
from sktime.forecasting.arima import AutoARIMA, ARIMA
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.compose import make_reduction
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

# %%
def main():
    from connection import config, create_db_connection, get_sql_data
    from utils import (logMessage, ad_test, get_first_date_of_prev_month, get_last_date_of_prev_month,
                    get_last_date_of_current_year, end_day_forecast_april, get_first_date_of_november, configLogging)
    from polyfit import PolynomRegressor
    import datetime

    # Logs Directory
    current_dir = Path(__file__).resolve()
    current_dir_parent_logs = current_dir.parent
    logs_folder = current_dir_parent_logs / "logs"
    logs_file_path = str(logs_folder/'lng_production_tangguh_forecasting.log')
    #logs_file_path = os.path.join('./logs', 'lng_production_tangguh_forecasting.log')

    # Configure logging
    configLogging(logs_file_path)
    
    # Connect to configuration file
    root_parent = current_dir.parent.parent.parent
    config_folder = root_parent / "config"
    config_forecast_tangguh_str = str(config_folder/'config_forecast_tangguh.ini')
    
    config_forecast = ConfigParser()
    config_forecast.read(config_forecast_tangguh_str)

    # Accessing sections
    section = config_forecast['config_tangguh']

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
    logMessage("Forecasting LNG Production BP Tangguh ...")
    
    # Connect to database
    # Exit program if not connected to database
    logMessage("Connecting to database ...")
    conn = create_db_connection(filename='database_tangguh.ini', section='postgresql_ml_lng_skk')
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
    
    sql_folder = current_dir_parent_logs / "sql"
    sql_file_path = str(sql_folder/'lng_prod_tangguh_data_query.sql')
    #query = os.path.join('./sql','lng_prod_tangguh_data_query.sql')
    query_1 = open(sql_file_path, mode="rt").read()
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
    logMessage("LNG Production Tangguh Null Value Cleaning ...")
    data_null_cleaning = data[['date', 'lng_production', 'unplanned_shutdown', 'planned_shutdown']].copy()
    data_null_cleaning['lng_production_copy'] = data[['lng_production']].copy()
    ds_null_cleaning = 'date'
    data_null_cleaning = data_null_cleaning.set_index(ds_null_cleaning)
    s = validate_series(data_null_cleaning)

    #%%
    threshold_ad = ThresholdAD(data_null_cleaning['lng_production_copy'].isnull())
    anomalies =  threshold_ad.detect(s)

    anomalies = anomalies.drop('lng_production', axis=1)
    anomalies = anomalies.drop('planned_shutdown', axis=1)
    anomalies = anomalies.drop('unplanned_shutdown', axis=1)

    # Copy data frame of anomalies
    copy_anomalies =  anomalies.copy()
    # Rename columns
    copy_anomalies.rename(columns={'lng_production_copy':'anomaly'}, inplace=True)
    # Merge original dataframe with anomalies
    new_s = pd.concat([s, copy_anomalies], axis=1)

    # Get only anomalies data
    anomalies_data = new_s[new_s['anomaly'].isnull()]

    # Copy data frame of anomalies
    copy_anomalies =  anomalies.copy()
    # Rename columns
    copy_anomalies.rename(columns={'lng_production_copy':'anomaly'}, inplace=True)
    # Merge original dataframe with anomalies
    new_s = pd.concat([s, copy_anomalies], axis=1)

    # Get only anomalies data
    #anomalies_data = new_s[new_s['anomaly'].isnull()]

    #%%
    for index, row in anomalies_data.iterrows():
        yr = index.year
        mt = index.month
        
        # Get last year start date month
        start_month = get_first_date_of_prev_month(yr,mt,step=-12)
        
        # Get last month last date
        end_month = get_last_date_of_prev_month(yr,mt,step=-1)
        
        # Get mean fead gas data for the month
        sql = "date>='"+start_month+ "' & "+ "date<='" +end_month+"'"
        mean_month=new_s['lng_production'].reset_index().query(sql).mean(skipna = True).values[0]
        
        # update value at specific location
        new_s.at[index,'lng_production'] = mean_month

    #%%
    logMessage("Unplanned Shutdown Cleaning ...")
    data_unplanned_cleaning = new_s[['lng_production', 'unplanned_shutdown', 'planned_shutdown']].copy()
    data_unplanned_cleaning['unplanned_shutdown'] = data_unplanned_cleaning['unplanned_shutdown'].astype('int')
    s2 = validate_series(data_unplanned_cleaning)

    #%%
    threshold_ad2 = ThresholdAD(data_unplanned_cleaning['unplanned_shutdown']==0)
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

    #print(yesterday_date)
    #print(prev_date_year)

    #%%
    for index, row in anomalies_data2.iterrows():
        yr = index.year
        mt = index.month
        
        # Get mean fead gas data for the month
        #sql = "date>='"+start_month+ "' & "+ "date<='" +end_month+"'"
        yesterday_date = index - datetime.timedelta(days=1)
        prev_date_year = yesterday_date - datetime.timedelta(days=364)
        
        yesterday_date = yesterday_date.strftime("%Y-%m-%d")
        prev_date_year = prev_date_year.strftime("%Y-%m-%d")
        sql = "date>='"+prev_date_year+ "' & "+ "date<='" +yesterday_date+"'"
        mean_month=new_s2['lng_production'].reset_index().query(sql).mean(skipna = True).values[0] 
        
        # update value at specific location
        new_s2.at[index,'lng_production'] = mean_month

    # Check if updated
    new_s2[new_s2['anomaly'] == False]

    #%%
    logMessage("LNG Production Tangguh Prepare Data ...")
    #prepare data
    data_cleaned = new_s2[['lng_production']].copy()
    data_cleaned = data_cleaned.reset_index()

    ds_cleaned = 'date'
    y_cleaned = 'lng_production'
    df_cleaned = data_cleaned[[ds_cleaned, y_cleaned]]
    df_cleaned = df_cleaned.set_index(ds_cleaned)
    df_cleaned.index = pd.DatetimeIndex(df_cleaned.index, freq='D')

    #Create column target
    train_df = df_cleaned['lng_production']

    #%%
    ### CREATE EXOGENOUS VARIABLE ###
    df_cleaned['planned_shutdown'] = data['planned_shutdown'].values
    df_cleaned['month'] = [i.month for i in df_cleaned.index]
    df_cleaned['day'] = [i.day for i in df_cleaned.index]
    train_exog = df_cleaned.iloc[:,1:]

    from sktime.forecasting.base import ForecastingHorizon

    #%%
    logMessage("Load Exogenous Data")
    #Load Data from Database
    from datetime import timedelta
    exog_forecast_start_date = ((pd.to_datetime(train_df.index[-1]).to_pydatetime()) + timedelta(days=1)).strftime("%Y-%m-%d")
    query_exog = str(sql_folder/'lng_prod_tangguh_exog_query.sql')
    #query_exog = os.path.join('./sql','lng_prod_tangguh_exog_query.sql')
    query_2 = open(query_exog, mode="rt").read()
    sql2 = ''
    if USE_DEFAULT_DATE == True:
        if current_date < date_nov:
            sql2 = query_2.format(exog_forecast_start_date, end_date)
        else :
            sql2 = query_2.format(exog_forecast_start_date, end_date_april)
    else :
        sql2 = query_2.format(FORECAST_START_DATE, FORECAST_END_DATE)
    
    data_exog = get_sql_data(sql2, conn)
    data_exog['date'] = pd.DatetimeIndex(data_exog['date'], freq='D')
    data_exog.sort_index(inplace=True)
    data_exog = data_exog.reset_index()

    ds_exog = 'date'
    y_exog = 'planned_shutdown'

    future_exog = data_exog[[ds_exog,y_exog]]
    future_exog = future_exog.set_index(ds_exog)
    future_exog.index = pd.DatetimeIndex(future_exog.index, freq='D')

    logMessage("Create Exogenous Features for future")
    #Create exogenous date index
    future_exog['planned_shutdown'] = future_exog['planned_shutdown']
    future_exog['month'] = [i.month for i in future_exog.index]
    future_exog['day'] = [i.day for i in future_exog.index]

    #Create Forecasting Horizon
    fh = ForecastingHorizon(future_exog.index, is_relative=False)

    #%%
    try:
        ##### FORECASTING #####
        ##### ARIMAX MODEL #####
        logMessage("Create Arimax Forecasting LNG Production BP Tangguh ...")
        # Get best parameter from database
        sql_arimax_model_param = """SELECT model_param_a 
                        FROM lng_analytics_model_param 
                        WHERE lng_plant = 'BP Tangguh' 
                        AND product = 'LNG Production'
                        ORDER BY running_date DESC 
                        LIMIT 1 OFFSET 0"""
                    
        arimax_model_param = get_sql_data(sql_arimax_model_param, conn)
        arimax_model_param = arimax_model_param['model_param_a'][0]
       
        # Convert string to tuple
        arimax_model_param = ast.literal_eval(arimax_model_param)

        # Get Adjustment Value Arimax
        sql_arimax_adj = """SELECT adj_forecast_a
                        FROM lng_analytics_adjustment
                        WHERE lng_plant = 'BP Tangguh' 
                        AND product = 'LNG Production'
                        ORDER BY running_date DESC 
                        LIMIT 1 OFFSET 0"""
        
        arimax_adj_value = get_sql_data(sql_arimax_adj, conn)
        arimax_adj_value = arimax_adj_value['adj_forecast_a'][0]

        #Set parameters
        arimax_suppress_warnings = True

        # Create ARIMAX Model
        logMessage("Creating ARIMAX Model ...")
        #arimax_model = AutoARIMA(d=arimax_differencing, suppress_warnings=arimax_suppress_warnings, error_action=arimax_error_action)
        arimax_model = ARIMA(order=arimax_model_param, suppress_warnings=arimax_suppress_warnings)
        arimax_model.fit(train_df, X=train_exog)
        future_exog = future_exog.sort_index()
        logMessage("ARIMAX Model Summary")
        logMessage(arimax_model.summary())
        
        logMessage("ARIMAX Model Prediction ..")
        arimax_forecast = arimax_model.predict(fh, X=future_exog)
        y_pred_arimax = pd.DataFrame(arimax_forecast).applymap('{:.2f}'.format)
        y_pred_arimax['day_num'] = [i.day for i in arimax_forecast.index]
        y_pred_arimax['month_num'] = [i.month for i in arimax_forecast.index]
        y_pred_arimax['year_num'] = [i.year for i in arimax_forecast.index]
        y_pred_arimax['date'] = y_pred_arimax['year_num'].astype(str) + '-' + y_pred_arimax['month_num'].astype(str) + '-' + y_pred_arimax['day_num'].astype(str)
        y_pred_arimax['date'] = pd.DatetimeIndex(y_pred_arimax['date'], freq='D')

        #Rename colum 0
        y_pred_arimax.rename(columns={0:'forecast_a'}, inplace=True)

        # Convert the 'forecast_a' column to float data type
        y_pred_arimax['forecast_a'] = y_pred_arimax['forecast_a'].astype(float)

        # Add adj value to all the values in the 'forecast_a' column
        y_pred_arimax['forecast_a'] = y_pred_arimax['forecast_a'] + arimax_adj_value


        ##### SARIMAX MODEL #####
        logMessage("Create Sarimax Forecasting LNG Production BP Tangguh ...")
        # Get best parameter from database
        sql_sarimax_model_param = """SELECT model_param_b 
                        FROM lng_analytics_model_param 
                        WHERE lng_plant = 'BP Tangguh' 
                        AND product = 'LNG Production'
                        ORDER BY running_date DESC 
                        LIMIT 1 OFFSET 0"""
                    
        sarimax_model_param = get_sql_data(sql_sarimax_model_param, conn)
        sarimax_model_param = sarimax_model_param['model_param_b'][0]
       
        # Convert string to tuple
        sarimax_model_param = ast.literal_eval(sarimax_model_param)

        # Get Adjustment Value Arimax
        sql_sarimax_adj = """SELECT adj_forecast_b
                        FROM lng_analytics_adjustment
                        WHERE lng_plant = 'BP Tangguh' 
                        AND product = 'LNG Production'
                        ORDER BY running_date DESC 
                        LIMIT 1 OFFSET 0"""
        
        sarimax_adj_value = get_sql_data(sql_sarimax_adj, conn)
        sarimax_adj_value = sarimax_adj_value['adj_forecast_b'][0]
        
        #Set parameters
        sarimax_suppress_warnings = True
        sarimax_order = sarimax_model_param['sarimax_order']
        sarimax_seasonal_order = sarimax_model_param['sarimax_seasonal_order']

        # Create SARIMAX Model
        logMessage("Creating SARIMAX Model ...")
        sarimax_model = ARIMA(order=sarimax_order, seasonal_order=sarimax_seasonal_order, suppress_warnings=sarimax_suppress_warnings)
        sarimax_model.fit(train_df, X=train_exog)
        logMessage("SARIMAX Model Summary")
        logMessage(sarimax_model.summary())
        
        logMessage("SARIMAX Model Prediction ..")
        #future_exog = future_exog.sort_index()
        #orde (1,0,1)(0,1,1)[12]
        sarimax_forecast = sarimax_model.predict(fh, X=future_exog)
        y_pred_sarimax = pd.DataFrame(sarimax_forecast).applymap('{:.2f}'.format)
        y_pred_sarimax['day_num'] = [i.day for i in sarimax_forecast.index]
        y_pred_sarimax['month_num'] = [i.month for i in sarimax_forecast.index]
        y_pred_sarimax['year_num'] = [i.year for i in sarimax_forecast.index]
        y_pred_sarimax['date'] = y_pred_sarimax['year_num'].astype(str) + '-' + y_pred_sarimax['month_num'].astype(str) + '-' + y_pred_sarimax['day_num'].astype(str)
        y_pred_sarimax['date'] = pd.DatetimeIndex(y_pred_sarimax['date'], freq='D')
        #Rename colum 0
        y_pred_sarimax.rename(columns={0:'forecast_b'}, inplace=True)

        # Convert the 'forecast_b' column to float data type
        y_pred_sarimax['forecast_b'] = y_pred_sarimax['forecast_b'].astype(float)

        # Add adj value to all the values in the 'forecast_b' column
        y_pred_sarimax['forecast_b'] = y_pred_sarimax['forecast_b'] + sarimax_adj_value


        ##### PROPHET MODEL #####
        logMessage("Create Prophet Forecasting LNG Production BP Tangguh ...")
        # Get best parameter from database
        sql_prophet_model_param = """SELECT model_param_c 
                        FROM lng_analytics_model_param 
                        WHERE lng_plant = 'BP Tangguh' 
                        AND product = 'LNG Production'
                        ORDER BY running_date DESC 
                        LIMIT 1 OFFSET 0"""
                    
        prophet_model_param = get_sql_data(sql_prophet_model_param, conn)
        prophet_model_param = prophet_model_param['model_param_c'][0]
        
        # Convert string to dictionary
        prophet_model_param = ast.literal_eval(prophet_model_param)

        # Get Adjustment Value Arimax
        sql_prophet_adj = """SELECT adj_forecast_c
                        FROM lng_analytics_adjustment
                        WHERE lng_plant = 'BP Tangguh' 
                        AND product = 'LNG Production'
                        ORDER BY running_date DESC 
                        LIMIT 1 OFFSET 0"""
        
        prophet_adj_value = get_sql_data(sql_prophet_adj, conn)
        prophet_adj_value = prophet_adj_value['adj_forecast_c'][0]
        
        # Set parameters
        prophet_seasonality_mode = prophet_model_param['seasonality_mode']
        prophet_n_changepoints = prophet_model_param['n_changepoints']
        prophet_seasonality_prior_scale = prophet_model_param['seasonality_prior_scale']
        prophet_changepoint_prior_scale = prophet_model_param['changepoint_prior_scale']
        prophet_daily_seasonality = prophet_model_param['daily_seasonality']
        prophet_weekly_seasonality = prophet_model_param['weekly_seasonality']
        prophet_yearly_seasonality = prophet_model_param['yearly_seasonality']

        #Create regressor forecasting
        logMessage("Creating Prophet Model ....")
        prophet_forecaster = Prophet(
                seasonality_mode=prophet_seasonality_mode,
                n_changepoints=prophet_n_changepoints,
                seasonality_prior_scale=prophet_seasonality_prior_scale, #Flexibility of the seasonality (0.01,10)
                changepoint_prior_scale=prophet_changepoint_prior_scale, #Flexibility of the trend (0.001,0.5)
                daily_seasonality=prophet_daily_seasonality,
                weekly_seasonality=prophet_weekly_seasonality,
                yearly_seasonality=prophet_yearly_seasonality)

        prophet_forecaster.fit(train_df, X=train_exog)
        logMessage("Prophet Model Prediction ...")
        prophet_forecast = prophet_forecaster.predict(fh, X=future_exog)
        y_pred_prophet = pd.DataFrame(prophet_forecast).applymap('{:.2f}'.format)
        y_pred_prophet['day_num'] = [i.day for i in prophet_forecast.index]
        y_pred_prophet['month_num'] = [i.month for i in prophet_forecast.index]
        y_pred_prophet['year_num'] = [i.year for i in prophet_forecast.index]
        y_pred_prophet['date'] = y_pred_prophet['year_num'].astype(str) + '-' + y_pred_prophet['month_num'].astype(str) + '-' + y_pred_prophet['day_num'].astype(str)
        y_pred_prophet['date'] = pd.DatetimeIndex(y_pred_prophet['date'], freq='D')
        #Rename colum 0
        y_pred_prophet.rename(columns={0:'forecast_c'}, inplace=True)

        # Convert the 'forecast_c' column to float data type
        y_pred_prophet['forecast_c'] = y_pred_prophet['forecast_c'].astype(float)

        # Add adj value to all the values in the 'forecast_c' column
        y_pred_prophet['forecast_c'] = y_pred_prophet['forecast_c'] + prophet_adj_value


        ##### RANDOM FOREST MODEL #####
        logMessage("Create Random Forest Forecasting LNG Production BP Tangguh ...")
        # Get best parameter from database
        sql_ranfor_model_param = """SELECT model_param_d 
                        FROM lng_analytics_model_param 
                        WHERE lng_plant = 'BP Tangguh' 
                        AND product = 'LNG Production'
                        ORDER BY running_date DESC 
                        LIMIT 1 OFFSET 0"""
                    
        ranfor_model_param = get_sql_data(sql_ranfor_model_param, conn)
        ranfor_model_param = ranfor_model_param['model_param_d'][0]
       
        # Convert string to tuple
        ranfor_model_param = ast.literal_eval(ranfor_model_param)

        # Get Adjustment Value Arimax
        sql_ranfor_adj = """SELECT adj_forecast_d
                        FROM lng_analytics_adjustment
                        WHERE lng_plant = 'BP Tangguh' 
                        AND product = 'LNG Production'
                        ORDER BY running_date DESC 
                        LIMIT 1 OFFSET 0"""
        
        ranfor_adj_value = get_sql_data(sql_ranfor_adj, conn)
        ranfor_adj_value = ranfor_adj_value['adj_forecast_d'][0]

        #Set parameters
        ranfor_n_estimators = ranfor_model_param['estimator__n_estimators']
        ranfor_lags = ranfor_model_param['window_length']
        ranfor_random_state = 0
        ranfor_criterion = "squared_error"
        ranfor_strategy = "recursive"

        # create regressor object
        logMessage("Creating Random Forest Model ...")
        ranfor_regressor = RandomForestRegressor(n_estimators = ranfor_n_estimators, random_state=ranfor_random_state, criterion=ranfor_criterion)
        ranfor_forecaster = make_reduction(ranfor_regressor, window_length=ranfor_lags, strategy=ranfor_strategy)
        ranfor_forecaster.fit(train_df, train_exog) #, X_train

        logMessage("Random Forest Model Prediction")
        ranfor_forecast = ranfor_forecaster.predict(fh, X=future_exog) #, X=X_test
        y_pred_ranfor = pd.DataFrame(ranfor_forecast).applymap('{:.2f}'.format)
        y_pred_ranfor['day_num'] = [i.day for i in ranfor_forecast.index]
        y_pred_ranfor['month_num'] = [i.month for i in ranfor_forecast.index]
        y_pred_ranfor['year_num'] = [i.year for i in ranfor_forecast.index]
        y_pred_ranfor['date'] = y_pred_ranfor['year_num'].astype(str) + '-' + y_pred_ranfor['month_num'].astype(str) + '-' + y_pred_ranfor['day_num'].astype(str)
        y_pred_ranfor['date'] = pd.DatetimeIndex(y_pred_ranfor['date'], freq='D')
        #Rename colum 0
        y_pred_ranfor.rename(columns={0:'forecast_d'}, inplace=True)

        # Convert the 'forecast_d' column to float data type
        y_pred_ranfor['forecast_d'] = y_pred_ranfor['forecast_d'].astype(float)

        # Add adj value to all the values in the 'forecast_d' column
        y_pred_ranfor['forecast_d'] = y_pred_ranfor['forecast_d'] + ranfor_adj_value


        ##### XGBOOST MODEL #####
        logMessage("Create XGBoost Forecasting LNG Production BP Tangguh ...")
        # Get best parameter from database
        sql_xgb_model_param = """SELECT model_param_e 
                        FROM lng_analytics_model_param 
                        WHERE lng_plant = 'BP Tangguh'
                        AND product = 'LNG Production'
                        ORDER BY running_date DESC 
                        LIMIT 1 OFFSET 0"""
                    
        xgb_model_param = get_sql_data(sql_xgb_model_param, conn)
        xgb_model_param = xgb_model_param['model_param_e'][0]
       
        # Convert string to tuple
        xgb_model_param = ast.literal_eval(xgb_model_param)

        # Get Adjustment Value Arimax
        sql_xgb_adj = """SELECT adj_forecast_e
                        FROM lng_analytics_adjustment
                        WHERE lng_plant = 'BP Tangguh' 
                        AND product = 'LNG Production'
                        ORDER BY running_date DESC 
                        LIMIT 1 OFFSET 0"""
        
        xgb_adj_value = get_sql_data(sql_xgb_adj, conn)
        xgb_adj_value = xgb_adj_value['adj_forecast_e'][0]

        #Set parameters
        xgb_objective = 'reg:squarederror'
        xgb_lags = xgb_model_param['window_length']
        xgb_strategy = "recursive"

        # Create regressor object
        logMessage("Creating XGBoost Model ...")
        xgb_regressor = XGBRegressor(objective=xgb_objective)
        xgb_forecaster = make_reduction(xgb_regressor, window_length=xgb_lags, strategy=xgb_strategy)
        xgb_forecaster.fit(train_df, X=train_exog)

        # Create forecasting
        logMessage("XGBoost Model Prediction ...")
        xgb_forecast = xgb_forecaster.predict(fh, X=future_exog) #, X=X_test
        y_pred_xgb = pd.DataFrame(xgb_forecast).applymap('{:.2f}'.format)
        y_pred_xgb['day_num'] = [i.day for i in xgb_forecast.index]
        y_pred_xgb['month_num'] = [i.month for i in xgb_forecast.index]
        y_pred_xgb['year_num'] = [i.year for i in xgb_forecast.index]
        y_pred_xgb['date'] = y_pred_xgb['year_num'].astype(str) + '-' + y_pred_xgb['month_num'].astype(str) + '-' + y_pred_xgb['day_num'].astype(str)
        y_pred_xgb['date'] = pd.DatetimeIndex(y_pred_xgb['date'], freq='D')
        #Rename colum 0
        y_pred_xgb.rename(columns={0:'forecast_e'}, inplace=True)

        # Convert the 'forecast_e' column to float data type
        y_pred_xgb['forecast_e'] = y_pred_xgb['forecast_e'].astype(float)

        # Add adj value to all the values in the 'forecast_e' column
        y_pred_xgb['forecast_e'] = y_pred_xgb['forecast_e'] + xgb_adj_value


        ##### LINEAR REGRESSION MODEL #####
        logMessage("Create Linear Regression Forecasting LNG Production BP Tangguh ...")
        # Get best parameter from database
        sql_linreg_model_param = """SELECT model_param_f 
                        FROM lng_analytics_model_param 
                        WHERE lng_plant = 'BP Tangguh' 
                        AND product = 'LNG Production'
                        ORDER BY running_date DESC 
                        LIMIT 1 OFFSET 0"""
                    
        linreg_model_param = get_sql_data(sql_linreg_model_param, conn)
        linreg_model_param = linreg_model_param['model_param_f'][0]
       
        # Convert string to tuple
        linreg_model_param = ast.literal_eval(linreg_model_param)

        # Get Adjustment Value Arimax
        sql_linreg_adj = """SELECT adj_forecast_f
                        FROM lng_analytics_adjustment
                        WHERE lng_plant = 'BP Tangguh' 
                        AND product = 'LNG Production'
                        ORDER BY running_date DESC 
                        LIMIT 1 OFFSET 0"""
        
        linreg_adj_value = get_sql_data(sql_linreg_adj, conn)
        linreg_adj_value = linreg_adj_value['adj_forecast_f'][0]

        #Set Parameters
        linreg_lags = linreg_model_param['window_length']
        linreg_strategy = "recursive"

        # Create regressor object
        logMessage("Creating Linear Regression Model ...")
        linreg_regressor = LinearRegression()
        linreg_forecaster = make_reduction(linreg_regressor, window_length=linreg_lags, strategy=linreg_strategy)
        linreg_forecaster.fit(train_df, X=train_exog)

        # Create forecasting
        logMessage("Linear Regression Model Prediction ...")
        linreg_forecast = linreg_forecaster.predict(fh, X=future_exog)
        y_pred_linreg = pd.DataFrame(linreg_forecast).applymap('{:.2f}'.format)
        y_pred_linreg['day_num'] = [i.day for i in linreg_forecast.index]
        y_pred_linreg['month_num'] = [i.month for i in linreg_forecast.index]
        y_pred_linreg['year_num'] = [i.year for i in linreg_forecast.index]
        y_pred_linreg['date'] = y_pred_linreg['year_num'].astype(str) + '-' + y_pred_linreg['month_num'].astype(str) + '-' + y_pred_linreg['day_num'].astype(str)
        y_pred_linreg['date'] = pd.DatetimeIndex(y_pred_linreg['date'], freq='D')
        #Rename colum 0
        y_pred_linreg.rename(columns={0:'forecast_f'}, inplace=True)

        # Convert the 'forecast_f' column to float data type
        y_pred_linreg['forecast_f'] = y_pred_linreg['forecast_f'].astype(float)

        # Add adj value to all the values in the 'forecast_f' column
        y_pred_linreg['forecast_f'] = y_pred_linreg['forecast_f'] + linreg_adj_value


        ##### POLYNOMIAL REGRESSION DEGREE=2 MODEL #####
        logMessage("Create Polynomial Regression Degree=2 Forecasting LNG Production BP Tangguh ...")
        # Get best parameter from database
        sql_poly2_model_param = """SELECT model_param_g 
                        FROM lng_analytics_model_param 
                        WHERE lng_plant = 'BP Tangguh' 
                        AND product = 'LNG Production'
                        ORDER BY running_date DESC 
                        LIMIT 1 OFFSET 0"""
                    
        poly2_model_param = get_sql_data(sql_poly2_model_param, conn)
        poly2_model_param = poly2_model_param['model_param_g'][0]
       
        # Convert string to tuple
        poly2_model_param = ast.literal_eval(poly2_model_param)

        # Get Adjustment Value Arimax
        sql_poly2_adj = """SELECT adj_forecast_g
                        FROM lng_analytics_adjustment
                        WHERE lng_plant = 'BP Tangguh' 
                        AND product = 'LNG Production'
                        ORDER BY running_date DESC 
                        LIMIT 1 OFFSET 0"""
        
        poly2_adj_value = get_sql_data(sql_poly2_adj, conn)
        poly2_adj_value = poly2_adj_value['adj_forecast_g'][0]

        #Set parameters
        poly2_lags = poly2_model_param['window_length']
        poly2_regularization = None
        poly2_interactions = False
        poly2_strategy = "recursive"

        # Create regressor object
        logMessage("Creating Polynomial Regression Orde 2 Model ...")
        poly2_regressor = PolynomRegressor(deg=2, regularization=poly2_regularization, interactions=poly2_interactions)
        poly2_forecaster = make_reduction(poly2_regressor, window_length=poly2_lags, strategy=poly2_strategy)
        poly2_forecaster.fit(train_df, X=train_exog)

        # Create forecasting
        logMessage("Polynomial Regression Orde 2 Model Prediction ...")
        poly2_forecast = poly2_forecaster.predict(fh, X=future_exog)
        y_pred_poly2 = pd.DataFrame(poly2_forecast).applymap('{:.2f}'.format)
        y_pred_poly2['day_num'] = [i.day for i in poly2_forecast.index]
        y_pred_poly2['month_num'] = [i.month for i in poly2_forecast.index]
        y_pred_poly2['year_num'] = [i.year for i in poly2_forecast.index]
        y_pred_poly2['date'] = y_pred_poly2['year_num'].astype(str) + '-' + y_pred_poly2['month_num'].astype(str) + '-' + y_pred_poly2['day_num'].astype(str)
        y_pred_poly2['date'] = pd.DatetimeIndex(y_pred_poly2['date'], freq='D')
        #Rename colum 0
        y_pred_poly2.rename(columns={0:'forecast_g'}, inplace=True)

        # Convert the 'forecast_g' column to float data type
        y_pred_poly2['forecast_g'] = y_pred_poly2['forecast_g'].astype(float)

        # Add adj value to all the values in the 'forecast_g' column
        y_pred_poly2['forecast_g'] = y_pred_poly2['forecast_g'] + poly2_adj_value


        ##### POLYNOMIAL REGRESSION DEGREE=3 MODEL #####
        logMessage("Create Polynomial Regression Degree=3 Forecasting LNG Production BP Tangguh ...")
        # Get best parameter from database
        sql_poly3_model_param = """SELECT model_param_h 
                        FROM lng_analytics_model_param 
                        WHERE lng_plant = 'BP Tangguh' 
                        AND product = 'LNG Production'
                        ORDER BY running_date DESC 
                        LIMIT 1 OFFSET 0"""
                    
        poly3_model_param = get_sql_data(sql_poly3_model_param, conn)
        poly3_model_param = poly3_model_param['model_param_h'][0]
       
        # Convert string to tuple
        poly3_model_param = ast.literal_eval(poly3_model_param)

        # Get Adjustment Value Arimax
        sql_poly3_adj = """SELECT adj_forecast_h
                        FROM lng_analytics_adjustment
                        WHERE lng_plant = 'BP Tangguh' 
                        AND product = 'LNG Production'
                        ORDER BY running_date DESC 
                        LIMIT 1 OFFSET 0"""
        
        poly3_adj_value = get_sql_data(sql_poly3_adj, conn)
        poly3_adj_value = poly3_adj_value['adj_forecast_h'][0]

        #Set parameters
        poly3_lags = poly3_model_param['window_length']
        poly3_regularization = None
        poly3_interactions = False
        poly3_strategy = "recursive"

        # Create regressor object
        logMessage("Creating Polynomial Regression Orde 3 Model ...")
        poly3_regressor = PolynomRegressor(deg=3, regularization=poly3_regularization, interactions=poly3_interactions)
        poly3_forecaster = make_reduction(poly3_regressor, window_length=poly3_lags, strategy=poly3_strategy)
        poly3_forecaster.fit(train_df, X=train_exog)

        # Create forecasting
        logMessage("Polynomial Regression Orde 3 Model Prediction ...")
        poly3_forecast = poly3_forecaster.predict(fh, X=future_exog)
        y_pred_poly3 = pd.DataFrame(poly3_forecast).applymap('{:.2f}'.format)
        y_pred_poly3['day_num'] = [i.day for i in poly3_forecast.index]
        y_pred_poly3['month_num'] = [i.month for i in poly3_forecast.index]
        y_pred_poly3['year_num'] = [i.year for i in poly3_forecast.index]
        y_pred_poly3['date'] = y_pred_poly3['year_num'].astype(str) + '-' + y_pred_poly3['month_num'].astype(str) + '-' + y_pred_poly3['day_num'].astype(str)
        y_pred_poly3['date'] = pd.DatetimeIndex(y_pred_poly3['date'], freq='D')
        #Rename colum 0
        y_pred_poly3.rename(columns={0:'forecast_h'}, inplace=True)

        # Convert the 'forecast_h' column to float data type
        y_pred_poly3['forecast_h'] = y_pred_poly3['forecast_h'].astype(float)

        # Add adj value to all the values in the 'forecast_h' column
        y_pred_poly3['forecast_h'] = y_pred_poly3['forecast_h'] + poly3_adj_value
        

        ##### JOIN PREDICTION RESULT TO DATAFRAME #####
        logMessage("Creating all model prediction result data frame ...")
        y_all_pred = pd.concat([y_pred_arimax[['forecast_a']],
                                y_pred_sarimax[['forecast_b']],
                                y_pred_prophet[['forecast_c']],
                                y_pred_ranfor[['forecast_d']],
                                y_pred_xgb[['forecast_e']],
                                y_pred_linreg[['forecast_f']],
                                y_pred_poly2[['forecast_g']],
                                y_pred_poly3[['forecast_h']]], axis=1)
        y_all_pred['date'] = future_exog.index.values

        # %%
        # Save forecast result to database
        logMessage("Updating forecast result to database ...")
        total_updated_rows = insert_forecast(conn, y_all_pred)
        logMessage("Updated rows: {}".format(total_updated_rows))
        
        logMessage("Done")
    except Exception as e:
        logMessage(e)

def insert_forecast(conn, y_pred):
    total_updated_rows = 0
    for index, row in y_pred.iterrows():
        prod_date = str(index) #row['date']
        forecast_a, forecast_b, forecast_c, forecast_d, forecast_e, forecast_f, forecast_g, forecast_h = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]
        
        #sql = f'UPDATE trir_monthly_test SET forecast_a = {} WHERE year_num = {} AND month_num = {}'.format(forecast, year_num, month_num)
        updated_rows = update_value(conn, forecast_a, forecast_b, forecast_c, forecast_d, forecast_e, forecast_f, forecast_g, forecast_h, prod_date)
        total_updated_rows = total_updated_rows + updated_rows 
        
    return total_updated_rows

def update_value(conn, forecast_a, forecast_b, forecast_c, 
                        forecast_d, forecast_e, forecast_f, forecast_g, forecast_h, prod_date):
    
    date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    created_by = 'PYTHON'
    
    """ insert forecasting result after last row in table """
    sql = """ UPDATE lng_production_daily
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
                WHERE prod_date = %s
                AND lng_plant = 'BP Tangguh'"""
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
