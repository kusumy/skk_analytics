# %%
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import seaborn as sns
from configparser import ConfigParser
import ast
from pathlib import Path

plt.style.use('fivethirtyeight')
from datetime import datetime
from tokenize import Ignore
from tracemalloc import start
from pmdarima.arima import auto_arima
from pmdarima.arima.auto import auto_arima

from sktime.forecasting.arima import AutoARIMA, ARIMA
from pmdarima.arima.utils import ndiffs, nsdiffs
import statsmodels.api as sm
from xgboost import XGBRegressor
from sktime.forecasting.compose import make_reduction
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# %%
def main():
    from connection import create_db_connection, get_sql_data
    from utils import logMessage, ad_test, configLogging
    from polyfit import PolynomRegressor
    import datetime
    
    # Logs Directory
    current_dir = Path(__file__).resolve()
    current_dir_parent_logs = current_dir.parent
    logs_folder = current_dir_parent_logs / "logs"
    logs_file_path = str(logs_folder/'incident_rate_yearly_forecasting.log')
    #logs_file_path = os.path.join('./logs', 'incident_rate_yearly_forecasting.log')

    # Configure logging
    configLogging(logs_file_path)

    # Connect to configuration file
    root_parent = current_dir.parent.parent.parent
    config_folder = root_parent / "config"
    config_forecast_yearly = str(config_folder/'config_forecast_hse_yearly.ini')

    config_forecast = ConfigParser()
    config_forecast.read(config_forecast_yearly)
    
    # Accessing sections
    section = config_forecast['config']

    USE_DEFAULT_DATE = section.getboolean('use_default_date')
    TRAIN_START_YEAR= section.getint('train_start_year')
    TRAIN_END_YEAR= section.getint('train_end_year')
    FORECAST_START_YEAR= section.getint('forecast_start_year')
    FORECAST_END_YEAR= section.getint('forecast_end_year')
        
    # Connect to database
    # Exit program if not connected to database
    logMessage("Connecting to database ...")
    conn = create_db_connection(filename='database_hse.ini', section='postgresql_ml_hse_skk')
    if conn == None:
        exit()
    
    from datetime import datetime
    current_year = datetime.now().year

    sql_folder = current_dir_parent_logs / "sql"
    sql_file_path = str(sql_folder/'query_yearly.sql')
    #query_data = os.path.join('./sql', 'query_yearly.sql')
    query_1 = open(sql_file_path, mode="rt").read()
    sql = ''
    if USE_DEFAULT_DATE == True:
        sql = query_1.format('2013', current_year)
    else :
        sql = query_1.format(TRAIN_START_YEAR, TRAIN_END_YEAR)
    
    data = get_sql_data(sql, conn)
    data['year_num'] = data['year_num'].astype(int)
    data['year_num'] = pd.to_datetime(data['year_num'], format='%Y')
    data = data.rename(columns=str.lower)
    data['year_num'] = pd.PeriodIndex(data['year_num'], freq='Y')
    data = data.reset_index()

    ds = 'year_num'
    y = 'trir' 
    df = data[[ds,y]]
    df = df.set_index(ds)
    df.index = pd.PeriodIndex(df.index, freq='Y')
    
    #Select target column
    train_df = df['trir']

    #%%
    #Ad Fuller Test
    ad_test(df['trir'])
 
    #%%
    #Exogenous for training
    train_exog = data[['year_num', 'survey_seismic', 'bor_eksplorasi', 'bor_eksploitasi', 'workover', 'wellservice']]
    train_exog = train_exog.set_index(train_exog['year_num'])
    train_exog['year_num'] = pd.PeriodIndex(train_exog['year_num'], freq='Y')
    train_exog.drop(columns=['year_num'], axis=1, inplace=True)
    train_exog.sort_index(inplace=True)
    
    # Convert the Period object to a Timestamp object
    last_index_timestamp = train_df.index[-1].to_timestamp()

    # Convert Timestamp object to datetime
    last_index_datetime = pd.to_datetime(last_index_timestamp)

    #Load Data from Database (create future exogenous)
    from datetime import timedelta
    from dateutil.relativedelta import relativedelta
    exog_forecast_start_date = (last_index_datetime + relativedelta(years=1)).year
    exog_forecast_end_date = exog_forecast_start_date + 1

    query_exog = str(sql_folder/'query_yearly_future.sql')
    #query_exog = os.path.join('./sql',"query_yearly_future.sql")
    query_2 = open(query_exog, mode="rt").read()
    sql2 = ''
    if USE_DEFAULT_DATE == True:
        sql2 = query_2.format(exog_forecast_start_date, exog_forecast_end_date)
    else :
        sql2 = query_2.format(FORECAST_START_YEAR, FORECAST_END_YEAR)
        
    future_exog = get_sql_data(sql2, conn)
    future_exog['year_num'] = future_exog['year_num'].astype(int)

    # Prepare data (future exogenous)
    future_exog['year_num'] = pd.to_datetime(future_exog['year_num'], format='%Y')
    future_exog = future_exog.set_index(future_exog['year_num'])
    future_exog.index = pd.PeriodIndex(future_exog.index, freq='Y')
    future_exog = future_exog.rename(columns=str.lower)
    future_exog['survey_seismic'] = data['survey_seismic'].iloc[-1]
    future_exog['bor_eksplorasi'] = data['bor_eksplorasi'].iloc[-1]
    future_exog['bor_eksploitasi'] = data['bor_eksploitasi'].iloc[-1]
    future_exog['workover'] = data['workover'].iloc[-1]
    future_exog['wellservice'] = data['wellservice'].iloc[-1]
    future_exog.drop(columns=['year_num'], axis=1, inplace=True)
    
    #Replace Data Null
    future_exog['survey_seismic'].fillna(method='ffill', inplace=True)
    future_exog['bor_eksplorasi'].fillna(method='ffill', inplace=True)
    future_exog['bor_eksploitasi'].fillna(method='ffill', inplace=True)
    future_exog['workover'].fillna(method='ffill', inplace=True)
    future_exog['wellservice'].fillna(method='ffill', inplace=True)

    from sktime.forecasting.base import ForecastingHorizon
    #time_predict = pd.period_range('2023', periods=2, freq='Y')
    fh = ForecastingHorizon(future_exog.index, is_relative=False)

    #%%
    try :
        ##### FORECASTING #####
        ##### ARIMAX MODEL #####
        logMessage("Create Arimax Forecasting IR Yearly ...")
        # Get best parameter from database
        sql_arimax_model_param = """SELECT best_param_a 
                                FROM hse_analytics_param 
                                WHERE ir_type = 'ir yearly' 
                                ORDER BY running_date DESC 
                                LIMIT 1 OFFSET 0"""
                    
        arimax_model_param = get_sql_data(sql_arimax_model_param, conn)
        arimax_model_param = arimax_model_param['best_param_a'][0]
        
        # Convert string to tuple
        arimax_model_param = ast.literal_eval(arimax_model_param)

        #Set parameters
        arimax_suppress_warnings = True

        # Create ARIMAX Model
        #arimax_model = auto_arima(train_df, exogenous=train_exog, d=1, trace=arimax_trace, error_action=arimax_error_action, suppress_warnings=arimax_suppress_warnings)
        arimax_model = ARIMA(order=arimax_model_param, suppress_warnings=arimax_suppress_warnings)
        logMessage("Creating ARIMAX Model ...")
        arimax_model.fit(train_df, X=train_exog)
        logMessage("ARIMAX Model Summary")
        logMessage(arimax_model.summary())
        
        logMessage("ARIMAX Model Prediction ..")
        arimax_forecast = arimax_model.predict(fh, X=future_exog) #len(fh)
        y_pred_arimax = pd.DataFrame(arimax_forecast).applymap('{:,.2f}'.format)
        y_pred_arimax['year_num'] = [i.year for i in arimax_forecast.index]
        #Rename colum 0
        y_pred_arimax.rename(columns={0:'forecast_a'}, inplace=True)


        ##### XGBOOST MODEL #####
        # Get best parameter from database
        sql_xgb_model_param = """SELECT best_param_b 
                                FROM hse_analytics_param 
                                WHERE ir_type = 'ir yearly' 
                                ORDER BY running_date DESC 
                                LIMIT 1 OFFSET 0"""
                    
        xgb_model_param = get_sql_data(sql_xgb_model_param, conn)
        xgb_model_param = xgb_model_param['best_param_b'][0]
       
        # Convert string to tuple
        xgb_model_param = ast.literal_eval(xgb_model_param)

        #Set Parameters
        xgb_lags = xgb_model_param['window_length']
        xgb_objective = 'reg:squarederror'
        xgb_strategy = "recursive"

        # Create regressor object
        xgb_regressor = XGBRegressor(objective=xgb_objective)
        xgb_forecaster = make_reduction(xgb_regressor, window_length=xgb_lags, strategy=xgb_strategy)
        
        logMessage("Creating XGBoost Model ....")
        xgb_forecaster.fit(train_df, X=train_exog) #, X=train_exog
        
        logMessage("XGBoost Model Prediction ...")
        xgb_forecast = xgb_forecaster.predict(fh, X=future_exog) #, X=future_exog
        y_pred_xgb = pd.DataFrame(xgb_forecast).applymap('{:,.2f}'.format)
        y_pred_xgb['year_num'] = [i.year for i in xgb_forecast.index]
        #Rename colum 0
        y_pred_xgb.rename(columns={0:'forecast_b'}, inplace=True)


        ##### RANDOM FOREST MODEL #####
        # Get best parameter from database
        sql_ranfor_model_param = """SELECT best_param_c
                                FROM hse_analytics_param 
                                WHERE ir_type = 'ir yearly' 
                                ORDER BY running_date DESC 
                                LIMIT 1 OFFSET 0"""
                    
        ranfor_model_param = get_sql_data(sql_ranfor_model_param, conn)
        ranfor_model_param = ranfor_model_param['best_param_c'][0]
       
        # Convert string to tuple
        ranfor_model_param = ast.literal_eval(ranfor_model_param)

        #Set parameters
        ranfor_n_estimators = ranfor_model_param['estimator__n_estimators']
        ranfor_random_state = 0
        ranfor_criterion = "squared_error"
        ranfor_lags = ranfor_model_param['window_length']
        ranfor_strategy = "recursive"

        # create regressor object
        ranfor_regressor = RandomForestRegressor(n_estimators = ranfor_n_estimators, random_state=ranfor_random_state, criterion=ranfor_criterion)
        ranfor_forecaster = make_reduction(ranfor_regressor, window_length= ranfor_lags, strategy=ranfor_strategy)
        
        logMessage("Creating Random Forest Model ...")
        ranfor_forecaster.fit(train_df, X=train_exog) #, X=train_exog
        
        logMessage("Random Forest Model Prediction ...")
        ranfor_forecast = ranfor_forecaster.predict(fh, X=future_exog) #, X=future_exog
        y_pred_ranfor = pd.DataFrame(ranfor_forecast).applymap('{:,.2f}'.format)
        y_pred_ranfor['year_num'] = [i.year for i in ranfor_forecast.index]
        #Rename colum 0
        y_pred_ranfor.rename(columns={0:'forecast_c'}, inplace=True)


        ##### LINEAR REGRESSION MODEL #####
        # Get best parameter from database
        sql_linreg_model_param = """SELECT best_param_d
                                FROM hse_analytics_param 
                                WHERE ir_type = 'ir yearly' 
                                ORDER BY running_date DESC 
                                LIMIT 1 OFFSET 0"""
                    
        linreg_model_param = get_sql_data(sql_linreg_model_param, conn)
        linreg_model_param = linreg_model_param['best_param_d'][0]
       
        # Convert string to tuple
        linreg_model_param = ast.literal_eval(linreg_model_param)

        #Set parameters
        linreg_normalize = True
        linreg_lags = linreg_model_param['window_length']
        linreg_strategy = "recursive"

        # Create regressor object
        linreg_regressor = LinearRegression()
        linreg_forecaster = make_reduction(linreg_regressor, window_length=linreg_lags, strategy=linreg_strategy)
        
        logMessage("Creating Linear Regression Model ...")
        linreg_forecaster.fit(train_df, X=train_exog) #, X=train_exog
        
        logMessage("Linear Regression Model Prediction ...")
        linreg_forecast = linreg_forecaster.predict(fh, X=future_exog) #, X=future_exog
        y_pred_linreg = pd.DataFrame(linreg_forecast).applymap('{:,.2f}'.format)
        y_pred_linreg['year_num'] = [i.year for i in linreg_forecast.index]
        #Rename colum 0
        y_pred_linreg.rename(columns={0:'forecast_d'}, inplace=True)


        ##### POLYNOMIAL REGRESSION DEGREE=2 MODEL #####
        # Get best parameter from database
        sql_poly2_model_param = """SELECT best_param_e
                                FROM hse_analytics_param 
                                WHERE ir_type = 'ir yearly' 
                                ORDER BY running_date DESC 
                                LIMIT 1 OFFSET 0"""
                    
        poly2_model_param = get_sql_data(sql_poly2_model_param, conn)
        poly2_model_param = poly2_model_param['best_param_e'][0]
       
        # Convert string to tuple
        poly2_model_param = ast.literal_eval(poly2_model_param)

        #Set parameters
        poly2_regularization = None
        poly2_interactions = False
        poly2_lags = poly2_model_param['window_length']
        poly2_strategy = "recursive"

        # Create regressor object
        poly2_regressor = PolynomRegressor(deg=2, regularization=poly2_regularization, interactions=poly2_interactions)
        poly2_forecaster = make_reduction(poly2_regressor, window_length=poly2_lags, strategy=poly2_strategy)
        
        logMessage("Creating Polynomial Regression Orde 2 Model ...")
        poly2_forecaster.fit(train_df, X=train_exog) #, X=train_exog
        
        logMessage("Polynomial Regression Orde 2 Model Prediction ...") 
        poly2_forecast = poly2_forecaster.predict(fh, X=future_exog) #, X=future_exog
        y_pred_poly2 = pd.DataFrame(poly2_forecast).applymap('{:,.2f}'.format)
        y_pred_poly2['year_num'] = [i.year for i in poly2_forecast.index]
        #Rename colum 0
        y_pred_poly2.rename(columns={0:'forecast_e'}, inplace=True)


        ##### POLYNOMIAL REGRESSION DEGREE=3 MODEL #####
        # Get best parameter from database
        sql_poly3_model_param = """SELECT best_param_f
                                FROM hse_analytics_param 
                                WHERE ir_type = 'ir yearly' 
                                ORDER BY running_date DESC 
                                LIMIT 1 OFFSET 0"""
                    
        poly3_model_param = get_sql_data(sql_poly3_model_param, conn)
        poly3_model_param = poly3_model_param['best_param_f'][0]
       
        # Convert string to tuple
        poly3_model_param = ast.literal_eval(poly3_model_param)
        
        #Set parameters
        poly3_regularization = None
        poly3_interactions = False
        poly3_lags = poly3_model_param['window_length']
        poly3_strategy = "recursive"

        # Create regressor object
        poly3_regressor = PolynomRegressor(deg=3, regularization=poly3_regularization, interactions=poly3_interactions)
        poly3_forecaster = make_reduction(poly3_regressor, window_length=poly3_lags, strategy=poly3_strategy)
        
        logMessage("Creating Polynomial Regression Orde 3 Model ...")
        poly3_forecaster.fit(train_df, X=train_exog) #, X=train_exog
        
        logMessage("Polynomial Regression Orde 3 Model Prediction ...")
        poly3_forecast = poly3_forecaster.predict(fh, X=future_exog) #, X=future_exog
        y_pred_poly3 = pd.DataFrame(poly3_forecast).applymap('{:,.2f}'.format)
        y_pred_poly3['year_num'] = [i.year for i in poly3_forecast.index]
        #Rename colum 0
        y_pred_poly3.rename(columns={0:'forecast_f'}, inplace=True)


        ##### JOIN PREDICTION RESULT TO DATAFRAME #####
        logMessage("Creating all model prediction result data frame ...")
        y_all_pred = pd.concat([y_pred_arimax[['forecast_a']],
                                y_pred_xgb[['forecast_b']],
                                y_pred_ranfor[['forecast_c']],
                                y_pred_linreg[['forecast_d']],
                                y_pred_poly2[['forecast_e']],
                                y_pred_poly3[['forecast_f']]], axis=1)
        #y_all_pred['year_num'] = future_exog.index.values

        #%%
        ##### PLOT PREDICTION #####
        fig, ax = plt.subplots(figsize=(20,8))
        ax.plot(train_df.to_timestamp(), label='train')
        ax.plot(arimax_forecast.to_timestamp(), label='pred_arimax')
        ax.plot(ranfor_forecast.to_timestamp(), label='pred_ranfor')
        ax.plot(xgb_forecast.to_timestamp(), label='pred_xgb')
        ax.plot(linreg_forecast.to_timestamp(), label='pred_linreg')
        ax.plot(poly2_forecast.to_timestamp(), label='pred_poly2')
        ax.plot(poly3_forecast.to_timestamp(), label='pred_poly3')
        title = 'Feed Gas BP Tangguh Forecasting with Exogenous Variable and Cleaning Data'
        ax.set_title(title)
        ax.set_ylabel("Feed Gas")
        ax.set_xlabel("Datestamp")
        ax.legend(loc='best')
        plt.close()
        
        #%%
        # Save forecast result to database
        logMessage("Updating forecast result to database ...")
        total_updated_rows = insert_forecast(conn, y_all_pred)
        logMessage("Updated rows: {}".format(total_updated_rows))

        logMessage("Done")
    except Exception as e:
        logMessage(e)

# %%
def insert_forecast(conn, y_pred):
    total_updated_rows = 0
    for index, row in y_pred.iterrows():
        year_num = index.year #row['date']
        #year_num = 2023  # dummy
        forecast_a, forecast_b, forecast_c, forecast_d, forecast_e, forecast_f = row[0], row[1], row[2], row[3], row[4], row[5]
        
        #sql = f'UPDATE trir_monthly_test SET forecast_a = {} WHERE year_num = {} AND month_num = {}'.format(forecast, year_num)
        updated_rows = update_value(conn, forecast_a, forecast_b, forecast_c, forecast_d, forecast_e, forecast_f, year_num)
        total_updated_rows = total_updated_rows + updated_rows 
        
    return total_updated_rows

def update_value(conn, forecast_a, forecast_b, forecast_c, 
                        forecast_d, forecast_e, forecast_f, year_num):
    
    date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    updated_by = 'PYTHON'
    
    """ insert forecasting result after last row in table """
    sql =  """ UPDATE hse_analytics_trir_yearly
                SET forecast_a = %s, 
                    forecast_b = %s, 
                    forecast_c = %s, 
                    forecast_d = %s, 
                    forecast_e = %s, 
                    forecast_f = %s,
                    updated_at = %s, 
                    updated_by = %s
                WHERE year_num = %s"""
    #year_num
    #conn = None
    updated_rows = 0
    try:
        # create a new cursor
        cur = conn.cursor()
        # execute the UPDATE  statement
        cur.execute(sql, (forecast_a, forecast_b, forecast_c, forecast_d, forecast_e, forecast_f, 
                          date_now, updated_by, year_num))
        # get the number of updated rows
        updated_rows = cur.rowcount
        # Commit the changes to the database
        conn.commit()
        # Close cursor
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        #logging.error(error)

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