#%%
#Import library
import logging
import configparser
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import psycopg2
import seaborn as sns
from datetime import datetime
from tokenize import Ignore
from tracemalloc import start

from pmdarima.arima.auto import auto_arima

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import pmdarima as pm
from pmdarima import model_selection 
from pmdarima.arima import auto_arima
plt.style.use('fivethirtyeight')

from openpyxl import Workbook

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
    
    from connection import config, retrieve_data, create_db_connection, get_sql_data
    from utils import configLogging, logMessage, ad_test
    from polyfit import PolynomRegressor, Constraints
    
    # Configure logging
    #configLogging("ir_monthly_cum_insample.log")
    
    # Connect to database
    # Exit program if not connected to database
    logMessage("Connecting to database ...")
    conn = create_db_connection(section='postgresql_ml_hse_skk')
    if conn == None:
        exit()
        
    #Load Data from Database
    #query_1 = open("query_month_cum.sql", mode="rt").read()
    query_1 = open(os.path.join('hse/sql', 'query_month_cum.sql'), mode="rt").read()
    data = get_sql_data(query_1, conn)
    #data = retrieve_data(query_1)
    data['year_num'] = data['year_num'].astype(int)
    data['month_num'] = data['month_num'].astype(int)
    data['date'] = data['year_num'].astype(str) + '-' + data['month_num'].astype(str)
    #data

    # Prepare data
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m')
    data = data[~(data['date'] > '2022-12')]
    data = data.rename(columns=str.lower)
    data['date'] = pd.PeriodIndex(data['date'], freq='M')
    data = data.reset_index()
    data.head()

    ds = 'date'
    y = 'trir_cum' 

    df = data[[ds,y]]
    df = df.set_index(ds)
    df.index = pd.PeriodIndex(df.index, freq='M')

    #%%
    #stationarity_check(df.to_timestamp())

    #%%
    #decomposition_plot(df.to_timestamp())

    #%%
    #plot_acf_pacf(df.to_timestamp())

    #%%
    #from chart_studio.plotly import plot_mpl
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(df.trir_cum.values, model="additive", period=24)
    fig = result.plot()
    #plt.show()
    plt.close()

    #%%
    ad_test(df['trir_cum'])

    #%%
    from sktime.utils.plotting import plot_series
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.model_selection import temporal_train_test_split
    from sktime.forecasting.compose import make_reduction

    # Test size
    test_size = 0.07

    # Split data
    y_train, y_test = temporal_train_test_split(df, test_size=test_size)
    #y_test.head(10)

    # Create forecasting Horizon
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    #fh

    ## Create Exogenous Variable
    df['drilling_explor_cum'] = data['bor_eksplorasi_cum'].values
    df['drilling_explot_cum'] = data['bor_eksploitasi_cum'].values
    df['workover_cum'] = data['workover_cum'].values
    df['wellservice_cum'] = data['wellservice_cum'].values
    df['survei_seismic_cum'] = data['survey_seismic_cum'].values
    df['bulan'] = [i.month for i in df.index]
    df['drilling_explor_cum'].fillna(method='ffill', inplace=True)
    df['drilling_explot_cum'].fillna(method='ffill', inplace=True)
    df['workover_cum'].fillna(method='ffill', inplace=True)
    df['wellservice_cum'].fillna(method='ffill', inplace=True)
    df['survei_seismic_cum'].fillna(method='ffill', inplace=True)
    #df.tail(20)

    # Split into train and test
    X_train, X_test = temporal_train_test_split(df.iloc[:,1:], test_size=test_size)
    #X_train.tail()

    #%%
    # plotting for illustration
    plt.figure(figsize=(20,8))
    plt.plot(y_train.to_timestamp(), label='train')
    plt.plot(y_test.to_timestamp(), label='test')
    plt.ylabel("Incident Rate Cumulative")
    plt.xlabel("Datestamp")
    plt.legend(loc='best')
    plt.close()

    # %%
    ##### FORECASTING #####
    from sklearn.metrics import mean_absolute_percentage_error

    ##### ARIMAX MODEL #####
    import pmdarima as pm
    from sktime.forecasting.arima import AutoARIMA

    #Set parameters
    arimax_differencing = 1
    arimax_seasonal = False
    arimax_error_action = 'warn'
    arimax_trace = True
    arimax_supress_warnings = True
    arimax_stepwise = True
    arimax_stationary = False

    # Create ARIMAX Model
    #arimax_model = pm.auto_arima(y=y_train, X=X_train, start_p=0, d=1, start_q=0, 
    #                            max_p=10, max_d=0, max_q=10,
    #                            m=0, seasonal=False, error_action='warn',trace=True,
    #                            supress_warnings=True,stepwise=True, stationary=False)
    arimax_model = AutoARIMA(d=arimax_differencing, start_p=0, start_q=0, max_p=10, max_q=10, suppress_warnings=arimax_supress_warnings, error_action=arimax_error_action, trace=arimax_trace, stationary=arimax_stationary) #If using SKTime AutoArima
    logMessage("Creating ARIMAX Model ...")

    arimax_model.fit(y_train, X=X_train)
    logMessage("ARIMAX Model Summary")
    logMessage(arimax_model.summary())

    logMessage("ARIMAX Model Prediction ..")
    arimax_forecast = arimax_model.predict(fh, X=X_test) #n_periods=len(fh)
    y_pred_arimax = pd.DataFrame(arimax_forecast).applymap('{:,.2f}'.format)
    y_pred_arimax['month_num'] = [i.month for i in arimax_forecast.index]
    y_pred_arimax['year_num'] = [i.year for i in arimax_forecast.index]
    #Rename colum 0
    y_pred_arimax.rename(columns={0:'forecast_a'}, inplace=True)

    mape_arimax = mean_absolute_percentage_error(y_test, arimax_forecast)
    mape_arimax_str = str('MAPE: %.4f' % mape_arimax)
    logMessage("ARIMAX Model "+mape_arimax_str)
    
    #Get parameter
    arimax_param = str(arimax_model.get_fitted_params()['order'])
    logMessage("Arimax Model Parameters "+arimax_param)


    ##### XGBOOST MODEL #####
    from xgboost import XGBRegressor

    #Set parameters
    xgb_objective = 'reg:squarederror'
    xgb_lags = 8 #6 8 10
    xgb_strategy = "recursive"

    # Create regressor object
    xgb_regressor = XGBRegressor(objective=xgb_objective)
    xgb_forecaster = make_reduction(xgb_regressor, window_length=xgb_lags, strategy=xgb_strategy)
    
    logMessage("Creating XGBoost Model ....")
    xgb_forecaster.fit(y_train, X=X_train)

    logMessage("XGBoost Model Prediction ...")
    xgb_forecast = xgb_forecaster.predict(fh, X=X_test)
    y_pred_xgb = pd.DataFrame(xgb_forecast).applymap('{:,.2f}'.format)
    y_pred_xgb['month_num'] = [i.month for i in xgb_forecast.index]
    y_pred_xgb['year_num'] = [i.year for i in xgb_forecast.index]
    #Rename colum 0
    y_pred_xgb.rename(columns={'trir_cum':'forecast_b'}, inplace=True)

    # Calculate model performance
    mape_xgb = mean_absolute_percentage_error(y_test, xgb_forecast)
    mape_xgb_str = str('MAPE: %.4f' % mape_xgb)
    logMessage("XGBoost Model "+mape_xgb_str)
    
    #Get Parameters
    xgb_param_lags = xgb_forecaster.get_params()['window_length']
    xgb_param = str({'window_length': xgb_param_lags})
    logMessage("XGBoost Model Parameters "+xgb_param)


    ##### RANDOM FOREST MODEL #####
    from sklearn.ensemble import RandomForestRegressor

    #Set parameters
    ranfor_n_estimators = 100
    ranfor_random_state = 0
    ranfor_criterion = "squared_error"
    ranfor_lags = 19 #9 17 19
    ranfor_strategy = "recursive"

    # create regressor object
    ranfor_regressor = RandomForestRegressor(n_estimators=ranfor_n_estimators, random_state=ranfor_random_state, criterion=ranfor_criterion)
    ranfor_forecaster = make_reduction(ranfor_regressor, window_length=ranfor_lags, strategy=ranfor_strategy)
    
    logMessage("Creating Random Forest Model ...")
    ranfor_forecaster.fit(y_train, X_train)

    logMessage("Random Forest Model Prediction ...")
    ranfor_forecast = ranfor_forecaster.predict(fh, X=X_test)
    y_pred_ranfor = pd.DataFrame(ranfor_forecast).applymap('{:,.2f}'.format)
    y_pred_ranfor['month_num'] = [i.month for i in ranfor_forecast.index]
    y_pred_ranfor['year_num'] = [i.year for i in ranfor_forecast.index]
    #Rename colum 0
    y_pred_ranfor.rename(columns={'trir_cum':'forecast_c'}, inplace=True)

    mape_ranfor = mean_absolute_percentage_error(y_test, ranfor_forecast)
    mape_ranfor_str = str('MAPE: %.4f' % mape_ranfor)
    logMessage("Random Forest Model "+mape_ranfor_str)
    
    #Get Parameters
    ranfor_n_estimator = ranfor_forecaster.get_params()['estimator__n_estimators']
    ranfor_window_length = ranfor_forecaster.get_params()['window_length']
    ranfor_param = str({'estimator__n_estimators': ranfor_n_estimator, 'window_length': ranfor_window_length})
    logMessage("Random Forest Model Parameters "+ranfor_param)


    ##### LINEAR REGRESSION MODEL #####
    from sklearn.linear_model import LinearRegression

    #Set parameters
    linreg_normalize = True
    linreg_lags = 1 #1 2
    linreg_strategy = "recursive"

    #Create regressor object
    linreg_regressor = LinearRegression(normalize=linreg_normalize)
    linreg_forecaster = make_reduction(linreg_regressor, window_length=linreg_lags, strategy=linreg_strategy)
    
    logMessage("Creating Linear Regression Model ...")
    linreg_forecaster.fit(y_train, X=X_train)

    logMessage("Linear Regression Model Prediction ...")
    linreg_forecast = linreg_forecaster.predict(fh, X=X_test)
    y_pred_linreg = pd.DataFrame(linreg_forecast).applymap('{:,.2f}'.format)
    y_pred_linreg['month_num'] = [i.month for i in linreg_forecast.index]
    y_pred_linreg['year_num'] = [i.year for i in linreg_forecast.index]
    #Rename colum 0
    y_pred_linreg.rename(columns={'trir_cum':'forecast_d'}, inplace=True)

    mape_linreg = mean_absolute_percentage_error(y_test, linreg_forecast)
    mape_linreg_str = str('MAPE: %.4f' % mape_linreg)
    logMessage("Linear Regression Model "+mape_linreg_str)
    
    #Get parameters
    linreg_window_length = linreg_forecaster.get_params()['window_length']
    linreg_param = str({'window_length': linreg_window_length})
    logMessage("Linear Regression Model Parameters "+linreg_param)


    ##### POLYNOMIAL REGRESSION DEGREE=2 MODEL #####
    #Set parameters
    poly2_regularization = None
    poly2_interactions = False
    poly2_lags = 1 #0.95 1 2
    poly2_strategy = "recursive"

    #Create regressor object
    poly2_regressor = PolynomRegressor(deg=2, regularization=poly2_regularization, interactions=poly2_interactions)
    poly2_forecaster = make_reduction(poly2_regressor, window_length=poly2_lags, strategy=poly2_strategy) 
    
    logMessage("Creating Polynomial Regression Orde 2 Model ...")
    poly2_forecaster.fit(y_train, X=X_train) #, X=X_train

    logMessage("Polynomial Regression Orde 2 Model Prediction ...")
    poly2_forecast = poly2_forecaster.predict(fh, X=X_test)
    y_pred_poly2 = pd.DataFrame(poly2_forecast).applymap('{:,.2f}'.format)
    y_pred_poly2['month_num'] = [i.month for i in poly2_forecast.index]
    y_pred_poly2['year_num'] = [i.year for i in poly2_forecast.index]
    #Rename colum 0
    y_pred_poly2.rename(columns={'trir_cum':'forecast_e'}, inplace=True)

    mape_poly2 = mean_absolute_percentage_error(y_test, poly2_forecast)
    mape_poly2_str = str('MAPE: %.4f' % mape_poly2)
    logMessage("Polynomial Regression Orde 2 Model "+mape_poly2_str)
    
    #Get parameters
    poly2_window_length = poly2_forecaster.get_params()['window_length']
    poly2_param = str({'window_length': poly2_window_length})
    logMessage("Polynomial Regression Orde 2 Model Parameters "+poly2_param)
    

    ##### POLYNOMIAL REGRESSION DEGREE=3 MODEL #####
    #Set parameters
    poly3_regularization = None
    poly3_interactions = False
    poly3_lags = 1 #0.94, 1
    poly3_strategy = "recursive"

    #Create regressor object
    poly3_regressor = PolynomRegressor(deg=3, regularization=poly3_regularization, interactions=poly3_interactions)
    poly3_forecaster = make_reduction(poly3_regressor, window_length=poly3_lags, strategy=poly3_strategy) 
    
    logMessage("Creating Polynomial Regression Orde 3 Model ...")
    poly3_forecaster.fit(y_train, X=X_train) #, X=X_train

    logMessage("Polynomial Regression Orde 3 Model Prediction ...")
    poly3_forecast = poly3_forecaster.predict(fh, X=X_test)
    y_pred_poly3 = pd.DataFrame(poly3_forecast).applymap('{:,.2f}'.format)
    y_pred_poly3['month_num'] = [i.month for i in poly3_forecast.index]
    y_pred_poly3['year_num'] = [i.year for i in poly3_forecast.index]
    #Rename colum 0
    y_pred_poly3.rename(columns={'trir_cum':'forecast_f'}, inplace=True)

    mape_poly3 = mean_absolute_percentage_error(y_test, poly3_forecast)
    mape_poly3_str = str('MAPE: %.4f' % mape_poly3)
    logMessage("Polynomial Regression Orde 3 Model "+mape_poly3_str)
    
    #Get parameters
    poly3_window_length = poly3_forecaster.get_params()['window_length']
    poly3_param = str({'window_length': poly3_window_length})
    logMessage("Polynomial Regression Orde 3 Model Parameters "+poly3_param)

    # %%
    #CREATE DATAFRAME MAPE
    logMessage("Creating all model mape result data frame ...")
    all_mape_pred = {'mape_forecast_a' : [mape_arimax],
                    'mape_forecast_b' : [mape_xgb],
                    'mape_forecast_c' : [mape_ranfor],
                    'mape_forecast_d' : [mape_linreg],
                    'mape_forecast_e' : [mape_poly2],
                    'mape_forecast_f' : [mape_poly3],
                    'type_id' : 1}
    
    all_mape_pred = pd.DataFrame(all_mape_pred)
    
    #CREATE PARAMETERS TO DATAFRAME
    logMessage("Creating all model params result data frame ...")
    all_model_param =  {'model_param_a': [arimax_param],
                        'model_param_b': [xgb_param],
                        'model_param_c': [ranfor_param],
                        'model_param_d': [linreg_param],
                        'model_param_e': [poly2_param],
                        'model_param_f': [poly3_param],
                        'ir_type' : 'ir monthly cumulative'}

    all_model_param = pd.DataFrame(all_model_param)

    
    # Save mape result to database
    logMessage("Updating MAPE result to database ...")
    total_updated_rows = insert_mape(conn, all_mape_pred)
    logMessage("Updated rows: {}".format(total_updated_rows))
    
    # Save param result to database
    logMessage("Updating Model Parameter result to database ...")
    total_updated_rows = insert_param(conn, all_model_param)
    logMessage("Updated rows: {}".format(total_updated_rows))
    
    logMessage("Done")
    
# %%
def insert_mape(conn, all_mape_pred):
    total_updated_rows = 0
    for index, row in all_mape_pred.iterrows():
        type_id = row['type_id']
        mape_forecast_a, mape_forecast_b, mape_forecast_c, mape_forecast_d, mape_forecast_e, mape_forecast_f = row[0], row[1], row[2], row[3], row[4], row[5]
        
        #sql = f'UPDATE trir_monthly_test SET forecast_a = {} WHERE year_num = {} AND month_num = {}'.format(forecast, year_num, month_num)
        updated_rows = update_mape_value(conn, mape_forecast_a, mape_forecast_b, mape_forecast_c, mape_forecast_d, mape_forecast_e, mape_forecast_f, type_id)
        total_updated_rows = total_updated_rows + updated_rows 
        
    return total_updated_rows

def insert_param(conn, all_model_param):
    total_updated_rows = 0
    for index, row in all_model_param.iterrows():
        ir_type = row['ir_type']
        model_param_a, model_param_b, model_param_c, model_param_d, model_param_e, model_param_f = row[0], row[1], row[2], row[3], row[4], row[5]
        
        #sql = f'UPDATE trir_monthly_test SET forecast_a = {} WHERE year_num = {} AND month_num = {}'.format(forecast, year_num, month_num)
        updated_rows = update_param_value(conn, model_param_a, model_param_b, model_param_c, model_param_d, model_param_e, model_param_f, ir_type)
        total_updated_rows = total_updated_rows + updated_rows 
        
    return total_updated_rows

def update_mape_value(conn, mape_forecast_a, mape_forecast_b, mape_forecast_c, 
                        mape_forecast_d, mape_forecast_e, mape_forecast_f, type_id):
    
    date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    created_by = 'PYTHON'
    
    """ insert mape result after last row in table """
    sql = """ INSERT INTO hse_analytics_mape
                    (type_id,
                    running_date,
                    mape_forecast_a,
                    mape_forecast_b,
                    mape_forecast_c,
                    mape_forecast_d,
                    mape_forecast_e,
                    mape_forecast_f,
                    created_at,
                    created_by)
                    VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
          """
                
    #conn = None
    updated_rows = 0
    try:
        # create a new cursor
        cur = conn.cursor()
        # execute the UPDATE  statement
        cur.execute(sql, (type_id, date_now, mape_forecast_a, mape_forecast_b, mape_forecast_c, mape_forecast_d, mape_forecast_e, mape_forecast_f,
                          date_now, created_by))
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
                        model_param_d, model_param_e, model_param_f, ir_type):
    
    date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    created_by = 'PYTHON'
    
    """ insert mape result after last row in table """
    sql = """ INSERT INTO hse_analytics_param
                    (ir_type,
                    running_date,
                    best_param_a,
                    best_param_b,
                    best_param_c,
                    best_param_d,
                    best_param_e,
                    best_param_f,
                    created_at,
                    created_by)
                    VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
          """
    
    #conn = None
    updated_rows = 0
    try:
        # create a new cursor
        cur = conn.cursor()
        # execute the UPDATE  statement
        cur.execute(sql, (ir_type, date_now, model_param_a, model_param_b, model_param_c, model_param_d, model_param_e, model_param_f,
                          date_now, created_by))
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
