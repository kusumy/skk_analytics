# %%
import logging
import configparser
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
#import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import psycopg2
import seaborn as sns
from datetime import datetime
from tokenize import Ignore
from tracemalloc import start

from pmdarima.arima.auto import auto_arima
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from connection import config, retrieve_data, create_db_connection, get_sql_data
from utils import configLogging, logMessage, ad_test

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
def main():
    # Configure logging
    configLogging("ir_yearly_insample.log")
    
    # Connect to database
    # Exit program if not connected to database
    logMessage("Connecting to database ...")
    conn = create_db_connection(section='postgresql_ml_hse_skk')
    if conn == None:
        exit()
    
    query_1 = open(os.path.join('hse/insample', 'query_yearly.sql'), mode="rt").read()    
    #query_1 = open("query_yearly.sql", mode="rt").read()
    data = get_sql_data(query_1, conn)
    #data = retrieve_data(query_1)
    data['year_num'] = data['year_num'].astype(int)
    data['year_num'] = pd.to_datetime(data['year_num'], format='%Y')
    data['year_num'] = pd.PeriodIndex(data['year_num'], freq='Y')
    data = data.rename(columns=str.lower)
    data = data.reset_index()
    #data

    ds = 'year_num'
    y = 'trir' 
    df = data[[ds,y]]
    df = df.set_index(ds)
    df.index = pd.PeriodIndex(df.index, freq='Y')
    #df

    #import chart_studio.plotly
    #import cufflinks as cf

    from plotly.offline import iplot
    #cf.go_offline()
    #cf.set_config_file(offline = False, world_readable = True)

    #df.iplot(title="Yearly Incident Rate SKK Migas")

    #%%
    #stationarity_check(df['trir'])

    #%%
    #decomposition_plot(df['trir'])

    #%%
    #plot_acf_pacf(df['trir'])

    #%%
    #from chart_studio.plotly import plot_mpl
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(df.trir.values, model="multiplicative", period=5)
    fig = result.plot()
    plt.close()

    #%%
    ad_test(df['trir'])

    from sktime.forecasting.model_selection import temporal_train_test_split
    # Test size
    test_size = 0.1

    # Split data
    y_train, y_test = temporal_train_test_split(df['trir'], test_size=test_size)
    #y_train.tail()

    #%%
    from sktime.forecasting.base import ForecastingHorizon
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    #fh

    #%%
    ## Create Exogenous Variable
    df['survey_seismic'] = data['survey_seismic'].values
    df['bor_eksplorasi'] = data['bor_eksplorasi'].values
    df['bor_eksploitasi'] = data['bor_eksploitasi'].values
    df['workover'] = data['workover'].values
    df['wellservice'] = data['wellservice'].values
    #df.tail(20)

    #%%
    # Split into train and test
    X_train, X_test = temporal_train_test_split(df.iloc[:,1:], test_size=test_size)
    #X_train

    #%%
    exogenous_features = ["survei_seismic", "bor_eksplorasi", "bor_eksploitasi",
                        "workover", "wellservice"]
    #exogenous_features

    # %%
    ##### FORECASTING #####

    ##### ARIMAX MODEL #####
    from pmdarima.arima.utils import ndiffs, nsdiffs
    from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
    import statsmodels.api as sm

    #Set parameters
    arimax_differencing = 1
    arimax_trace = True
    arimax_error_action = "ignore"
    arimax_suppress_warnings = True

    # Create SARIMA Model
    arimax_model = auto_arima(y_train, exogenous=X_train, d=arimax_differencing, trace=arimax_trace, error_action=arimax_error_action, suppress_warnings=arimax_suppress_warnings)
    arimax_model.fit(y_train, exogenous=X_train)

    arimax_forecast = arimax_model.predict(n_periods=len(fh), exogenous=X_test)
    y_pred_arimax = pd.DataFrame(arimax_forecast).applymap('{:,.2f}'.format)
    y_pred_arimax['year_num'] = [i.year for i in arimax_forecast.index]
    #Rename colum 0
    y_pred_arimax.rename(columns={0:'forecast_a'}, inplace=True)
    mape_arimax = mean_absolute_percentage_error(y_test, arimax_forecast)

    ##### XGBOOST MODEL #####
    from xgboost import XGBRegressor
    from sktime.forecasting.compose import make_reduction

    #Set parameters
    xgb_objective = 'reg:squarederror'
    xgb_lags = 6
    xgb_strategy = "recursive"

    #Create regressor object
    xgb_regressor = XGBRegressor(objective=xgb_objective)
    xgb_forecaster = make_reduction(xgb_regressor, window_length=xgb_lags, strategy=xgb_strategy)
    xgb_forecaster.fit(y_train, X=X_train)

    xgb_forecast = xgb_forecaster.predict(fh, X=X_test)
    y_pred_xgb = pd.DataFrame(xgb_forecast).applymap('{:,.2f}'.format)
    y_pred_xgb['month_num'] = [i.month for i in xgb_forecast.index]
    y_pred_xgb['year_num'] = [i.year for i in xgb_forecast.index]
    #Rename colum 0
    y_pred_xgb.rename(columns={0:'forecast_b'}, inplace=True)

    # Calculate model performance
    mape_xgb = mean_absolute_percentage_error(y_test, xgb_forecast)

    ##### RANDOM FOREST MODEL #####
    from sklearn.ensemble import RandomForestRegressor

    #Set parameters
    ranfor_n_estimators = 100
    ranfor_random_state = 0
    ranfor_criterion = "squared_error"
    ranfor_lags = 0.8
    ranfor_strategy = "recursive"

    # create regressor object
    ranfor_regressor = RandomForestRegressor(n_estimators=ranfor_n_estimators, random_state=ranfor_random_state, criterion=ranfor_criterion)
    ranfor_forecaster = make_reduction(ranfor_regressor, window_length=ranfor_lags, strategy=ranfor_strategy)
    ranfor_forecaster.fit(y_train, X_train)

    ranfor_forecast = ranfor_forecaster.predict(fh, X=X_test)
    y_pred_ranfor = pd.DataFrame(ranfor_forecast).applymap('{:,.2f}'.format)
    y_pred_ranfor['month_num'] = [i.month for i in ranfor_forecast.index]
    y_pred_ranfor['year_num'] = [i.year for i in ranfor_forecast.index]
    #Rename colum 0
    y_pred_ranfor.rename(columns={0:'forecast_c'}, inplace=True)

    mape_ranfor = mean_absolute_percentage_error(y_test, ranfor_forecast)


    ##### LINEAR REGRESSION MODEL #####
    from sklearn.linear_model import LinearRegression

    #Set parameters
    linreg_normalize = True
    linreg_lags = 0.8
    linreg_strategy = "recursive"

    #Create regressor object
    linreg_regressor = LinearRegression(normalize=linreg_normalize)
    linreg_forecaster = make_reduction(linreg_regressor, window_length=linreg_lags, strategy=linreg_strategy)
    linreg_forecaster.fit(y_train, X=X_train)

    linreg_forecast = linreg_forecaster.predict(fh, X=X_test)
    y_pred_linreg = pd.DataFrame(linreg_forecast).applymap('{:,.2f}'.format)
    y_pred_linreg['month_num'] = [i.month for i in linreg_forecast.index]
    y_pred_linreg['year_num'] = [i.year for i in linreg_forecast.index]
    #Rename colum 0
    y_pred_linreg.rename(columns={0:'forecast_d'}, inplace=True)

    mape_linreg = mean_absolute_percentage_error(y_test, linreg_forecast)


    ##### POLYNOMIAL REGRESSION DEGREE=2 MODEL #####
    from polyfit import PolynomRegressor, Constraints

    #Set parameters
    poly2_regularization = None
    poly2_interactions = False
    poly2_lags = 0.7
    poly2_strategy = "recursive"

    #Create regressor object
    poly2_regressor = PolynomRegressor(deg=2, regularization=poly2_regularization, interactions=poly2_interactions)
    poly2_forecaster = make_reduction(poly2_regressor, window_length=poly2_lags, strategy=poly2_strategy) 
    poly2_forecaster.fit(y_train, X=X_train) #, X=X_train

    # Create forecasting
    poly2_forecast = poly2_forecaster.predict(fh, X=X_test)
    y_pred_poly2 = pd.DataFrame(poly2_forecast).applymap('{:,.2f}'.format)
    y_pred_poly2['month_num'] = [i.month for i in poly2_forecast.index]
    y_pred_poly2['year_num'] = [i.year for i in poly2_forecast.index]
    #Rename colum 0
    y_pred_poly2.rename(columns={0:'forecast_e'}, inplace=True)

    mape_poly2 = mean_absolute_percentage_error(y_test, poly2_forecast)


    ##### POLYNOMIAL REGRESSION DEGREE=3 MODEL #####
    from polyfit import PolynomRegressor, Constraints

    #Set parameters
    poly3_regularization = None
    poly3_interactions = False
    poly3_lags = 0.6
    poly3_strategy = "recursive"

    #Create regressor object
    poly3_regressor = PolynomRegressor(deg=3, regularization=poly3_regularization, interactions=poly3_interactions)
    poly3_forecaster = make_reduction(poly3_regressor, window_length=poly3_lags, strategy=poly3_strategy) 
    poly3_forecaster.fit(y_train, X=X_train) #, X=X_train

    # Create forecasting
    poly3_forecast = poly3_forecaster.predict(fh, X=X_test)
    y_pred_poly3 = pd.DataFrame(poly3_forecast).applymap('{:,.2f}'.format)
    y_pred_poly3['month_num'] = [i.month for i in poly3_forecast.index]
    y_pred_poly3['year_num'] = [i.year for i in poly3_forecast.index]
    #Rename colum 0
    y_pred_poly3.rename(columns={0:'forecast_f'}, inplace=True)

    mape_poly3 = mean_absolute_percentage_error(y_test, poly3_forecast)

    # %%
    #logMessage("Creating all model prediction result data frame ...")
    y_all_pred = pd.concat([y_pred_arimax[['forecast_a']], 
                            y_pred_xgb[['forecast_b']], 
                            y_pred_ranfor[['forecast_c']], 
                            y_pred_linreg[['forecast_d']], 
                            y_pred_poly2[['forecast_e']], 
                            y_pred_poly3[['forecast_f']]], axis=1)
    #y_all_pred


    #Create Dataframe Mape All Method
    dict_mape_pred = {'mape_forecast_a' : [mape_arimax],
                    'mape_forecast_b' : [mape_xgb],
                    'mape_forecast_c' : [mape_ranfor],
                    'mape_forecast_d' : [mape_linreg],
                    'mape_forecast_e' : [mape_poly2],
                    'mape_forecast_f' : [mape_poly3],
                    'type_id' : 2}
    all_mape_pred = pd.DataFrame(dict_mape_pred)
    #all_mape_pred

    # Save forecast result to database
    logMessage("Updating forecast result to database ...")
    total_updated_rows = insert_forecast(conn, y_all_pred)
    logMessage("Updated rows: {}".format(total_updated_rows))
    
    # Save mape result to database
    logMessage("Updating MAPE result to database ...")
    total_updated_rows = insert_mape(conn, all_mape_pred)
    logMessage("Updated rows: {}".format(total_updated_rows))
    
    logMessage("Done")
    
# %%
def insert_forecast(conn, y_all_pred):
    total_updated_rows = 0
    for index, row in y_all_pred.iterrows():
        year_num = index.year #row['date']
        forecast_a, forecast_b, forecast_c, forecast_d, forecast_e, forecast_f = row[0], row[1], row[2], row[3], row[4], row[5]
        
        #sql = f'UPDATE trir_monthly_test SET forecast_a = {} WHERE year_num = {} AND month_num = {}'.format(forecast, year_num, month_num)
        updated_rows = update_value(conn, forecast_a, forecast_b, forecast_c, forecast_d, forecast_e, forecast_f, year_num)
        total_updated_rows = total_updated_rows + updated_rows 

def insert_mape(conn, all_mape_pred):
    total_updated_rows = 0
    for index, row in all_mape_pred.iterrows():
        type_id = row['type_id']
        mape_forecast_a, mape_forecast_b, mape_forecast_c, mape_forecast_d, mape_forecast_e, mape_forecast_f = row[0], row[1], row[2], row[3], row[4], row[5]
        
        #sql = f'UPDATE trir_monthly_test SET forecast_a = {} WHERE year_num = {} AND month_num = {}'.format(forecast, year_num, month_num)
        updated_rows = update_mape_value(conn, mape_forecast_a, mape_forecast_b, mape_forecast_c, mape_forecast_d, mape_forecast_e, mape_forecast_f, type_id)
        total_updated_rows = total_updated_rows + updated_rows 
        
    return total_updated_rows

def update_value(conn, forecast_a, forecast_b, forecast_c, 
                        forecast_d, forecast_e, forecast_f, year_num):
    
    date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    updated_by = 'PYTHON'
    
    """ insert forecasting result after last row in table """
    sql =  """ UPDATE hse_analytics_trir_monthly_cum
                SET forecast_a = %s, 
                    forecast_b = %s, 
                    forecast_c = %s, 
                    forecast_d = %s, 
                    forecast_e = %s, 
                    forecast_f = %s,
                    updated_at = %s, 
                    updated_by = %s
                WHERE year_num = %s"""
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

def update_mape_value(conn, mape_forecast_a, mape_forecast_b, mape_forecast_c, 
                        mape_forecast_d, mape_forecast_e, mape_forecast_f, type):
    
    date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    created_by = 'PYTHON'
    
    """ insert mape result after last row in table """
    sql = """ UPDATE hse_analytics_mape
                SET mape_forecast_a = %s, 
                    mape_forecast_b = %s, 
                    mape_forecast_c = %s, 
                    mape_forecast_d = %s, 
                    mape_forecast_e = %s, 
                    mape_forecast_f = %s,
                    updated_at = %s, 
                    updated_by = %s
                WHERE type_id = %s"""
    updated_rows = 0
    try:
        # create a new cursor
        cur = conn.cursor()
        # execute the UPDATE  statement
        cur.execute(sql, (mape_forecast_a, mape_forecast_b, mape_forecast_c, mape_forecast_d, mape_forecast_e, mape_forecast_f,
                          date_now, created_by, type))
        # get the number of updated rows
        updated_rows = cur.rowcount
        # Commit the changes to the database
        conn.commit()
        # Close cursor
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        logMessage(error)

    return updated_rows















