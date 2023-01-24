# %%
import configparser
import logging 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
import plotly.express as px
import pmdarima as pm
import psycopg2
import seaborn as sns
import ast
from datetime import datetime
from humanfriendly import format_timespan
from tokenize import Ignore

from pmdarima import model_selection
from pmdarima.arima import auto_arima
from pmdarima.arima.auto import auto_arima
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sktime.utils.plotting import plot_series
from sktime.forecasting.base import ForecastingHorizon
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

import pmdarima as pm
from sktime.forecasting.arima import AutoARIMA, ARIMA
from pmdarima.arima.utils import ndiffs, nsdiffs
import statsmodels.api as sm
from xgboost import XGBRegressor
from sktime.forecasting.compose import make_reduction
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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
    #plt.show(block=False)
    
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

def main():
    
    from connection import config, retrieve_data, create_db_connection, get_sql_data
    from utils import configLogging, logMessage, ad_test
    from polyfit import PolynomRegressor, Constraints
    
    # Configure logging
    #configLogging("incident_rate_trir.log")
    
    # %%
    # Connect to database
    # Exit program if not connected to database
    logMessage("Connecting to database ...")
    conn = create_db_connection(section='postgresql_ml_hse_skk')
    if conn == None:
        exit()
        
    # Load Data from Database
    query_1 = open(os.path.join('hse/sql', 'query_month_cum.sql'), mode="rt").read()
    data = get_sql_data(query_1, conn)
    #data = retrieve_data(query_1, section='postgresql_ml_hse')
    data['year_num'] = data['year_num'].astype(int)
    data['month_num'] = data['month_num'].astype(int)
    data['date'] = data['year_num'].astype(str) + '-' + data['month_num'].astype(str)
    #data

    #%%
    # Prepare data
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m')
    data = data[~(data['date'] > '2022-12')]
    data = data.rename(columns=str.lower)

    data['date'] = pd.PeriodIndex(data['date'], freq='M')
    data = data.reset_index()
    #data.head()

    #%%
    ds = 'date'
    y = 'trir_cum' 

    df = data[[ds,y]]
    df = df.set_index(ds)
    df.index = pd.PeriodIndex(df.index, freq='M')
    #df

    #%%
    train_df = df['trir_cum']
    #train_df

    #stationary check
    #%%
    #stationarity_check(df.to_timestamp())
    #decomposition_plot(df.to_timestamp())
    #plot_acf_pacf(df.to_timestamp())

    #%%
    #from chart_studio.plotly import plot_mpl
    from statsmodels.tsa.seasonal import seasonal_decompose

    result = seasonal_decompose(df.trir_cum.values, model="additive", period=24)
    #fig = result.plot()
    #plt.show()

    #%%
    ad_test(df['trir_cum'])


    #%%
    # Create forecasting Horizon
    time_predict = pd.period_range('2022-11', periods=14, freq='M')
    # Create forecasting Horizon
    fh = ForecastingHorizon(time_predict, is_relative=False)

    #%%
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
    train_exog = df.iloc[:,1:]

    #%%
    #import exogenous for predict
    query_exog = open(os.path.join('hse/sql', 'query_month_cum3.sql'), mode="rt").read()
    data2 = get_sql_data(query_exog, conn)
    #data2 = retrieve_data(query_2, section='postgresql_ml_hse')
    data2['year_num'] = data2['year_num'].astype(int)
    data2['month_num'] = data2['month_num'].astype(int)
    data2['date'] = data2['year_num'].astype(str) + '-' + data2['month_num'].astype(str)
    # Prepare data
    data2['date'] = pd.to_datetime(data2['date'], format='%Y-%m')

    #%%
    future_exog = data2[['date', 'drilling_explor_cum', 'drilling_explot_cum', 'workover_cum',
                    'wellservice_cum', 'survei_seismic_cum']].copy()
    future_exog = future_exog.set_index(future_exog['date'])
    future_exog.index = pd.PeriodIndex(future_exog.index, freq='M')
    #future_exog.sort_index(inplace=True)
    future_exog.drop(['date'], axis=1, inplace=True)
    future_exog = future_exog.iloc[-14:]
    future_exog['bulan'] = [i.month for i in future_exog.index]

    # %%
    try :
        ##### FORECASTING #####
        ##### ARIMAX MODEL #####
        # Get best parameter from database
        sql_arimax_model_param = """SELECT best_param_a 
                                FROM hse_analytics_param 
                                WHERE ir_type = 'ir monthly cumulative' 
                                ORDER BY running_date DESC 
                                LIMIT 1 OFFSET 0"""
                    
        arimax_model_param = get_sql_data(sql_arimax_model_param, conn)
        arimax_model_param = arimax_model_param['best_param_a'][0]
        
        # Convert string to tuple
        arimax_model_param = ast.literal_eval(arimax_model_param)
        
        #Set parameters
        arimax_suppress_warnings = True
        
        arimax_model = ARIMA(order=arimax_model_param, suppress_warnings=arimax_suppress_warnings)
        logMessage("Creating ARIMAX Model ...")
        arimax_model.fit(df['trir_cum'], X=train_exog) # , X=train_exog
        logMessage("ARIMAX Model Summary")
        logMessage(arimax_model.summary())
        
        logMessage("ARIMAX Model Prediction ..")
        arimax_forecast = arimax_model.predict(fh, X=future_exog) #len(fh)
        y_pred_arimax = pd.DataFrame(arimax_forecast).applymap('{:,.2f}'.format)
        y_pred_arimax['month_num'] = [i.month for i in future_exog.index]
        y_pred_arimax['year_num'] = [i.year for i in future_exog.index]

        # Rename column to forecast_a
        y_pred_arimax.rename(columns={0:'forecast_a'}, inplace=True)
        
        #%%
        ##### XGBOOST MODEL #####
        # Get best parameter from database
        sql_xgb_model_param = """SELECT best_param_b 
                            FROM hse_analytics_param 
                            WHERE ir_type = 'ir monthly cumulative'
                            ORDER BY running_date DESC 
                            LIMIT 1 OFFSET 0"""
                        
        xgb_model_param = get_sql_data(sql_xgb_model_param, conn)
        xgb_model_param = xgb_model_param['best_param_b'][0]
        
        # Convert string to tuple
        xgb_model_param = ast.literal_eval(xgb_model_param)

        #Set parameters
        xgb_lags = xgb_model_param['window_length']
        xgb_objective = 'reg:squarederror'
        xgb_strategy = "recursive"
        
        # Create regressor object
        xgb_regressor = XGBRegressor(objective=xgb_objective)
        xgb_forecaster = make_reduction(xgb_regressor, window_length=xgb_lags, strategy=xgb_strategy)
        logMessage("Creating XGBoost Model ....")
        xgb_forecaster.fit(train_df, X=train_exog) #, X=train_exog

        # Create forecasting
        logMessage("XGBoost Model Prediction ...")
        xgb_forecast = xgb_forecaster.predict(fh, X=future_exog) #, X=future_exog
        y_pred_xgb = pd.DataFrame(xgb_forecast).applymap('{:,.2f}'.format)
        y_pred_xgb['month_num'] = [i.month for i in xgb_forecast.index]
        y_pred_xgb['year_num'] = [i.year for i in xgb_forecast.index]

        # Rename column to forecast_e
        y_pred_xgb.rename(columns={0:'forecast_b'}, inplace=True)
        
        #%%
        ##### RANDOM FOREST MODEL #####
        # Get best parameter from database
        sql_ranfor_model_param = """SELECT best_param_c 
                            FROM hse_analytics_param 
                            WHERE ir_type = 'ir monthly cumulative'
                            ORDER BY running_date DESC 
                            LIMIT 1 OFFSET 0"""
                        
        ranfor_model_param = get_sql_data(sql_ranfor_model_param, conn)
        ranfor_model_param = ranfor_model_param['best_param_c'][0]
        
        # Convert string to tuple
        ranfor_model_param = ast.literal_eval(ranfor_model_param)
        
        #Set Parameter
        ranfor_n_estimators = ranfor_model_param['estimator__n_estimators']
        random_state = 0
        ranfor_criterion = "squared_error"
        ranfor_lags = ranfor_model_param['window_length']
        ranfor_strategy = "recursive"

        # create regressor object
        ranfor_regressor = RandomForestRegressor(n_estimators = ranfor_n_estimators, random_state=random_state, criterion=ranfor_criterion)
        ranfor_forecaster = make_reduction(ranfor_regressor, window_length= ranfor_lags, strategy=ranfor_strategy)
        logMessage("Creating Random Forest Model ...")
        ranfor_forecaster.fit(train_df, X=train_exog) #, X=train_exog

        # Create forecasting
        logMessage("Random Forest Model Prediction")
        ranfor_forecast = ranfor_forecaster.predict(fh, X=future_exog) #, X=future_exog
        y_pred_ranfor = pd.DataFrame(ranfor_forecast).applymap('{:,.2f}'.format)
        y_pred_ranfor['month_num'] = [i.month for i in ranfor_forecast.index]
        y_pred_ranfor['year_num'] = [i.year for i in ranfor_forecast.index]

        # Rename column to forecast_c
        y_pred_ranfor.rename(columns={0:'forecast_c'}, inplace=True)    


        #%%
        ##### LINEAR REGRESSION MODEL #####
        # Get best parameter from database
        sql_linreg_model_param = """SELECT best_param_d
                                FROM hse_analytics_param 
                                WHERE ir_type = 'ir monthly cumulative' 
                                ORDER BY running_date DESC 
                                LIMIT 1 OFFSET 0"""
                    
        linreg_model_param = get_sql_data(sql_linreg_model_param, conn)
        linreg_model_param = linreg_model_param['best_param_d'][0]
       
        # Convert string to tuple
        linreg_model_param = ast.literal_eval(linreg_model_param)
        
        #Set parameter
        linreg_normalize = True
        linreg_lags = linreg_model_param['window_length']
        linreg_strategy = "recursive"

        # create regressor object
        linreg_regressor = LinearRegression(normalize=linreg_normalize)
        linreg_forecaster = make_reduction(linreg_regressor, window_length=linreg_lags, strategy=linreg_strategy)
        logMessage("Creating Linear Regression Model ...")
        linreg_forecaster.fit(train_df, X=train_exog) #, X=train_exog
        
        logMessage("Linear Regression Model Prediction ...")
        linreg_forecast = linreg_forecaster.predict(fh, X=future_exog) #, X=future_exog
        y_pred_linreg = pd.DataFrame(linreg_forecast).applymap('{:,.2f}'.format)
        y_pred_linreg['month_num'] = [i.month for i in linreg_forecast.index]
        y_pred_linreg['year_num'] = [i.year for i in linreg_forecast.index]

        # Rename column to forecast_d
        y_pred_linreg.rename(columns={0:'forecast_d'}, inplace=True)

        #%%
        ##### POLYNOMIAL REGRESSION DEGREE=2 MODEL #####
        # Get best parameter from database
        sql_poly2_model_param = """SELECT best_param_e
                                FROM hse_analytics_param 
                                WHERE ir_type = 'ir monthly cumulative' 
                                ORDER BY running_date DESC 
                                LIMIT 1 OFFSET 0"""
                    
        poly2_model_param = get_sql_data(sql_poly2_model_param, conn)
        poly2_model_param = poly2_model_param['best_param_e'][0]
       
        # Convert string to tuple
        poly2_model_param = ast.literal_eval(poly2_model_param)
        
        #Set parameter
        poly2_regularization = None
        poly2_interactions= False
        poly2_lags = poly2_model_param['window_length']
        poly2_strategy = "recursive"

        # Create regressor object
        poly2_regressor = PolynomRegressor(deg=2, regularization=poly2_regularization, interactions=poly2_interactions)
        poly2_forecaster = make_reduction(poly2_regressor, window_length=poly2_lags, strategy=poly2_strategy) #WL=0.9 (degree 2), WL=0.7 (degree 3)
        logMessage("Creating Polynomial Regression Orde 2 Model ...")
        poly2_forecaster.fit(train_df, X=train_exog) #, X=train_exog
        
        logMessage("Polynomial Regression Orde 2 Model Prediction ...")
        poly2_forecast = poly2_forecaster.predict(fh, X=future_exog) #, X=future_exog
        y_pred_poly2 = pd.DataFrame(poly2_forecast).applymap('{:,.2f}'.format)
        y_pred_poly2['month_num'] = [i.month for i in poly2_forecast.index]
        y_pred_poly2['year_num'] = [i.year for i in poly2_forecast.index]

        # Rename column to forecast_e
        y_pred_poly2.rename(columns={0:'forecast_e'}, inplace=True)

        #%%
        ##### POLYNOMIAL REGRESSION DEGREE=3 MODEL #####
        # Get best parameter from database
        sql_poly3_model_param = """SELECT best_param_f
                                FROM hse_analytics_param 
                                WHERE ir_type = 'ir monthly cumulative' 
                                ORDER BY running_date DESC 
                                LIMIT 1 OFFSET 0"""
                    
        poly3_model_param = get_sql_data(sql_poly3_model_param, conn)
        poly3_model_param = poly3_model_param['best_param_f'][0]
       
        # Convert string to tuple
        poly3_model_param = ast.literal_eval(poly3_model_param)
        
        #Set parameter
        poly3_regularization = None
        poly3_interactions= False
        poly3_lags = poly3_model_param['window_length']
        poly3_strategy = "recursive"

        # Create regressor object
        poly3_regressor = PolynomRegressor(deg=3, regularization=poly3_regularization, interactions=poly3_interactions)
        poly3_forecaster = make_reduction(poly3_regressor, window_length=poly3_lags, strategy=poly3_strategy) #WL=0.9 (degree 2), WL=0.7 (degree 3)
        logMessage("Creating Polynomial Regression Orde 3 Model ...")
        poly3_forecaster.fit(train_df, X=train_exog) #, X=train_exog
        
        logMessage("Polynomial Regression Orde 3 Model Prediction ...")
        poly3_forecast = poly3_forecaster.predict(fh, X=future_exog) #, X=future_exog
        y_pred_poly3 = pd.DataFrame(poly3_forecast).applymap('{:,.2f}'.format)
        y_pred_poly3['month_num'] = [i.month for i in poly3_forecast.index]
        y_pred_poly3['year_num'] = [i.year for i in poly3_forecast.index]

        # Rename column to forecast_f
        y_pred_poly3.rename(columns={0:'forecast_f'}, inplace=True)
        
        logMessage("Creating all model prediction result data frame ...")
        y_all_pred = pd.concat([y_pred_arimax[['forecast_a']], 
                                y_pred_xgb[['forecast_b']], 
                                y_pred_ranfor[['forecast_c']], 
                                y_pred_linreg[['forecast_d']], 
                                y_pred_poly2[['forecast_e']], 
                                y_pred_poly3[['forecast_f']]
                            ], axis=1)
        
        #%%
        # Plot prediction
        plt.figure(figsize=(20,8))
        plt.plot(train_df.to_timestamp(), label='train')
        plt.plot(arimax_forecast.to_timestamp(), label='pred', marker="o", markersize=4)
        plt.plot(xgb_forecast.to_timestamp(), label='pred', marker="o", markersize=4)
        plt.plot(ranfor_forecast.to_timestamp(), label='pred', marker="o", markersize=4)
        plt.plot(linreg_forecast.to_timestamp(), label='pred', marker="o", markersize=4)
        plt.plot(poly2_forecast.to_timestamp(), label='pred', marker="o", markersize=4)
        plt.plot(poly3_forecast.to_timestamp(), label='pred', marker="o", markersize=4)
        plt.ylabel("Incident Rate")
        plt.xlabel("Datestamp")
        plt.legend(loc='best')
        plt.title(' Incident Rate Monthly Cumulative (Arimax Model) with Exogenous Variables')
        #plt.savefig("Incident Rate Monthly Cumulative (Arimax Model) with Exogenous" + ".jpg")
        #plt.show()
        plt.close()

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
        month_num = index.month
        forecast_a, forecast_b, forecast_c, forecast_d, forecast_e, forecast_f = row[0], row[1], row[2], row[3], row[4], row[5]
        
        #sql = f'UPDATE trir_monthly_test SET forecast_a = {} WHERE year_num = {} AND month_num = {}'.format(forecast, year_num, month_num)
        updated_rows = update_value(conn, forecast_a, forecast_b, forecast_c, forecast_d, forecast_e, forecast_f, year_num, month_num)
        total_updated_rows = total_updated_rows + updated_rows 
        
    return total_updated_rows

def update_value(conn, forecast_a, forecast_b, forecast_c, 
                        forecast_d, forecast_e, forecast_f, year_num, month_num):
    
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
                WHERE year_num = %s
                AND month_num = %s"""
    #conn = None
    updated_rows = 0
    try:
        # create a new cursor
        cur = conn.cursor()
        # execute the UPDATE  statement
        cur.execute(sql, (forecast_a, forecast_b, forecast_c, forecast_d, forecast_e, forecast_f, 
                          date_now, updated_by, year_num, month_num))
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