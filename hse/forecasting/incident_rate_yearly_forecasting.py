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
import pmdarima as pm
import psycopg2
import seaborn as sns

plt.style.use('fivethirtyeight')
from datetime import datetime
from tokenize import Ignore
from tracemalloc import start
from pmdarima.arima import auto_arima
from pmdarima.arima.auto import auto_arima
from pmdarima import model_selection
from connection import config, retrieve_data, create_db_connection, get_sql_data
from utils import configLogging, logMessage, ad_test

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# %%
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
    #Configure logging
    configLogging("yearly_incident_rate.log")
    
    # Connect to database
    # Exit program if not connected to database
    logMessage("Connecting to database ...")
    conn = create_db_connection(section='postgresql_ml_hse_skk')
    if conn == None:
        exit()
       
    #Load Data from Database
    query_1 = open(os.path.join('hse\\sql', 'query_yearly.sql'), mode="rt").read()
    data = get_sql_data(query_1, conn)
    #data = retrieve_data(query_1)
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

    #stationarity_check(df.to_timestamp())
    #decomposition_plot(df.to_timestamp())
    #plot_acf_pacf(df.to_timestamp())

    #from chart_studio.plotly import plot_mpl
    #from statsmodels.tsa.seasonal import seasonal_decompose
    #result = seasonal_decompose(df.trir.values, model="multiplicative", period=5)
    #fig = result.plot()
    #plt.show()

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

    #Load Data from Database (create future exogenous)
    query_exog = open(os.path.join('hse\\sql', 'query_yearly_future.sql'), mode="rt").read()
    #query_2 = open("query_yearly_future.sql", mode="rt").read()
    future_exog = get_sql_data(query_exog, conn)
    #future_exog = retrieve_data(query_2)
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

    from sktime.forecasting.base import ForecastingHorizon
    time_predict = pd.period_range('2023', periods=2, freq='Y')
    fh = ForecastingHorizon(time_predict, is_relative=False)

    #%%
    ##### ARIMAX MODEL #####
    from pmdarima.arima.utils import ndiffs, nsdiffs
    from sklearn.metrics import mean_squared_error
    import statsmodels.api as sm

    #Set parameters
    arimax_trace = True
    arimax_error_action = "ignore"
    arimax_suppress_warnings = True

    # Create ARIMAX Model
    arimax_model = auto_arima(train_df, exogenous=train_exog, d=1, trace=arimax_trace, error_action=arimax_error_action, suppress_warnings=arimax_suppress_warnings)
    logMessage("Creating ARIMAX Model ...")
    arimax_model.fit(train_df, exogenous=train_exog)
    logMessage("ARIMAX Model Summary")
    logMessage(arimax_model.summary())
    
    logMessage("ARIMAX Model Prediction ..")
    arimax_forecast = arimax_model.predict(len(fh), exogenous=future_exog)
    y_pred_arimax = pd.DataFrame(arimax_forecast).applymap('{:,.2f}'.format)
    y_pred_arimax['year_num'] = [i.year for i in arimax_forecast.index]
    #Rename colum 0
    y_pred_arimax.rename(columns={0:'forecast_a'}, inplace=True)


    ##### XGBOOST MODEL #####
    from xgboost import XGBRegressor
    from sktime.forecasting.compose import make_reduction

    #Set Parameters
    xgb_objective = 'reg:squarederror'
    xgb_lags = 6

    # Create regressor object
    xgb_regressor = XGBRegressor(objective='reg:squarederror')
    xgb_forecaster = make_reduction(xgb_regressor, window_length=6, strategy="recursive")
    
    logMessage("Creating XGBoost Model ....")
    xgb_forecaster.fit(train_df, X=train_exog) #, X=train_exog
    
    logMessage("XGBoost Model Prediction ...")
    xgb_forecast = xgb_forecaster.predict(fh, X=future_exog) #, X=future_exog
    y_pred_xgb = pd.DataFrame(xgb_forecast).applymap('{:,.2f}'.format)
    y_pred_xgb['year_num'] = [i.year for i in xgb_forecast.index]
    #Rename colum 0
    y_pred_xgb.rename(columns={0:'forecast_b'}, inplace=True)


    ##### RANDOM FOREST MODEL #####
    from sklearn.ensemble import RandomForestRegressor

    #Set parameters
    ranfor_n_estimators = 100
    ranfor_random_state = 0
    ranfor_criterion = "squared_error"
    ranfor_lags = 0.8
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
    from sklearn.linear_model import LinearRegression

    #Set parameters
    linreg_normalize = True
    linreg_lags = 0.8
    linreg_strategy = "recursive"

    # Create regressor object
    linreg_regressor = LinearRegression(normalize=linreg_normalize)
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
    from polyfit import PolynomRegressor, Constraints

    #Set parameters
    poly2_regularization = None
    poly2_interactions = False
    poly2_lags = 0.7
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
    #Set parameters
    poly3_regularization = None
    poly3_interactions = False
    poly3_lags = 0.6
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