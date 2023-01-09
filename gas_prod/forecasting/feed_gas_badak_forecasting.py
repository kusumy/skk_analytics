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
import matplotlib.pyplot as plt
import matplotlib as mpl
from connection import config, retrieve_data, create_db_connection, get_sql_data
from utils import *

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

#%%
def main():
    # Configure logging
    #configLogging("feed_gas_badak_forecasting.log")
    logMessage("Forecasting Feed Gas PT Badak ...")
    
    # Connect to database
    # Exit program if not connected to database
    logMessage("Connecting to database ...")
    conn = create_db_connection(section='postgresql_ml_lng_skk')
    if conn == None:
        exit()
    
    #Load data from database
    query_data = os.path.join('gas_prod/sql','feed_gas_badak_data_query.sql')
    query_1 = open(query_data, mode="rt").read()
    data = get_sql_data(query_1, conn)
    data['date'] = pd.DatetimeIndex(data['date'], freq='D')
    data = data.reset_index()

    #%%
    data_null_cleaning = data[['date', 'feed_gas']].copy()
    data_null_cleaning['feed_gas_copy'] = data[['feed_gas']].copy()
    ds_null_cleaning = 'date'
    data_null_cleaning = data_null_cleaning.set_index(ds_null_cleaning)
    s = validate_series(data_null_cleaning)  
    
    threshold_ad = ThresholdAD(data_null_cleaning['feed_gas_copy'].isnull())
    anomalies =  threshold_ad.detect(s)
    anomalies = anomalies.drop('feed_gas', axis=1)

    # Create anomaly detection model
    #threshold_ad = ThresholdAD(high=high_limit2, low=low_limit1)
    #anomalies =  threshold_ad.detect(s)

    # Copy data frame of anomalies
    copy_anomalies =  anomalies.copy()
    # Rename columns
    copy_anomalies.rename(columns={'feed_gas_copy':'anomaly'}, inplace=True)
    # Merge original dataframe with anomalies
    new_s = pd.concat([s, copy_anomalies], axis=1)

    # Get only anomalies data
    anomalies_data = new_s[new_s['anomaly'].isnull()]

    # Create anomaly detection model
    #threshold_ad = ThresholdAD(high=high_limit2, low=low_limit1)
    #anomalies =  threshold_ad.detect(s)

    # Copy data frame of anomalies
    copy_anomalies =  anomalies.copy()
    # Rename columns
    copy_anomalies.rename(columns={'feed_gas_copy':'anomaly'}, inplace=True)
    # Merge original dataframe with anomalies
    new_s = pd.concat([s, copy_anomalies], axis=1)

    # Get only anomalies data
    #anomalies_data = new_s[new_s['anomaly'].isnull()]
    
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
        mean_month=new_s['feed_gas'].reset_index().query(sql).mean(skipna = True).values[0]
        
        # update value at specific location
        new_s.at[index,'feed_gas'] = mean_month
        
        print(index), print(sql), print(mean_month)
    
    #%%
    #prepare data
    data_cleaned = new_s[['feed_gas']].copy()
    data_cleaned = data_cleaned.reset_index()

    ds_cleaned = 'date'
    y_cleaned = 'feed_gas'
    df_cleaned = data_cleaned[[ds_cleaned, y_cleaned]]
    df_cleaned = df_cleaned.set_index(ds_cleaned)
    df_cleaned.index = pd.DatetimeIndex(df_cleaned.index, freq='D')

    #%%
    # Smooth time series signal using polynomial smoothing
    from tsmoothie.smoother import PolynomialSmoother,  LowessSmoother

    #smoother = PolynomialSmoother(degree=1, copy=True)
    smoother = LowessSmoother(smooth_fraction=0.01, iterations=1)
    smoother.smooth(df_cleaned)

    # generate intervals
    low, up = smoother.get_intervals('prediction_interval')

    # plotting for illustration
    # plt.style.use('fivethirtyeight')
    # fig1, ax = plt.subplots(figsize=(18,7))
    # ax.plot(df_cleaned.index, df_cleaned[y_cleaned], label='original')
    # ax.plot(df_cleaned.index, smoother.smooth_data[0], linewidth=3, color='blue', label='smoothed')
    # ax.fill_between(df_cleaned.index, low[0], up[0], alpha=0.3)
    # ax.set_ylabel("Feed Gas")
    # ax.set_xlabel("Datestamp")
    # ax.legend(loc='best')
    # title = ("PT Badak Feed Gas Production")
    # ax.set_title(title)
    # plt.savefig("ptbadak_smoothed.jpg")
    # plt.show()
    # plt.close()

    #%%
    # Copy data from original
    df_smoothed = df_cleaned.copy()
    # Replace original with smoothed data
    df_smoothed[y_cleaned] = smoother.smooth_data[0]

    #%%
    #import chart_studio.plotly
    #import cufflinks as cf

    #from plotly.offline import iplot
    #cf.go_offline()
    #cf.set_config_file(offline = False, world_readable = True)

    #%%
    #df_smoothed.iplot(title="Feed Gas PT Badak")

    #%%
    #stationarity_check(df_smoothed)

    #%%
    #decomposition_plot(df_smoothed)

    #%%
    #plot_acf_pacf(df_smoothed)

    #%%
    #from chart_studio.plotly import plot_mpl
    #from statsmodels.tsa.seasonal import seasonal_decompose
    #result = seasonal_decompose(df_smoothed.feed_gas.values, model="multiplicative", period=365)
    #fig = result.plot()
    #plt.show()
    #plt.close()

    #%%
    #Ad Fuller Test
    ad_test(df_smoothed['feed_gas'])

    #%%
    #Select target column after smoothing data
    train_df = df_smoothed['feed_gas']

    #%%
    # create features from date
    df_cleaned['month'] = [i.month for i in df_cleaned.index]
    df_cleaned['day'] = [i.day for i in df_cleaned.index]
    train_exog = df_cleaned.iloc[:,1:]

    #%%
    query_exog = os.path.join('gas_prod/sql','feed_gas_badak_exog_query.sql')
    query_2 = open(query_exog, mode="rt").read()
    data_exog = get_sql_data(query_2, conn)
    data_exog['date'] = pd.DatetimeIndex(data_exog['date'], freq='D')
    data_exog.sort_index(inplace=True)
    data_exog = data_exog.reset_index()
            
    ds_exog = 'date'
    x_exog = 'feed_gas'
    future_exog = data_exog[[ds_exog, x_exog]]
    future_exog = future_exog.set_index(ds_exog)
    future_exog['month'] = [i.month for i in future_exog.index]
    future_exog['day'] = [i.day for i in future_exog.index]
    future_exog.drop(['feed_gas'], axis=1, inplace=True)

    #%%
    from sktime.forecasting.base import ForecastingHorizon
    time_predict = pd.period_range('2022-11-11', periods=51, freq='D')
    fh = ForecastingHorizon(future_exog.index, is_relative=False)

    # %%
    try:
        ##### FORECASTING #####

        ##### ARIMAX MODEL #####
        from sktime.forecasting.arima import ARIMA
        from pmdarima.arima.utils import ndiffs, nsdiffs
        from sklearn.metrics import mean_squared_error
        import statsmodels.api as sm

        #Set parameters
        arimax_differencing = 1
        arimax_trace = True
        arimax_error_action = "ignore"
        arimax_suppress_warnings = True

        #Create ARIMAX Model
        #ARIMA(4,1,5)
        #arimax_model = auto_arima(train_df, exogenous=future_exog, d=arimax_differencing, trace=arimax_trace, error_action=arimax_error_action, suppress_warnings=arimax_suppress_warnings)
        logMessage("Creating ARIMAX Model ...")
        arimax_model = ARIMA(order=(3, 1, 2), suppress_warnings=arimax_suppress_warnings)
        arimax_model.fit(train_df, X=train_exog)
        logMessage("ARIMAX Model Summary")
        logMessage(arimax_model.summary())
        
        logMessage("ARIMAX Model Prediction ..")
        arimax_forecast = arimax_model.predict(fh, X=future_exog) #if pmdarima using len(fh)
        y_pred_arimax = pd.DataFrame(arimax_forecast).applymap('{:.2f}'.format)
        y_pred_arimax['day_num'] = [i.day for i in arimax_forecast.index]
        y_pred_arimax['month_num'] = [i.month for i in arimax_forecast.index]
        y_pred_arimax['year_num'] = [i.year for i in arimax_forecast.index]
        y_pred_arimax['date'] = y_pred_arimax['year_num'].astype(str) + '-' + y_pred_arimax['month_num'].astype(str) + '-' + y_pred_arimax['day_num'].astype(str)
        y_pred_arimax['date'] = pd.DatetimeIndex(y_pred_arimax['date'], freq='D')
        #Rename colum 0
        y_pred_arimax.rename(columns={0:'forecast_a'}, inplace=True)


        ##### SARIMAX MODEL #####

        #Set parameters
        sarimax_differencing = 1
        sarimax_seasonal_differencing = 1
        sarimax_seasonal = True
        sarimax_m = 12
        sarimax_trace = True
        sarimax_error_action = "ignore"
        sarimax_suppress_warnings = True

        # Create SARIMAX Model
        #sarimax_model = auto_arima(train_df, exogenous=future_exog, d=sarimax_differencing, D=sarimax_seasonal_differencing, seasonal=sarimax_seasonal, 
        #                            m=sarimax_m, trace=sarimax_trace, error_action=sarimax_error_action, suppress_warnings=sarimax_suppress_warnings)
        logMessage("Creating SARIMAX Model ...")
        sarimax_model = ARIMA(order=(3, 1, 2), seasonal_order=(1, 1, 1, 12), suppress_warnings=sarimax_suppress_warnings)
        sarimax_model.fit(train_df, X=train_exog)
        logMessage("SARIMAX Model Summary")
        logMessage(sarimax_model.summary())
        
        logMessage("SARIMAX Model Prediction ..")
        sarimax_forecast = sarimax_model.predict(fh, X=future_exog)
        y_pred_sarimax = pd.DataFrame(sarimax_forecast).applymap('{:.2f}'.format)
        y_pred_sarimax['day_num'] = [i.day for i in sarimax_forecast.index]
        y_pred_sarimax['month_num'] = [i.month for i in sarimax_forecast.index]
        y_pred_sarimax['year_num'] = [i.year for i in sarimax_forecast.index]
        y_pred_sarimax['date'] = y_pred_sarimax['year_num'].astype(str) + '-' + y_pred_sarimax['month_num'].astype(str) + '-' + y_pred_sarimax['day_num'].astype(str)
        y_pred_sarimax['date'] = pd.DatetimeIndex(y_pred_sarimax['date'], freq='D')
        #Rename colum 0
        y_pred_sarimax.rename(columns={0:'forecast_b'}, inplace=True)


        ##### PROPHET MODEL #####
        from sktime.forecasting.fbprophet import Prophet
        from sktime.forecasting.compose import make_reduction

        #Set parameters
        prophet_seasonality_mode = 'additive'
        prophet_n_changepoints = 9
        prophet_seasonality_prior_scale = 10
        prophet_changepoint_prior_scale = 0.5
        prophet_holidays_prior_scale = 2
        prophet_daily_seasonality = True
        prophet_weekly_seasonality = False
        prophet_yearly_seasonality = True

        #Create regressor object
        logMessage("Creating Prophet Model ....")
        prophet_forecaster = Prophet(
                            seasonality_mode=prophet_seasonality_mode,
                            n_changepoints=prophet_n_changepoints,
                            seasonality_prior_scale=prophet_seasonality_prior_scale, #Flexibility of the seasonality (0.01,10)
                            changepoint_prior_scale=prophet_changepoint_prior_scale, #Flexibility of the trend (0.001,0.5)
                            holidays_prior_scale=prophet_holidays_prior_scale, #Flexibility of the holiday effects (0.01,10)
                            #changepoint_range=0.8, #proportion of the history in which the trend is allowed to change
                            daily_seasonality=prophet_daily_seasonality,
                            weekly_seasonality=prophet_weekly_seasonality,
                            yearly_seasonality=prophet_yearly_seasonality)
        prophet_forecaster.fit(train_df, train_exog) #, X_train

        logMessage("Prophet Model Prediction ...")
        future_exog.sort_index(inplace=True)
        prophet_forecast = prophet_forecaster.predict(fh, X=future_exog) #, X=X_test
        y_pred_prophet = pd.DataFrame(prophet_forecast).applymap('{:.2f}'.format)
        y_pred_prophet['day_num'] = [i.day for i in prophet_forecast.index]
        y_pred_prophet['month_num'] = [i.month for i in prophet_forecast.index]
        y_pred_prophet['year_num'] = [i.year for i in prophet_forecast.index]
        y_pred_prophet['date'] = y_pred_prophet['year_num'].astype(str) + '-' + y_pred_prophet['month_num'].astype(str) + '-' + y_pred_prophet['day_num'].astype(str)
        y_pred_prophet['date'] = pd.DatetimeIndex(y_pred_prophet['date'], freq='D')
        #Rename colum 0
        y_pred_prophet.rename(columns={0:'forecast_c'}, inplace=True)


        ##### RANDOM FOREST MODEL #####
        from sklearn.ensemble import RandomForestRegressor

        #Set parameters
        ranfor_lags = 7
        ranfor_n_estimators = 80
        ranfor_random_state = 0
        ranfor_criterion = "squared_error"
        ranfor_strategy = "recursive"

        # create regressor object
        logMessage("Creating Random Forest Model ...")
        ranfor_regressor = RandomForestRegressor(n_estimators=ranfor_n_estimators, random_state=ranfor_random_state, criterion=ranfor_criterion)
        ranfor_forecaster = make_reduction(ranfor_regressor, window_length=ranfor_lags, strategy=ranfor_strategy)
        ranfor_forecaster.fit(train_df, train_exog) #, X_train

        logMessage("Random Forest Model Prediction")
        future_exog.sort_index(inplace=True)
        ranfor_forecast = ranfor_forecaster.predict(fh, X=future_exog) #, X=X_test
        y_pred_ranfor = pd.DataFrame(ranfor_forecast).applymap('{:.2f}'.format)
        y_pred_ranfor['day_num'] = [i.day for i in ranfor_forecast.index]
        y_pred_ranfor['month_num'] = [i.month for i in ranfor_forecast.index]
        y_pred_ranfor['year_num'] = [i.year for i in ranfor_forecast.index]
        y_pred_ranfor['date'] = y_pred_ranfor['year_num'].astype(str) + '-' + y_pred_ranfor['month_num'].astype(str) + '-' + y_pred_ranfor['day_num'].astype(str)
        y_pred_ranfor['date'] = pd.DatetimeIndex(y_pred_ranfor['date'], freq='D')
        #Rename colum 0
        y_pred_ranfor.rename(columns={0:'forecast_d'}, inplace=True)


        ##### XGBOOST MODEL #####
        from xgboost import XGBRegressor

        #Set parameters
        xgb_lags = 14
        xgb_objective = 'reg:squarederror'
        xgb_strategy = "recursive"

        #Create regressor object
        logMessage("Creating XGBoost Model ...")
        xgb_regressor = XGBRegressor(objective=xgb_objective)
        xgb_forecaster = make_reduction(xgb_regressor, window_length=xgb_lags, strategy=xgb_strategy)
        xgb_forecaster.fit(train_df, train_exog) #, X_train
        
        logMessage("XGBoost Model Prediction ...")
        future_exog.sort_index(inplace=True)
        xgb_forecast = xgb_forecaster.predict(fh, X=future_exog) #, X=X_test
        y_pred_xgb = pd.DataFrame(xgb_forecast).applymap('{:.2f}'.format)
        y_pred_xgb['day_num'] = [i.day for i in xgb_forecast.index]
        y_pred_xgb['month_num'] = [i.month for i in xgb_forecast.index]
        y_pred_xgb['year_num'] = [i.year for i in xgb_forecast.index]
        y_pred_xgb['date'] = y_pred_xgb['year_num'].astype(str) + '-' + y_pred_xgb['month_num'].astype(str) + '-' + y_pred_xgb['day_num'].astype(str)
        y_pred_xgb['date'] = pd.DatetimeIndex(y_pred_xgb['date'], freq='D')
        #Rename colum 0
        y_pred_xgb.rename(columns={0:'forecast_e'}, inplace=True)


        #### LINEAR REGRESSION MODEL #####
        from sklearn.linear_model import LinearRegression

        #Set parameters
        linreg_lags = 0.96
        linreg_normalize = True
        linreg_strategy = "recursive"

        # Create regressor object
        logMessage("Creating Linear Regression Model ...")
        linreg_regressor = LinearRegression(normalize=linreg_normalize)
        linreg_forecaster = make_reduction(linreg_regressor, window_length=linreg_lags, strategy=linreg_strategy)
        linreg_forecaster.fit(train_df, train_exog)

        logMessage("Linear Regression Model Prediction ...")
        future_exog.sort_index(inplace=True)
        linreg_forecast = linreg_forecaster.predict(fh, X=future_exog) #, X=X_test
        y_pred_linreg = pd.DataFrame(linreg_forecast).applymap('{:.2f}'.format)
        y_pred_linreg['day_num'] = [i.day for i in linreg_forecast.index]
        y_pred_linreg['month_num'] = [i.month for i in linreg_forecast.index]
        y_pred_linreg['year_num'] = [i.year for i in linreg_forecast.index]
        y_pred_linreg['date'] = y_pred_linreg['year_num'].astype(str) + '-' + y_pred_linreg['month_num'].astype(str) + '-' + y_pred_linreg['day_num'].astype(str)
        y_pred_linreg['date'] = pd.DatetimeIndex(y_pred_linreg['date'], freq='D')
        #Rename colum 0
        y_pred_linreg.rename(columns={0:'forecast_f'}, inplace=True)


        ##### POLYNOMIAL REGRESSION DEGREE=2 MODEL #####
        from polyfit import PolynomRegressor, Constraints

        #Set parameters
        poly2_regularization = None
        poly2_interactions = False
        poly_lags = 24
        poly2_strategy = "recursive"

        # Create regressor object
        logMessage("Creating Polynomial Regression Orde 2 Model ...")
        poly2_regressor = PolynomRegressor(deg=2, regularization=poly2_regularization, interactions=poly2_interactions)
        poly2_forecaster = make_reduction(poly2_regressor, window_length=poly_lags, strategy=poly2_strategy) #WL=0.9 (degree 2), WL=0.7 (degree 3)
        poly2_forecaster.fit(train_df, train_exog) #, X=X_train

        logMessage("Polynomial Regression Orde 2 Model Prediction ...")
        future_exog.sort_index(inplace=True)
        poly2_forecast = poly2_forecaster.predict(fh, X=future_exog) #, X=X_test
        y_pred_poly2 = pd.DataFrame(poly2_forecast).applymap('{:.2f}'.format)
        y_pred_poly2['day_num'] = [i.day for i in poly2_forecast.index]
        y_pred_poly2['month_num'] = [i.month for i in poly2_forecast.index]
        y_pred_poly2['year_num'] = [i.year for i in poly2_forecast.index]
        y_pred_poly2['date'] = y_pred_poly2['year_num'].astype(str) + '-' + y_pred_poly2['month_num'].astype(str) + '-' + y_pred_poly2['day_num'].astype(str)
        y_pred_poly2['date'] = pd.DatetimeIndex(y_pred_poly2['date'], freq='D')
        #Rename colum 0
        y_pred_poly2.rename(columns={0:'forecast_g'}, inplace=True)


        ##### POLYNOMIAL REGRESSION DEGREE=3 MODEL #####
        from polyfit import PolynomRegressor, Constraints

        #Set parameters
        poly3_regularization = None
        poly3_interactions = False
        poly3_lags = 27
        poly3_strategy = "recursive"

        # Create regressor object
        logMessage("Creating Polynomial Regression Orde 3 Model ...")
        poly3_regressor = PolynomRegressor(deg=3, regularization=poly3_regularization, interactions=poly3_interactions)
        poly3_forecaster = make_reduction(poly3_regressor, window_length=poly3_lags, strategy=poly3_strategy) #WL=0.9 (degree 2), WL=0.7 (degree 3)
        poly3_forecaster.fit(train_df, train_exog) #, X=X_train

        logMessage("Polynomial Regression Orde 3 Model Prediction ...")
        future_exog.sort_index(inplace=True)
        poly3_forecast = poly3_forecaster.predict(fh, X=future_exog) #, X=X_test
        y_pred_poly3 = pd.DataFrame(poly3_forecast).applymap('{:.2f}'.format)
        y_pred_poly3['day_num'] = [i.day for i in poly3_forecast.index]
        y_pred_poly3['month_num'] = [i.month for i in poly3_forecast.index]
        y_pred_poly3['year_num'] = [i.year for i in poly3_forecast.index]
        y_pred_poly3['date'] = y_pred_poly3['year_num'].astype(str) + '-' + y_pred_poly3['month_num'].astype(str) + '-' + y_pred_poly3['day_num'].astype(str)
        y_pred_poly3['date'] = pd.DatetimeIndex(y_pred_poly3['date'], freq='D')
        #Rename colum 0
        y_pred_poly3.rename(columns={0:'forecast_h'}, inplace=True)

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

        #%%
        # Plot prediction
        fig, ax = plt.subplots(figsize=(20,8))
        ax.plot(train_df, label='train')
        ax.plot(arimax_forecast, label='arimax_pred')
        ax.plot(sarimax_forecast, label='sarimax_pred')
        ax.plot(prophet_forecast, label='prophet_pred')
        ax.plot(ranfor_forecast, label='ranfor_pred')
        ax.plot(xgb_forecast, label='xgb_pred')
        ax.plot(linreg_forecast, label='linreg_pred')
        ax.plot(poly2_forecast, label='poly2_pred')
        ax.plot(poly3_forecast, label='poly3_pred')
        title = 'Feed Gas PT Badak with Smoothing Data & Exogenous Variables (Day & Month)'
        ax.set_title(title)
        ax.set_ylabel("Feed Gas")
        ax.set_xlabel("Datestamp")
        ax.legend(loc='best')
        #plt.savefig("Feed Gas PT Badak with Smoothing Data & Exogenous Variables (Day & Month)" + ".jpg")
        plt.close()
        
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
    sql = """ UPDATE lng_feed_gas_daily
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
                AND lng_plant = 'PT Badak'"""
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

# if __name__ == "__main__":
#     #main(sys.argv[1], sys.argv[2], sys.argv[3])
#     main()
