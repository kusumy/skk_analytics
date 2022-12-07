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
from pmdarima.arima.auto import auto_arima
import matplotlib as mpl
import matplotlib.pyplot as plt
from connection import config, retrieve_data, create_db_connection, get_sql_data
from utils import configLogging, logMessage, ad_test, stationarity_check, decomposition_plot, plot_acf_pacf

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.style.use('fivethirtyeight')

import pmdarima as pm
from pmdarima import model_selection 
from pmdarima.arima import auto_arima

# %%
def main():
    # Configure logging
    #configLogging("lng_prod_badak_forecasting.log")
    logMessage("Forecasting LNG Production PT Badak ...")
    
    # Connect to database
    # Exit program if not connected to database
    logMessage("Connecting to database ...")
    conn = create_db_connection(section='postgresql_ml_lng_skk')
    if conn == None:
        exit()
    
    query_data = os.path.join('gas_prod/sql','lng_prod_badak_data_query.sql')
    query_1 = open(query_data, mode="rt").read()
    data = get_sql_data(query_1, conn)
    data['date'] = pd.DatetimeIndex(data['date'], freq='D')
    data = data.reset_index()
    
    ds = 'date'
    y = 'lng_production' 

    df = data[[ds,y]]
    df = df.set_index(ds)
    df.index = pd.DatetimeIndex(df.index, freq='D')

    #Create column target
    train_df = df['lng_production']

    #%%
    #stationarity_check(train_df)

    #%%
    #decomposition_plot(train_df)

    #%%
    #plot_acf_pacf(train_df)

    #%%
    # from chart_studio.plotly import plot_mpl
    # from statsmodels.tsa.seasonal import seasonal_decompose
    # result = seasonal_decompose(df.lng_production.values, model="multiplicative", period=365)
    # fig = result.plot()
    # plt.close()

    #%%
    #Ad Fuller Test
    ad_test(train_df)

    #%%
    #CREATE EXOGENOUS VARIABLES
    df['fg_exog'] = data['fg_exog'].values
    df['month'] = [i.month for i in df.index]
    df['day'] = [i.day for i in df.index]
    train_exog = df.iloc[:,1:]

    from sktime.forecasting.base import ForecastingHorizon
    time_predict = pd.period_range('2022-11-11', periods=51, freq='D')

    #%%
    query_exog = os.path.join('gas_prod/sql','lng_prod_badak_exog_query.sql')
    query_2 = open(query_exog, mode="rt").read()
    data_exog = get_sql_data(query_2, conn)
    data_exog['date'] = pd.DatetimeIndex(data_exog['date'], freq='D')
    data_exog.sort_index(inplace=True)
    data_exog = data_exog.reset_index()

    ds_exog = 'date'
    y_exog = 'fg_exog'

    future_exog = data_exog[[ds_exog,y_exog]]
    future_exog = future_exog.set_index(ds_exog)
    future_exog.index = pd.DatetimeIndex(future_exog.index, freq='D')

    #Create exogenous date index
    future_exog['fg_exog'] = future_exog['fg_exog']
    future_exog['month'] = [i.month for i in future_exog.index]
    future_exog['day'] = [i.day for i in future_exog.index]

    #Create Forecasting Horizon
    fh = ForecastingHorizon(future_exog.index, is_relative=False)

    # plotting for illustration
    # fig1, ax = plt.subplots(figsize=(20,8))
    # ax.plot(train_df, label='train')
    # ax.set_ylabel("LNG Production")
    # ax.set_xlabel("Datestamp")
    # ax.legend(loc='best')
    # plt.close()

    try:
        ##### FORECASTING #####
        #%%
        ##### ARIMAX MODEL #####
        from pmdarima.arima.utils import ndiffs, nsdiffs
        import statsmodels.api as sm
        from sktime.forecasting.arima import AutoARIMA, ARIMA

        #Set parameters
        arimax_differencing = 1
        arimax_stationary = False
        arimax_trace = True
        arimax_error_action = "ignore"
        arimax_suppress_warnings = True

        # Create ARIMA Model
        #ARIMA(1,1,3)(0,0,0)[0]
        #arimax_model = AutoARIMA(d=arimax_differencing, stationary=arimax_stationary, trace=arimax_trace, error_action=arimax_error_action, suppress_warnings=arimax_suppress_warnings)
        arimax_model = ARIMA(order=(1, 1, 3), suppress_warnings=arimax_suppress_warnings)
        logMessage("Creating ARIMAX Model ...")
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


        #%%
        ##### SARIMAX MODEL #####

        #Set parameters
        sarimax_differencing = 1
        sarimax_seasonal_differencing = 1
        sarimax_sp = 12
        sarimax_stationary = False
        sarimax_seasonal = True
        sarimax_trace = True
        sarimax_error_action = "ignore"
        sarimax_suppress_warnings = True

        # Create SARIMA Model
        #ARIMA(2,1,3)(0,0,0)[12] 
        #sarimax_model = AutoARIMA(d=sarimax_differencing, D=sarimax_seasonal_differencing, sp=sarimax_sp, stationary=sarimax_stationary,
        #                seasonal=sarimax_seasonal, trace=sarimax_trace, error_action=sarimax_error_action, suppress_warnings=sarimax_suppress_warnings)
        sarimax_model = ARIMA(order=(1, 1, 3), seasonal_order=(2, 1, 0, 12), suppress_warnings=sarimax_suppress_warnings)
        logMessage("Creating SARIMAX Model ...")
        sarimax_model.fit(train_df, X=train_exog)
        logMessage("SARIMAX Model Summary")
        logMessage(arimax_model.summary())
        
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

        #%%
        ##### PROPHET MODEL #####
        from sktime.forecasting.fbprophet import Prophet
        from sktime.forecasting.compose import make_reduction

        #Set parameters
        prophet_seasonality_mode = 'additive'
        prophet_n_changepoints = 2
        prophet_seasonality_prior_scale = 0.05
        prophet_changepoint_prior_scale = 0.4
        prophet_holidays_prior_scale = 8
        prophet_daily_seasonality = 7
        prophet_weekly_seasonality = 1
        prophet_yearly_seasonality = 10

        #Create regressor object
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

        logMessage("Creating Prophet Model ....")
        prophet_forecaster.fit(train_df, train_exog) #, X_train
        logMessage("Prophet Model Prediction ...")
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
        ranfor_n_estimators = 100
        ranfor_random_state = 0
        ranfor_criterion =  "squared_error"
        ranfor_lags = 32
        ranfor_strategy = "recursive"

        #Create regressor object
        ranfor_regressor = RandomForestRegressor(n_estimators = ranfor_n_estimators, random_state=ranfor_random_state, criterion=ranfor_criterion)
        ranfor_forecaster = make_reduction(ranfor_regressor, window_length=ranfor_lags, strategy=ranfor_strategy) #30, nonexog=30
        logMessage("Creating Random Forest Model ...")
        
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


        ##### XGBOOST MODEL #####
        from xgboost import XGBRegressor

        #Set parameters
        xgb_objective = 'reg:squarederror'
        xgb_lags = 42
        xgb_strategy = "recursive"

        #Create regressor object
        xgb_regressor = XGBRegressor(objective=xgb_objective)
        xgb_forecaster = make_reduction(xgb_regressor, window_length=xgb_lags, strategy=xgb_strategy)
        logMessage("Creating XGBoost Model ...")
        xgb_forecaster.fit(train_df, train_exog) #, X_train
        
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


        ##### LINEAR REGRESSION MODEL #####
        from sklearn.linear_model import LinearRegression

        #Set parameters
        linreg_normalize = True
        linreg_lags = 44
        linreg_strategy = "recursive"

        # Create regressor object
        linreg_regressor = LinearRegression(normalize=linreg_normalize)
        linreg_forecaster = make_reduction(linreg_regressor, window_length=linreg_lags, strategy=linreg_strategy)
        logMessage("Creating Linear Regression Model ...")
        linreg_forecaster.fit(train_df, X=train_exog)
        
        logMessage("Linear Regression Model Prediction ...")
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
        poly2_lags = 7
        poly2_strategy = "recursive"

        # Create regressor object
        poly2_regressor = PolynomRegressor(deg=2, regularization=poly2_regularization, interactions=poly2_interactions)
        poly2_forecaster = make_reduction(poly2_regressor, window_length=poly2_lags, strategy=poly2_strategy)
        logMessage("Creating Polynomial Regression Orde 2 Model ...")
        poly2_forecaster.fit(train_df, X=train_exog) #, X=X_train
        
        logMessage("Polynomial Regression Orde 2 Model Prediction ...")
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
        poly3_lags = 2
        poly3_strategy = "recursive"

        # Create regressor object
        poly3_regressor = PolynomRegressor(deg=2, regularization=poly3_regularization, interactions=poly3_interactions)
        poly3_forecaster = make_reduction(poly3_regressor, window_length=poly3_lags, strategy=poly3_strategy)
        logMessage("Creating Polynomial Regression Orde 3 Model ...")
        poly3_forecaster.fit(train_df, X=train_exog) #, X=X_train
        
        logMessage("Polynomial Regression Orde 3 Model Prediction ...")
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
        title = 'LNG Production PT Badak with Exogenous Variable (Feed Gas, Day & Month)'
        ax.set_title(title)
        ax.set_ylabel("LNG Production")
        ax.set_xlabel("Datestamp")
        ax.legend(loc='best')
        #plt.savefig("LNG Production PT Badak Arimax Model with Exogenous Variables (Feed Gas + Day-Month)" + ".jpg")
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
