# %%
import logging
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

from connection import *
from utils import *

from pmdarima import model_selection
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)

from adtk.detector import ThresholdAD
from adtk.visualization import plot
from adtk.data import validate_series
pd.options.plotting.backend = "plotly"
from dateutil.relativedelta import *


# %%
def main():
    # Configure logging
    #configLogging("feed_gas_tangguh.log")
<<<<<<< HEAD
    logMessage("Creating Feed Gas BP Tangguh Model ...")
=======
>>>>>>> 21af3625c9ee755c28de8726159af49de0c06b45
    
    # Connect to database
    # Exit program if not connected to database
    logMessage("Connecting to database ...")
    conn = create_db_connection(section='postgresql_ml_lng_skk_dev')
    if conn == None:
        exit()
        
    logMessage("Cleaning data ...")
    ##### CLEANING FEED GAS DATA #####
    #Load Data from Database
    query_1 = open(os.path.join('gas_prod/sql', 'fg_tangguh_data_query_insample.sql'), mode="rt").read()
    data = get_sql_data(query_1, conn)

    data['date'] = pd.DatetimeIndex(data['date'], freq='D')
    data = data.reset_index()
    data.head()

    #%%
    ds = 'date'
    y = 'feed_gas' 

    df = data[[ds,y]]
    df = df.set_index(ds)
    df.index = pd.DatetimeIndex(df.index, freq='D')
    #df

    #%%
    data_cleaning = data[['date', 'feed_gas', 'wpnb_gas', 'unplanned_shutdown', 'planned_shutdown']].copy()
    ds_cleaning = 'date'
    data_cleaning = data_cleaning.set_index(ds_cleaning)
    data_cleaning['unplanned_shutdown'] = data_cleaning['unplanned_shutdown'].astype('int')
    s = validate_series(data_cleaning)

    # plotting for illustration
    plt.style.use('fivethirtyeight')

    fig1, ax = plt.subplots(figsize=(20,8))
    ax.plot(data_cleaning['feed_gas'], label='feed gas')
    ax.set_ylabel("feed gas - condensate")
    ax.set_xlabel("Datestamp")
    ax.legend(loc='best')
    plt.close()

    #%%
    # Calculate standar deviation
    fg_std = data_cleaning['feed_gas'].std()
    fg_mean = data_cleaning['feed_gas'].mean()

    #Detect Anomaly Values
    # Create anomaly detection model
    high_limit1 = fg_mean+3*fg_std
    low_limit1 = fg_mean-3*fg_std
    high_limit2 = fg_mean+fg_std
    low_limit2 = fg_mean-fg_std

    threshold_ad = ThresholdAD(data_cleaning['unplanned_shutdown']==0)
    anomalies = threshold_ad.detect(s)

    anomalies = anomalies.drop('feed_gas', axis=1)
    anomalies = anomalies.drop('wpnb_gas', axis=1)
    anomalies = anomalies.drop('planned_shutdown', axis=1)
    #anomalies

    #%%
    # Create anomaly detection model
    #threshold_ad = ThresholdAD(high=high_limit2, low=low_limit1)
    #anomalies =  threshold_ad.detect(s)

    # Copy data frame of anomalies
    copy_anomalies =  anomalies.copy()
    # Rename columns
    copy_anomalies.rename(columns={'unplanned_shutdown':'anomaly'}, inplace=True)
    # Merge original dataframe with anomalies
    new_s = pd.concat([s, copy_anomalies], axis=1)

    # Get only anomalies data
    anomalies_data = new_s[new_s['anomaly'] == False]
    #anomalies_data.tail(100)

    #%%
    # Plot data and its anomalies
    from cProfile import label
    from imaplib import Time2Internaldate

    fig = px.line(new_s, y='feed_gas')

    # Add horizontal line for 3 sigma
    fig.add_hline(y=high_limit2, line_color='red', line_dash="dot",
                annotation_text="Mean + std", 
                annotation_position="top right")
    fig.add_hline(y=low_limit1, line_color='red', line_dash="dot",
                annotation_text="Mean - 3*std", 
                annotation_position="bottom right")
    fig.add_scatter(x=anomalies_data.index, y=anomalies_data['feed_gas'], mode='markers', marker=dict(color='red'), name="Unplanned Shutdown", showlegend=True)
    fig.update_layout(title_text='Feed Gas Tangguh', title_font_size=24)

    #fig.show()
    plt.close()

    #%%
    #REPLACE ANOMALY VALUES
    from datetime import date, datetime, timedelta

    def get_first_date_of_current_month(year, month):
        """Return the first date of the month.

        Args:
            year (int): Year
            month (int): Month

        Returns:
            date (datetime): First date of the current month
        """
        first_date = datetime(year, month, 1)
        return first_date.strftime("%Y-%m-%d")

    def get_last_date_of_month(year, month):
        """Return the last date of the month.
        
        Args:
            year (int): Year, i.e. 2022
            month (int): Month, i.e. 1 for January

        Returns:
            date (datetime): Last date of the current month
        """
        
        if month == 12:
            last_date = datetime(year, month, 31)
        else:
            last_date = datetime(year, month + 1, 1) + timedelta(days=-1)
        
        return last_date.strftime("%Y-%m-%d")

    
    def get_first_date_of_prev_month(year, month, step=-1):
        """Return the first date of the month.

        Args:
            year (int): Year
            month (int): Month

        Returns:
            date (datetime): First date of the current month
        """
        first_date = datetime(year, month, 1)
        first_date = first_date + relativedelta(months=step)
        return first_date.strftime("%Y-%m-%d")

    def get_last_date_of_prev_month(year, month, step=-1):
        """Return the last date of the month.
        
        Args:
            year (int): Year, i.e. 2022
            month (int): Month, i.e. 1 for January

        Returns:
            date (datetime): Last date of the current month
        """
        
        if month == 12:
            last_date = datetime(year, month, 31)
        else:
            last_date = datetime(year, month + 1, 1) + timedelta(days=-1)
            
        last_date = last_date + relativedelta(months=step)
        
        return last_date.strftime("%Y-%m-%d")

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
        
        print(sql), print(mean_month)

    # Check if updated
    new_s[new_s['anomaly'] == False]

    anomaly_upd = new_s[new_s['anomaly'] == False]

    #%%
    # Plot data and its anomalies
    from cProfile import label
    from imaplib import Time2Internaldate


    fig = px.line(new_s, y='feed_gas')

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
    fig.add_scatter(x=anomalies_data.index, y=anomalies_data['feed_gas'], mode='markers', marker=dict(color='red'), name="Unplanned Shutdown", showlegend=True)
    fig.add_scatter(x=anomaly_upd.index, y=anomaly_upd['feed_gas'], mode='markers', marker=dict(color='green'), name="Unplanned Cleaned", showlegend=True)
    fig.update_layout(title_text='Feed Gas BP Tangguh', title_font_size=24)

    #fig.show()
    plt.close()


    #%%
    data_cleaned = new_s[['feed_gas', 'wpnb_gas', 'planned_shutdown']].copy()
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
<<<<<<< HEAD
    # from chart_studio.plotly import plot_mpl
    # from statsmodels.tsa.seasonal import seasonal_decompose
    # result = seasonal_decompose(train_df.values, model="additive", period=365)
    # fig = result.plot()
    # plt.close()
=======
    #from chart_studio.plotly import plot_mpl
    #from statsmodels.tsa.seasonal import seasonal_decompose
    #result = seasonal_decompose(train_df.values, model="additive", period=365)
    #fig = result.plot()
    #plt.close()
>>>>>>> 21af3625c9ee755c28de8726159af49de0c06b45

    #%%
    from statsmodels.tsa.stattools import adfuller
    def ad_test(dataset):
        dftest = adfuller(df, autolag = 'AIC')
        print("1. ADF : ",dftest[0])
        print("2. P-Value : ", dftest[1])
        print("3. Num Of Lags : ", dftest[2])
        print("4. Num Of Observations Used For ADF Regression:", dftest[3])
        print("5. Critical Values :")
        for key, val in dftest[4].items():
            print("\t",key, ": ", val)
    ad_test(train_df)

    #%%
    from sktime.forecasting.model_selection import temporal_train_test_split
    from sktime.forecasting.base import ForecastingHorizon

    # Test size
    test_size = 0.2
    # Split data (original data)
    y_train, y_test = temporal_train_test_split(df, test_size=test_size)
    # Split data (original data)
    y_train_cleaned, y_test_cleaned = temporal_train_test_split(df_cleaned, test_size=test_size)
    # Horizon
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    #%%
    # create features (exog) from date
    df_cleaned['month'] = [i.month for i in df_cleaned.index]
    df_cleaned['planned_shutdown'] = data['planned_shutdown'].values
    df_cleaned['day'] = [i.day for i in df_cleaned.index]
    df_cleaned['wpnb_gas'] = data['wpnb_gas'].values
    #df['day_of_year'] = [i.dayofyear for i in df.index]
    #df['week_of_year'] = [i.weekofyear for i in df.index]
    #df_cleaned.tail(20)

    #%%
    # Split into train and test
    X_train, X_test = temporal_train_test_split(df_cleaned.iloc[:,1:], test_size=test_size)
    #X_train

    #%%
    exogenous_features = ["month", "planned_shutdown", "day", "wpnb_gas"]
    #exogenous_features

    #%%
    # plotting for illustration
    fig1, ax = plt.subplots(figsize=(20,8))
    ax.plot(y_train, label='train')
    ax.plot(y_test, label='test')
    ax.set_ylabel("Feed Gas")
    ax.set_xlabel("Datestamp")
    ax.legend(loc='best')
    plt.close()


    ##### ARIMAX MODEL (forecast_a) #####
    # %%
    from pmdarima.arima.utils import ndiffs, nsdiffs
    from sklearn.metrics import mean_squared_error
    import statsmodels.api as sm
    from sktime.forecasting.arima import AutoARIMA
    from sktime.forecasting.statsforecast import StatsForecastAutoARIMA

    # Create ARIMAX (forecast_a) Model
    arimax_model = AutoARIMA(d=0, suppress_warnings=True, error_action='ignore')
    logMessage("Creating ARIMAX Model ...")
    arimax_model.fit(y_train_cleaned.feed_gas, X=X_train[exogenous_features])
    logMessage("ARIMAX Model Summary")
    logMessage(arimax_model.summary())
    
    logMessage("ARIMAX Model Prediction ..")
    arimax_forecast = arimax_model.predict(fh, X=X_test[exogenous_features])
    y_pred_arimax = pd.DataFrame(arimax_forecast).applymap('{:.2f}'.format)
    y_pred_arimax['day_num'] = [i.day for i in arimax_forecast.index]
    y_pred_arimax['month_num'] = [i.month for i in arimax_forecast.index]
    y_pred_arimax['year_num'] = [i.year for i in arimax_forecast.index]
    y_pred_arimax['date'] = y_pred_arimax['year_num'].astype(str) + '-' + y_pred_arimax['month_num'].astype(str) + '-' + y_pred_arimax['day_num'].astype(str)
    y_pred_arimax['date'] = pd.DatetimeIndex(y_pred_arimax['date'], freq='D')
    
    #Rename colum 0
    y_pred_arimax.rename(columns={0:'forecast_a'}, inplace=True)

    #Create MAPE
    arimax_mape = mean_absolute_percentage_error(y_test.feed_gas, arimax_forecast)
    arimax_mape_str = str('MAPE: %.4f' % arimax_mape)
    logMessage("ARIMAX Model "+arimax_mape_str)
    
    #Get parameter
    arimax_param = str(arimax_model.get_fitted_params()['order'])
    logMessage("Arimax Model Parameters "+arimax_param)


    ##### SARIMAX MODEL (forecast_b) #####
    ##### ARIMA(2,0,0)(2,1,0)[12] #####
    #%%
    from pmdarima.arima import auto_arima
    from sktime.forecasting.arima import ARIMA
    
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
    
    #sarimax_model = auto_arima(y=y_train_cleaned.feed_gas, X=X_train[exogenous_features], d=0, D=1, seasonal=True, m=12, trace=True, error_action="ignore", suppress_warnings=True)
    sarimax_model = AutoARIMA(d=sarimax_differencing, D=sarimax_seasonal_differencing, seasonal=sarimax_seasonal, sp=sarimax_sp, trace=sarimax_trace, n_fits=sarimax_n_fits, stepwise=sarimax_stepwise, error_action=sarimax_error_action, suppress_warnings=sarimax_suppress_warnings)
    #sarimax_model = ARIMA(order=(2, 0, 0), seasonal_order=(2, 1, 0, 12), suppress_warnings=sarimax_suppress_warnings)
    logMessage("Creating SARIMAX Model ...") 
    sarimax_model.fit(y_train_cleaned.feed_gas, X=X_train[exogenous_features])
    logMessage("SARIMAX Model Summary")
    logMessage(sarimax_model.summary())
    
    logMessage("SARIMAX Model Prediction ..")
    sarimax_forecast = sarimax_model.predict(fh, X=X_test[exogenous_features]) #len(fh)
    y_test["Forecast_SARIMAX"] = sarimax_forecast
    y_pred_sarimax = pd.DataFrame(sarimax_forecast).applymap('{:.2f}'.format)
    y_pred_sarimax['day_num'] = [i.day for i in sarimax_forecast.index]
    y_pred_sarimax['month_num'] = [i.month for i in sarimax_forecast.index]
    y_pred_sarimax['year_num'] = [i.year for i in sarimax_forecast.index]
    y_pred_sarimax['date'] = y_pred_sarimax['year_num'].astype(str) + '-' + y_pred_sarimax['month_num'].astype(str) + '-' + y_pred_sarimax['day_num'].astype(str)
    y_pred_sarimax['date'] = pd.DatetimeIndex(y_pred_sarimax['date'], freq='D')
    #Rename colum 0
    y_pred_sarimax.rename(columns={0:'forecast_b'}, inplace=True)

    #Create MAPE
    sarimax_mape = mean_absolute_percentage_error(y_test.feed_gas, y_test.Forecast_SARIMAX)
    sarimax_mape_str = str('MAPE: %.4f' % sarimax_mape)
    logMessage("SARIMAX Model "+sarimax_mape_str)
    
    #Get parameters
    sarimax_param_order = str(sarimax_model.get_fitted_params()['order'])
    sarimax_param_order_seasonal = str(sarimax_model.get_fitted_params()['seasonal_order'])
    sarimax_param = sarimax_param_order + sarimax_param_order_seasonal
    logMessage("Sarimax Model Parameters "+sarimax_param)
    

    ##### PROPHET MODEL (forecast_c) #####
    #%%
    # Create model
    from sktime.forecasting.fbprophet import Prophet
    from sktime.forecasting.compose import make_reduction

    #Set Parameters
    seasonality_mode = 'multiplicative'
    n_changepoints = 30
    seasonality_prior_scale = 0.1
    changepoint_prior_scale = 0.1
    holidays_prior_scale = 8
    daily_seasonality = 10
    weekly_seasonality = 1
    yearly_seasonality = 10

    # create regressor object
    prophet_forecaster = Prophet(
        seasonality_mode=seasonality_mode,
        n_changepoints=n_changepoints,
        seasonality_prior_scale=seasonality_prior_scale, #Flexibility of the seasonality (0.01,10)
        changepoint_prior_scale=changepoint_prior_scale, #Flexibility of the trend (0.001,0.5)
        holidays_prior_scale=holidays_prior_scale, #Flexibility of the holiday effects (0.01,10)
        #changepoint_range=0.8, #proportion of the history in which the trend is allowed to change
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality)

    logMessage("Creating Prophet Model ....")
    prophet_forecaster.fit(y_train_cleaned, X_train) #, X_train
    
    logMessage("Prophet Model Prediction ...")
    prophet_forecast = prophet_forecaster.predict(fh, X=X_test) #, X=X_test
    y_pred_prophet = pd.DataFrame(prophet_forecast).applymap('{:.2f}'.format)
    y_pred_prophet['day_num'] = [i.day for i in prophet_forecast.index]
    y_pred_prophet['month_num'] = [i.month for i in prophet_forecast.index]
    y_pred_prophet['year_num'] = [i.year for i in prophet_forecast.index]
    y_pred_prophet['date'] = y_pred_prophet['year_num'].astype(str) + '-' + y_pred_prophet['month_num'].astype(str) + '-' + y_pred_prophet['day_num'].astype(str)
    y_pred_prophet['date'] = pd.DatetimeIndex(y_pred_prophet['date'], freq='D')
    #Rename colum 0
    y_pred_prophet.rename(columns={'feed_gas':'forecast_c'}, inplace=True)

    #Create MAPE
    prophet_mape = mean_absolute_percentage_error(y_test['feed_gas'], prophet_forecast)
    prophet_mape_str = str('MAPE: %.4f' % prophet_mape)
    logMessage("Prophet Model "+prophet_mape_str)
    
    #Get parameters
    prophet_param_seasonality_mode = str(prophet_forecaster.get_params()['seasonality_mode'])
    prophet_param_n_changepoints = str(prophet_forecaster.get_params()['n_changepoints'])
    prophet_param_seasonality_prior_scale = str(prophet_forecaster.get_params()['seasonality_prior_scale'])
    prophet_param_changepoint_prior_scale = str(prophet_forecaster.get_params()['changepoint_prior_scale'])
    prophet_param_holidays_prior_scale = str(prophet_forecaster.get_params()['holidays_prior_scale'])
    prophet_param_daily_seasonality = str(prophet_forecaster.get_params()['daily_seasonality'])
    prophet_param_weekly_seasonality = str(prophet_forecaster.get_params()['weekly_seasonality'])
    prophet_param_yearly_seasonality = str(prophet_forecaster.get_params()['yearly_seasonality'])
    prophet_param = prophet_param_seasonality_mode + ', ' + prophet_param_n_changepoints + ', ' + prophet_param_seasonality_prior_scale + ', ' + prophet_param_changepoint_prior_scale + ', ' + prophet_param_holidays_prior_scale + ', ' + prophet_param_daily_seasonality + ', ' + prophet_param_weekly_seasonality + ', ' + prophet_param_yearly_seasonality
    logMessage("Prophet Model Parameters "+prophet_param)


    ##### RANDOM FOREST MODEL (forecast_d) #####
    #%%
    from sklearn.ensemble import RandomForestRegressor

    #Set Parameters
    ranfor_n_estimators = 150
    ranfor_lags = 41
    ranfor_random_state = 0
    ranfor_criterion = "squared_error"
    ranfor_strategy = "recursive"

    # create regressor object
    ranfor_regressor = RandomForestRegressor(n_estimators = ranfor_n_estimators, random_state = ranfor_random_state, criterion = ranfor_criterion)
    ranfor_forecaster = make_reduction(ranfor_regressor, window_length = ranfor_lags, strategy = ranfor_strategy)

    logMessage("Creating Random Forest Model ...")
    ranfor_forecaster.fit(y_train_cleaned, X_train) #, X_train
    
    logMessage("Random Forest Model Prediction ...")
    ranfor_forecast = ranfor_forecaster.predict(fh, X=X_test) #, X=X_test
    y_pred_ranfor = pd.DataFrame(ranfor_forecast).applymap('{:.2f}'.format)
    y_pred_ranfor['day_num'] = [i.day for i in ranfor_forecast.index]
    y_pred_ranfor['month_num'] = [i.month for i in ranfor_forecast.index]
    y_pred_ranfor['year_num'] = [i.year for i in ranfor_forecast.index]
    y_pred_ranfor['date'] = y_pred_ranfor['year_num'].astype(str) + '-' + y_pred_ranfor['month_num'].astype(str) + '-' + y_pred_ranfor['day_num'].astype(str)
    y_pred_ranfor['date'] = pd.DatetimeIndex(y_pred_ranfor['date'], freq='D')
    #Rename colum 0
    y_pred_ranfor.rename(columns={'feed_gas':'forecast_d'}, inplace=True)

    #Create MAPE
    ranfor_mape = mean_absolute_percentage_error(y_test['feed_gas'], ranfor_forecast)
    ranfor_mape_str = str('MAPE: %.4f' % ranfor_mape)
    logMessage("Random Forest Model "+ranfor_mape_str)
    
    #Get Parameters
    ranfor_param_estimator = str(ranfor_forecaster.get_fitted_params()['estimator'])
    ranfor_param_lags = str(ranfor_forecaster.get_fitted_params()['window_length'])
    ranfor_param = ranfor_param_estimator + ', ' + ranfor_param_lags
    logMessage("Random Forest Model Parameters "+ranfor_param)


    ##### XGBOOST MODEL (forecast_e) #####
    #%%
    # Create model
    from xgboost import XGBRegressor

    #Set Parameters
    xgb_objective = 'reg:squarederror'
    xgb_lags = 46
    xgb_strategy = "recursive"

    xgb_regressor = XGBRegressor(objective=xgb_objective)
    xgb_forecaster = make_reduction(xgb_regressor, window_length=xgb_lags, strategy=xgb_strategy)
    
    logMessage("Creating XGBoost Model ....")
    xgb_forecaster.fit(y_train_cleaned, X=X_train) #, X_train
    
    logMessage("XGBoost Model Prediction ...")
    xgb_forecast = xgb_forecaster.predict(fh, X=X_test) #, X=X_test
    y_pred_xgb = pd.DataFrame(xgb_forecast).applymap('{:.2f}'.format)
    y_pred_xgb['day_num'] = [i.day for i in xgb_forecast.index]
    y_pred_xgb['month_num'] = [i.month for i in xgb_forecast.index]
    y_pred_xgb['year_num'] = [i.year for i in xgb_forecast.index]
    y_pred_xgb['date'] = y_pred_xgb['year_num'].astype(str) + '-' + y_pred_xgb['month_num'].astype(str) + '-' + y_pred_xgb['day_num'].astype(str)
    y_pred_xgb['date'] = pd.DatetimeIndex(y_pred_xgb['date'], freq='D')
    #Rename colum 0
    y_pred_xgb.rename(columns={'feed_gas':'forecast_e'}, inplace=True)

    #Create MAPE
    xgb_mape = mean_absolute_percentage_error(y_test['feed_gas'], xgb_forecast)
    xgb_mape_str = str('MAPE: %.4f' % xgb_mape)
    logMessage("XGBoost Model "+xgb_mape_str)
    
    #Get Parameters
    xgb_param_lags = str(xgb_forecaster.get_params()['window_length'])
    xgb_param_objective = str(xgb_forecaster.get_params()['estimator__objective'])
    xgb_param = xgb_param_lags + ', ' + xgb_param_objective
    logMessage("XGBoost Model Parameters "+xgb_param)


    ##### LINEAR REGRESSION MODEL (forecast_f) #####
    #%%
    # Create model
    from sklearn.linear_model import LinearRegression

    #Set Parameters
    linreg_lags = 33
    linreg_strategy = "recursive"

    linreg_regressor = LinearRegression(normalize=True)
    linreg_forecaster = make_reduction(linreg_regressor, window_length=linreg_lags, strategy=linreg_strategy)

    logMessage("Creating Linear Regression Model ...")
    linreg_forecaster.fit(y_train_cleaned, X=X_train) #, X=X_train
    
    logMessage("Linear Regression Model Prediction ...")
    linreg_forecast = linreg_forecaster.predict(fh, X=X_test) #, X=X_test
    y_pred_linreg = pd.DataFrame(linreg_forecast).applymap('{:.2f}'.format)
    y_pred_linreg['day_num'] = [i.day for i in linreg_forecast.index]
    y_pred_linreg['month_num'] = [i.month for i in linreg_forecast.index]
    y_pred_linreg['year_num'] = [i.year for i in linreg_forecast.index]
    y_pred_linreg['date'] = y_pred_linreg['year_num'].astype(str) + '-' + y_pred_linreg['month_num'].astype(str) + '-' + y_pred_linreg['day_num'].astype(str)
    y_pred_linreg['date'] = pd.DatetimeIndex(y_pred_linreg['date'], freq='D')
    #Rename colum 0
    y_pred_linreg.rename(columns={'feed_gas':'forecast_f'}, inplace=True)

    #Create MAPE
    linreg_mape = mean_absolute_percentage_error(y_test['feed_gas'], linreg_forecast)
    linreg_mape_str = str('MAPE: %.4f' % linreg_mape)
    logMessage("Linear Regression Model "+linreg_mape_str)
    
    #Get parameters
    linreg_param_estimator = str(linreg_forecaster.get_fitted_params()['estimator'])
    linreg_param_lags = str(linreg_forecaster.get_fitted_params()['window_length'])
    linreg_param = linreg_param_estimator + ', ' + linreg_param_lags
    logMessage("Linear Regression Model Parameters "+linreg_param)


    ##### POLYNOMIAL REGRESSION DEGREE=2 MODEL (forecast_g) #####
    #%%
    #Create model
    from polyfit import PolynomRegressor, Constraints

    #Set Parameters
    poly2_lags = 16
    poly2_regularization = None
    poly2_interactions = False
    poly2_strategy = "recursive"

    poly2_regressor = PolynomRegressor(deg=2, regularization=poly2_regularization, interactions=poly2_interactions)
    poly2_forecaster = make_reduction(poly2_regressor, window_length=poly2_lags, strategy=poly2_strategy)

    logMessage("Creating Polynomial Regression Orde 2 Model ...")
    poly2_forecaster.fit(y_train_cleaned, X=X_train) #, X=X_train
    
    logMessage("Polynomial Regression Orde 2 Model Prediction ...")
    poly2_forecast = poly2_forecaster.predict(fh, X=X_test) #, X=X_test
    y_pred_poly2 = pd.DataFrame(poly2_forecast).applymap('{:.2f}'.format)
    y_pred_poly2['day_num'] = [i.day for i in poly2_forecast.index]
    y_pred_poly2['month_num'] = [i.month for i in poly2_forecast.index]
    y_pred_poly2['year_num'] = [i.year for i in poly2_forecast.index]
    y_pred_poly2['date'] = y_pred_poly2['year_num'].astype(str) + '-' + y_pred_poly2['month_num'].astype(str) + '-' + y_pred_poly2['day_num'].astype(str)
    y_pred_poly2['date'] = pd.DatetimeIndex(y_pred_poly2['date'], freq='D')
    #Rename colum 0
    y_pred_poly2.rename(columns={'feed_gas':'forecast_g'}, inplace=True)

    #Create MAPE
    poly2_mape = mean_absolute_percentage_error(y_test['feed_gas'], poly2_forecast)
    poly2_mape_str = str('MAPE: %.4f' % poly2_mape)
    logMessage("Polynomial Regression Degree=2 Model "+poly2_mape_str)
    
    #Get parameters
    poly2_param_estimator = str(poly2_forecaster.get_fitted_params()['estimator'])
    poly2_param_lags = str(poly2_forecaster.get_fitted_params()['window_length'])
    poly2_param = poly2_param_estimator + ', ' + poly2_param_lags
    logMessage("Polynomial Regression Orde 2 Model Parameters "+poly2_param)


    ##### POLYNOMIAL REGRESSION DEGREE=3 MODEL (forecast_h) #####
    #%%
    #Create model
    from polyfit import PolynomRegressor, Constraints

    #Set Parameters
    poly3_lags = 0.6
    poly3_regularization = None
    poly3_interactions = False
    poly3_strategy = "recursive"

    poly3_regressor = PolynomRegressor(deg=3, regularization=poly3_regularization, interactions=poly3_interactions)
    poly3_forecaster = make_reduction(poly3_regressor, window_length=poly3_lags, strategy=poly3_strategy)

    logMessage("Creating Polynomial Regression Orde 3 Model ...")
    poly3_forecaster.fit(y_train_cleaned, X=X_train) #, X=X_train
    
    logMessage("Polynomial Regression Orde 3 Model Prediction ...")
    poly3_forecast = poly3_forecaster.predict(fh, X=X_test) #, X=X_test
    y_pred_poly3 = pd.DataFrame(poly3_forecast).applymap('{:.2f}'.format)
    y_pred_poly3['day_num'] = [i.day for i in poly3_forecast.index]
    y_pred_poly3['month_num'] = [i.month for i in poly3_forecast.index]
    y_pred_poly3['year_num'] = [i.year for i in poly3_forecast.index]
    y_pred_poly3['date'] = y_pred_poly3['year_num'].astype(str) + '-' + y_pred_poly3['month_num'].astype(str) + '-' + y_pred_poly3['day_num'].astype(str)
    y_pred_poly3['date'] = pd.DatetimeIndex(y_pred_poly3['date'], freq='D')
    #Rename colum 0
    y_pred_poly3.rename(columns={'feed_gas':'forecast_h'}, inplace=True)

    #Create MAPE
    poly3_mape = mean_absolute_percentage_error(y_test['feed_gas'], poly3_forecast)
    poly3_mape_str = str('MAPE: %.4f' % poly3_mape)
    logMessage("Polynomial Regression Degree=3 Model "+poly3_mape_str)
    
    #Get parameters
    poly3_param_estimator = str(poly3_forecaster.get_fitted_params()['estimator'])
    poly3_param_lags = str(poly3_forecaster.get_fitted_params()['window_length'])
    poly3_param = poly3_param_estimator + ', ' + poly3_param_lags
    logMessage("Polynomial Regression Orde 3 Model Parameters "+poly3_param)

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
    title = 'Feed Gas BP Tangguh Forecasting with Exogenous Variable and Cleaning Data'
    ax.set_title(title)
    ax.set_ylabel("Feed Gas")
    ax.set_xlabel("Datestamp")
    ax.legend(loc='best')
    plt.close()
    
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
    y_all_pred['date'] = y_test.index.values

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
                        'model_param_c': [prophet_param],
                        'model_param_d': [ranfor_param],
                        'model_param_e': [xgb_param],
                        'model_param_f': [linreg_param],
                        'model_param_g': [poly2_param],
                        'model_param_h': [poly3_param],
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
        logMessage(error)

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
        logMessage(error)

    return updated_rows