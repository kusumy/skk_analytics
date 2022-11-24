# %%
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
import pmdarima as pm
import psycopg2
import seaborn as sns

plt.style.use('fivethirtyeight')
from datetime import datetime
from tokenize import Ignore
from tracemalloc import start

from connection import config, retrieve_data, create_db_connection
from pmdarima import model_selection
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

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
    #plt.legend(loc='best')
    #plt.tight_layout()

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
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(message)s", 
        filename="condensate_tangguh.log",
        filemode="w" #, 
        #handlers=[
        #    logging.FileHandler("condensate_tangguh.log"),
        #    logging.StreamHandler(sys.stdout) ]
    ) #filename="condensate_tangguh.log",
    
    # Connect to database
    # Exit program if not connected to database
    conn = create_db_connection(section='postgresql_ml_lng')
    if conn == None:
        exit()
        
    # Prepare data
    file = os.path.join('gas_prod','condensate_unplanned-planned_shutdown_cleaned_with.csv')
    data = pd.read_csv(file, sep=',')

    data['date'] = pd.DatetimeIndex(data['date'], freq='D')
    data = data.reset_index()
    #data.head()

    #%%
    ds = 'date'
    y = 'condensate' 

    df = data[[ds,y]]
    df = df.set_index(ds)
    df.index = pd.DatetimeIndex(df.index, freq='D')
    #df

    #%%
    stationarity_check(df)

    #%%
    decomposition_plot(df)

    #%%
    plot_acf_pacf(df)

    #%%
    #from chart_studio.plotly import plot_mpl
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(df.condensate.values, model="additive", period=365)
    fig = result.plot()
    #plt.show()

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
    ad_test(df['condensate'])

    #%%
    from sktime.forecasting.model_selection import temporal_train_test_split
    from sktime.forecasting.base import ForecastingHorizon

    # Test size
    test_size = 0.2
    # Split data
    y_train, y_test = temporal_train_test_split(df, test_size=test_size)
    # Horizon
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    #%%
    # create features (exog) from date
    df['month'] = [i.month for i in df.index]
    df['planned_shutdown'] = data['planned_shutdown'].values
    df['day'] = [i.day for i in df.index]
    df['wpnb_oil'] = data['wpnb_oil'].values
    #df['day_of_year'] = [i.dayofyear for i in df.index]
    #df['week_of_year'] = [i.weekofyear for i in df.index]
    #df.tail(20)

    #%%
    # Split into train and test
    X_train, X_test = temporal_train_test_split(df.iloc[:,1:], test_size=test_size)
    #X_train

    #%%
    exogenous_features = ["month", "day", "planned_shutdown", "wpnb_oil"]
    #exogenous_features

    #%%
    # plotting for illustration
    fig1, ax = plt.subplots(figsize=(20,8))
    ax.plot(y_train, label='train')
    ax.plot(y_test, label='test')
    ax.set_ylabel("Condensate")
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
    arimax_model.fit(y_train.condensate, X=X_train[exogenous_features])
    logging.info("ARIMAX Model Summary")
    logging.info(arimax_model.summary())

    arimax_forecast = arimax_model.predict(fh, X=X_test[exogenous_features])
    y_test["Forecast_ARIMAX"] = arimax_forecast
    y_pred_arimax = pd.DataFrame(arimax_forecast).applymap('{:.2f}'.format)
    y_pred_arimax['day_num'] = [i.day for i in arimax_forecast.index]
    y_pred_arimax['month_num'] = [i.month for i in arimax_forecast.index]
    y_pred_arimax['year_num'] = [i.year for i in arimax_forecast.index]
    y_pred_arimax['date'] = y_pred_arimax['year_num'].astype(str) + '-' + y_pred_arimax['month_num'].astype(str) + '-' + y_pred_arimax['day_num'].astype(str)
    y_pred_arimax['date'] = pd.DatetimeIndex(y_pred_arimax['date'], freq='D')

    #Create MAPE
    arimax_mape = mean_absolute_percentage_error(y_test.condensate, y_test.Forecast_ARIMAX)
    arimax_mape_100 = 100*arimax_mape
    arimax_mape_str = str('MAPE: %.4f' % arimax_mape_100) + '%'
    logging.info("ARIMAX Model "+arimax_mape_str)

    # Rename column to forecast_a
    y_pred_arimax.rename(columns={0:'forecast_a'}, inplace=True)

    ##### SARIMAX MODEL (forecast_b) #####
    #%%
    from pmdarima.arima import auto_arima, ARIMA
    
    #sarimax_model = auto_arima(y=y_train.condensate, X=X_train[exogenous_features], d=0, D=1, seasonal=True, m=12, trace=True, error_action="ignore", suppress_warnings=True)
    sarimax_model = ARIMA(order=(2,0,2), seasonal_order=(2,1,0,12),  suppress_warnings=True)
    sarimax_model.fit(y_train.condensate, X=X_train[exogenous_features])
    sarimax_model.summary()
    logging.info("SARIMAX Model Summary")
    logging.info(sarimax_model.summary())

    sarimax_forecast = sarimax_model.predict(len(fh), X=X_test[exogenous_features])
    y_test["Forecast_SARIMAX"] = sarimax_forecast
    y_pred_sarimax = pd.DataFrame(sarimax_forecast).applymap('{:.2f}'.format)
    y_pred_sarimax['day_num'] = [i.day for i in sarimax_forecast.index]
    y_pred_sarimax['month_num'] = [i.month for i in sarimax_forecast.index]
    y_pred_sarimax['year_num'] = [i.year for i in sarimax_forecast.index]
    y_pred_sarimax['date'] = y_pred_sarimax['year_num'].astype(str) + '-' + y_pred_sarimax['month_num'].astype(str) + '-' + y_pred_sarimax['day_num'].astype(str)
    y_pred_sarimax['date'] = pd.DatetimeIndex(y_pred_sarimax['date'], freq='D')

    #Create MAPE
    sarimax_mape = mean_absolute_percentage_error(y_test.condensate, y_test.Forecast_SARIMAX)
    sarimax_mape_100 = 100*sarimax_mape
    sarimax_mape_str = str('MAPE: %.4f' % sarimax_mape_100) + '%'
    logging.info("SARIMAX Model "+sarimax_mape_str)
    
    # Rename column to forecast_b
    y_pred_sarimax.rename(columns={0:'forecast_b'}, inplace=True)

    ##### PROPHET MODEL (forecast_c) #####
    #%%
    # Create model
    from sktime.forecasting.fbprophet import Prophet
    from sktime.forecasting.compose import make_reduction

    #Set Parameters
    seasonality_mode = 'multiplicative'
    n_changepoints = 40
    seasonality_prior_scale = 0.2
    changepoint_prior_scale = 0.1
    holidays_prior_scale = 8
    daily_seasonality = 5
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

    logging.info("Creating Prophet Model ....")
    prophet_forecaster.fit(y_train, X_train) #, X_train
    logging.info(prophet_forecaster._get_fitted_params)
    
    logging.info("Prophet Model Prediction")
    prophet_forecast = prophet_forecaster.predict(fh, X=X_test) #, X=X_test
    y_pred_prophet = pd.DataFrame(prophet_forecast).applymap('{:.2f}'.format)
    y_pred_prophet['day_num'] = [i.day for i in prophet_forecast.index]
    y_pred_prophet['month_num'] = [i.month for i in prophet_forecast.index]
    y_pred_prophet['year_num'] = [i.year for i in prophet_forecast.index]
    y_pred_prophet['date'] = y_pred_prophet['year_num'].astype(str) + '-' + y_pred_prophet['month_num'].astype(str) + '-' + y_pred_prophet['day_num'].astype(str)
    y_pred_prophet['date'] = pd.DatetimeIndex(y_pred_prophet['date'], freq='D')

    #Create MAPE
    prophet_mape = mean_absolute_percentage_error(y_test['condensate'], prophet_forecast)
    prophet_mape_100 = 100*prophet_mape
    prophet_mape_str = str('MAPE: %.4f' % prophet_mape_100) + '%'
    logging.info("Prophet Model "+prophet_mape_str)

    # Rename column to forecast_c
    y_pred_prophet.rename(columns={'condensate':'forecast_c'}, inplace=True)
    
    ##### RANDOM FOREST MODEL (forecast_d) #####
    #%%
    from sklearn.ensemble import RandomForestRegressor

    #Set Parameters
    ranfor_n_estimators = 150
    ranfor_lags = 53
    ranfor_random_state = 0
    ranfor_criterion = "squared_error"
    ranfor_strategy = "recursive"

    # create regressor object
    ranfor_regressor = RandomForestRegressor(n_estimators = ranfor_n_estimators, random_state = ranfor_random_state, criterion = ranfor_criterion)
    ranfor_forecaster = make_reduction(ranfor_regressor, window_length = ranfor_lags, strategy = ranfor_strategy)

    logging.info("Creating Prophet Model ....")
    ranfor_forecaster.fit(y_train, X_train) #, X_train
    
    logging.info("Random Forest Model Prediction")
    ranfor_forecast = ranfor_forecaster.predict(fh, X=X_test) #, X=X_test
    y_pred_ranfor = pd.DataFrame(ranfor_forecast).applymap('{:.2f}'.format)
    y_pred_ranfor['day_num'] = [i.day for i in ranfor_forecast.index]
    y_pred_ranfor['month_num'] = [i.month for i in ranfor_forecast.index]
    y_pred_ranfor['year_num'] = [i.year for i in ranfor_forecast.index]
    y_pred_ranfor['date'] = y_pred_ranfor['year_num'].astype(str) + '-' + y_pred_ranfor['month_num'].astype(str) + '-' + y_pred_ranfor['day_num'].astype(str)
    y_pred_ranfor['date'] = pd.DatetimeIndex(y_pred_ranfor['date'], freq='D')

    #Create MAPE
    ranfor_mape = mean_absolute_percentage_error(y_test['condensate'], ranfor_forecast)
    ranfor_mape_100 = 100*ranfor_mape
    ranfor_mape_str = str('MAPE: %.4f' % ranfor_mape_100) + '%'
    logging.info("Random Forest Model "+ranfor_mape_str)

    # Rename column to forecast_e
    y_pred_ranfor.rename(columns={'condensate':'forecast_d'}, inplace=True)    

    ##### XGBOOST MODEL (forecast_e) #####
    #%%
    # Create model
    from xgboost import XGBRegressor

    #Set Parameters
    xgb_objective = 'reg:squarederror'
    xgb_lags = 25
    xgb_strategy = "recursive"

    xgb_regressor = XGBRegressor(objective=xgb_objective)
    xgb_forecaster = make_reduction(xgb_regressor, window_length=xgb_lags, strategy=xgb_strategy)

    logging.info("Creating XGBoost Model ....")
    xgb_forecaster.fit(y_train, X=X_train) #, X_train
    
    logging.info("XGBoost Model Prediction")
    xgb_forecast = xgb_forecaster.predict(fh, X=X_test) #, X=X_test
    y_pred_xgb = pd.DataFrame(xgb_forecast).applymap('{:.2f}'.format)
    y_pred_xgb['day_num'] = [i.day for i in xgb_forecast.index]
    y_pred_xgb['month_num'] = [i.month for i in xgb_forecast.index]
    y_pred_xgb['year_num'] = [i.year for i in xgb_forecast.index]
    y_pred_xgb['date'] = y_pred_xgb['year_num'].astype(str) + '-' + y_pred_xgb['month_num'].astype(str) + '-' + y_pred_xgb['day_num'].astype(str)
    y_pred_xgb['date'] = pd.DatetimeIndex(y_pred_xgb['date'], freq='D')

    #Create MAPE
    xgb_mape = mean_absolute_percentage_error(y_test['condensate'], xgb_forecast)
    xgb_mape_100 = 100*xgb_mape
    xgb_mape_str = str('MAPE: %.4f' % xgb_mape_100) + '%'
    logging.info("XGBoost Model "+xgb_mape_str)
    
    # Rename column to forecast_e
    y_pred_xgb.rename(columns={'condensate':'forecast_e'}, inplace=True)

    ##### LINEAR REGRESSION MODEL (forecast_f) #####
    #%%
    # Create model
    from sklearn.linear_model import LinearRegression

    #Set Parameters
    linreg_lags = 22
    linreg_strategy = "recursive"

    linreg_regressor = LinearRegression(normalize=True)
    linreg_forecaster = make_reduction(linreg_regressor, window_length=linreg_lags, strategy=linreg_strategy)

    logging.info("Creating Linear Regression Model ....")
    linreg_forecaster.fit(y_train, X=X_train) #, X=X_train
    
    logging.info("Linear Regression Model Prediction")
    linreg_forecast = linreg_forecaster.predict(fh, X=X_test) #, X=X_test
    y_pred_linreg = pd.DataFrame(linreg_forecast).applymap('{:.2f}'.format)
    y_pred_linreg['day_num'] = [i.day for i in linreg_forecast.index]
    y_pred_linreg['month_num'] = [i.month for i in linreg_forecast.index]
    y_pred_linreg['year_num'] = [i.year for i in linreg_forecast.index]
    y_pred_linreg['date'] = y_pred_linreg['year_num'].astype(str) + '-' + y_pred_linreg['month_num'].astype(str) + '-' + y_pred_linreg['day_num'].astype(str)
    y_pred_linreg['date'] = pd.DatetimeIndex(y_pred_linreg['date'], freq='D')

    #Create MAPE
    linreg_mape = mean_absolute_percentage_error(y_test['condensate'], linreg_forecast)
    linreg_mape_100 = 100*linreg_mape
    linreg_mape_str = str('MAPE: %.4f' % linreg_mape_100) + '%'
    logging.info("Linear Regression Model "+linreg_mape_str)

    # Rename column to forecast_f
    y_pred_linreg.rename(columns={'condensate':'forecast_f'}, inplace=True)

    ##### POLYNOMIAL REGRESSION DEGREE=2 MODEL (forecast_g) #####
    #%%
    #Create model
    from polyfit import PolynomRegressor, Constraints

    #Set Parameters
    poly2_lags = 5
    poly2_regularization = None
    poly2_interactions = False
    poly2_strategy = "recursive"

    poly2_regressor = PolynomRegressor(deg=2, regularization=poly2_regularization, interactions=poly2_interactions)
    poly2_forecaster = make_reduction(poly2_regressor, window_length=poly2_lags, strategy=poly2_strategy)

    logging.info("Creating Polynomial Regression Orde 2 Model ....")
    poly2_forecaster.fit(y_train, X=X_train) #, X=X_train
    
    logging.info("Polynomial Regression Orde 2 Model Prediction")
    poly2_forecast = poly2_forecaster.predict(fh, X=X_test) #, X=X_test
    y_pred_poly2 = pd.DataFrame(poly2_forecast).applymap('{:.2f}'.format)
    y_pred_poly2['day_num'] = [i.day for i in poly2_forecast.index]
    y_pred_poly2['month_num'] = [i.month for i in poly2_forecast.index]
    y_pred_poly2['year_num'] = [i.year for i in poly2_forecast.index]
    y_pred_poly2['date'] = y_pred_poly2['year_num'].astype(str) + '-' + y_pred_poly2['month_num'].astype(str) + '-' + y_pred_poly2['day_num'].astype(str)
    y_pred_poly2['date'] = pd.DatetimeIndex(y_pred_poly2['date'], freq='D')

    #Create MAPE
    poly2_mape = mean_absolute_percentage_error(y_test['condensate'], poly2_forecast)
    poly2_mape_100 = 100*poly2_mape
    poly2_mape_str = str('MAPE: %.4f' % poly2_mape_100) + '%'
    logging.info("Polynomial Regression Orde 2 Model "+poly2_mape_str)

    # Rename column to forecast_g
    y_pred_poly2.rename(columns={'condensate':'forecast_g'}, inplace=True)

    ##### POLYNOMIAL REGRESSION DEGREE=3 MODEL (forecast_h) #####
    #%%
    #Create model
    from polyfit import PolynomRegressor, Constraints

    #Set Parameters
    poly3_lags = 0.55
    poly3_regularization = None
    poly3_interactions = False
    poly3_strategy = "recursive"

    poly3_regressor = PolynomRegressor(deg=3, regularization=poly3_regularization, interactions=poly3_interactions)
    poly3_forecaster = make_reduction(poly3_regressor, window_length=poly3_lags, strategy=poly3_strategy)

    logging.info("Creating Polynomial Regression Orde 3 Model ....")
    poly3_forecaster.fit(y_train, X=X_train) #, X=X_train
    
    logging.info("Polynomial Regression Orde 3 Model Prediction")
    poly3_forecast = poly3_forecaster.predict(fh, X=X_test) #, X=X_test
    y_pred_poly3 = pd.DataFrame(poly3_forecast).applymap('{:.2f}'.format)
    y_pred_poly3['day_num'] = [i.day for i in poly3_forecast.index]
    y_pred_poly3['month_num'] = [i.month for i in poly3_forecast.index]
    y_pred_poly3['year_num'] = [i.year for i in poly3_forecast.index]
    y_pred_poly3['date'] = y_pred_poly3['year_num'].astype(str) + '-' + y_pred_poly3['month_num'].astype(str) + '-' + y_pred_poly3['day_num'].astype(str)
    y_pred_poly3['date'] = pd.DatetimeIndex(y_pred_poly3['date'], freq='D')

    #Create MAPE
    poly3_mape = mean_absolute_percentage_error(y_test['condensate'], poly3_forecast)
    poly3_mape_100 = 100*poly3_mape
    poly3_mape_str = str('MAPE: %.4f' % poly3_mape_100) + '%'

    logging.info("Polynomial Regression Orde 3 Model "+poly3_mape_str)

    # Rename column to forecast_h
    y_pred_poly3.rename(columns={'condensate':'forecast_h'}, inplace=True)
    
    # %%
    # Join prediction data frame
    logging.info("Creating all model prediction result data frame ..")
    y_all_pred = pd.concat([y_pred_arimax[['forecast_a']], 
                            y_pred_sarimax[['forecast_b']], 
                            y_pred_prophet[['forecast_c']], 
                            y_pred_ranfor[['forecast_d']], 
                            y_pred_xgb[['forecast_e']], 
                            y_pred_linreg[['forecast_f']], 
                            y_pred_poly2[['forecast_g']], 
                            y_pred_poly3[['forecast_h']]
                           ], axis=1)

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
    #plt.show()

    #%%
    #CREATE DATAFRAME MAPE
    mape_data_condensate =  {'arimax': [arimax_mape_100/100],
                'sarimax': [sarimax_mape_100/100],
                'prophet': [prophet_mape_100/100],
                'random_forest': [ranfor_mape_100/100],
                'xgboost': [xgb_mape_100/100],
                'linear_regression': [linreg_mape_100/100],
                'polynomial_degree_2': [poly2_mape_100/100],
                'polynomial_degree_3': [poly3_mape_100/100]}

    all_mape_condensate = pd.DataFrame(mape_data_condensate)
    #all_mape_condensate
    
    # Save forecast result to database
    logging.info("Updating forecast result to database ...")
    total_updated_rows = insert_forecast(conn, y_all_pred)
    logging.info("Updated rows: {}".format(total_updated_rows))

# %%
# %%
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
    created_by = 'python'
    
    """ insert forecasting result after last row in table """
    sql = """ UPDATE lng_condensate_daily
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
