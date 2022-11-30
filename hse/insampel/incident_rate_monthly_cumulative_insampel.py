#%%
#Import library
import pandas as pd
import numpy as np
import plotly.express as px
from pmdarima.arima.auto import auto_arima
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from connection import config, retrieve_data

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
#Load Data from Database
query_1 = open("query_month_cum.sql", mode="rt").read()
data = retrieve_data(query_1)
data['year_num'] = data['year_num'].astype(int)
data['month_num'] = data['month_num'].astype(int)
data['date'] = data['year_num'].astype(str) + '-' + data['month_num'].astype(str)
data

# %%
# Prepare data
data['date'] = pd.to_datetime(data['date'], format='%Y-%m')
data = data[~(data['date'] > '2022-09')]
data = data.rename(columns=str.lower)

data['date'] = pd.PeriodIndex(data['date'], freq='M')
data = data.reset_index()
data.head()

# %%
ds = 'date'
y = 'trir_cum' 

df = data[[ds,y]]
df = df.set_index(ds)
df.index = pd.PeriodIndex(df.index, freq='M')
df

#%%
stationarity_check(df.to_timestamp())

#%%
decomposition_plot(df.to_timestamp())

#%%
plot_acf_pacf(df.to_timestamp())

#%%
from chart_studio.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df.trir_cum.values, model="additive", period=24)
fig = result.plot()
plt.show()

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
ad_test(df['trir_cum'])

#%%
from sktime.utils.plotting import plot_series
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.compose import make_reduction

#%%
# Test size
test_size = 0.07

# Split data
y_train, y_test = temporal_train_test_split(df, test_size=test_size)
y_test.head(10)

#%%
# Create forecasting Horizon
fh = ForecastingHorizon(y_test.index, is_relative=False)
fh

#%%
## Create Exogenous Variable
#df['jam_kerja_cum'] = data['jam_kerja_cum'].values
#df['freq_rate_cum'] = data['freq_rate_cum'].values
df['bor_eksplorasi_cum'] = data['bor_eksplorasi_cum'].values
df['bor_eksploitasi_cum'] = data['bor_eksploitasi_cum'].values
df['workover_cum'] = data['workover_cum'].values
df['wellservice_cum'] = data['wellservice_cum'].values
df['survey_seismic_cum'] = data['survey_seismic_cum'].values
df['bulan'] = [i.month for i in df.index]
df.tail(20)

#%%
# Split into train and test
X_train, X_test = temporal_train_test_split(df.iloc[:,1:], test_size=test_size)
X_train.tail()

#%%
# plotting for illustration
plt.figure(figsize=(20,8))
plt.plot(y_train.to_timestamp(), label='train')
plt.plot(y_test.to_timestamp(), label='test')
plt.ylabel("Incident Rate Cumulative")
plt.xlabel("Datestamp")
plt.legend(loc='best')
plt.show()

# %%
##### FORECASTING #####
from sklearn.metrics import mean_absolute_percentage_error

##### ARIMAX MODEL #####
import pmdarima as pm

#Set parameters
arimax_seasonal = False
arimax_error_action = 'warn'
arimax_trace = True
arimax_supress_warnings = True
arimax_stepwise = True
arimax_stationary = False

# Create ARIMAX Model
arimax_model = pm.auto_arima(y=y_train, X=X_train, start_p=0, d=1, start_q=0, 
                    max_p=10, max_d=0, max_q=10,
                    m=0, seasonal=False, error_action='warn',trace=True,
                    supress_warnings=True,stepwise=True, stationary=False)
#arimax_model.summary()
#print("arimax order is : ", arimax_model.order)

arimax_model.fit(y_train, X=X_train)
arimax_forecast = arimax_model.predict(n_periods=len(fh), X=X_test)
y_pred_arimax = pd.DataFrame(arimax_forecast).applymap('{:,.2f}'.format)
y_pred_arimax['month_num'] = [i.month for i in arimax_forecast.index]
y_pred_arimax['year_num'] = [i.year for i in arimax_forecast.index]
#Rename colum 0
y_pred_arimax.rename(columns={0:'forecast_a'}, inplace=True)

mape_arimax = mean_absolute_percentage_error(y_test, arimax_forecast)


##### XGBOOST MODEL #####
from xgboost import XGBRegressor

#Set parameters
xgb_objective = 'reg:squarederror'
xgb_lags = 19
xgb_strategy = "recursive"

# Create regressor object
xgb_regressor = XGBRegressor(objective=xgb_objective)
xgb_forecaster = make_reduction(xgb_regressor, window_length=xgb_lags, strategy=xgb_strategy)
xgb_forecaster.fit(y_train, X=X_train)

xgb_forecast = xgb_forecaster.predict(fh, X=X_test)
y_pred_xgb = pd.DataFrame(xgb_forecast).applymap('{:,.2f}'.format)
y_pred_xgb['month_num'] = [i.month for i in xgb_forecast.index]
y_pred_xgb['year_num'] = [i.year for i in xgb_forecast.index]
#Rename colum 0
y_pred_xgb.rename(columns={'trir_cum':'forecast_b'}, inplace=True)

# Calculate model performance
mape_xgb = mean_absolute_percentage_error(y_test, xgb_forecast)


##### RANDOM FOREST MODEL #####
from sklearn.ensemble import RandomForestRegressor

#Set parameters
ranfor_n_estimators = 100
ranfor_random_state = 0
ranfor_criterion = "squared_error"
ranfor_lags = 19
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
y_pred_ranfor.rename(columns={'trir_cum':'forecast_c'}, inplace=True)

mape_ranfor = mean_absolute_percentage_error(y_test, ranfor_forecast)


##### LINEAR REGRESSION MODEL #####
from sklearn.linear_model import LinearRegression

#Set parameters
linreg_normalize = True
linreg_lags = 0.9
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
y_pred_linreg.rename(columns={'trir_cum':'forecast_d'}, inplace=True)

mape_linreg = mean_absolute_percentage_error(y_test, linreg_forecast)


##### POLYNOMIAL REGRESSION DEGREE=2 MODEL #####
from polyfit import PolynomRegressor, Constraints

#Set parameters
poly2_regularization = None
poly2_interactions = False
poly2_lags = 0.95
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
y_pred_poly2.rename(columns={'trir_cum':'forecast_e'}, inplace=True)

mape_poly2 = mean_absolute_percentage_error(y_test, poly2_forecast)


##### POLYNOMIAL REGRESSION DEGREE=3 MODEL #####
from polyfit import PolynomRegressor, Constraints

#Set parameters
poly3_regularization = None
poly3_interactions = False
poly3_lags = 0.94
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
y_pred_poly3.rename(columns={'trir_cum':'forecast_f'}, inplace=True)

mape_poly3 = mean_absolute_percentage_error(y_test, poly3_forecast)


# %%
#logMessage("Creating all model prediction result data frame ...")
y_all_pred = pd.concat([y_pred_arimax[['forecast_a']], 
                        y_pred_xgb[['forecast_b']], 
                        y_pred_ranfor[['forecast_c']], 
                        y_pred_linreg[['forecast_d']], 
                        y_pred_poly2[['forecast_e']], 
                        y_pred_poly3[['forecast_f']]], axis=1)

# %%
#Create Dataframe Mape All Method
all_mape_pred = {'arimax' : [mape_arimax],
                 'xgboost' : [mape_xgb],
                 'random_forest' : [mape_ranfor],
                 'linear_regression' : [mape_linreg],
                 'polynomial_degree_2' : [mape_poly2],
                 'polynomial_degree_3' : [mape_poly3]}
all_mape_pred = pd.DataFrame(all_mape_pred)
all_mape_pred