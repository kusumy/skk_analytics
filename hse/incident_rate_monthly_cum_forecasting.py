# %%
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
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
plt.style.use('fivethirtyeight')

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
#Load Data from Database
query_1 = open("query_month_cum.sql", mode="rt").read()

data = retrieve_data(query_1)
data['year_num'] = data['year_num'].astype(int)
data['month_num'] = data['month_num'].astype(int)

data['date'] = data['year_num'].astype(str) + '-' + data['month_num'].astype(str)
data

#%%
# Prepare data
data['date'] = pd.to_datetime(data['date'], format='%Y-%m')
data = data[~(data['date'] > '2022-09')]
data = data.rename(columns=str.lower)

data['date'] = pd.PeriodIndex(data['date'], freq='M')
data = data.drop([data.index[0], data.index[1]])
data = data.reset_index()
data.head()

#%%
ds = 'date'
y = 'trir_cum' 

df = data[[ds,y]]
df = df.set_index(ds)
df.index = pd.PeriodIndex(df.index, freq='M')
df

#%%
train_df = df['trir_cum']
train_df

#stationary check
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
from sktime.forecasting.compose import make_reduction

#%%
# Create forecasting Horizon
time_predict = pd.period_range('2022-10', periods=3, freq='M')
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
train_exog = df.iloc[:,1:]

#%%
#import exogenous for predict
query_2 = open("query_month_cum3.sql", mode="rt").read()
data2 = retrieve_data(query_2)
data2['year_num'] = data2['year_num'].astype(int)
data2['month_num'] = data2['month_num'].astype(int)
data2['date'] = data2['year_num'].astype(str) + '-' + data2['month_num'].astype(str)
# Prepare data
data2['date'] = pd.to_datetime(data2['date'], format='%Y-%m')

#%%
test_exog = data2[['date', 'drilling_explor_cum', 'drilling_explot_cum', 'workover_cum',
                   'wellservice_cum', 'survei_seismic_cum']].copy()
test_exog = test_exog.set_index(test_exog['date'])
test_exog.index = pd.PeriodIndex(test_exog.index, freq='M')
test_exog.drop(['date'], axis=1, inplace=True)
test_exog = test_exog.iloc[-3:]
test_exog['bulan'] = [i.month for i in test_exog.index]

# %%
##### ARIMAx MODEL #####
import pmdarima as pm
arimax_model = pm.auto_arima(y=df['trir_cum'], X=train_exog, start_p=0, d=1, start_q=0, 
                    max_p=10, max_d=0, max_q=10,
                    m=0, seasonal=False, error_action='warn',trace=True,
                    supress_warnings=True,stepwise=True, stationary=False)
arimax_model.summary()

arimax_model.fit(df['trir_cum'], X=train_exog)
arimax_forecast = arimax_model.predict(n_periods=len(fh), X=test_exog)
y_pred_arimax = pd.DataFrame(arimax_forecast).applymap('{:,.2f}'.format)
y_pred_arimax['month_num'] = [i.month for i in arimax_forecast.index]
y_pred_arimax['year_num'] = [i.year for i in arimax_forecast.index]


#%%
##### XGBOOST MODEL #####
# Create model
from xgboost import XGBRegressor
from sktime.forecasting.compose import make_reduction

#Set Parameter
xgb_objective = 'reg:squarederror'
xgb_lags = 19
xgb_strategy = "recursive"

# Create regressor object
xgb_regressor = XGBRegressor(objective=xgb_objective)
xgb_forecaster = make_reduction(xgb_regressor, window_length=xgb_lags, strategy=xgb_strategy)
xgb_forecaster.fit(train_df, X=train_exog)

# Create forecasting
xgb_forecast = xgb_forecaster.predict(fh, X=test_exog)
y_pred_xgb = pd.DataFrame(xgb_forecast).applymap('{:,.2f}'.format)
y_pred_xgb['month_num'] = [i.month for i in xgb_forecast.index]
y_pred_xgb['year_num'] = [i.year for i in xgb_forecast.index]


#%%
##### RANDOM FOREST MODEL #####
# Create model
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.compose import make_reduction

#Set Parameter
ranfor_n_estimators = 100
random_state = 0
ranfor_criterion = "squared_error"
ranfor_lags = 18
ranfor_strategy = "recursive"

# create regressor object
ranfor_regressor = RandomForestRegressor(n_estimators = ranfor_n_estimators, random_state=random_state, criterion=ranfor_criterion)
ranfor_forecaster = make_reduction(ranfor_regressor, window_length= ranfor_lags, strategy=ranfor_strategy)
ranfor_forecaster.fit(train_df, X=train_exog)

# Create forecasting
ranfor_forecast = ranfor_forecaster.predict(fh, X=test_exog)
y_pred_ranfor = pd.DataFrame(ranfor_forecast).applymap('{:,.2f}'.format)
y_pred_ranfor['month_num'] = [i.month for i in ranfor_forecast.index]
y_pred_ranfor['year_num'] = [i.year for i in ranfor_forecast.index]


#%%
##### LINEAR REGRESSION MODEL #####
# Create model
from sklearn.linear_model import LinearRegression
from sktime.forecasting.compose import make_reduction

#Set parameter
linreg_normalize = True
linreg_lags = 0.9
linreg_strategy = "recursive"

# create regressor object
linreg_regressor = LinearRegression(normalize=linreg_normalize)
linreg_forecaster = make_reduction(linreg_regressor, window_length=linreg_lags, strategy=linreg_strategy)
linreg_forecaster.fit(train_df, X=train_exog)
linreg_forecast = linreg_forecaster.predict(fh, X=test_exog)
y_pred_linreg = pd.DataFrame(linreg_forecast).applymap('{:,.2f}'.format)
y_pred_linreg['month_num'] = [i.month for i in linreg_forecast.index]
y_pred_linreg['year_num'] = [i.year for i in linreg_forecast.index]


#%%
##### POLYNOMIAL REGRESSION DEGREE=2 MODEL #####
# Create model
from polyfit import PolynomRegressor, Constraints
from sktime.forecasting.compose import make_reduction

#Set parameter
poly2_regularization = None
poly2_interactions= False
poly2_lags = 0.95
poly2_strategy = "recursive"

# Create regressor object
poly2_regressor = PolynomRegressor(deg=2, regularization=poly2_regularization, interactions=poly2_interactions)
poly2_forecaster = make_reduction(poly2_regressor, window_length=poly2_lags, strategy=poly2_strategy) #WL=0.9 (degree 2), WL=0.7 (degree 3)
poly2_forecaster.fit(train_df, X=train_exog)
poly2_forecast = poly2_forecaster.predict(fh, X=test_exog)
y_pred_poly2 = pd.DataFrame(poly2_forecast).applymap('{:,.2f}'.format)
y_pred_poly2['month_num'] = [i.month for i in poly2_forecast.index]
y_pred_poly2['year_num'] = [i.year for i in poly2_forecast.index]


#%%
##### POLYNOMIAL REGRESSION DEGREE=3 MODEL #####
# Create model
from polyfit import PolynomRegressor, Constraints
from sktime.forecasting.compose import make_reduction

#Set parameter
poly3_regularization = None
poly3_interactions= False
poly3_lags = 0.94
poly3_strategy = "recursive"

# Create regressor object
poly3_regressor = PolynomRegressor(deg=3, regularization=poly3_regularization, interactions=poly3_interactions)
poly3_forecaster = make_reduction(poly3_regressor, window_length=poly3_lags, strategy=poly3_strategy) #WL=0.9 (degree 2), WL=0.7 (degree 3)
poly3_forecaster.fit(train_df, X=train_exog)
poly3_forecast = poly3_forecaster.predict(fh, X=test_exog)
y_pred_poly3 = pd.DataFrame(poly3_forecast).applymap('{:,.2f}'.format)
y_pred_poly3['month_num'] = [i.month for i in poly3_forecast.index]
y_pred_poly3['year_num'] = [i.year for i in poly3_forecast.index]


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
plt.show()

#%%
import psycopg2

def update_trir(forecast, year_num, month_num):
    """ update vendor name based on the vendor id """
    sql = """ UPDATE hse_analytics_trir_monthly_cum
                SET forecast_a = %s
                WHERE year_num = %s
                AND month_num = %s"""
    conn = None
    updated_rows = 0
    try:
        # connect to the PostgreSQL database
        conn = psycopg2.connect(
            host = '188.166.239.112',
            dbname = 'skk_da_hse_analytics',
            user = 'postgres',
            password = 'rahasia',
            port = 5432)
        # create a new cursor
        cur = conn.cursor()
        # execute the UPDATE  statement
        cur.execute(sql, (forecast, year_num, month_num))
        # get the number of updated rows
        updated_rows = cur.rowcount
        # Commit the changes to the database
        conn.commit()
        # Close communication with the PostgreSQL database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

    return updated_rows

#%%
for index, row in y_pred.iterrows():
    year_num = row['year_num']
    month_num = row['month_num']
    forecast = row[0]
  
    #sql = f'UPDATE trir_monthly_test SET forecast_a = {} WHERE year_num = {} AND month_num = {}'.format(forecast, year_num, month_num)
    update_trir(forecast, year_num, month_num)
# %%
