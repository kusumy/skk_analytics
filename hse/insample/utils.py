import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np 
import pandas as pd

from datetime import date, datetime, timedelta
from dateutil.relativedelta import *

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

def configLogging(filename="log.log"):
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG, 
        format="%(asctime)s %(message)s", 
        filename=filename,
        filemode="w" #, 
        #handlers=[
        #    logging.FileHandler("condensate_tangguh.log"),
        #    logging.StreamHandler(sys.stdout) ]
    ) 

    logger = logging.getLogger('cmdstanpy')
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)

def logMessage(messages):
    import sys
    
    logging.info(messages)
    print(messages, file=sys.stdout)
    #sys.stdout.write(messages)
    
def ad_test(dataset):
    from statsmodels.tsa.stattools import adfuller
    
    dftest = adfuller(dataset, autolag = 'AIC')
    logMessage("1. ADF : {}".format(dftest[0]))
    logMessage("2. P-Value : {}".format(dftest[1]))
    logMessage("3. Num Of Lags : {}".format(dftest[2]))
    logMessage("4. Num Of Observations Used For ADF Regression: {}".format(dftest[3]))
    logMessage("5. Critical Values : ")
    for key, val in dftest[4].items():
        logMessage("\t {}: {}".format(key, val))
    return dftest
        
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

def check_stationarity(df):
    kps = kpss(df)
    adf = adfuller(df)

    kpss_pv, adf_pv = kps[1], adf[1]
    kpssh, adfh = 'Stationary', 'Non-stationary'

    if adf_pv < 0.05:
        # Reject ADF Null Hypothesis
        adfh = 'Stationary'
    if kpss_pv < 0.05:
        # Reject KPSS Null Hypothesis
        kpssh = 'Non-stationary'
    return (kpssh, adfh)

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
    
    return plt

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

def get_last_date_of_current_year():
    return datetime.now().date().replace(month=12, day=31).strftime("%Y-%m-%d")

def end_day_forecast_april():
    return (datetime.now().date() + timedelta(days=365)).replace(month=4, day=30).strftime("%Y-%m-%d")

def get_first_date_of_november():
    return datetime.now().date().replace(month=11, day=1).strftime("%Y-%m-%d")