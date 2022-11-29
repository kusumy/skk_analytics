import logging
import os

def configLogging(filename="log.log"):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(message)s", 
        filename=filename,
        filemode="w" #, 
        #handlers=[
        #    logging.FileHandler("condensate_tangguh.log"),
        #    logging.StreamHandler(sys.stdout) ]
    ) 

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
    logMessage("3. Num Of Lags {}: ".format(dftest[2]))
    logMessage("4. Num Of Observations Used For ADF Regression: {}".format(dftest[3]))
    logMessage("5. Critical Values : ")
    for key, val in dftest[4].items():
        logMessage("\t {}: {}".format(key, val))