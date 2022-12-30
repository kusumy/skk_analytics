# Construct the argument parser
import argparse 
import itertools
import logging
import ast
import arrow
import sys
import time
from humanfriendly import format_timespan
from utils import configLogging, logMessage, ad_test

import warnings
warnings.filterwarnings('ignore')


#import gas_prod.forecasting.feed_gas_tangguh_forecasting as feed_gas_tangguh_forecasting
#import gas_prod.forecasting.lng_production_tangguh_forecasting as lng_tangguh_forecasting
#import gas_prod.forecasting.condensate_tangguh_forecasting as condensate_tangguh_forecasting
import gas_prod.forecasting.lng_prod_badak_forecasting as lng_badak_forecasting
#import gas_prod.forecasting.feed_gas_badak_forecasting as feed_gas_badak_forecasting
#import gas_prod.forecasting.condensate_badak_forecasting as condensate_badak_forecasting
#import gas_prod.forecasting.c3_badak_forecasting as c3_badak_forecasting
#import gas_prod.forecasting.c4_badak_forecasting as c4_badak_forecasting

# adding gas prod to the system path
sys.path.insert(0, './gas_prod')
sys.path.insert(0, './hse')

# Add the arguments to the parser
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--startdate", required=True, help="Start date test data")
ap.add_argument("-e", "--enddate", required=True, help="End date test data")

# Get argument
#args = vars(ap.parse_args())
#startDate = str(args['startdate'])
#endDate = str(args['enddate'])
# do whatever the script does

# Configure logging
configLogging("main_lng.log")
    
# Forecasting Feed Gas BP Tangguh ...
#t0 = time.process_time()
#feed_gas_tangguh_forecasting.main()
#t1 = time.process_time()
#exec_time = format_timespan(t1-t0, max_units=3)
#logMessage("Forecasting Feed Gas BP Tangguh : " + exec_time)

#logMessage('\n')

#"Forecasting Feed Gas PT Badak ...
#t2 = time.process_time()
#feed_gas_badak_forecasting.main() 
#t3 = time.process_time()
#exec_time = format_timespan(t3-t2, max_units=3)
#logMessage("Forecasting Feed Gas PT Badak : " + exec_time)

#logMessage('\n')

# Forecasting LNG Production BP Tangguh ...
#t4 = time.process_time()
#lng_tangguh_forecasting.main() 
#t5 = time.process_time()
#exec_time = format_timespan(t5-t4, max_units=3)
#logMessage("Forecasting LNG Production BP Tangguh : " + exec_time)

#logMessage('\n')

# Forecasting LNG Production PT Badak ...
t6 = time.process_time()
lng_badak_forecasting.main()
t7 = time.process_time()
exec_time = format_timespan(t7-t6, max_units=3)
logMessage("Forecasting LNG Production PT Badak : " + exec_time)

logMessage('\n')

# Forecasting Condensate BP Tangguh ...
#t8 = time.process_time()
#condensate_tangguh_forecasting.main()
#t9 = time.process_time()
#exec_time = format_timespan(t9-t8, max_units=3)
#logMessage("Forecasting Condensate BP Tangguh : " + exec_time)

#logMessage('\n')

# Forecasting Condensate PT Badak ...
#t10 = time.process_time()
#condensate_badak_forecasting.main()
#t11 = time.process_time()
#exec_time = format_timespan(t11-t10, max_units=3)
#logMessage("Forecasting Condensate PT Badak : " + exec_time)

#logMessage('\n')

# Forecasting C3 PT Badak ...
#t12 = time.process_time()
#c3_badak_forecasting.main()
#t13 = time.process_time()
#exec_time = format_timespan(t13-t12, max_units=3)
#logMessage("Forecasting C3 PT Badak : " + exec_time)

#logMessage('\n')

# Forecasting C4 PT Badak ...
#t14 = time.process_time()
#c4_badak_forecasting.main()
#t15 = time.process_time()
#exec_time = format_timespan(t15-t14, max_units=3)
#logMessage("Forecasting C4 PT Badak : " + exec_time)

#logMessage('\n')

#total_exec_time = format_timespan(t15-t0, max_units=3)
#logMessage("Total execution time : " + total_exec_time)

exit()

# def main():
#     #print()
#     feed_gas_tangguh.main()
    
# if __name__ == "__main__":
#     #main(sys.argv[1], sys.argv[2], sys.argv[3])
#     main()
