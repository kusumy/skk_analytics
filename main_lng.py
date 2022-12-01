# Construct the argument parser
import argparse 
import itertools
import logging
import ast
import arrow
import sys
import time
from humanfriendly import format_timespan

import warnings
warnings.filterwarnings('ignore')


#import gas_prod.forecasting.feed_gas_tangguh_forecasting as feed_gas_tangguh_forecasting
import gas_prod.forecasting.lng_production_tangguh_forecasting as lng_tangguh_forecasting
#import gas_prod.forecasting.condensate_tangguh_forecasting as condensate_tangguh_forecasting

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

#t0 = time.process_time()
#feed_gas_tangguh_forecasting.main()
#t1 = time.process_time()
#exec_time = format_timespan(t1-t0, max_units=3)
#print("Forecasting Feed Gas Tangguh : " + exec_time)

#print('\n')

t2 = time.process_time()
lng_tangguh_forecasting.main()
t3 = time.process_time()
exec_time = format_timespan(t3-t2, max_units=3)
print("Forecasting LNG Production Tangguh : " + exec_time)

#print('\n')

#t4 = time.process_time()
#condensate_tangguh_forecasting.main()
#t5 = time.process_time()
#exec_time = format_timespan(t5-t4, max_units=3)
#print("Forecasting Condensate Tangguh : " + exec_time)

#print('\n')

#total_exec_time = format_timespan(t3-t0, max_units=3)
#print("Total execution time : " + total_exec_time)

exit()

# def main():
#     #print()
#     feed_gas_tangguh.main()
    
# if __name__ == "__main__":
#     #main(sys.argv[1], sys.argv[2], sys.argv[3])
#     main()
