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

import hse.insample.incident_rate_monthly_cumulative_insample as ir_monthly_cum_insample
import hse.insample.yearly_incident_rate_insample as ir_yearly_insample

# adding gas prod to the system path
sys.path.insert(0, './gas_prod')
sys.path.insert(0, './hse')
sys.path.insert(0, './hse/insample')

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
configLogging("main_hse_insample.log")

#t0 = time.process_time()
#ir_monthly_cum_insample.main()
#t1 = time.process_time()
#exec_time = format_timespan(t1-t0, max_units=3)
#print("Forecasting incident rate monthly cumulative : " + exec_time)

#print('\n')

t2 = time.process_time()
ir_yearly_insample.main()
t3 = time.process_time()
exec_time = format_timespan(t3-t2, max_units=3)
print("Forecasting incident rate yearly : " + exec_time)

print('\n')

#total_exec_time = format_timespan(t3-t0, max_units=3)
#print("Total execution time : " + total_exec_time)

exit()

# def main():
#     #print()
#     feed_gas_tangguh.main()
    
# if __name__ == "__main__":
#     #main(sys.argv[1], sys.argv[2], sys.argv[3])
#     main()
