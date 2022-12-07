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

#import gas_prod.insample.feed_gas_tangguh_forecasting_insample as feed_gas_tangguh_insample
import gas_prod.insample.condensate_tangguh_forecasting_insample as condensate_tangguh_insample
#import gas_prod.insample.feed_gas_badak_forecasting_insample as feed_gas_badak_insample

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

#feed_gas_tangguh_insample.main()
#t1 = time.process_time()
#exec_time = format_timespan(t1-t0, max_units=3)
#print("Forecasting Feed Gas BP Tangguh : " + exec_time)

#print('\n')

t2 = time.process_time()
condensate_tangguh_insample.main()
t3 = time.process_time()
exec_time = format_timespan(t3-t2, max_units=3)
print("Forecasting Condensate Tangguh : " + exec_time)

#print('\n')

#t4 = time.process_time()
#feed_gas_badak_insample.main()
#t5 = time.process_time()
#exec_time = format_timespan(t5-t4, max_units=3)
#print("Forecasting Feed gas PT Badak : " + exec_time)

#total_exec_time = format_timespan(t5-t0, max_units=3)
#print("Total execution time : " + total_exec_time)

exit()

# def main():
#     #print()
#     feed_gas_tangguh.main()
    
# if __name__ == "__main__":
#     #main(sys.argv[1], sys.argv[2], sys.argv[3])
#     main()
