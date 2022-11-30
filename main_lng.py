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

#In sampel
import gas_prod.insample.feed_gas_tangguh_with_planned_shutdown_exog_all_method as feed_gas_tangguh
#In sampel
import gas_prod.insample.condensate_tangguh_with_planned_shutdown_exog_all_method as condensate_tangguh
#Out sampel
import gas_prod.forecasting.feed_gas_tangguh_forecasting as feed_gas_tangguh_forecasting
#import gas_prod.lng_tangguh as lng_tangguh
#Out sampel
import gas_prod.forecasting.condensate_tangguh_forecasting as condensate_tangguh_forecasting

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

t0 = time.process_time()
#feed_gas_tangguh.main()
#condensate_tangguh.main()
feed_gas_tangguh_forecasting.main()
t1 = time.process_time()
exec_time = format_timespan(t1-t0, max_units=3)
print("Forecasting Feed gas Tangguh : " + exec_time)

t2 = time.process_time()
condensate_tangguh_forecasting.main()
t3 = time.process_time()
exec_time = format_timespan(t1-t0, max_units=3)
print("Forecasting Condensate Tangguh : " + exec_time)

total_exec_time = format_timespan(t3-t0, max_units=3)
print("Total execution time : " + total_exec_time)

exit()

# def main():
#     #print()
#     feed_gas_tangguh.main()
    
# if __name__ == "__main__":
#     #main(sys.argv[1], sys.argv[2], sys.argv[3])
#     main()
