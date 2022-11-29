# Construct the argument parser
import argparse 
import itertools
import logging
import ast
import arrow
import sys

import warnings
warnings.filterwarnings('ignore')

import gas_prod.feed_gas_tangguh_with_planned_shutdown_exog_all_method as feed_gas_tangguh
import gas_prod.condensate_tangguh_with_planned_shutdown_exog_all_method as condensate_tangguh
import gas_prod.feed_gas_tangguh_forecasting as feed_gas_tangguh_forecasting
#import gas_prod.lng_tangguh as lng_tangguh

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

feed_gas_tangguh_forecasting.main()
#feed_gas_tangguh.main()
#condensate_tangguh.main()
exit()

# def main():
#     #print()
#     feed_gas_tangguh.main()
    
# if __name__ == "__main__":
#     #main(sys.argv[1], sys.argv[2], sys.argv[3])
#     main()
