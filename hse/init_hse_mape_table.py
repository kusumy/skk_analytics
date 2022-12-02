
import psycopg2

import warnings
warnings.filterwarnings('ignore')


import sys
 
# setting path
sys.path.append('..')

from connection import exe

sql1 = """
        INSERT INTO public.hse_analytics_mape
            (type_id, 
            created_at, 
            created_by)
        VALUES(1, '2022-12-02 00:00:00', 'PYTHON');

        INSERT INTO public.hse_analytics_mape
            (type_id, 
            created_at, 
            created_by)
        VALUES(2, '2022-12-02 00:00:00', 'PYTHON');
        
       """


# Add the arguments to the parser
#ap = argparse.ArgumentParser()
#ap.add_argument("-y", "--year", required=True, help="Year data to truncate")

# Connect to database
# Exit program if not connected to database
conn = create_db_connection(section='postgresql_ml_hse')
if conn == None:
    exit()
else:
    updated_rows = execute_sql(conn, sql1)
    if updated_rows != None:
        print("Initial rows has been inserted")
