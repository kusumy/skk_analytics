
import psycopg2

import warnings
warnings.filterwarnings('ignore')

import sys
 
# setting path
sys.path.append('.')

from connection import create_db_connection

def execute_sql(conn, sql):
    # Create a new cursor
    try:
        cur = conn.cursor()
        cur.execute(sql)
        updated_rows = cur.rowcount
        # Commit the changes to the database
        conn.commit()
        # Close cursor
        cur.close()
        return updated_rows
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        #logging.error(error)
        return None
    
sql1 = """
        INSERT INTO public.hse_analytics_mape
        ( type_id,
          created_at, 
          created_by)
        VALUES(1, '2022-12-02 00:00:00', 'PYTHON');

        INSERT INTO public.hse_analytics_mape
        ( type_id,
          created_at, 
          created_by)
        VALUES(2, '2022-12-02 00:00:00', 'PYTHON');
        
       """


# Add the arguments to the parser
#ap = argparse.ArgumentParser()
#ap.add_argument("-y", "--year", required=True, help="Year data to truncate")

# Connect to database
# Exit program if not connected to database
conn = create_db_connection(section='postgresql_ml_hse_skk')
if conn == None:
    exit()
else:
    updated_rows = execute_sql(conn, sql1)
    if updated_rows != None:
        print("Initial rows has been inserted")
