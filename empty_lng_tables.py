import argparse 
import configparser
import itertools
import logging
import ast
import arrow
import sys
import psycopg2

import warnings
warnings.filterwarnings('ignore')

def create_db_connection():
    # Read database configuration INI
    config = configparser.ConfigParser()
    config.read('database.ini')
    postgresql = config['postgresql_ml_lng_skk']
    host = postgresql['host']
    dbname = postgresql['database']
    user = postgresql['user']
    password = postgresql['password']
    port = int(postgresql['port'])
    
    try:
        # connect to the PostgreSQL database
        conn = psycopg2.connect(
            host = host, 
            dbname = dbname, 
            user = user, 
            password = password, 
            port = port)
        
        logging.info("Database connected ...")
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        #logging.error(error)
        return None

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
        UPDATE lng_feed_gas_daily
        SET forecast_a = null, 
            forecast_b = null, 
            forecast_c = null, 
            forecast_d = null, 
            forecast_e = null, 
            forecast_f = null, 
            forecast_g = null, 
            forecast_h = null, 
            updated_at = null, 
            updated_by = null
       """

sql2 = """
        UPDATE lng_production_daily
        SET forecast_a = null, 
            forecast_b = null, 
            forecast_c = null, 
            forecast_d = null, 
            forecast_e = null, 
            forecast_f = null, 
            forecast_g = null, 
            forecast_h = null, 
            updated_at = null, 
            updated_by = null
       """

sql3 = """
        UPDATE lng_condensate_daily
        SET forecast_a = null, 
            forecast_b = null, 
            forecast_c = null, 
            forecast_d = null, 
            forecast_e = null, 
            forecast_f = null, 
            forecast_g = null, 
            forecast_h = null, 
            updated_at = null, 
            updated_by = null
       """

sql4 = """
        UPDATE lng_lpg_c3_daily
        SET forecast_a = null, 
            forecast_b = null, 
            forecast_c = null, 
            forecast_d = null, 
            forecast_e = null, 
            forecast_f = null, 
            forecast_g = null, 
            forecast_h = null, 
            updated_at = null, 
            updated_by = null
       """

sql5 = """
        UPDATE lng_lpg_c4_daily
        SET forecast_a = null, 
            forecast_b = null, 
            forecast_c = null, 
            forecast_d = null, 
            forecast_e = null, 
            forecast_f = null, 
            forecast_g = null, 
            forecast_h = null, 
            updated_at = null, 
            updated_by = null
       """

# Add the arguments to the parser
#ap = argparse.ArgumentParser()
#ap.add_argument("-y", "--year", required=True, help="Year data to truncate")

# Connect to database
# Exit program if not connected to database
conn = create_db_connection()
if conn == None:
    exit()
else:
    updated_rows = execute_sql(conn, sql1)
    if updated_rows != None:
        print("lng_feed_gas_daily table has been emptied")
    updated_rows = execute_sql(conn, sql2)
    if updated_rows != None:
        print("lng_production_daily table has been emptied")
    updated_rows = execute_sql(conn, sql3)
    if updated_rows != None:
        print("lng_condensate_daily table has been emptied")
    updated_rows = execute_sql(conn, sql4)
    if updated_rows != None:
        print("lng_lpg_c3_daily table has been emptied")
    updated_rows = execute_sql(conn, sql5)
    if updated_rows != None:
        print("lng_lpg_c4_daily table has been emptied")