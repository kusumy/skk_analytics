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
    postgresql = config['postgresql']
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
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        #logging.error(error)
        return None

sql1 = '''
        UPDATE hse_analytics_trir_monthly
        SET forecast_a = null, 
            forecast_b = null, 
            forecast_c = null, 
            forecast_d = null, 
            forecast_e = null, 
            forecast_f = null
       '''

sql2 = '''
        UPDATE hse_analytics_trir_monthly_cum
        SET forecast_a = null, 
            forecast_b = null, 
            forecast_c = null, 
            forecast_d = null, 
            forecast_e = null, 
            forecast_f = null
       '''

sql3 = '''
        UPDATE hse_analytics_trir_yearly
        SET forecast_a = null, 
            forecast_b = null, 
            forecast_c = null, 
            forecast_d = null, 
            forecast_e = null, 
            forecast_f = null
       '''

# Add the arguments to the parser
#ap = argparse.ArgumentParser()
#ap.add_argument("-y", "--year", required=True, help="Year data to truncate")

# Connect to database
# Exit program if not connected to database
conn = create_db_connection()
if conn == None:
    exit()
else:
    execute_sql(conn, sql1)
    print("hse_analytics_trir_monthly table has been emptied")
    #execute_sql(conn, sql2)
    #print("hse_analytics_trir_monthly_cum table has been emptied")
    #execute_sql(conn, sql3)
    #print("hse_analytics_trir_yearly table has been emptied")