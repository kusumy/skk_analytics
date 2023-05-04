#%%
#Import library
import pandas as pd
import numpy as np
import os
import psycopg2
import psycopg2.extras as extras
from io import StringIO
from pandas.io.sql import execute
from pathlib import Path

from configparser import ConfigParser

def config(filename='database.ini', section='postgresql'):
    """
        database configuration parser from file that contains database connection
        Input:
            - filename: path to config file
            - section: section of config file that contains desired database connection
        Output:
            - db: key-value pairs of database configuration
    """
    # create a parser
    parser = ConfigParser()

    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception(
            'Section {0} not found in the {1} file'.format(section, filename))

    return db

def retrieve_data(sql=None, db_connection='database.ini', section='postgresql'):
    params = config(filename=db_connection, section=section)
    with psycopg2.connect(**params) as conn:
        if sql == None:
            sql = "select now();"
        data = pd.read_sql_query(sql, conn)
    return data

def get_sql_data(sql, conn):
    if conn != None:
        data = pd.read_sql_query(sql, conn)
    else:
        data = None
    
    return data

#query = open("query_fg_tangguh.sql", mode="rt").read()
#df = retrieve_data(query)

def create_db_connection(filename='database_tangguh.ini', section='postgresql_ml_lng_skk'):
    # Connect to configuration file
    current_dir = Path(__file__).resolve()
    root_parent = current_dir.parent.parent.parent
    config_folder = root_parent / "config"
    config_db_str = str(config_folder/'database_tangguh.ini')

    # Read database configuration INI
    config = ConfigParser()
    config.read(config_db_str)
    postgresql = config[section]
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
        
        #logging.info("Database connected ...")
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        #print(error)
        #logging.error(error)
        return None
