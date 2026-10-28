#Author: Dunyin Gu
#Creation Date: 01/06/2022

# import libraries
import os
import boto3
from pathlib import Path
import yaml
import snowflake.connector
import re
import multiprocessing

import utils

import pandas as pd
import numpy as np
import time
import datetime
from io import StringIO, BytesIO
import s3fs


# add string concatenation function to parse the config file
def concat(loader, node):
    seq = loader.construct_sequence(node)
    seq = [str(tmp) for tmp in seq]
    return ''.join(seq)


# add toupper function to parse the config file
def toupper(loader, node):
    seq = loader.construct_sequence(node)
    seq = [str(tmp).upper() for tmp in seq]
    return ''.join(seq)


# add system date function to parse the yaml file
def sys_date(loader, node):
    return str(datetime.date.today())


# run python script
def execute(script):
    os.system(f'python {script}')

# run python script
#def execute(script):
 #   with open(script,"r") as rnf:
 #       exec(rnf.read())


def main():
    #' -----------------------------
    # 0.0 SETUP                ----
    #' -----------------------------
    utils.utils_log_step(x='Start runing 00_master_score.py')
    
    utils.utils_log_step(x='Parsing config file and getting environmental parameters')
    
    # sql file path
    sql_file_path='sql/score'
    
    # load functions in config file
    yaml.add_constructor('!toupper', toupper)
    yaml.add_constructor('!concat', concat)
    yaml.add_constructor('!sys_date', sys_date)
    
    # read config file
    with open(os.path.join(str(Path.cwd()),'config.yml'), 'r') as config_file:
        config = yaml.load(config_file)
    
    lifestage_label_dict = config['lifestage_label_dict']
    db_tables_dict = config['db_tables']
    score_freq = config['score_freq']
    
    # eastablish connection to snowflake
    con = utils.sf_connect()
    cur = con.cursor()
    
    # get s3 and sagemaker environmental variables
    region = 'ap-southeast-2'
    s3_client = boto3.client('s3',region_name=region)
    #s3_client = boto3.client('s3')
    s3_bucket = utils.s3_bucket()
    
    # define file path
    score_data_file_path = 'Program/ds013_lifestage_rebuild/data/score'
    model_file_path = 'Program/ds013_lifestage_rebuild/models'
    
    #' -----------------------------
    # 1.0 CHECK & RUN           ----
    #' -----------------------------
    # check if scoring job is required
    
    # get the number of days since last scoring
    sql=utils.read_sql(sql_file_path,'check_last_score_date.sql',db_tables_dict)
    df_check_score = utils.load_data(con,sql)
    days_in_btw = df_check_score['DAYS_IN_BETWEEN'][0]
    
    if np.isnan(days_in_btw):
        utils.utils_log_step(x='No socring has been performed before - start running the scoring job')
        execute('03_model_score.py')
        execute('04_check_notif.py')
    elif days_in_btw >= score_freq *7:
        utils.utils_log_step(x=f'Last scoring was performed {days_in_btw} days ago >= 91 days - start running the scoring job')
        execute('03_model_score.py')
        execute('04_check_notif.py')
    else:
        utils.utils_log_step(x=f'Last scoring was performed {days_in_btw} days ago < 91 days - scoring job not required')
    
    
    #' --------------------------------------------------------
    # 2.0 END
    #' --------------------------------------------------------
    utils.utils_log_step(x='00_master_score.py has been successfully completed')




if __name__ == '__main__':
    main()