#Author: Dunyin Gu
#Creation Date: 12/24/2021

# import libraries
import os
import boto3
from pathlib import Path
import yaml
import snowflake.connector
import re
import multiprocessing
import subprocess

import utils

import pandas as pd
import numpy as np
import time
import datetime
from io import StringIO, BytesIO
import s3fs

import sklearn
from sklearn.ensemble import IsolationForest
from collections import Counter
import imblearn
from imblearn.over_sampling import SMOTE

from scipy import stats
from sklearn.utils.fixes import loguniform
import math

import ast

import pickle
import joblib
import tarfile
import gzip


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
    utils.utils_log_step(x='Start runing 00_master_train.py')
    
    utils.utils_log_step(x='Parsing config file and getting environmental parameters')
    
    # sql file path
    sql_file_path='sql/train'
    
    # load functions in config file
    yaml.add_constructor('!toupper', toupper)
    yaml.add_constructor('!concat', concat)
    yaml.add_constructor('!sys_date', sys_date)
    
    # read config file
    with open(os.path.join(str(Path.cwd()),'config.yml'), 'r') as config_file:
        config = yaml.load(config_file)
    
    lifestage_label_dict = config['lifestage_label_dict']
    db_tables_dict = config['db_tables']
    
    # eastablish connection to snowflake
    con = utils.sf_connect()
    cur = con.cursor()
    
    # get s3 and sagemaker environmental variables
    region = 'ap-southeast-2'
    s3_client = boto3.client('s3',region_name=region)
    #s3_client = boto3.client('s3')
    s3_bucket = utils.s3_bucket()
    
    # define file path
    train_data_file_path = 'Program/ds013_lifestage_rebuild/data/train'
    model_file_path = 'Program/ds013_lifestage_rebuild/models'
    
    #' -----------------------------
    # 1.0 CHECK & RUN           ----
    #' -----------------------------
    # check last train date vs. train source table update date
    utils.utils_log_step(x='Checking if the train data/model need to be refreshed')
    
    # get last train date
    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=train_data_file_path)
    if 'Contents' in response:
        at_list = pd.DataFrame.from_dict(response['Contents'])
        at_list.columns= at_list.columns.str.strip().str.lower()
        at_list.rename(columns={'key':'filename','lastmodified':'file_createtime'}, inplace=True)
        at_list = at_list[['filename','file_createtime']]
        at_list['file_createtime'] = at_list['file_createtime'].dt.tz_convert("Australia/Melbourne")
        at_list = at_list[at_list.filename.str.contains('df_train')]
        at_list = at_list[at_list['file_createtime']==max(at_list.file_createtime)].reset_index(drop=True)
        
        train_date = at_list['file_createtime'][0].to_pydatetime().date()
        
        # get train source table update date
        sql=utils.read_sql(sql_file_path,'check_update_src_srvy_tbl.sql',db_tables_dict)
        df_check_update = utils.load_data(con,sql)
        df_check_update['LAST_UPDATE_DATE'] = pd.to_datetime(df_check_update['LAST_UPDATE_DATE'])
        
        # define train script running sequence
        #script_runner = multiprocessing.Pool(processes = 3)
        #data_prep_script = ['01_train_data_prep.py']
        #train_scripts = ['02_init_classifier_model_train.py','02_transition_model_train.py']
        
        # run the train scripts when conditions are met
        if (df_check_update['LAST_UPDATE_DATE'][0] > train_date) or (len(at_list)==0):
            utils.utils_log_step(x='Running train data prep job')
            #script_runner.map(execute, data_prep_script)
            execute('01_train_data_prep.py')
            
            utils.utils_log_step(x='Running model train jobs')
            #script_runner.map(execute, train_scripts)
            subprocess.run('python 02_init_classifier_model_train.py & python 02_transition_model_train.py', shell=True)
        else:
            utils.utils_log_step(x='The source table has not been updated since last train - re-train is not required')
    
    else:
        utils.utils_log_step(x='No train output is found')
        utils.utils_log_step(x='Running train data prep job')
        execute('01_train_data_prep.py')
        
        utils.utils_log_step(x='Running model train jobs')
        subprocess.run('python 02_init_classifier_model_train.py & python 02_transition_model_train.py', shell=True)
    
    
    #' --------------------------------------------------------
    # 2.0 END
    #' --------------------------------------------------------
    utils.utils_log_step(x='00_master_train.py has been successfully completed')





if __name__ == '__main__':
    main()