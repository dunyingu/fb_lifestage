# import libraries
import os
import time
import datetime
from datetime import datetime
from datetime import date
import json
#!pip install snowflake-connector-python[pandas]
import snowflake.connector

from pathlib import Path
import yaml
import re
import s3fs
from io import StringIO, BytesIO

import pandas as pd
import numpy as np

import boto3
import sagemaker
from sagemaker.xgboost.estimator import XGBoost

from sagemaker.serializers import CSVSerializer
from sagemaker.predictor import csv_serializer
from sagemaker.predictor import Predictor
import ast


os.environ['TZ'] = 'Australia/Melbourne'
time.tzset()
s3_location = 'Program/ds013_lifestage_rebuild'
s3_client = boto3.client("s3")


# establish connection to snowflake via AWS secret manager credentials
def sf_connect():
    session = boto3.Session()
    region_name = 'ap-southeast-2' #region_name =boto3.Session().region_name
    
    client = session.client(service_name='secretsmanager',region_name=region_name)
    secret_name = 'db/flybuys/analytics/mig_analytics_rw'
    get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    
    secret = get_secret_value_response['SecretString']
    secret_json = json.loads(secret)
    account = secret_json['account'] + '.' + secret_json['region']
    
    # establish connection to snowflake
    con = snowflake.connector.connect(
      user=secret_json['username'],
      account=account,
      password=secret_json['password'],
      role=secret_json['role']
      )
    
    return (con)


# get s3 bucket name
def s3_bucket():
    session = boto3.Session()
    region_name = 'ap-southeast-2' #region_name =boto3.Session().region_name
    ssm_client = session.client(service_name='ssm',region_name=region_name)
    param_name = '/analytics/env/s3/flybuys/ctrlm/output/prod'
    get_param_value_response = ssm_client.get_parameter(Name=param_name)
    s3_bucket_name = get_param_value_response['Parameter']['Value']
    return (s3_bucket_name)


# get snowflake stage location
def sf_stage():
    stg = '@pub_flybuys.dl_utility.flybuys_ctrlm_prod'
    return (stg)


# print log message
def utils_log_step(x):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = timestamp + ' ' + x
    print(log_msg)

def utils_file_name(arg_list=[]):
    output_file_name = '/'.join(arg_list)
    output_file_name = output_file_name.replace('//','/')
    return output_file_name


# read sql file to fetch the data
def read_sql (sql_file_path,sql_file_name,db_tables_dict):
    if os.path.exists(os.path.join(str(Path.cwd()),sql_file_path,sql_file_name)):
        sql_file = open(os.path.join(str(Path.cwd()),sql_file_path,sql_file_name),'r')
        sql_list = sql_file.readlines()
        
        tbl_list = [item.replace('(','').replace(')','') for item in re.split(' |\n|/',''.join(sql_list)) if item.startswith('?') ]
        #tbl_list = [item.replace('(','').replace(')','') .replace('?','') for item in re.split(' |\n',''.join(sql_list)) if item.startswith('?') ]
        
        repl = {}
        for item in tbl_list:
            repl[item]=db_tables_dict[item]
        
        sql_qry = ''.join(sql_list)
        sql_qry = re.sub('/\\*(.*\n){1,}\\*/|/\\*.*\\*/|--.*\n','',sql_qry)
        sql = re.sub('[?](\\w+)',lambda x: repl[x.group()], sql_qry)
        
        #cur = con.cursor()
        #cur.execute(sql)
        #df = cur.fetch_pandas_all()
        #cur.close()
        # con.close()
        #return (df)
    else:
        print('Cannot find file ', sql_file_name, '. Please check file name')
    
    return(sql)


# load data from snowflake table 
def load_data(con,sql):
    cur = con.cursor()
    cur.execute(sql)
    df = cur.fetch_pandas_all()
    cur.close()
    return df


# pivot table
def pivot_table (df_input,pivot_col,index_col=[],values=[]):
    df_output = df_input
    df_output['GROUP_LOYALTY_MEMBER_NUMBER'] = df_output['GROUP_LOYALTY_MEMBER_NUMBER'].astype(str)
    df_output = df_output[~df_output[pivot_col].isnull()].pivot(index=index_col, columns=pivot_col, values=values)
    #df_output = df_output.drop('NaN', axis=1, level=1)
    df_output.columns = ['_'.join(col).strip() for col in df_output.columns.values]
    df_output = df_output.reset_index()
    #df_output = df_output[~df_output['RESPONSE_DATE'].isnull()] 'SYSTEMDATE'
    return df_output


# save df to a specified s3 path
def save_df_to_s3 (df, header, s3_bucket, data_file_path, filename):
    csv_buffer = df.to_csv(header=header, index=False)
    bytes_obj = BytesIO(csv_buffer.encode('utf-8'))
    raw_data_file_key= f'{data_file_path}/{filename}'
    s3_client.upload_fileobj(bytes_obj,s3_bucket, raw_data_file_key)


# define oversampling size for each class
def sampling_strategy(class_cnt_dict,class_num=6,ratio=0.2):
    max_class_pair = max(class_cnt_dict.items(), key=lambda k: k[1])
    sampling_strategy_dict={}
    
    for i in range(class_num):
        if i == int(max_class_pair[0]):
            sampling_strategy_dict[i]=max_class_pair[1]
        elif int(max_class_pair[1]*ratio) > class_cnt_dict[i]:
            sampling_strategy_dict[i]=int(max_class_pair[1]*ratio)
        else:
            sampling_strategy_dict[i]=class_cnt_dict[i]
            
    return sampling_strategy_dict


# run transition train job via SageMaker
def lifestage_transition_train(prior_lifestage,train_date,
                               s3_bucket,train_data_file_path,
                               model_file_path,
                               script_path, hyperparameters,role,instance_type):
    train_file_name = f'df_train_smote_{prior_lifestage}_transition_{train_date}.csv' #{str(date.today())}
    train_path = f's3://{s3_bucket}/{train_data_file_path}/{train_file_name}'
    train_input = sagemaker.inputs.TrainingInput(s3_data=train_path,  content_type='csv')
    
    model_output_path = 's3://{}/{}/transition'.format(s3_bucket, model_file_path)
    
    xgb_script_mode_estimator = XGBoost(
    entry_point=script_path,
    framework_version='1.3-1', #'1.4-2'
    #source_dir = model_source_path,
    hyperparameters=hyperparameters,
    role=role,
    instance_type=instance_type,
    instance_count=1,
    output_path=model_output_path
    )

    job_name = f'{prior_lifestage}_TRANSITION_TRAIN_{str(date.today())}'.replace('_','-')
    
    xgb_script_mode_estimator.fit({'train': train_input}
                              ,job_name=job_name
                              ,wait=False)
    
    return xgb_script_mode_estimator


# check SageMaker train job status
def wait_for_training_job_to_complete(estimator):
    job = estimator.latest_training_job.job_name
    print(f'Waiting for job: {job}')
    status = estimator.latest_training_job.describe()['TrainingJobStatus']
    while status == 'InProgress':
        time.sleep(45)
        status = estimator.latest_training_job.describe()['TrainingJobStatus']
        if status == 'InProgress':
            print(f'{job} job status: {status}')
    print(f'DONE. Status for {job} is {status}\n')