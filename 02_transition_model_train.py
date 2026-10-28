#Author: Dunyin Gu
#Creation Date: 12/30/2021

# import libraries
import os
import boto3
from pathlib import Path
import yaml
import snowflake.connector
import re

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

from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV, GridSearchCV

import xgboost
from xgboost import XGBClassifier

import sagemaker
from sagemaker import get_execution_role
# from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker import image_uris
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
from sagemaker import AlgorithmEstimator

from sagemaker.multidatamodel import MultiDataModel
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import CSVSerializer
from sagemaker.predictor import csv_serializer
from sagemaker.predictor import Predictor
from sagemaker.transformer import Transformer

import sys
sys.modules['sklearn.externals.joblib'] = joblib
from sklearn.externals.joblib import parallel_backend, Parallel, delayed
from sklearn.externals.joblib import hash


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


def main():
    #' -----------------------------
    # 0.0 SETUP                ----
    #' -----------------------------
    utils.utils_log_step(x='Start runing 02_transition_model_train.py')
    
    utils.utils_log_step(x='Parsing config file and getting environmental parameters')
    
    #sql_file_path='sql/train'
    
    # load functions in config file
    yaml.add_constructor('!toupper', toupper)
    yaml.add_constructor('!concat', concat)
    yaml.add_constructor('!sys_date', sys_date)
    
    # read config file
    with open(os.path.join(str(Path.cwd()),'config.yml'), 'r') as config_file:
        config = yaml.load(config_file)
    
    lifestage_label_dict = config['lifestage_label_dict']
    db_tables_dict = config['db_tables']
    
    #con = utils.sf_connect()
    #cur = con.cursor()
    
    # get s3 and sagemaker environmental variables
    region = 'ap-southeast-2'
    s3_client = boto3.client('s3',region_name=region)
    #s3_client = boto3.client('s3')
    s3_bucket = utils.s3_bucket()
    
    #region = boto3.Session().region_name
    sagemaker_session = sagemaker.Session(boto_session=boto3.session.Session(region_name=region),
                                          default_bucket=s3_bucket)
    #sagemaker_session = sagemaker.Session()
    sagemaker_client = boto3.client('sagemaker',region_name=region)
    #sagemaker_client = boto3.client('sagemaker')
    role = get_execution_role(sagemaker_session=sagemaker_session)
    #role = get_execution_role()
    
    # define file path
    train_data_file_path = 'Program/ds013_lifestage_rebuild/data/train'
    model_file_path = 'Program/ds013_lifestage_rebuild/models'
    
    #' -----------------------------
    # 1.0 CLEANUP               ----
    #' -----------------------------
    # clean up previous transition endpoint/models
    utils.utils_log_step(x='Clean up previous transition endpoint/models')
    df_transition_endpoints = pd.DataFrame(sagemaker_session.sagemaker_client.list_endpoints()['Endpoints'])
    df_transition_endpoints=df_transition_endpoints[df_transition_endpoints['EndpointName'].str.contains('lifestage-transition')].sort_values(by=['LastModifiedTime'], ascending=False)
    if len(df_transition_endpoints) > 0:
        # delete previous transition endpoint/models
        prev_transition_endpoint = df_transition_endpoints['EndpointName'][0]
        endpoint_response = sagemaker_client.describe_endpoint_config(EndpointConfigName=prev_transition_endpoint)
        models = endpoint_response['ProductionVariants'][0]['ModelName']
        sagemaker_client.delete_model(ModelName=models)
        sagemaker_client.delete_endpoint_config(EndpointConfigName=prev_transition_endpoint)
        sagemaker_client.delete_endpoint(EndpointName=prev_transition_endpoint)
        
        # delete model objects
        s3_response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=f'{model_file_path}/transition')
        if 'Contents' in response:
            for model_obj in s3_response['Contents']:
                s3_client.delete_object(Bucket=s3_bucket, Key=model_obj['Key'])
    else:
        print('No previous transition endpoint has been found - continue...')
    
    
    #' -----------------------------
    # 2.0 MODEL TRAIN           ----
    #' -----------------------------
    # define SageMaker training job parameters
    utils.utils_log_step(x='Set up SageMaker training job')
    
    train_date = str(datetime.date.today())
    script_path = 'sm_transition_train.py'
    hyperparameters = {'n_iter': '10',
                       'cv': "3",
                       'seed': str(datetime.date.today().day)
                      }
    instance_type = 'ml.m5.12xlarge'
    
    # run transition train jobs
    utils.utils_log_step(x='Running transition train jobs via SageMaker')
    
    estimators=[]
    sm_train_jobs=[]
    
    for i in range(len(lifestage_label_dict)):
        lifestage_label = lifestage_label_dict[i]
        xgb_script_mode_estimator = utils.lifestage_transition_train(lifestage_label,train_date,
                                                                     s3_bucket,train_data_file_path,
                                                                     model_file_path,
                                                                     script_path, hyperparameters,role,instance_type)
        
        estimators.append(xgb_script_mode_estimator)
        sm_train_jobs.append(xgb_script_mode_estimator.latest_training_job.job_name)
        time.sleep(10)
    
    # check train jobs' status and waiting for the jobs to complete
    utils.utils_log_step(x='Waiting for transition train jobs to complete...')
    
    for est in estimators:
        utils.wait_for_training_job_to_complete(est)
    
    
    #' -----------------------------
    # 3.0 MODEL DEPLOY          ----
    #' -----------------------------
    utils.utils_log_step(x='Deploy trained transition models as endpoint')
    
    # create transition models
    estimator = estimators[0]
    XGBOOST_IMAGE = image_uris.retrieve(region=region, framework='xgboost', version='1.3-1')
    transition_model = estimator.create_model(role=role, image_uri=XGBOOST_IMAGE)
    
    # path of model objects for endpoint
    model_data_prefix = 's3://{}/{}/transition'.format(s3_bucket, model_file_path)
    
    # define and create endpoint
    ENDPOINT_NAME = 'lifestage-transition-' + str(datetime.date.today())
    
    transition_endpoint = MultiDataModel(
    name=ENDPOINT_NAME,
    model_data_prefix=model_data_prefix,
    model=transition_model,
    sagemaker_session=sagemaker_session
    )
    
    ENDPOINT_INSTANCE_TYPE = 'ml.m5.4xlarge'
    
    # deploy the endpoint and add model objects to the endpoint
    transition_endpoint_predictor = transition_endpoint.deploy(
    initial_instance_count=1, instance_type=ENDPOINT_INSTANCE_TYPE, endpoint_name=ENDPOINT_NAME
    )
    
    for est in estimators:
        artifact_path = est.latest_training_job.describe()['ModelArtifacts']['S3ModelArtifacts']
        model_name = artifact_path.split('/')[-3] + '.tar.gz'
        transition_endpoint.add_model(model_data_source=artifact_path, model_data_path=model_name)
    
    # define the serializer type for the transition endpoint predictor
    transition_endpoint_predictor.serializer=CSVSerializer()
    
    
    #' -----------------------------
    # 4.0 END                   ----
    #' -----------------------------
    utils.utils_log_step(x='02_transition_model_train.py has been successfully completed')




if __name__ == '__main__':
    main()


