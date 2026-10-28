#Author: Dunyin Gu
#Creation Date: 12/29/2021

# import libraries
import os
import boto3
from pathlib import Path
import yaml

import utils

import pandas as pd
import numpy as np
import time
import datetime
from io import StringIO, BytesIO
import s3fs

import sklearn

from scipy import stats
from sklearn.utils.fixes import loguniform
import math

import pickle
import joblib

from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import StratifiedKFold

import xgboost
from xgboost import XGBClassifier

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

# split dataset into cross-validation sets
def cross_validation_split(n_splits, random_int, X, y):
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state=random_int)
    
    i = 0
    X_train_list = []
    X_val_list = []
    y_train_list = []
    y_val_list= []
    
    for train_index, val_index in skf.split(X,y):
        X_train, X_val = X.iloc[train_index,:], X.iloc[val_index,:]
        y_train, y_val = y.iloc[train_index].values.ravel(), y.iloc[val_index].values.ravel()
        X_train_list.append(X_train)
        X_val_list.append(X_val)
        y_train_list.append(y_train)
        y_val_list.append(y_val)
        i+=1
        
    return X_train_list,X_val_list,y_train_list,y_val_list,i

# check the train and validate score for each cross-validation set
def xgb_random_search_cv(xgb_model, params, split_idx, X_train_list, y_train_list, X_val_list, y_val_list):
    
    X_train, y_train, X_val, y_val = X_train_list[split_idx], y_train_list[split_idx], X_val_list[split_idx], y_val_list[split_idx]
    t0 = time.time()
    
    xgb=xgb_model(**params)
    
    xgb.fit(X_train, y_train,
            early_stopping_rounds=100,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric='merror',
            verbose=False)
    
    train_time = time.time() - t0
    
    for item in xgb.evals_result()['validation_0'].items():
        item_list_train = item
    
    train_score = min(item_list_train[1]) 
    
    for item in xgb.evals_result()['validation_1'].items():
        item_list_val = item
    
    val_score = min(item_list_val[1])
    
    results = {'hyperparameters':params,
               'hyperparameters_hash':hash(params),
               'split_idx':split_idx,
               'train_time':train_time,
               'train_score':train_score,
               'val_score':val_score
              }
    return results

# get the best hyperparameters
def xgb_best_hyperparams(X,y,n_iter,seed,cv):
    
    X_train_list,X_val_list,y_train_list,y_val_list,cv_num = cross_validation_split(n_splits=cv, random_int=seed, X=X, y=y)
    
    param_dist = {'n_estimators': stats.randint(600, 1200),
                'max_depth': [3, 4, 5, 6, 7, 8],
                'colsample_bytree': stats.uniform(0.2, 0.7), #0.2 - 0.9
                'min_child_weight': stats.randint(0, 20),
                'learning_rate': stats.uniform(0.01, 0.49), #0.01 - 0.5
                'gamma': stats.randint(1, 5),
                'subsample': stats.uniform(0.5, 0.3), #0.5 - 0.8
                'reg_alpha': loguniform(0.01,100),
                'objective': ['multi:softprob']
                }
    param_space = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=seed))
    
    print('Starting Randomized Search CV process in parallel...')
    job_num = n_iter * cv
    with parallel_backend(n_jobs=job_num,backend='threading'):
        xgb_random_search_cv_results = joblib.Parallel(verbose=100)([joblib.delayed(xgb_random_search_cv)(XGBClassifier, params, split_idx, X_train_list, y_train_list, X_val_list, y_val_list)  
                                                                     for params in param_space
                                                                     for split_idx in range(cv_num)
                                                                    ]
                                                                    )
    print('Randomized Search CV process completed.')
    
    df_xgb_random_search_cv_results = pd.DataFrame.from_dict(xgb_random_search_cv_results)
    df_xgb_random_search_cv_evaluation = df_xgb_random_search_cv_results.groupby('hyperparameters_hash').agg(
                                                                                                            {'train_time': np.mean,
                                                                                                             'train_score': np.mean,
                                                                                                             'val_score': np.mean}
                                                                                                             ).reset_index()
    all_hyperparams = pd.DataFrame()
    all_hyperparams['hyperparameters'] = df_xgb_random_search_cv_results['hyperparameters']
    all_hyperparams['hyperparameters_hash'] = df_xgb_random_search_cv_results['hyperparameters_hash']
    
    df_xgb_random_search_cv_best_hyperparams = df_xgb_random_search_cv_evaluation.merge(all_hyperparams)
    df_xgb_random_search_cv_best_hyperparams = df_xgb_random_search_cv_best_hyperparams.sort_values(by='val_score',ascending=True)
    
    best_hyperparams = df_xgb_random_search_cv_best_hyperparams.iloc[0]['hyperparameters']
    
    return best_hyperparams


def main():
    #' -----------------------------
    # 0.0 SETUP                ----
    #' -----------------------------
    utils.utils_log_step(x='Start runing 02_init_classifier_model_train.py')
    
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
    
    # define file path
    train_data_file_path = 'Program/ds013_lifestage_rebuild/data/train'
    model_file_path = 'Program/ds013_lifestage_rebuild/models'
    
    
    #' -----------------------------
    # 1.0 LOAD DATA             ----
    #' -----------------------------
    # load data for initial classification model
    utils.utils_log_step(x='Load train data for initial classifier model')
    train_data_filename='df_train_initi_classifier_' + str(datetime.date.today()) + '.csv'
    df_train_initi_classifier = pd.read_csv(f's3://{s3_bucket}/{train_data_file_path}/{train_data_filename}',header=0)
    X = df_train_initi_classifier.iloc[:,:-1]
    y = df_train_initi_classifier.iloc[:,-1]
    
    #' -----------------------------
    # 2.0 MODEL TRAIN           ----
    #' -----------------------------
    # set up training parameters
    utils.utils_log_step(x='Set up training')
    seed = datetime.date.today().day
    n_iter = 10
    cv = 3
    
    # search for best hyperparameters
    utils.utils_log_step(x='Runing 3-fold Cross Validation to search for the best hyperparameters')
    train_hp = xgb_best_hyperparams(X,y,n_iter,seed,cv)
    
    # use the best hyperparameters to train the classification model
    utils.utils_log_step(x='Fit the model using the best hyperparameters')
    xgb_classifier = XGBClassifier(**train_hp)
    xgb_classifier.fit(X,y)
    
    # save the output model
    utils.utils_log_step(x='Save the output classification model')
    model_name = 'init_classifier_xgboost_model_' + str(datetime.date.today())
    model_loc = f's3://{s3_bucket}/{model_file_path}/init_classifier/{model_name}'
    fs = s3fs.S3FileSystem()
    pickle.dump(xgb_classifier, fs.open(model_loc, "wb"))
    
    #' -----------------------------
    # 3.0 END                   ----
    #' -----------------------------
    utils.utils_log_step(x='02_init_classifier_model_train.py has been successfully completed')




if __name__ == '__main__':
    main()
