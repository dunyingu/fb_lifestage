from __future__ import print_function

import argparse
import json
import logging
import os
import pickle as pkl
import joblib
import time

import pandas as pd
import numpy as np
import math
from scipy import stats
from sklearn.utils.fixes import loguniform

import sys
sys.modules['sklearn.externals.joblib'] = joblib
from sklearn.externals.joblib import parallel_backend, Parallel, delayed
from sklearn.externals.joblib import hash

# import dask # dask module not installed on xgboost instance
# import dask.distributed
# import dask.array as da
# import dask.dataframe as dd
# from dask.distributed import Client, LocalCluster
# from dask.distributed import Executor, progress
# from dask import delayed

import sklearn
import xgboost
from xgboost import XGBClassifier
# from xgboost.dask import DaskXGBClassifier

from sklearn.model_selection import StratifiedKFold#, KFold, RandomizedSearchCV
from sklearn.model_selection import ParameterSampler
# from sklearn.experimental import enable_halving_search_cv
# from sklearn.model_selection import HalvingRandomSearchCV

from sagemaker_containers import entry_point
# from sagemaker_xgboost_container.data_utils import get_dmatrix
from sagemaker_xgboost_container import distributed


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


def xgb_random_search_cv(xgb_model, params, split_idx, X_train_list, y_train_list, X_val_list, y_val_list):
    
    X_train, y_train, X_val, y_val = X_train_list[split_idx], y_train_list[split_idx], X_val_list[split_idx], y_val_list[split_idx]
    
    t0 = time.time()
    xgb = xgb_model(**params)
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
    
    results = {"hyperparameters":params,
               "hyperparameters_hash":hash(params),
               "split_idx":split_idx,
               "train_time":train_time,
               "train_score":train_score,
               "val_score":val_score
              }
    return results


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
    
    #delayed_xgb_random_search_cv = [delayed(xgb_random_search_cv)(XGBClassifier, params, split_idx, X_train_list, y_train_list, X_val_list, y_val_list)  
     #                               for params in param_space
     #                               for split_idx in range(cv_num)
     #                              ]
    
    print("Starting Randomized Search CV process in parallel...")
    job_num = n_iter * cv
    with parallel_backend(n_jobs=job_num,backend='threading'):
        xgb_random_search_cv_results = joblib.Parallel(verbose=100)([joblib.delayed(xgb_random_search_cv)(XGBClassifier, params, split_idx, X_train_list, y_train_list, X_val_list, y_val_list)  
                                                                     for params in param_space
                                                                     for split_idx in range(cv_num)
                                                                    ]
                                                                    )
    print("Randomized Search CV process completed.")
    
    df_xgb_random_search_cv_results = pd.DataFrame.from_dict(xgb_random_search_cv_results)
    df_xgb_random_search_cv_evaluation = df_xgb_random_search_cv_results.groupby('hyperparameters_hash').agg(
                                                                                                            {"train_time": np.mean,
                                                                                                             "train_score": np.mean,
                                                                                                            "val_score": np.mean}
                                                                                                             ).reset_index()
    all_hyperparams = pd.DataFrame()
    all_hyperparams['hyperparameters'] = df_xgb_random_search_cv_results['hyperparameters']
    all_hyperparams['hyperparameters_hash'] = df_xgb_random_search_cv_results['hyperparameters_hash']
    
    df_xgb_random_search_cv_best_hyperparams = df_xgb_random_search_cv_evaluation.merge(all_hyperparams)
    df_xgb_random_search_cv_best_hyperparams = df_xgb_random_search_cv_best_hyperparams.sort_values(by='val_score',ascending=True) 
    
    best_hyperparams = df_xgb_random_search_cv_best_hyperparams.iloc[0]['hyperparameters']
    
    return best_hyperparams 


def _xgb_train(params,X,y,model_dir,is_master):
    
    booster = xgboost.XGBClassifier(**params)
    booster.fit(X,y)
    
    if is_master:
        model_location = model_dir + "/xgboost-model"
        # pkl.dump(booster, open(model_location, "wb"))
        booster.save_model(model_location)
        logging.info("Stored trained model at {}".format(model_location))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Hyperparameters are described here.
    # parser.add_argument("--num_round", type=int)
    # parser.add_argument("--max_depth", type=int)
    # parser.add_argument("--colsample_bytree", type=float)
    # parser.add_argument("--min_child_weight", type=int)
    # parser.add_argument("--learning_rate", type=float) # eta
    # parser.add_argument("--gamma", type=int)
    # parser.add_argument("--subsample", type=float)
    # parser.add_argument("--reg_alpha", type=float) # alpha
    # parser.add_argument("--objective", type=str)
    # parser.add_argument("--verbosity", type=int)
    
    parser.add_argument("--n_iter", type=int)
    parser.add_argument("--cv", type=int)
    parser.add_argument("--seed", type=int)
    
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument(
        "--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR")
    )
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    # parser.add_argument(
     #   "--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION")
    # )
    
    parser.add_argument("--sm_hosts", type=str, default=os.environ.get("SM_HOSTS"))
    parser.add_argument(
       "--sm_current_host", type=str, default=os.environ.get("SM_CURRENT_HOST")
    )
    
    args, _ = parser.parse_known_args()
    
    # Get SageMaker host information from runtime environment variables
    sm_hosts = json.loads(args.sm_hosts)
    sm_current_host = args.sm_current_host
    
    print(args.train)
    logging.info("train path is {}".format(args.train))
    
    if os.path.isfile(args.train):
        files_path = args.train
    else:
        for root, dirs, files in os.walk(args.train):
            if dirs == []:
                files_path = root
                break
    
    print(files_path)
    logging.info("files path is {}".format(files_path))
    
    csv_file = files_path if os.path.isfile(files_path) else [
        f for f in os.listdir(files_path) if os.path.isfile(os.path.join(files_path, f))][0]
    
    print(csv_file)
    logging.info("csv file is {}".format(csv_file))
    
    df_train = pd.read_csv(os.path.join(files_path, csv_file),header=None)
    X = df_train.iloc[:, 1:]
    y = df_train.iloc[:, 0]
    
    # train_hp = {
      #  "n_estimators": args.num_round,
      #  "max_depth": args.max_depth,
      #  "colsample_bytree": args.colsample_bytree,
      #  "min_child_weight": args.min_child_weight,
      #  "learning_rate": args.learning_rate, # eta
      #  "gamma": args.gamma,
      #  "subsample": args.subsample,
      #  "reg_alpha": args.reg_alpha, # alpha
      #  "objective": args.objective,
      #  "verbosity": args.verbosity
    #}
    
    # n_iter = 10
    # cv = 3
    
    train_hp = xgb_best_hyperparams(X,y,args.n_iter,args.seed,args.cv)
    
    xgb_train_args = dict(
        params=train_hp,
        X = X,
        y = y,
        model_dir=args.model_dir
    )
    
    if len(sm_hosts) > 1:
        # Wait until all hosts are able to find each other
        entry_point._wait_hostname_resolution()
        
        # Execute training function after initializing rabit.
        distributed.rabit_run(
            exec_fun=_xgb_train,
            args=xgb_train_args,
            include_in_training=(df_train is not None), #(dtrain is not None)
            hosts=sm_hosts,
            current_host=sm_current_host,
            update_rabit_args=True
        )
    else:
        # If single node training, call training method directly.
        if df_train is not None: #dtrain
            xgb_train_args["is_master"] = True
            _xgb_train(**xgb_train_args)
        else:
            raise ValueError("Training channel must have data to train model.")


def model_fn(model_dir):
    model_file = "xgboost-model"
    booster = xgboost.XGBClassifier()
    booster.load_model(os.path.join(model_dir, model_file))
    return booster

def input_fn(request_body, request_content_type):
    if request_content_type == 'text/csv':
        samples = []
        for r in request_body.split('\n'):
            samples.append(list(map(float,r.split(','))))
        return np.array(samples)
    else:
        raise ValueError("Thie model only supports text/csv input")

def predict_fn(input_data, model):
    return model.predict_proba(input_data) #.predict()