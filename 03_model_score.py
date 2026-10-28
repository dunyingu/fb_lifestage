#Author: Dunyin Gu
#Creation Date: 01/04/2022

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


# use transition endpoint for prediction
def transition_predict_prob(data, lifestage_trans_predictor, model, rows=500):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    pred_prob = np.array([])
    for array in split_array:
        pred_prob = np.concatenate([pred_prob.reshape(-1,6),np.array(ast.literal_eval(lifestage_trans_predictor.predict(array, target_model=model).decode('utf-8')))],axis=0)
    return pred_prob


def main():
    #' -----------------------------
    # 0.0 SETUP                 ----
    #' -----------------------------
    utils.utils_log_step(x='Start runing 03_model_score.py')
    
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
    files_dict = config['files']
    
    # eastablish connection to snowflake
    con = utils.sf_connect()
    cur = con.cursor()
    
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
    score_data_file_path = 'Program/ds013_lifestage_rebuild/data/score'
    model_file_path = 'Program/ds013_lifestage_rebuild/models'
    
    
    #' -----------------------------
    # 1.0 EXTRACT DATA          ----
    #' -----------------------------
    utils.utils_log_step(x='Extracting and Preparing score data from source tables')
    
    #sql_list = []
    for sql_file in sorted(os.listdir(os.path.join(str(Path.cwd()),sql_file_path))):
        if sql_file.startswith('create_'):
            sql=utils.read_sql(sql_file_path,sql_file,db_tables_dict)
            #sql_list.append(sql)
            utils.utils_log_step(x=f'running sql: {sql_file}')
            cur.execute(sql)
        else:
            continue
    
    
    #' -----------------------------
    # 2.0 Transform Data        ----
    #' -----------------------------
    utils.utils_log_step(x='Transforming data for score feature preparation')
    
    utils.utils_log_step(x='Getting active score members attributes')
    sql=utils.read_sql(sql_file_path,'get_score_mbr_attributes.sql',db_tables_dict)
    df_score_mbr_attributes = utils.load_data(con,sql)
    df_score_mbr_attributes['GROUP_LOYALTY_MEMBER_NUMBER'] = df_score_mbr_attributes['GROUP_LOYALTY_MEMBER_NUMBER'].astype(str)
    
    utils.utils_log_step(x='Getting active score members summarized and staple transaction features')
    sql=utils.read_sql(sql_file_path,'get_score_mbr_summ_staple_txn.sql',db_tables_dict)
    df_score_mbr_agg_txn_summ = utils.load_data(con,sql)
    df_score_mbr_agg_txn_summ['GROUP_LOYALTY_MEMBER_NUMBER'] = df_score_mbr_agg_txn_summ['GROUP_LOYALTY_MEMBER_NUMBER'].astype(str)
    
    utils.utils_log_step(x='Getting active score members differentating aggregate prod category transaction features')
    sql=utils.read_sql(sql_file_path,'get_score_mbr_agg_diff_txn.sql',db_tables_dict)
    df_score_mbr_diff_agg_txn = utils.load_data(con,sql)
    df_score_mbr_diff_agg_txn['GROUP_LOYALTY_MEMBER_NUMBER'] = df_score_mbr_diff_agg_txn['GROUP_LOYALTY_MEMBER_NUMBER'].astype(str)
    
    utils.utils_log_step(x='Getting active score members differentating aggregate prod category ipi')
    sql=utils.read_sql(sql_file_path,'get_score_mbr_agg_diff_ipi.sql',db_tables_dict)
    df_score_mbr_diff_agg_ipi = utils.load_data(con,sql)
    df_score_mbr_diff_agg_ipi['GROUP_LOYALTY_MEMBER_NUMBER'] = df_score_mbr_diff_agg_ipi['GROUP_LOYALTY_MEMBER_NUMBER'].astype(str)
    
    utils.utils_log_step(x='Getting active score members necessity prod category ipi')
    sql=utils.read_sql(sql_file_path,'get_score_mbr_nec_ipi.sql',db_tables_dict)
    df_score_mbr_nec_txn_ipi = utils.load_data(con,sql)
    df_score_mbr_nec_txn_ipi['GROUP_LOYALTY_MEMBER_NUMBER'] = df_score_mbr_nec_txn_ipi['GROUP_LOYALTY_MEMBER_NUMBER'].astype(str)
    
    utils.utils_log_step(x='Getting active score members selected necessity prod category pack size')
    sql=utils.read_sql(sql_file_path,'get_score_mbr_selected_staple_packsize.sql',db_tables_dict)
    df_score_mbr_nec_pack_size = utils.load_data(con,sql)
    df_score_mbr_nec_pack_size['GROUP_LOYALTY_MEMBER_NUMBER'] = df_score_mbr_nec_pack_size['GROUP_LOYALTY_MEMBER_NUMBER'].astype(str)
    
    # pivot txn/ipi/pack size feat tables
    utils.utils_log_step(x='Pivotting active score members txn/ipi/pack size data')
    
    df_score_mbr_diff_agg_txn_pivot = utils.pivot_table(df_input=df_score_mbr_diff_agg_txn,
                                                        pivot_col='LABEL_DIFF_MERCH_HIER_AGG',
                                                        index_col = ['GROUP_LOYALTY_MEMBER_NUMBER','SYSTEMDATE'],
                                                        values=['TXN_CNT', 'SPEND','FREQ_RATIO','SPEND_RATIO']
                                                       )
    
    df_score_mbr_diff_agg_ipi_pivot = utils.pivot_table(df_input=df_score_mbr_diff_agg_ipi,
                                                        pivot_col='LABEL_DIFF_MERCH_HIER_AGG',
                                                        index_col = ['GROUP_LOYALTY_MEMBER_NUMBER','SYSTEMDATE'],
                                                        values=['MIN_IPI','AVG_IPI','MEDIAN_IPI','MAX_IPI']
                                                       )
    
    df_score_mbr_nec_txn_ipi_pivot = utils.pivot_table(df_input=df_score_mbr_nec_txn_ipi,
                                                       pivot_col='NEC_SKUPLUS3_MERCH_HIER',
                                                       index_col = ['GROUP_LOYALTY_MEMBER_NUMBER','SYSTEMDATE'],
                                                       values=['MIN_IPI','AVG_IPI','MEDIAN_IPI','MAX_IPI']
                                                      )
    
    df_score_mbr_nec_pack_size_pivot = utils.pivot_table(df_input=df_score_mbr_nec_pack_size,
                                                         pivot_col='SKUPLUS3_MERCH_HIER_DESC',
                                                         index_col = ['GROUP_LOYALTY_MEMBER_NUMBER','SYSTEMDATE'],
                                                         values=['MODE_PACK_SIZE']
                                                        )
    
    # join all the feat tables
    utils.utils_log_step(x='Joining all the features for active score members')
    df_score_mbr_agg_txn_feat= pd.merge(df_score_mbr_agg_txn_summ.fillna(0),
                                        df_score_mbr_diff_agg_txn_pivot.fillna(0), 
                                        on=['GROUP_LOYALTY_MEMBER_NUMBER','SYSTEMDATE']
                                        ,how='left'
                                        )
    
    df_score_mbr_agg_txn_ipi_feat = pd.merge(df_score_mbr_agg_txn_feat.fillna(0),
                                             df_score_mbr_diff_agg_ipi_pivot.fillna(184), 
                                             on=['GROUP_LOYALTY_MEMBER_NUMBER','SYSTEMDATE']
                                            ,how='left'
                                            )
    
    df_score_mbr_agg_txn_ipi_feat = pd.merge(df_score_mbr_agg_txn_ipi_feat.fillna(184),
                                             df_score_mbr_nec_txn_ipi_pivot.fillna(184), 
                                             on=['GROUP_LOYALTY_MEMBER_NUMBER','SYSTEMDATE']
                                            ,how='left'
                                            )
    
    df_score_mbr_agg_txn_ipi_pack_feat = pd.merge(df_score_mbr_agg_txn_ipi_feat.fillna(184),
                                                  df_score_mbr_nec_pack_size_pivot.fillna(0), 
                                                  on=['GROUP_LOYALTY_MEMBER_NUMBER','SYSTEMDATE']
                                                 ,how='left'
                                                 )
    
    df_score_mbr_all_feat = pd.merge(df_score_mbr_agg_txn_ipi_pack_feat.fillna(0),
                                     df_score_mbr_attributes[['GROUP_LOYALTY_MEMBER_NUMBER','SYSTEMDATE',
                                                              'PRMY_MEMBER_GENDER','MEMBER_AGE_AT_SURVEY',
                                                              'RFM_TILE','ACTIVE_CARDS']], 
                                     on=['GROUP_LOYALTY_MEMBER_NUMBER','SYSTEMDATE']
                                     ,how='left'
                                    )
    
    # remove score mbrs with less than 3 txns or no age
    utils.utils_log_step(x='Include the active members with age and more than 2 transactions for scoring')
    df_score_mbr_all_feat_filtered = df_score_mbr_all_feat[(df_score_mbr_all_feat['TOTAL_TXN_CNT']>=3)]
    df_score_mbr_all_feat_filtered = df_score_mbr_all_feat_filtered[(df_score_mbr_all_feat_filtered['TOTAL_SPEND']>0)]
    df_score_mbr_all_feat_filtered = df_score_mbr_all_feat_filtered[~df_score_mbr_all_feat_filtered['MEMBER_AGE_AT_SURVEY'].isnull()]
    
    # get prior lifestage for previously scored members if any
    utils.utils_log_step(x='Getting prior lifestage for previously scored members')
    sql=utils.read_sql(sql_file_path,'get_prior_lifestage.sql',db_tables_dict)
    df_prior_lifestage = utils.load_data(con,sql)
    df_prior_lifestage['GROUP_LOYALTY_MEMBER_NUMBER'] = df_prior_lifestage['GROUP_LOYALTY_MEMBER_NUMBER'].astype(str)
    df_prior_lifestage['MONTHS_IN_BETWEEN'] = df_prior_lifestage['MONTHS_IN_BETWEEN'].astype(float)
    
    df_score_mbr_all_feat_prior = pd.merge(df_score_mbr_all_feat_filtered,
                                           df_prior_lifestage,
                                           on='GROUP_LOYALTY_MEMBER_NUMBER'
                                           ,how='left')
    
    # split the active score base into score members for initial classifier and those for transition
    utils.utils_log_step(x='Create score members for initial classifier')
    df_score_mbr_all_feat_init = df_score_mbr_all_feat_prior[df_score_mbr_all_feat_prior['PRIOR_LIFESTAGE_LABEL'].isnull()]
    utils.save_df_to_s3(df=df_score_mbr_all_feat_init,
                        header=True, 
                        s3_bucket=s3_bucket,
                        data_file_path=score_data_file_path, 
                        filename='score_mbr_all_feat_init_' + str(datetime.date.today()) + '.csv'
                       )

    utils.utils_log_step(x='Create score members for transition model')
    df_score_mbr_all_feat_trans = df_score_mbr_all_feat_prior[~df_score_mbr_all_feat_prior['PRIOR_LIFESTAGE_LABEL'].isnull()]
    df_score_mbr_all_feat_trans['PRIOR_LIFESTAGE_LABEL']=df_score_mbr_all_feat_trans['PRIOR_LIFESTAGE_LABEL'].astype(int)
    utils.save_df_to_s3(df=df_score_mbr_all_feat_trans,
                        header=True, 
                        s3_bucket=s3_bucket,
                        data_file_path=score_data_file_path, 
                        filename='score_mbr_all_feat_trans_' + str(datetime.date.today()) + '.csv'
                       )
    
    #' -----------------------------
    # 3.0 MODEL SCORE           ----
    #' -----------------------------
    # get latest initial classifier model
    utils.utils_log_step(x='Load the most recent initial classifier')
    
    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=os.path.join(model_file_path,'init_classifier'))
    at_list = pd.DataFrame.from_dict(response['Contents'])
    at_list.columns= at_list.columns.str.strip().str.lower()
    at_list.rename(columns={'key':'filename','lastmodified':'file_createtime'}, inplace=True)
    at_list = at_list[['filename','file_createtime']]
    at_list['file_createtime'] = at_list['file_createtime'].dt.tz_convert("Australia/Melbourne")
    at_list = at_list[at_list.filename.str.contains('init_classifier_')]
    at_list = at_list[at_list['file_createtime']==max(at_list.file_createtime)].reset_index(drop=True)
    
    train_date = at_list['file_createtime'][0].to_pydatetime().date()
    
    model_name = 'init_classifier_xgboost_model_' + str(train_date)
    model_loc = f's3://{s3_bucket}/{model_file_path}/init_classifier/{model_name}'
    
    fs = s3fs.S3FileSystem()
    
    xgb_classifier = pickle.load(fs.open(model_loc, 'rb'))
    
    # Use the latest initial classifier to score and store the output in a dataframe
    utils.utils_log_step(x='Use the most recent initial classifier for scoring')
    
    X_init = df_score_mbr_all_feat_init.iloc[:,2:-3].values
    
    df_lifestage_init_score_output = pd.DataFrame()
    df_lifestage_init_score_output['GROUP_LOYALTY_MEMBER_NUMBER'] = df_score_mbr_all_feat_init['GROUP_LOYALTY_MEMBER_NUMBER']
    df_lifestage_init_score_output['GROUP_LOYALTY_MEMBER_NUMBER'] = df_lifestage_init_score_output['GROUP_LOYALTY_MEMBER_NUMBER'].astype(str)
    df_lifestage_init_score_output['MBR_AGE'] = df_score_mbr_all_feat_init['MEMBER_AGE_AT_SURVEY']
    df_lifestage_init_score_output['SYSTEMDATE'] = df_score_mbr_all_feat_init['SYSTEMDATE']
    
    y_lifestage_init_pred = xgb_classifier.predict(X_init)
    df_lifestage_init_score_output['PRED_LABEL'] = y_lifestage_init_pred
    
    init_lifestage_label = [(df_lifestage_init_score_output['PRED_LABEL']==0),
                            (df_lifestage_init_score_output['PRED_LABEL']==1),
                            (df_lifestage_init_score_output['PRED_LABEL']==2),
                            (df_lifestage_init_score_output['PRED_LABEL']==3),
                            (df_lifestage_init_score_output['PRED_LABEL']==4),
                            (df_lifestage_init_score_output['PRED_LABEL']==5)
                           ]
    
    lifestage_label_list = lifestage_label_dict.values()
    
    df_lifestage_init_score_output['PRED_LIFESTAGE']=np.select(init_lifestage_label,lifestage_label_list)
    
    y_lifestage_init_pred_prob = xgb_classifier.predict_proba(X_init)
    df_lifestage_init_score_output['0.SINGLE'],df_lifestage_init_score_output['1.COUPLE_NO_KIDS'],df_lifestage_init_score_output['2.FAMILY_W_BABY'],df_lifestage_init_score_output['3.FAMILY_W_CHILDREN'],df_lifestage_init_score_output['4.FAMILY_W_BABY_OTHER_CHILDREN'],df_lifestage_init_score_output['5.MULTI-ADULT_FAMILY'] =y_lifestage_init_pred_prob.T
    
    utils.utils_log_step(x='Save the initial classifier scoring output')
    utils.save_df_to_s3(df=df_lifestage_init_score_output,
                        header=True, 
                        s3_bucket=s3_bucket,
                        data_file_path=score_data_file_path, 
                        filename='active_12m_mbr_lifestage_init_score_output_' + str(datetime.date.today()) + '.csv'
                       )
    
    # get latest transition model
    utils.utils_log_step(x='Load the most recent transition endpoint')
    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=os.path.join(model_file_path,'transition'))
    at_list = pd.DataFrame.from_dict(response['Contents'])
    at_list.columns= at_list.columns.str.strip().str.lower()
    at_list.rename(columns={'key':'filename','lastmodified':'file_createtime'}, inplace=True)
    at_list = at_list[['filename','file_createtime']]
    at_list['file_createtime'] = at_list['file_createtime'].dt.tz_convert("Australia/Melbourne")
    at_list = at_list[at_list.filename.str.contains('-TRANSITION-')]
    
    for i in at_list.index:
        at_list.loc[i,'model_date'] = re.search('(\d{4}-\d{2}-\d{2})', at_list['filename'][i]).group(1)
    
    at_list['model_date']=pd.to_datetime(at_list['model_date'])
    # at_list['model_date'] = at_list['model_date'].dt.tz_localize("Australia/Melbourne")
    
    # at_list = at_list[at_list['file_createtime']==max(at_list.file_createtime)].reset_index(drop=True)
    at_list = at_list[at_list['model_date']==max(at_list.model_date)].reset_index(drop=True)
    
    # model_date = at_list['file_createtime'][0].to_pydatetime().date()
    model_date = at_list['model_date'][0].to_pydatetime().date()
    
    lifestage_trans_endpoint_name = "lifestage-transition-" + str(model_date)
    
    lifestage_trans_predictor = Predictor(endpoint_name=lifestage_trans_endpoint_name,sagemaker_session=sagemaker_session,serializer=CSVSerializer())
    
    # Use the latest transition model to score and store the output in a dataframe
    utils.utils_log_step(x='Use the most recent transition model for scoring')
    
    df_score_mbr_all_feat_trans=df_score_mbr_all_feat_trans.drop(['PRMY_MEMBER_GENDER','LAST_SCORE_DATE'], axis=1)
    
    df_lifestage_trans_score_output = pd.DataFrame()
    for i in range(len(lifestage_label_dict)):
        df_score_input = df_score_mbr_all_feat_trans[df_score_mbr_all_feat_trans['PRIOR_LIFESTAGE_LABEL']==i]
        X_trans =  df_score_input.iloc[:,2:-1].values 
        
        lifestage_trans_model = lifestage_label_dict[i].replace('_','-') + '-TRANSITION-TRAIN-' + str(model_date) + '.tar.gz'
        y_prob = transition_predict_prob(X_trans, lifestage_trans_predictor, model='/'+lifestage_trans_model, rows=500)
        y_pred = np.argmax(y_prob, axis=1)
        
        df_score_output_temp = pd.DataFrame()
        df_score_output_temp['GROUP_LOYALTY_MEMBER_NUMBER']=df_score_input['GROUP_LOYALTY_MEMBER_NUMBER']#.str.split('.').str[0]
        df_score_output_temp['MBR_AGE']=df_score_input['MEMBER_AGE_AT_SURVEY']
        df_score_output_temp['SYSTEMDATE']=df_score_input['SYSTEMDATE']
        df_score_output_temp['PRIOR_LIFESTAGE_LABEL']=df_score_input['PRIOR_LIFESTAGE_LABEL']
        df_score_output_temp['PRED_LABEL']=y_pred.reshape(-1,1)
        df_score_output_temp['0.SINGLE'],df_score_output_temp['1.COUPLE_NO_KIDS'],df_score_output_temp['2.FAMILY_W_BABY'],df_score_output_temp['3.FAMILY_W_CHILDREN'],df_score_output_temp['4.FAMILY_W_BABY_OTHER_CHILDREN'],df_score_output_temp['5.MULTI-ADULT_FAMILY']=y_prob.T
        
        df_lifestage_trans_score_output = pd.concat([df_lifestage_trans_score_output, df_score_output_temp], axis=0)
    
    prior_lifestage_label = [(df_lifestage_trans_score_output['PRIOR_LIFESTAGE_LABEL']==0),
                             (df_lifestage_trans_score_output['PRIOR_LIFESTAGE_LABEL']==1),
                             (df_lifestage_trans_score_output['PRIOR_LIFESTAGE_LABEL']==2),
                             (df_lifestage_trans_score_output['PRIOR_LIFESTAGE_LABEL']==3),
                             (df_lifestage_trans_score_output['PRIOR_LIFESTAGE_LABEL']==4),
                             (df_lifestage_trans_score_output['PRIOR_LIFESTAGE_LABEL']==5)
                            ]
    
    pred_lifestage_label = [(df_lifestage_trans_score_output['PRED_LABEL']==0),
                            (df_lifestage_trans_score_output['PRED_LABEL']==1),
                            (df_lifestage_trans_score_output['PRED_LABEL']==2),
                            (df_lifestage_trans_score_output['PRED_LABEL']==3),
                            (df_lifestage_trans_score_output['PRED_LABEL']==4),
                            (df_lifestage_trans_score_output['PRED_LABEL']==5)
                           ]
    
    df_lifestage_trans_score_output['PRIOR_LIFESTAGE']=np.select(prior_lifestage_label,lifestage_label_list)
    df_lifestage_trans_score_output['PRED_LIFESTAGE']=np.select(pred_lifestage_label,lifestage_label_list)
    
    utils.utils_log_step(x='Save the transition scoring output')
    utils.save_df_to_s3(df=df_lifestage_trans_score_output,
                        header=True, 
                        s3_bucket=s3_bucket,
                        data_file_path=score_data_file_path, 
                        filename='active_12m_mbr_lifestage_trans_score_output_' + str(datetime.date.today()) + '.csv')
    
    
    #' -----------------------------
    # 4.0 WRITE OUTPUT          ----
    #' -----------------------------
    utils.utils_log_step(x='Compile the scoring output')
    df_lifestage_score_all_output = pd.concat([df_lifestage_init_score_output,
                                               df_lifestage_trans_score_output[['GROUP_LOYALTY_MEMBER_NUMBER','MBR_AGE','SYSTEMDATE','PRED_LABEL','PRED_LIFESTAGE','0.SINGLE','1.COUPLE_NO_KIDS','2.FAMILY_W_BABY','3.FAMILY_W_CHILDREN','4.FAMILY_W_BABY_OTHER_CHILDREN','5.MULTI-ADULT_FAMILY']]
                                              ], 
                                              axis=0)
    df_lifestage_score_all_output['MBR_AGE']= df_lifestage_score_all_output['MBR_AGE'].astype(int)
    df_lifestage_score_all_output['SYSTEMDATE']= pd.to_datetime(df_lifestage_score_all_output['SYSTEMDATE'])
    
    utils.utils_log_step(x='Save all scoring output')
    utils.save_df_to_s3(df=df_lifestage_score_all_output,
                        header=True, 
                        s3_bucket=s3_bucket,
                        data_file_path=score_data_file_path, 
                        filename='active_12m_mbr_lifestage_all_score_output_' + str(datetime.date.today()) + '.csv')
    
    utils.utils_log_step(x='Write the scoring output to the snowflake table')
    sql=utils.read_sql(sql_file_path,'write_score_output_tbl.sql',db_tables_dict)
    sql=re.sub('score_output_file',files_dict['score_output_file'],sql)
    sql=re.sub('@stage_datascience',utils.sf_stage(),sql)
    cur.execute(sql)
    
    utils.utils_log_step(x='Update the type2 lifestage table')
    sql=utils.read_sql(sql_file_path,'update_lifestage_typ2_tbl.sql',db_tables_dict)
    cur.execute(sql)
    
    utils.utils_log_step(x='Insert into the type2 lifestage table')
    sql=utils.read_sql(sql_file_path,'insert_lifestage_typ2_tbl.sql',db_tables_dict)
    cur.execute(sql)

    utils.utils_log_step(x='Update the type2 lifestage high name table')
    sql=utils.read_sql(sql_file_path,'update_lifestage_high_name_typ2_tbl.sql',db_tables_dict)
    cur.execute(sql)

    utils.utils_log_step(x='Insert into the type2 lifestage high name table')
    sql=utils.read_sql(sql_file_path,'insert_lifestage_high_name_typ2_tbl.sql',db_tables_dict)
    cur.execute(sql)
    
    cur.close()
    
    
    #' -----------------------------
    # 5.0 END                   ----
    #' -----------------------------
    utils.utils_log_step(x='03_model_score.py has been successfully completed')





if __name__ == '__main__':
    main()












