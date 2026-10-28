#Author: Dunyin Gu
#Creation Date: 12/17/2021

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
    utils.utils_log_step(x='Start runing 01_train_data_prep.py')
    
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
    # 1.0 EXTRACT DATA          ----
    #' -----------------------------
    utils.utils_log_step(x='Extracting and Preparing train data from source tables')
    #sql_list = []
    for sql_file in sorted(os.listdir(os.path.join(str(Path.cwd()),sql_file_path))):
        if sql_file.startswith('create_'):
            sql=utils.read_sql(sql_file_path,sql_file,db_tables_dict)
            #sql_list.append(sql)
            utils.utils_log_step(x=f'running sql: {sql_file}')
            cur.execute(sql)
        else:
            continue
    cur.close()
    
    #' -----------------------------
    # 2.0 Transform Data      ----
    #' -----------------------------
    utils.utils_log_step(x='Transforming data for feature preparation')
    
    utils.utils_log_step(x='Getting train members attributes and lifestage labels from survey')
    sql=utils.read_sql(sql_file_path,'get_train_mbr_attributes_lifestage_label.sql',db_tables_dict)
    df_srvy_lifestage_label_eng = utils.load_data(con,sql)
    df_srvy_lifestage_label_eng['GROUP_LOYALTY_MEMBER_NUMBER'] = df_srvy_lifestage_label_eng['GROUP_LOYALTY_MEMBER_NUMBER'].astype(str)
    
    utils.utils_log_step(x='Getting train members summarized and staple transaction features')
    sql=utils.read_sql(sql_file_path,'get_train_mbr_summ_staple_txn.sql',db_tables_dict)
    df_srvy_lifestage_agg_txn_summ = utils.load_data(con,sql)
    df_srvy_lifestage_agg_txn_summ['GROUP_LOYALTY_MEMBER_NUMBER'] = df_srvy_lifestage_agg_txn_summ['GROUP_LOYALTY_MEMBER_NUMBER'].astype(str)
    
    utils.utils_log_step(x='Getting train members differentating aggregate prod category transaction features')
    sql=utils.read_sql(sql_file_path,'get_train_mbr_agg_diff_txn.sql',db_tables_dict)
    df_srvy_lifestage_diff_agg_txn = utils.load_data(con,sql)
    df_srvy_lifestage_diff_agg_txn['GROUP_LOYALTY_MEMBER_NUMBER'] = df_srvy_lifestage_diff_agg_txn['GROUP_LOYALTY_MEMBER_NUMBER'].astype(str)
    
    utils.utils_log_step(x='Getting train members differentating aggregate prod category ipi')
    sql=utils.read_sql(sql_file_path,'get_train_mbr_agg_diff_ipi.sql',db_tables_dict)
    df_srvy_lifestage_diff_agg_ipi = utils.load_data(con,sql)
    df_srvy_lifestage_diff_agg_ipi['GROUP_LOYALTY_MEMBER_NUMBER'] = df_srvy_lifestage_diff_agg_ipi['GROUP_LOYALTY_MEMBER_NUMBER'].astype(str)
    
    utils.utils_log_step(x='Getting train members necessity prod category ipi')
    sql=utils.read_sql(sql_file_path,'get_train_mbr_nec_ipi.sql',db_tables_dict)
    df_srvy_lifestage_nec_txn_ipi = utils.load_data(con,sql)
    df_srvy_lifestage_nec_txn_ipi['GROUP_LOYALTY_MEMBER_NUMBER'] = df_srvy_lifestage_nec_txn_ipi['GROUP_LOYALTY_MEMBER_NUMBER'].astype(str)
    
    utils.utils_log_step(x='Getting train members selected necessity prod category pack size')
    sql=utils.read_sql(sql_file_path,'get_train_mbr_selected_staple_packsize.sql',db_tables_dict)
    df_srvy_lifestage_nec_pack_size = utils.load_data(con,sql)
    df_srvy_lifestage_nec_pack_size['GROUP_LOYALTY_MEMBER_NUMBER'] = df_srvy_lifestage_nec_pack_size['GROUP_LOYALTY_MEMBER_NUMBER'].astype(str)
    
    # pivot txn/ipi/pack size feat tables
    utils.utils_log_step(x='Pivotting train members txn/ipi/pack size data')
    df_srvy_lifestage_diff_agg_txn_pivot = utils.pivot_table(df_input=df_srvy_lifestage_diff_agg_txn,
                                                             pivot_col='LABEL_DIFF_MERCH_HIER_AGG',
                                                             index_col = ['GROUP_LOYALTY_MEMBER_NUMBER','RESPONSE_DATE'],
                                                             values=['TXN_CNT', 'SPEND','FREQ_RATIO','SPEND_RATIO'])
    
    df_srvy_lifestage_diff_agg_ipi_pivot = utils.pivot_table(df_input=df_srvy_lifestage_diff_agg_ipi,
                                                             pivot_col='LABEL_DIFF_MERCH_HIER_AGG',
                                                             index_col = ['GROUP_LOYALTY_MEMBER_NUMBER','RESPONSE_DATE'],
                                                             values=['MIN_IPI','AVG_IPI','MEDIAN_IPI','MAX_IPI'])
    
    df_srvy_lifestage_nec_txn_ipi_pivot = utils.pivot_table(df_input=df_srvy_lifestage_nec_txn_ipi,
                                                            pivot_col='NEC_SKUPLUS3_MERCH_HIER',
                                                            index_col = ['GROUP_LOYALTY_MEMBER_NUMBER','RESPONSE_DATE'],
                                                            values=['MIN_IPI','AVG_IPI','MEDIAN_IPI','MAX_IPI'])
    
    df_srvy_lifestage_nec_pack_size_pivot = utils.pivot_table(df_input=df_srvy_lifestage_nec_pack_size,
                                                              pivot_col='SKUPLUS3_MERCH_HIER_DESC',
                                                              index_col = ['GROUP_LOYALTY_MEMBER_NUMBER','RESPONSE_DATE'],
                                                              values=['MODE_PACK_SIZE'])
    
    # join all the feat tables with label
    utils.utils_log_step(x='Joining all the features with lifestage label')
    
    df_srvy_lifestage_agg_txn_feat = pd.merge(df_srvy_lifestage_agg_txn_summ.fillna(0),
                                              df_srvy_lifestage_diff_agg_txn_pivot.fillna(0), 
                                              on=['GROUP_LOYALTY_MEMBER_NUMBER','RESPONSE_DATE'],
                                              how='left')
    
    df_srvy_lifestage_agg_txn_ipi_feat = pd.merge(df_srvy_lifestage_agg_txn_feat.fillna(0),
                                                  df_srvy_lifestage_diff_agg_ipi_pivot.fillna(184), 
                                                  on=['GROUP_LOYALTY_MEMBER_NUMBER','RESPONSE_DATE'],
                                                  how='left')
    
    df_srvy_lifestage_agg_txn_ipi_feat = pd.merge(df_srvy_lifestage_agg_txn_ipi_feat.fillna(184),
                                                  df_srvy_lifestage_nec_txn_ipi_pivot.fillna(184), 
                                                  on=['GROUP_LOYALTY_MEMBER_NUMBER','RESPONSE_DATE'],
                                                  how='left')
    
    df_srvy_lifestage_agg_txn_ipi_pack_feat = pd.merge(df_srvy_lifestage_agg_txn_ipi_feat.fillna(184),
                                                       df_srvy_lifestage_nec_pack_size_pivot.fillna(0), 
                                                       on=['GROUP_LOYALTY_MEMBER_NUMBER','RESPONSE_DATE'],
                                                       how='left')
    
    df_srvy_lifestage_all_feat_label = pd.merge(df_srvy_lifestage_agg_txn_ipi_pack_feat.fillna(0),
                             df_srvy_lifestage_label_eng[['GROUP_LOYALTY_MEMBER_NUMBER','RESPONSE_DATE','PRMY_MEMBER_GENDER','MEMBER_AGE_AT_SURVEY','RFM_TILE','ACTIVE_CARDS','LIFESTAGE_LABEL']], 
                                                on=['GROUP_LOYALTY_MEMBER_NUMBER','RESPONSE_DATE'],
                                                how='left')
    
    # save all available train members feat and label df to s3
    utils.utils_log_step(x='Saving all train members (from survey) feature and label table to S3')
    utils.save_df_to_s3(df=df_srvy_lifestage_all_feat_label, 
                        header=True, 
                        s3_bucket=s3_bucket,
                        data_file_path=train_data_file_path, 
                        filename='srvy_lifestage_all_feat_label_' + str(datetime.date.today()) + '.csv'
                       )
    #' --------------------------------------------------------
    # 3.0 PREPARE TRAIN DATA FOR INITIAL CLASSIFIER MODEL  ----
    #' --------------------------------------------------------
    utils.utils_log_step(x='Preparing train data for initial life stage classifier model')
    
    # remove train mbrs with less than 3 txns/txn outliers or null attributes
    utils.utils_log_step(x='Removing train memebrs with less than 3 txns/txn outliers or null attributes')
    df_srvy_lifestage_all_feat_label_filter = df_srvy_lifestage_all_feat_label[(df_srvy_lifestage_all_feat_label['TOTAL_TXN_CNT']>=3) 
                                                                               & (df_srvy_lifestage_all_feat_label['TOTAL_TXN_CNT']<=df_srvy_lifestage_all_feat_label['TOTAL_TXN_CNT'].quantile([.99]).iloc[0])
                                                                              ]
    df_srvy_lifestage_all_feat_label_filter  = df_srvy_lifestage_all_feat_label_filter[(df_srvy_lifestage_all_feat_label_filter['TOTAL_SPEND']>0) 
                                                                                       & (df_srvy_lifestage_all_feat_label_filter['TOTAL_SPEND']<=df_srvy_lifestage_all_feat_label['TOTAL_SPEND'].quantile([.99]).iloc[0])
                                                                                      ]
    df_srvy_lifestage_all_feat_label_filter = df_srvy_lifestage_all_feat_label_filter[~df_srvy_lifestage_all_feat_label_filter['PRMY_MEMBER_GENDER'].isnull()]
    df_srvy_lifestage_all_feat_label_filter = df_srvy_lifestage_all_feat_label_filter[~df_srvy_lifestage_all_feat_label_filter['MEMBER_AGE_AT_SURVEY'].isnull()]
    df_srvy_lifestage_all_feat_label_filter = df_srvy_lifestage_all_feat_label_filter[~df_srvy_lifestage_all_feat_label_filter['RFM_TILE'].isnull()]
    df_srvy_lifestage_all_feat_label_filter = df_srvy_lifestage_all_feat_label_filter[~df_srvy_lifestage_all_feat_label_filter['ACTIVE_CARDS'].isnull()]
    
    isf = IsolationForest(contamination=0.4)
    
    ary_srvy_lifestage_all_feat_label_filter_isf = np.array([])
    
    for i in np.unique(df_srvy_lifestage_all_feat_label_filter.iloc[:,-1].values):
        x = df_srvy_lifestage_all_feat_label_filter[df_srvy_lifestage_all_feat_label_filter.iloc[:,-1]==i].iloc[:,2:-1].values
        y = df_srvy_lifestage_all_feat_label_filter[df_srvy_lifestage_all_feat_label_filter.iloc[:,-1]==i].iloc[:,-1].values
        y_hat = isf.fit_predict(x)
        mask = y_hat != -1
        temp = np.concatenate([x[mask,:],y[mask].reshape(-1,1)], axis=1)
        ary_srvy_lifestage_all_feat_label_filter_isf = np.vstack([ary_srvy_lifestage_all_feat_label_filter_isf,temp]) if ary_srvy_lifestage_all_feat_label_filter_isf.size else temp
    
    df_srvy_lifestage_all_feat_label_filter_isf = pd.DataFrame(data=ary_srvy_lifestage_all_feat_label_filter_isf, columns = list(df_srvy_lifestage_all_feat_label_filter.columns)[2:])
    df_srvy_lifestage_all_feat_label_filter_isf['LIFESTAGE_LABEL'] = df_srvy_lifestage_all_feat_label_filter_isf['LIFESTAGE_LABEL'].astype(int)
    
    # save filtered train members feat and label df to s3
    utils.utils_log_step(x='Saving filtered train members feature and label table to S3')
    #utils.save_df_to_s3(df=df_srvy_lifestage_all_feat_label_filter_isf, 
    #                   header=True, 
    #                   s3_bucket=s3_bucket,
    #                   data_file_path=train_data_file_path, 
    #                   filename='srvy_lifestage_all_feat_label_filter_isf_' + str(datetime.date.today()) + '.csv'
    #                  )
    
    # use SMOTE to oversample the minority class
    #utils.utils_log_step(x='Using SMOTE to synthesize train samples for minority class')
    
    #X = df_srvy_lifestage_all_feat_label_filter_isf.iloc[:,:-1]
    #y = df_srvy_lifestage_all_feat_label_filter_isf.iloc[:,-1]
    #oversample = SMOTE() #'not majority'
    #X_smote, y_smote = oversample.fit_resample(X, y)
    #df_srvy_lifestage_all_feat_label_filter_isf_smote=pd.concat([X_smote, y_smote], axis=1)
    # save oversampled train members feat and label df to s3
    #df_train_initi_classifier = df_srvy_lifestage_all_feat_label_filter_isf_smote
    df_train_initi_classifier = df_srvy_lifestage_all_feat_label_filter_isf
    
    utils.utils_log_step(x='Saving oversampled train members feature and label table to S3')
    utils.save_df_to_s3(df=df_train_initi_classifier, 
                        header=True, 
                        s3_bucket=s3_bucket,
                        data_file_path=train_data_file_path, 
                        filename='df_train_initi_classifier_' + str(datetime.date.today()) + '.csv'
                       )
    
    #' --------------------------------------------------------
    # 4.0 PREPARE TRAIN DATA FOR TRANSITION MODEL          ----
    #' --------------------------------------------------------
    utils.utils_log_step(x='Getting train members attributes and lifestage transition labels from survey')
    sql=utils.read_sql(sql_file_path,'get_train_mbr_attributes_lifestage_transition_label.sql',db_tables_dict)
    df_srvy_lifestage_transition = utils.load_data(con,sql)
    df_srvy_lifestage_transition['GROUP_LOYALTY_MEMBER_NUMBER'] = df_srvy_lifestage_transition['GROUP_LOYALTY_MEMBER_NUMBER'].astype(str)
    
    utils.utils_log_step(x='Joining feature table with train members with lifestage transition')
    df_srvy_lifestage_transition_feat = pd.merge(df_srvy_lifestage_all_feat_label_filter.iloc[:,:-1],
                                                 df_srvy_lifestage_transition[['GROUP_LOYALTY_MEMBER_NUMBER','RESPONSE_DATE','MONTHS_BETWEEN_TRANSITION','PRIOR_LIFESTAGE_LABEL','TARGET_LIFESTAGE_LABEL']], 
                                                 on=['GROUP_LOYALTY_MEMBER_NUMBER','RESPONSE_DATE'],
                                                 how = 'inner'
                                               )
    df_srvy_lifestage_transition_feat=df_srvy_lifestage_transition_feat.drop(['PRMY_MEMBER_GENDER'], axis=1)
    
    utils.utils_log_step(x='Saving train members feature and transition label table to S3')
    utils.save_df_to_s3(df=df_srvy_lifestage_transition_feat,
                        header=True, 
                        s3_bucket=s3_bucket,
                        data_file_path=train_data_file_path, 
                        filename='srvy_lifestage_transition_feat_' + str(datetime.date.today()) + '.csv'
                       )
    
    # use SMOTE to oversample the minority class for life stage transition
    utils.utils_log_step(x='Using SMOTE to synthesize train samples for minority class per each life stage transition')
    
    class_num = len(lifestage_label_dict)
    for i in range(class_num):
        df_trans = df_srvy_lifestage_transition_feat[df_srvy_lifestage_transition_feat['PRIOR_LIFESTAGE_LABEL']==i]
        X=  df_trans.iloc[:,2:-2]
        y=  df_trans.iloc[:,-1]
        
        class_cnt_dict = Counter(y)
        sampling_strategy_dict = utils.sampling_strategy(class_cnt_dict=class_cnt_dict,class_num=class_num,ratio=0.2)
        oversample = SMOTE(sampling_strategy=sampling_strategy_dict) #'not majority'
        X_smote, y_smote = oversample.fit_resample(X, y)
        df_train_smote = pd.concat([y_smote, X_smote], axis=1)
        
        lifestage_label=lifestage_label_dict[i]
        data_file_name = f'df_train_smote_{lifestage_label}_transition_{str(datetime.date.today())}.csv'
        
        utils.utils_log_step(x= f'Saving oversampled train data with prior life stage of {lifestage_label} to S3')
        utils.save_df_to_s3(df=df_train_smote, 
                            header=None, 
                            s3_bucket=s3_bucket,
                            data_file_path=train_data_file_path, 
                            filename=data_file_name
                           )
    
    #' --------------------------------------------------------
    # 5.0 END
    #' --------------------------------------------------------
    utils.utils_log_step(x='01_train_data_prep.py has been successfully completed')





if __name__ == '__main__':
    main()
