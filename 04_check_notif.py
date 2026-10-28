#Author: Dunyin Gu
#Creation Date: 07/18/2022

# import libraries
import os
import boto3
from pathlib import Path
import yaml
import snowflake.connector

import utils

import pandas as pd
import numpy as np
import datetime
import s3fs
import re

import email,smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import matplotlib.pyplot as plt
import seaborn as sns


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
    # 0.0 SETUP                 ----
    #' -----------------------------
    utils.utils_log_step(x='Start runing 04_check_notif.py')
    
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
    email_list = config['email']
    
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
    # 1.0 MODEL OUTPUT CHECK    ----
    #' -----------------------------
    utils.utils_log_step(x='Count new members scored for the first time')
    # count of new members scored
    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=score_data_file_path)
    at_list = pd.DataFrame.from_dict(response['Contents'])
    at_list.columns= at_list.columns.str.strip().str.lower()
    at_list.rename(columns={'key':'filename','lastmodified':'file_createtime'}, inplace=True)
    at_list = at_list[['filename','file_createtime']]
    at_list['file_createtime'] = at_list['file_createtime'].dt.tz_convert("Australia/Melbourne")
    at_list_init = at_list[at_list.filename.str.contains('active_12m_mbr_lifestage_init_score_output_')]
    at_list_init = at_list_init[at_list_init['file_createtime']==max(at_list_init.file_createtime)].reset_index(drop=True)
    init_score_output_name = at_list_init['filename'][0]
    init_score_output_loc = f's3://{s3_bucket}/{init_score_output_name}'
    df_init_score_output = pd.read_csv(init_score_output_loc,header=0)
    new_mbr_cnt = '{:,.0f}'.format(df_init_score_output['GROUP_LOYALTY_MEMBER_NUMBER'].nunique())
    
    utils.utils_log_step(x='Count members scored previously')
    # count of members that were scored previously
    at_list_trans = at_list[at_list.filename.str.contains('active_12m_mbr_lifestage_trans_score_output_')]
    at_list_trans = at_list_trans[at_list_trans['file_createtime']==max(at_list_trans.file_createtime)].reset_index(drop=True)
    trans_score_output_name = at_list_trans['filename'][0]
    trans_score_output_loc = f's3://{s3_bucket}/{trans_score_output_name}'
    df_trans_score_output = pd.read_csv(trans_score_output_loc,header=0)
    prev_scored_mbr_cnt = '{:,.0f}'.format(df_trans_score_output['GROUP_LOYALTY_MEMBER_NUMBER'].nunique())
    
    utils.utils_log_step(x='Count members whose lifestage have changed')
    # count of members whose high name lifestage have transitioned to a different segment than previous
    trans_mbr_cnt = '{:,.0f}'.format(df_trans_score_output.loc[~(df_trans_score_output['PRED_LIFESTAGE']==df_trans_score_output['PRIOR_LIFESTAGE'])]['GROUP_LOYALTY_MEMBER_NUMBER'].nunique())
    
    utils.utils_log_step(x='Count members for each lifestage segment change')
    # summary of member count for each high name lifestage segment transition
    df_trans_score_output_t = df_trans_score_output.loc[~(df_trans_score_output['PRED_LIFESTAGE']==df_trans_score_output['PRIOR_LIFESTAGE'])][['GROUP_LOYALTY_MEMBER_NUMBER','PRIOR_LIFESTAGE_LABEL','PRIOR_LIFESTAGE','PRED_LABEL','PRED_LIFESTAGE']]
    df_trans_score_output_t['LIFESTAGE_SEGMENT_TRANSITION']=df_trans_score_output_t['PRIOR_LIFESTAGE_LABEL'].astype(str)+'.'+df_trans_score_output_t['PRIOR_LIFESTAGE']+' -> '+df_trans_score_output_t['PRED_LABEL'].astype(str)+'.'+df_trans_score_output_t['PRED_LIFESTAGE']
    df_trans_score_summ=df_trans_score_output_t.groupby('LIFESTAGE_SEGMENT_TRANSITION')['GROUP_LOYALTY_MEMBER_NUMBER'].nunique().reset_index().rename(columns={'GROUP_LOYALTY_MEMBER_NUMBER':'TOTAL_MEMBER_CNT'})
    
    utils.utils_log_step(x='Summarize members lifestage distribution of current scoring job')
    sql=utils.read_sql(sql_file_path,'check_lifestage_dist.sql',db_tables_dict)
    df_score_dist = utils.load_data(con,sql)
    
    utils.utils_log_step(x='Compare members lifestage between current and last scoring job')
    # query the lifestage segments change between current scoring cycle vs last scoring cycle
    sql=utils.read_sql(sql_file_path,'comp_score_output_current_last.sql',db_tables_dict)
    df_comp_score_cur_last = utils.load_data(con,sql)
    df_comp_score_cur_last['PREVIOUS_SCORE_DATE'] = pd.to_datetime(df_comp_score_cur_last['PREVIOUS_SCORE_DATE'])
    df_comp_score_cur_last['CURRENT_SCORE_DATE'] = pd.to_datetime(df_comp_score_cur_last['CURRENT_SCORE_DATE'])
    
    prev_score_date = str(df_comp_score_cur_last['PREVIOUS_SCORE_DATE'].min().date())
    curr_score_date = str(df_comp_score_cur_last['CURRENT_SCORE_DATE'].max().date())
    
    utils.utils_log_step(x='Count members scored in last scoring cycle but not included in the current job')
    # count of members who have been scored in last scoring cycle but not included in the current job
    df_mbrs_disappear = df_comp_score_cur_last.loc[df_comp_score_cur_last['PREVIOUS_SCORE_DATE']==df_comp_score_cur_last['PREVIOUS_SCORE_DATE'].min()]
    df_mbrs_disappear = df_mbrs_disappear.loc[df_mbrs_disappear['CURRENT_LIFESTAGE_SEGMENT'].isnull()]
    disappear_mbrs_cont = '{:,.0f}'.format(df_mbrs_disappear['GROUP_LOYALTY_MEMBER_NUMBER'].nunique())
    
    utils.utils_log_step(x='Plot the 3M transition matrix from last to current scoring')
    df_comp = df_comp_score_cur_last.loc[~df_comp_score_cur_last['CURRENT_LIFESTAGE_SEGMENT'].isnull()][['PREVIOUS_LIFESTAGE_SEGMENT','CURRENT_LIFESTAGE_SEGMENT']].reset_index()
    # plot the overview of 3M lifestage segments change between current scoring cycle vs last scoring cycle
    comp_matrix = pd.crosstab(df_comp['PREVIOUS_LIFESTAGE_SEGMENT'],df_comp['CURRENT_LIFESTAGE_SEGMENT'],
                              rownames=["Previous Lifestage High Name"],colnames=["Current Lifestage High Name"],
                              normalize="index"
                              )
    
    plt.rcParams["figure.figsize"] = (44,17)
    plt.rcParams['xtick.labelsize']= 19
    plt.rcParams['ytick.labelsize']= 19
    
    fig, ax = plt.subplots()
    sns.set(font_scale=2.4)
    sns.heatmap(comp_matrix, annot=True, annot_kws = {'size':24} ,fmt=".3f",cmap="BuPu")
    plt.title('Overview of 3M Lifestage Segment Movement',fontsize=30)
    plt.xlabel(f'Current {curr_score_date} Lifestage High Name',fontsize=27)
    plt.ylabel(f'Previous {prev_score_date} Lifestage High Name',fontsize=27)
    
    plot_name = "comp_matrix.png"
    fig.savefig(plot_name,dpi=fig.dpi)
    
    # write the scoring output to the member segment file for FBDL layer
    utils.utils_log_step(x='Write to member segment file')
    sql=utils.read_sql(sql_file_path,'write_mbr_segment_file.sql',db_tables_dict)
    sql=re.sub('current_date',config['run_date'],sql)
    cur.execute(sql)
    
    cur.close()
    
    #' -----------------------------
    # 2.0 SEND OUT NOTIFICATION ----
    #' -----------------------------
    utils.utils_log_step(x='Email send statistics & summary to stakeholders')
    message = MIMEMultipart()
    
    message_subject = f"{curr_score_date} Lifestage Scoring Job Summary"
    
    message['FROM'] = email_list['stream']
    message['TO']= email_list['developer']
    message['Cc'] = email_list['stakeholders']
    message['Subject']=message_subject
    
    nchrTO  = email_list['developer'] + ',' + email_list['stakeholders']
    
    with open(plot_name, 'rb') as fp:
        img = MIMEImage(fp.read())
        img.add_header('Content-Disposition', 'attachment', filename=plot_name)
        img.add_header('X-Attachment-Id', '0')
        img.add_header('Content-ID', '<0>')
        fp.close()
        message.attach(img)
    
    html = f"""\
    <html>
        <head>
        </head><body>
            <p>Lifestage v2.0 scoring job has been completed on {curr_score_date}.</p>
            <p>Statistics of scoring output:<br />{new_mbr_cnt} members are scored by lifestage model for the first time and {prev_scored_mbr_cnt} members were scored previously.<br />{trans_mbr_cnt} members are predicted a different lifestage segment than the previous scoring cycle.<br />{disappear_mbrs_cont} members who were scored in the previous scoring cycle on {prev_score_date} are not eligible for the current scoring cycle.</p>
            <p>Summary of lifestage segment distribution in current vs. previous scoring cycle in the below table:</p>
            <p>{df_score_dist.to_html(index=False).replace('<th>','<th style = "background-color: royalblue; color: white">')}</p>
            <p>Member count for each pair of lifestage segment change in the below table:</p>
            <p>{df_trans_score_summ.to_html(index=False).replace('<th>','<th style = "background-color: royalblue; color: white">')}</p>
            <p>Overview of 3-month lifestage transition from last to current scoring cycle in the below plot:</p>
            <p><img src="cid:0"></p>
        </body>
    </html>
    """
    
    message.attach(MIMEText(html, 'html', 'utf-8'))
    
    port = 25
    host = "email-smtp.ap-southeast-2.amazonaws.com"
    
    server = smtplib.SMTP(host=host, port=port)
    server.sendmail(email_list['stream'], nchrTO, message.as_string())
    server.quit()
    
    #' -----------------------------
    # 3.0 END                   ----
    #' -----------------------------
    utils.utils_log_step(x='04_check_notif.py has been successfully completed')
    

if __name__ == '__main__':
    main()
