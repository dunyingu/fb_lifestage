COPY INTO ?score_output_tbl
FROM 
@stage_datascience/Program/ds013_lifestage_rebuild/data/score/score_output_file
--@pub_flybuys.dl_utility.flybuys_ctrlm_dev/Program/ds013_lifestage_rebuild/data/score/active_12m_mbr_lifestage_score_all_output_2021-12-08.csv
FILE_FORMAT=exp_all.data_science.ds003_csv
ON_ERROR = 'continue'