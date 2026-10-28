#' ----------------------------------------------------------------------
#' Title: user_credentials.R
#' Author: Sunil Godavani
#' Date: 2020-05-28
#'
#' Description:
#' Function code to define connections based on credentials for CONTROL-M
#' 
#' Notes:
#' - To be deployed into the same directory as the running script
#' - Code to add to running script: source("user_credentials.R")
#' ----------------------------------------------------------------------

# load libraries
suppressPackageStartupMessages({
  library(reticulate)
  library(jsonlite)
  library(DBI)
  library(odbc)
})

# define snowflake connection from credentials in secrets manager
.sf_connect <- function(account) {
  ### create a Secrets Manager client
  boto3 <- import('boto3')
  session <- boto3$session$Session()
  client <- session$client(
    service_name='secretsmanager',
    region_name='ap-southeast-2'
  )

  ### get secret
  secret_name = "db/flybuys/analytics/mig_analytics_rw"
  get_secret_value_response <- client$get_secret_value(SecretId=secret_name)
  secret <- get_secret_value_response['SecretString']
  secret_json <- fromJSON(secret$SecretString)

  # create a direct connection using credentials from secret
  dbConnect(
    drv = odbc()
    , Server = paste(secret_json$account,'ap-southeast-2.snowflakecomputing.com', sep='.')
    , driver = "SnowflakeDSIIDriver"
    , UID = secret_json$username
    , PWD = secret_json$password
  )
}

# define s3 bucket from parameter store
.s3_bucket <- function() {
  ### create an SSM client
  ssm_boto3 <- import('boto3')
  ssm_session <- ssm_boto3$session$Session()
  ssm_client <- ssm_session$client(
    service_name='ssm',
    region_name="ap-southeast-2"
  )

  ### get param
  param_name = "/analytics/env/s3/flybuys/ctrlm/output/prod"
  get_param_value_response <- ssm_client$get_parameter(Name=param_name, WithDecryption=FALSE)
  s3_bucket_name <- gsub('data-platform','analytics-platform', get_param_value_response$Parameter$Value)

  return(s3_bucket_name)
}

# define snowflake external stage
.sf_stage <- function() {
  stg = '@pub_flybuys.dl_utility.flybuys_ctrlm_prod'
  return(stg)
}