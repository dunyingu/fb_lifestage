# Lifestage Rebuild

This is the code repository for the Flybuys data science member lifestage segment prediction framework, providing the Flybuys Analytics Community with a more stable and reasonable active member lifestage prediction vs. the previous version inherited from Coles. This framework is an end-to-end process, enabling the automatic execution of quarterly scoring job as well as model traning jobs (based on flybuys survey data and Coles transaction data).

## Job Design

This framework has been developed to fit into the current Sagemaker and scheduling pipeline setup - each run can be scheduled as notebook jobs by control-m using a upgraded CI/CD scheduling pipeline through Gitlab. The setup involves ECR registered python docker image installed with a list of required packages and creation of notebook instance in Sagemaker environment which is loaded with the python image via Cloudformation, followed by the execution of python scripts in the docker container.

The detailed methodology and development framework can be found in this pack: https://flybuys.sharepoint.com/:p:/r/sites/dl-flybuysdataanalytics/_layouts/15/doc2.aspx?sourcedoc=%7B403F1598-3443-43E3-AEEC-6755D277EC59%7D&file=Lifestage_Rebuild_Final_202202_V2.pptx&action=edit&mobileredirect=true&DefaultItemOpen=1&PreviousSessionID=732fc852-d181-23bd-2496-5576fdc90ccd

## Quaterly Run

For the Sagmaker job runs, the scoring module will be run separately from model training module which is only triggered when there is new survey date coming through to the source table. This means that module 00_master_score.py will act as the wrapper to complete scoring job every quarter. To run this manually in a python docker image, take the following steps on a Sagemaker Jupyterlab instance: 

1. Open up a terminal console window
2. Navigate to the supplier triggers folder e.g. cd Sagemaker/ds013_lifestage_rebuild
3. Run the shell script: 
- Running an end-to-end module: ./run_pyscript.sh 00_master_score.py
- Running an individual script: ./run_pyscript.sh 03_model_score.py

Please note the script needs to be executed manually only when the scheduled job fails. The manual execution of those scripts will write the data to the Snowflake output tables in prod, causing confusions.

## Debugging / Reviewing

Logs will be stored in S3 bucket. Logs are only written back when the scheduling pipeline has successfully kicked off.


