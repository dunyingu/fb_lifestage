# Lifestage Exploration

This is the code repository for the lifestage segment prediction framework, providing the FB Analytics Community with a more stable and reasonable active member lifestage prediction vs. the previous legacy version. This framework is an end-to-end process, enabling the automatic execution of quarterly scoring job as well as model traning jobs (based on survey data and transaction data).

## Job Design

This framework has been developed to fit into the current Sagemaker and scheduling pipeline setup - each run can be scheduled as notebook jobs by control-m using a upgraded CI/CD scheduling pipeline through Gitlab. The setup involves ECR registered python docker image installed with a list of required packages and creation of notebook instance in Sagemaker environment which is loaded with the python image via Cloudformation, followed by the execution of python scripts in the docker container.

The detailed methodology and development framework can be found in the paper.

## Debugging / Reviewing

Logs will be stored in S3 bucket. Logs are only written back when the scheduling pipeline has successfully kicked off.


