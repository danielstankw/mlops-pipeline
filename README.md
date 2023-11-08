for airflow data is stored in the scheduler container /opt/airflow 

## setting up the s3 bucket
1.  create an s3 bucket called mlops-s3-bucket-daniel
2. https://github.com/danielstankw/mlops/blob/main/AWS_setup.md  --> follow every step!
3. AWS Credentials in MLflow Service: While you have set the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in the Airflow services, these environment variables are not being set in the mlflow-server service. MLflow needs these credentials to access your S3 bucket. Add them to the mlflow-server service:

4. 127.0.0.1 might not work as it refers to the localhost within each container. 
Instead, you'd typically use the name of the MLflow container
MLFLOW_TRACKING_URI = "http://mlflow-server:5000"

5. 
