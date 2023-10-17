from datetime import datetime, timedelta
import os
from airflow.decorators import task, dag
import mlflow
from datetime import datetime
import logging



logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=2)
}

AIRFLOW_HOME  = os.getenv('AIRFLOW_HOME')
EXPERIMENT_NAME = 'test-experiment-dags'
MLFLOW_TRACKING_URI = "http://mlflow-server:5000"
# MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"


@dag('mlflow_dag_v5',
    default_args=default_args,
    description='A MLflow test dag',
    schedule_interval=None,#timedelta(days=1),
    start_date=datetime(2023, 9, 25),
    catchup=False,
    tags = ['diabetes'],
    dagrun_timeout=timedelta(minutes=10))

def mlflow_dag():
    
    @task()
    def mlfow_setup():
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logging.info("MLflow trackng URI: %s", mlflow.get_tracking_uri())

        logging.info("Experiment name is %s", EXPERIMENT_NAME)
        mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
        logging.info("Experiment was set as: %s", EXPERIMENT_NAME)


    mlfow_setup()

mlflow_dag_obj = mlflow_dag()

