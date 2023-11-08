from datetime import datetime, timedelta
import os
from airflow.decorators import task, dag
import pandas as pd
import optuna
import xgboost as xgb
from airflow.utils.dates import days_ago

import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import  f1_score
from datetime import datetime
import logging


EXPERIMENT_NAME = 'experiment-0'

AIRFLOW_HOME  = os.getenv('AIRFLOW_HOME')
DATASET = 'dataset'
MLFLOW_TRACKING_URI = "http://mlflow-server:5000"


logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')



default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=2)}


@dag('training_dag_v6',
    default_args=default_args,
    description='A DAG for optuna hyperprameter tunning of the model',
    schedule_interval=None,#timedelta(days=1),
    start_date=days_ago(1),
    catchup=False,
    tags = ['diabetes-mlops'],
    dagrun_timeout=timedelta(minutes=10))

def train_models():

    @task()
    def init_mlflow():
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logging.info("MLflow tracking URI set to: %s", mlflow.get_tracking_uri())

        mlflow.set_experiment(EXPERIMENT_NAME)
        logging.info("Experiment was set to: %s", EXPERIMENT_NAME)

        # Get the experiment details
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        # Extract the artifact location
        artifact_location = experiment.artifact_location
        logging.info("Artifact location was set to: %s", artifact_location)

    def load_prepare_data_for_training():

        dataset_path = os.path.join(AIRFLOW_HOME, DATASET)

        df_train = pd.read_csv(os.path.join(dataset_path, 'df_train.csv'))
        logging.info("Training data loaded from: %s", os.path.join(dataset_path, 'df_train.csv'))

        df_val = pd.read_csv(os.path.join(dataset_path, 'df_val.csv'))
        logging.info("Validation data loaded from: %s", os.path.join(dataset_path, 'df_val.csv'))


        target = 'diabetes'
        y_train = df_train[target].values
        y_val = df_val[target].values

        df_train = df_train.drop(target, axis=1)
        df_val = df_val.drop(target, axis=1)
        
        # one-hot encoding categorical features
        dv = DictVectorizer()
        train_dicts = df_train.to_dict(orient='records')
        X_train = dv.fit_transform(train_dicts)
        logging.info("Encoded Train Data")

        val_dicts = df_val.to_dict(orient='records')
        X_val = dv.transform(val_dicts)
        logging.info("Encoded Validation Data")


        # os.makedirs("model", exist_ok=True)
        # with open("model/preprocessor.b", "wb") as f:
        #     pickle.dump(dv, f)

        return X_train, X_val, y_train, y_val
  
          
    @task()
    def optimize_params(n_trials=3):
            

        logging.info("Default artifacts URI: %s", mlflow.get_artifact_uri())

        X_train, X_val, y_train, y_val = load_prepare_data_for_training()

        # Creating DMatrix objects inside the task
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_val, label=y_val)

        mlflow.autolog(disable=True)
        def objective(trial):

            if mlflow.active_run():
                mlflow.end_run()

            with mlflow.start_run():

                logging.info("Starting trial #: %s", trial.number)
                
                mlflow.set_tag("model", "xgboost")

                constant_params = {
                    "objective": "binary:logistic",
                    "eval_metric" : "logloss"}
                
                hyper_params = {
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                    "max_depth": trial.suggest_int("max_depth", 1, 10),
                    "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 20)
                }

                combined_params = {**constant_params, **hyper_params}

                mlflow.log_params(combined_params)

                classifier = xgb.train(
                    params=combined_params,
                    dtrain=dtrain,
                    evals=[(dvalid, "validation")],
                    verbose_eval=100,
                    early_stopping_rounds=10,
                    num_boost_round=500#1000
                )
                
                # output are the probabilities, 
                # we need to convert to the binary classes
                y_pred_proba = classifier.predict(dvalid)
                y_pred = (y_pred_proba > 0.5).astype(int)
                
                f1_metric = f1_score(y_true=y_val, y_pred=y_pred) 

                mlflow.log_metric('f1', f1_metric)

                mlflow.xgboost.log_model(classifier, 
                                         artifact_path="models")

                
            return f1_metric
        
        # To execute the optimization, we create a study object and pass 
        # the objective function to the optimize method.
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        return 

  
    # study = optimize_params(dtrain, y_train, dvalid, y_val, n_trials=2)
    initialize_mlflow_task = init_mlflow()
    optimization_task = optimize_params(n_trials=2)

    # Set the task dependencies
    initialize_mlflow_task >> optimization_task

training_dag = train_models()

