import os 
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import requests
import zipfile
import pickle

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, roc_auc_score, precision_score, average_precision_score

from datetime import datetime


DATA_LINK = "https://raw.githubusercontent.com/danielstankw/data-mlops/5d593d3e7ea9c19bb154a3eb5a8527ce88543a97/archive.zip"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def download_data(url: str) -> pd.DataFrame:
    """
    Download data from a given URL, unzip it to data_dir folder, and return the data as a DataFrame.
    
    Parameters:
    - url (str): The URL of the data to be downloaded.
    
    Returns:
    - pd.DataFrame: The downloaded data as a DataFrame.
    """

    data_dir = 'data'
    
    # Create 'data' directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Download the file
    zip_path = os.path.join(data_dir, 'archive.zip')
    response = requests.get(url)
    with open(zip_path, 'wb') as file:
        file.write(response.content)

    # Unzip the file into the 'data' directory
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    # Remove the residual archive.zip file
    os.remove(zip_path)

    # Fetch the name of the .csv files
    csv_files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]
    if not csv_files:
        raise ValueError("No CSV files found in the downloaded data.")
    
    # Read the first CSV file into a DataFrame
    df = pd.read_csv(os.path.join(data_dir, csv_files[0]))

    return df

def preprocess_data(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """
    Preprocess the given data by handling duplicates, and separating the target column.
    
    Parameters:
    - df (pd.DataFrame): The input data to be preprocessed.
    
    Returns:
    - pd.DataFrame: The preprocessed data.
    - pd.Series: The target column values.
    """
    
    # Handling duplicates
    df = df.drop_duplicates()

    # Drop rows where gender is 'Other'
    df = df[df['gender'] != 'Other']

    # Recategorize the 'smoking_history' column
    def recategorize_smoking_status(smoking_status: str) -> str:
        """Recategorize smoking status based on given conditions."""
        mapping = {
            'never': 'non_smoker',
            'No Info': 'no_info',
            'current': 'current',
            'ever': 'past_smoker',
            'former': 'past_smoker',
            'not current': 'past_smoker'
        }
        return mapping.get(smoking_status, smoking_status)

    df['smoking_history'] = df['smoking_history'].apply(recategorize_smoking_status)

    # Separate the target column
    target = 'diabetes'
    y = df[target].copy()
    df.drop(target, axis=1, inplace=True)

    return df, y

def split_data(df: pd.DataFrame, y: pd.Series, verbose: bool = False) -> tuple:
    """
    Split the data into training, validation, and test sets.
    
    Parameters:
    - df (pd.DataFrame): The input data.
    - y (pd.Series): The target values.
    - verbose (bool): If True, print the percentage of diabetic values in each set.
    
    Returns:
    - tuple: Training, validation, and test sets for both data and target values.
    """
    
    df_train, df_temp, y_train, y_temp = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    df_val, df_test, y_val, y_test = train_test_split(
        df_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    if verbose:
        print(f'[%] of diabetic values in the train set: {np.round(y_train.sum() / len(y_train) * 100, 2)}')
        print(f'[%] of diabetic values in the validation set: {np.round(y_val.sum() / len(y_val) * 100, 2)}')
        print(f'[%] of diabetic values in the test set: {np.round(y_test.sum() / len(y_test) * 100, 2)}')

    return df_train, df_val, df_test, y_train, y_val, y_test

def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> tuple:
    """
    Calculate various metrics to evaluate the performance of a classification model.
    
    Parameters:
    - y_true (pd.Series): The true target values.
    - y_pred (pd.Series): The predicted target values.
    
    Returns:
    - tuple: AUC, F1 score, precision, recall, average precision, and accuracy.
    """
    
    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    return auc, f1, precision, recall, avg_precision, accuracy


def optimize_params(dtrain, y_train, dvalid, y_val, n_trials=3):
    """
    Optimize hyperparameters for an XGBoost classifier using Optuna.
    
    Parameters:
    - X_train, y_train: Training data and labels.
    - X_val, y_val: Validation data and labels.
    - n_trials (int): Number of trials for hyperparameter optimization.
    
    Returns:
    - optuna.study.Study: Study object with optimization results.
    """

    # dtrain = xgb.DMatrix(X_train, label=y_train)
    # dvalid = xgb.DMatrix(X_val, label=y_val)
    
    def objective(trial):
        """
        Objective function for Optuna optimization.
        
        Parameters:
        - trial: Optuna trial object.
        
        Returns:
        - float: F1 score for the given hyperparameters.
        """

        with mlflow.start_run():
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
                combined_params,
                dtrain,
                evals=[(dvalid, "validation")],
                verbose_eval=1,
                early_stopping_rounds=10,
                num_boost_round=1000
            )
            
            # output are the probabilities, we need to convert to the binary classes
            y_pred_proba = classifier.predict(dvalid)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            f1_metric = f1_score(y_true=y_val, y_pred=y_pred) 

            mlflow.log_metric('f1', f1_metric)
            
        return f1_metric
    
    # To execute the optimization, we create a study object and pass 
    # the objective function to the optimize method.
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    return study




def train_optimized_model(study, dtrain, y_train, dvalid, y_val):
    """
    Train an XGBoost classifier using the best parameters from a given study.
    
    Parameters:
    - study: Optuna study object with optimization results.
    - train_data, val_data: Training and validation data.
    - y_train, y_val: Training and validation labels.
    """
    
    # dtrain = xgb.DMatrix(X_train, label=y_train)
    # dvalid = xgb.DMatrix(X_val, label=y_val)
    
    with mlflow.start_run():
        
        constant_params = {
            "objective": "binary:logistic"}

        # Combine constant and hyperopt parameters
        combined_params = {**study.best_params, **constant_params}
        mlflow.log_params(combined_params)

        classifier = xgb.train(
            combined_params,
            dtrain,
            evals=[(dvalid, "validation")],
            verbose_eval=1,
            early_stopping_rounds=10,
            num_boost_round=1000
        )
        
        y_pred_proba = classifier.predict(dvalid)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        f1_metric = f1_score(y_val, y_pred)
        print(f'F1 Score: {f1_metric}')

        mlflow.log_metric("f1", f1_metric)
        mlflow.sklearn.log_model(classifier, artifact_path="models_mlflow")

        print(f"Default artifacts URI: '{mlflow.get_artifact_uri()}'")

        

def predict(run_id):

    # RUN_ID = '956c7d6a75d24c76b06606eeec98fd86'
    model_uri = f'runs:/{run_id}/models_mlflow'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    # loaded_model = mlflow.xgboost.load_model(model_uri)


    y_pred_proba = loaded_model.predict(dvalid)
    y_pred = (y_pred_proba > 0.5).astype(int)
    f1 = f1_score(y_val, y_pred)
    print('F1 Score: ',f1)

    return y_pred


def train_pipeline():
    df_original = download_data(url=DATA_LINK)
    df = df_original.copy()

    # preprocess data
    df, y = preprocess_data(df)

    # data split
    df_train, df_val, df_test, y_train, y_val, y_test = split_data(df, y)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(mlflow.get_tracking_uri())
    # set an experiment, if it doesnt exist create one
    mlflow.set_experiment(experiment_name='final-experiment')

        # features
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()
    all_features = categorical_cols + numerical_cols

    # one-hot encoding categorical features
    dv = DictVectorizer()
    train_dicts = df_train[all_features].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[all_features].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    os.makedirs("model", exist_ok=True)
    with open("model/preprocessor.b", "wb") as f:
        pickle.dump(dv, f)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_val, label=y_val)

    study = optimize_params(dtrain, y_train, dvalid, y_val, n_trials=2)



    # train best model
    train_optimized_model(study, dtrain, y_train, dvalid, y_val)






