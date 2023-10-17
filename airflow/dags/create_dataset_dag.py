# Import necessary libraries
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import os
import requests
import zipfile
from airflow.utils.dates import days_ago
from sklearn.model_selection import train_test_split
from airflow.decorators import task, dag
import logging


# Data URL dataset
DATA_LINK = "https://raw.githubusercontent.com/danielstankw/data-mlops/5d593d3e7ea9c19bb154a3eb5a8527ce88543a97/archive.zip"
AIRFLOW_HOME   = os.getenv('AIRFLOW_HOME')
DATASET = 'dataset'

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=2)
}

@dag('data_prep_v1',
    default_args=default_args,
    description='A DAG to download, preprocess, and split data',
    schedule_interval=None,#timedelta(days=1),
    start_date=days_ago(1),#datetime(2023, 10, 9),
    catchup=False,
    tags = ['diabetes-mlops'],
    dagrun_timeout=timedelta(minutes=10))


def create_datasets():
    
    @task()
    def download_data(url: str) -> str:
        try:
            dataset_path = os.path.join(AIRFLOW_HOME, DATASET)

            logging.info("Data dir set to: %s", AIRFLOW_HOME)

            # Create directory if it doesn't exist
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)

            logging.info("Starting data download from URL: %s", url)
            # Download the file
            zip_path = os.path.join(dataset_path, 'archive.zip')
            response = requests.get(url)
            # Raise an HTTPError for bad responses
            response.raise_for_status()  

            with open(zip_path, 'wb') as file:
                file.write(response.content)

            # Unzip the file into the 'data' directory
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_path)

            # Remove the residual archive.zip file
            os.remove(zip_path)

            # Fetch the name of the .csv files
            csv_files = [file for file in os.listdir(dataset_path) if file.endswith('.csv')]
            if not csv_files:
                raise ValueError("No CSV files found in the downloaded data.")
            
            data_path = os.path.join(dataset_path, csv_files[0])
            logging.info("Data downloaded and saved to: %s", data_path) 

            return data_path

        except requests.exceptions.RequestException as re:
            logging.error("Error occurred while making a request to URL: %s, Error: %s", url, re)
        except zipfile.BadZipFile as bzf:
            logging.error("Error occurred while extracting the zip file, Error: %s", bzf)
        except Exception as e:
            logging.error("An unexpected error occurred, Error: %s", e)
   
    @task()
    def preprocess_data(data_path: str) -> str:

        df = pd.read_csv(data_path)
        logging.info("Starting data preprocessing. \n Data was succesfully read to the DataFrame")

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

        logging.info("Data preprocessing completed.")

        data_processed_url = os.path.join(os.path.dirname(data_path), 'preprocessed_data.csv')
        
        # Save the DataFrame to a CSV file
        df.to_csv(data_processed_url, index=False)
        logging.info("Saved preprocessed data as: %s", data_processed_url)
        
        # Check whether the file was created successfully
        if os.path.exists(data_processed_url):
            logging.info("File was created successfully.")
        else:
            logging.error("Failed to create the file.")
            raise FileNotFoundError(f"Failed to create the file at {data_processed_url}")
    
        return data_processed_url

    @task()
    def split_data(processed_url : str):
        try:
            absolute_path = os.path.abspath(processed_url)
            logging.info("Trying to read preprocessed data from: %s", processed_url)
            logging.info("Absolute path: %s", absolute_path)
        
            # Check if the file exists
            if not os.path.exists(absolute_path):
                logging.error("File does not exist: %s", absolute_path)
                raise FileNotFoundError(f"No such file or directory: '{absolute_path}'")
        
            df = pd.read_csv(processed_url)
            logging.info("Successfully loaded pre-processed data.")
                # Separate the target column
            target = 'diabetes'
            y = df[target].copy()
            # df.drop(target, axis=1, inplace=True)

            df_train, df_temp, y_train, y_temp = train_test_split(
                df, y, test_size=0.2, random_state=42, stratify=y
            )
            
            df_val, df_test, y_val, y_test = train_test_split(
                df_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )
            logging.info("Successfully Split the Data")
            dataset_path = os.path.join(AIRFLOW_HOME, DATASET)
            # Save the split datasets as CSV files
            df_train.to_csv(os.path.join(dataset_path, 'df_train.csv'), index=False)
            df_val.to_csv(os.path.join(dataset_path, 'df_val.csv'), index=False)
            df_test.to_csv(os.path.join(dataset_path, 'df_test.csv'), index=False)

            logging.info("Saved df_train to path: %s", os.path.join(dataset_path, 'df_train.csv'))
            logging.info("Successfully Saved the Split Data as CSV Files")

        except Exception as e:
            logging.error("An error occurred: %s", e)
            raise
        
    data_path = download_data(DATA_LINK)
    data_preprocess_path = preprocess_data(data_path)
    split_data(data_preprocess_path)

data_dag = create_datasets() 