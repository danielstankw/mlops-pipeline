# # Import necessary libraries
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from datetime import datetime, timedelta
# import pandas as pd
# import os
# import requests
# import zipfile
# from sklearn.model_selection import train_test_split
# import logging

# DATA_LINK = "https://raw.githubusercontent.com/danielstankw/data-mlops/5d593d3e7ea9c19bb154a3eb5a8527ce88543a97/archive.zip"
# DATASET_PATH = 'datasets'



# logging.basicConfig(level=logging.INFO,
#                     format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')


# def download_data(url: str, **kwargs):

#         data_dir = f'{DATASET_PATH}'
        
#         # Create 'data' directory if it doesn't exist
#         if not os.path.exists(data_dir):
#             os.makedirs(data_dir)

#         logging.info("Starting data download from URL: %s", url)
#         # Download the file
#         zip_path = os.path.join(data_dir, 'archive.zip')
#         response = requests.get(url)
#         with open(zip_path, 'wb') as file:
#             file.write(response.content)

#         # Unzip the file into the 'data' directory
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             zip_ref.extractall(data_dir)

#         # Remove the residual archive.zip file
#         os.remove(zip_path)

#         # Fetch the name of the .csv files
#         csv_files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]
#         if not csv_files:
#             raise ValueError("No CSV files found in the downloaded data.")
        
#         # Read the first CSV file into a DataFrame
#         data_path = os.path.join(data_dir, csv_files[0])
#         logging.info("Data downloaded and saved to: %s", data_path) 

#         task_instance = kwargs['ti']
#         task_instance.xcom_push(key='data_path', value=data_path)

#         return data_path


# # def save_datasets(df_train, df_val, df_test):
# #     '''
# #     Save datasets to folder
# #     '''
# #     print('Save Datasets')
# #     df_train.to_parquet(
# #         f'{AIRFLOW_HOME}/{DATASET_PATH}/{TRAIN_DATASET_PQ}', index=False
# #     )
# #     df_val.to_parquet(f'{AIRFLOW_HOME}/{DATASET_PATH}/{VAL_DATASET_PQ}', index=False)
# #     df_test.to_parquet(
# #         f'{AIRFLOW_HOME}/{DATASET_PATH}/{TEST_DATASET_PQ}', index=False
# #     )

# def preprocess_data(**kwargs):
#     """
#     Preprocess the given data by handling duplicates, and separating the target column.
    
#     Parameters:
#     - df (pd.DataFrame): The input data to be preprocessed.
    
#     Returns:
#     - pd.DataFrame: The preprocessed data.
#     - pd.Series: The target column values.
#     """
#     # Pull the data path from XCom
#     task_instance = kwargs['ti']
#     data_path = task_instance.xcom_pull(task_ids='download_data', key='data_path')
    
#     df = pd.read_csv(data_path)
#     logging.info("Starting data preprocessing. Data was succesfully read to the DataFrame")

#     # Handling duplicates
#     df = df.drop_duplicates()

#     # Drop rows where gender is 'Other'
#     df = df[df['gender'] != 'Other']

#     # Recategorize the 'smoking_history' column
#     def recategorize_smoking_status(smoking_status: str) -> str:
#         """Recategorize smoking status based on given conditions."""
#         mapping = {
#             'never': 'non_smoker',
#             'No Info': 'no_info',
#             'current': 'current',
#             'ever': 'past_smoker',
#             'former': 'past_smoker',
#             'not current': 'past_smoker'
#         }
#         return mapping.get(smoking_status, smoking_status)

#     df['smoking_history'] = df['smoking_history'].apply(recategorize_smoking_status)

#     logging.info("Data preprocessing completed.")

#     path_parts = data_path.split('/')[:-1]
#     # Join the remaining parts to get the directory path
#     directory_path = '/'.join(path_parts)

#     data_processed_url = directory_path + 'preprocessed.csv' 
#     df.to_csv(data_processed_url)
#     logging.info("Saved preprocessed data as:", data_processed_url)


# def split_data(df: pd.DataFrame, y: pd.Series, verbose: bool = False) -> tuple:
#     """
#     Split the data into training, validation, and test sets.
    
#     Parameters:
#     - df (pd.DataFrame): The input data.
#     - y (pd.Series): The target values.
#     - verbose (bool): If True, print the percentage of diabetic values in each set.
    
#     Returns:
#     - tuple: Training, validation, and test sets for both data and target values.
#     """
#         # Separate the target column
#     target = 'diabetes'
#     y = df[target].copy()
#     df.drop(target, axis=1, inplace=True)

#     df_train, df_temp, y_train, y_temp = train_test_split(
#         df, y, test_size=0.2, random_state=42, stratify=y
#     )
    
#     df_val, df_test, y_val, y_test = train_test_split(
#         df_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
#     )

#     if verbose:
#         print(f'[%] of diabetic values in the train set: {np.round(y_train.sum() / len(y_train) * 100, 2)}')
#         print(f'[%] of diabetic values in the validation set: {np.round(y_val.sum() / len(y_val) * 100, 2)}')
#         print(f'[%] of diabetic values in the test set: {np.round(y_test.sum() / len(y_test) * 100, 2)}')

#     return df_train, df_val, df_test, y_train, y_val, y_test





# # Define default arguments for the DAG
# default_args = {
#     'owner': 'airflow',
#     'depends_on_past': False,
#     'email_on_failure': False,
#     'email_on_retry': False,
#     'retries': 2,
#     'retry_delay': timedelta(minutes=5),
# }

# # Initialize the DAG
# dag = DAG(
#     'testing_dag_v04',
#     default_args=default_args,
#     description='A DAG to download, preprocess, and split data',
#     schedule_interval=timedelta(days=1),
#     start_date=datetime(2023, 9, 19),
#     catchup=False,
# )

# # Define the tasks using the PythonOperator
# task_download_data = PythonOperator(
#     task_id='download_data',
#     python_callable=download_data,
#     op_args=[DATA_LINK],  # Replace with your data link
#     provide_context=True,
#     dag=dag,
# )

# task_preprocess_data = PythonOperator(
#     task_id='preprocess_data',
#     python_callable=preprocess_data,
#     provide_context=True,
#     dag=dag,
# )

# # task_split_data = PythonOperator(
# #     task_id='split_data',
# #     python_callable=split_data,
# #     provide_context=True,
# #     dag=dag,
# # )

# # provide_context=True: allows to exchange data using XComs

# # Set task dependencies
# task_download_data >> task_preprocess_data 

# # df_original = download_data(url=DATA_LINK)
# # df = df_original.copy()
# # df, y = preprocess_data(df)
# # df_train, df_val, df_test, y_train, y_val, y_test = split_data(df, y)