#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os 
from sklearn.feature_extraction import DictVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import mlflow
from sklearn.pipeline import make_pipeline


# In[3]:


DATA_LINK = "https://raw.githubusercontent.com/danielstankw/data-mlops/5d593d3e7ea9c19bb154a3eb5a8527ce88543a97/archive.zip"


# In[4]:


# Check if the folder 'data' exists; if not, create it
if not os.path.exists('data'):
    os.makedirs('data')

# Download the file
get_ipython().system('wget -O data/archive.zip $DATA_LINK')

# Unzip the file into the 'data' folder
get_ipython().system('unzip -o data/archive.zip -d data/')

# Remove the archive.zip file
os.remove('data/archive.zip')

# Fetch and display the name of the .csv files
csv_files = [file for file in os.listdir('data') if file.endswith('.csv')]
print("\nCSV files in the 'data' folder:")
print(csv_files)


# ### Reading DataFrame

# In[5]:


df = pd.read_csv("./data/" + csv_files[0])


# In[6]:


df.head()


# In[7]:


df.shape


# In[8]:


# Handle duplicates
duplicate_rows_data = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_data.shape)
df = df.drop_duplicates()
print('Shape of df after removal', df.shape)


# ### EDA

# In[9]:


df.dtypes


# In[10]:


# separate numerical and categorical features
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()
print('categorical', categorical_cols)
print('numerical', numerical_cols)


# ### Check unique values

# Columns we need to encode
# * `gender`
# * `smoking history` 
# 

# In[11]:


df[categorical_cols].nunique()


# In[12]:


# check unique values of categorical columns
df.gender.unique()


# In[13]:


# remove the Other value
len(df[df.gender=='Other'])
# remove those cases


# In[14]:


df.drop(df[df['gender'] == 'Other'].index, inplace = True)
print(df.shape)


# In[15]:


df.gender.unique()


# In[16]:


df.smoking_history.unique()


# I will put into the same category:
# * Never
# * No Info
# * Current
# * Former, Ever, Not Current 

# In[17]:


# Define a function to map the existing categories to new ones
def recategorize_smoking(smoking_status):
    if smoking_status == 'never':
        return 'non_smoker'
    elif smoking_status == "No Info":
        return 'no_info'
    elif smoking_status == 'current':
        return 'current'
    elif smoking_status in ['ever', 'former', 'not current']:
        return 'past_smoker'

# Apply the function to the 'smoking_history' column
df['smoking_history'] = df['smoking_history'].apply(recategorize_smoking)

# Check the new value counts
print(df['smoking_history'].value_counts())


# In[18]:


df[categorical_cols].nunique()


# ### Check Missing Values

# In[19]:


df.columns


# In[20]:


df.isnull().sum()


# In[21]:


df.describe()


# ### Make sure labels are ok

# In[22]:


df.diabetes.nunique()


# ### Plots

# In[23]:


plt.hist(df['age'], bins=30, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[24]:


# Bar plot for gender
sns.countplot(x='diabetes', data=df)
plt.title('Diabetes Distribution')
plt.show()


# ### Model Training

# In[25]:


df.columns


# In[26]:


print(categorical_cols)
print(numerical_cols)


# In[27]:


# remove target from features
numerical_cols.remove('diabetes')


# In[28]:


print(numerical_cols)


# In[29]:


len_df = len(df)
print(len_df)


# ### Careful with regard to the split as we have imbalanced classification

# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


print('% of diabetics', df.diabetes.sum()/len(df)*100)


# In[32]:


target = 'diabetes'
y = df[target].values


# In[33]:


df.drop(['diabetes'], axis='columns', inplace=True)


# In[34]:


df_train, df_test_val, y_train, y_test_val = train_test_split(df, y, test_size=0.2, random_state=42, stratify=y)
df_val, df_test, y_val, y_test = train_test_split(df_test_val, y_test_val, test_size=0.5, random_state=42, stratify=y_test_val)


# In[35]:


len(df_train), len(df_val), len(df_test)


# In[36]:


# df_train, df_val, y_train, y_val = train_test_split(df, y, test_size=0.2, random_state=42, stratify=y)


# In[37]:


# len(df_train), len(df_val), len(y_train), len(y_val)


# In[38]:


print('[%] of diabetic values in the train set: ', np.round(y_train.sum()/len(y_train)*100, 2))
print('[%] of diabetic values in the validation set: ', np.round(y_val.sum()/len(y_val)*100, 2))
print('[%] of diabetic values in the test set: ', np.round(y_test.sum()/len(y_test)*100, 2))


# In[39]:


all_features = categorical_cols + numerical_cols

dv = DictVectorizer()

train_dicts = df_train[all_features].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[all_features].to_dict(orient='records')
X_val = dv.transform(val_dicts)


# In[40]:


print(dv.feature_names_)


# ### Train

# In[41]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score


# Hyperparameters/ CrossValdidation/ Various Models

# In[42]:


lr = RandomForestClassifier(n_jobs=-1)
lr.fit(X_train, y_train)


# In[43]:


y_pred = lr.predict(X_val)


# In[44]:


from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score, average_precision_score


# In[45]:


auc = roc_auc_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
prc = average_precision_score(y_val, y_pred)

print((auc, f1, precision, recall, prc))


# In[46]:


# Create a dataframe for feature importance
importance_df = pd.DataFrame({'Feature': dv.feature_names_, 'Importance': lr.feature_importances_})
# Sort the dataframe by importance
importance_df = importance_df.sort_values('Importance', ascending=False)


# In[47]:


importance_df


# In[48]:


# Plot the feature importances
plt.figure(figsize=(8, 4))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances')
plt.show()


# ### Metric

# In[49]:


# Evaluate the model
print(f"Model Accuracy: {round(accuracy_score(y_val, y_pred),2)} ", )
print(classification_report(y_val, y_pred))
print('F1 Score: ',f1_score(y_val, y_pred))


# In[50]:


print('Perfect Conf Matrix')
display(confusion_matrix(y_val, y_val))
print()
print('Validation Conf Matrix')
display(confusion_matrix(y_val, y_pred))


# In[51]:


# Plot confusion matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(5, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# ### Hyperparam optimization

# In[52]:


import xgboost as xgb
import optuna


# In[53]:


# dtrain = xgb.DMatrix(X_train, label=y_train)
# dvalid = xgb.DMatrix(X_val, label=y_val)


# https://forecastegy.com/posts/xgboost-hyperparameter-tuning-with-optuna/

# * `n_estimators`: determines number of trees [fast:200 - slow:5000]: fix the number and tune learning rate which determines how much each tree contributes to final prediction. The more trees that smaller the learning rate. [0.001-0.1]
# * `max_depth`: complexity of each tree in the model, how deep each tree can grow.[1-10]
# * `subsample`: usually choose between [0.05-1] fraction determining the amount of data used for each building tree. This introduces diversity that helps to handle overfit.
# * `colsample_bytree`: proportion of features to be considered by each tree
# * `min_child_weight`: minimum sum of instance weights that must be present in a child note in each tree: [1-20]

# ### MLflow
# * Backend: local sqlite
# * Artifact: local
# * Registry: localhost
# 
# mlflow server --backend-store-uri sqlite:///backend.db

# #### Tracking

# In[54]:


MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"


# In[55]:


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(mlflow.get_tracking_uri())


# In[56]:


# set an experiment, if it doesnt exist create one
mlflow.set_experiment(experiment_name='final-experiment')


# In[57]:


# obtain a list of existing experiments
exp = mlflow.search_experiments()
for e in exp:
    print(e)
    print()


# In[58]:


# # First, we’ll define the objective function, which Optuna will aim to optimize.
# # primary goal is to maximize F1 on validation set
# def objective(trial):
    
#     with mlflow.start_run():
#         mlflow.set_tag("model", "xgboost")
        
#         params = {
#             "objective": "binary:logistic",
#             "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
#             "max_depth": trial.suggest_int("max_depth", 1, 10),
#             "subsample": trial.suggest_float("subsample", 0.05, 1.0),
#             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
#             "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
#         }

#         mlflow.log_params(params)
        
#         # n_boost_round = n_estimator
#         booster = xgb.train(params=params, 
#                             dtrain=dtrain, 
#                             num_boost_round=1000,
#                             evals=[(dvalid,'validation')] , 
#                             early_stopping_rounds=50)
    
#         # output are the probabilities, we need to convert to the binary classes
#         y_pred_proba = booster.predict(dvalid)
#         y_pred = (y_pred_proba > 0.5).astype(int)
#         f1 = f1_score(y_true=y_val, y_pred=y_pred) 

#         mlflow.log_metric('f1', f1)
        
#     return f1


# In[59]:


# First, we’ll define the objective function, which Optuna will aim to optimize.
# primary goal is to maximize F1 on validation set
def objective(trial):
    
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")

        constant_params = {
            "objective": "binary:logistic",
            "n_estimators" : 1000,
            "eval_metric" : "logloss",
            "early_stopping_rounds": 50}
        
        hyper_params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20)
        }

        params = {**constant_params, **hyper_params}

        mlflow.log_params(params)

        clf = xgb.XGBClassifier(**params)
        
        clf.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True)

        # output are the probabilities, we need to convert to the binary classes
        y_pred_proba = clf.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        f1 = f1_score(y_true=y_val, y_pred=y_pred) 

        mlflow.log_metric('f1', f1)
        
    return f1


# The `direction` parameter specifies whether we want to minimize or maximize the objective function.
# * **Regression**: Lower RMSE --> better model we want to minimize it
# * **Classificaition** Higher AUROC/ Accuracy --> Better model to maximize!
# 
# The `n_trials` parameter defines the number of times the model will be trained with different hyperparameter values.
# 
# In practice, about 30 trials are usually sufficient to find a solid set of hyperparameters.
# 
# 

# In[60]:


# To execute the optimization, we create a study object and pass 
# the objective function to the optimize method.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=3)


# Once the optimization is complete, we can display the best hyperparameters and the RMSE score.

# In[62]:


print('Best hyperparameters:', study.best_params)
print()
print('Best f1:', study.best_value)
# print()
# print("Best trial:", study.best_trial.params)


# An additional tip: if most of the best trials utilize a specific hyperparameter near the minimum or maximum value, consider expanding the search space for that hyperparameter.
# 
# For example, if most of the best trials use learning_rate close to 0.001, you should probably restart the optimization with the search space trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True).

# ### Final Model

# In[70]:


with mlflow.start_run():

    constant_params = {
        "objective": "binary:logistic",
        "n_estimators" : 1000}

    # combine constant and hyperopt params
    params = {**study.best_params, **constant_params}

    mlflow.log_params(params)

    pipeline = make_pipeline(
        DictVectorizer(),
        xgb.XGBClassifier(**params, n_jobs=-1)
    )
    
    pipeline.fit(train_dicts, y_train)
    
    
    y_pred_proba = pipeline.predict_proba(val_dicts)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    f1 = f1_score(y_val, y_pred)
    print('F1 Score: ',f1)

    mlflow.log_metric("f1", f1)

    mlflow.sklearn.log_model(pipeline, artifact_path="models_mlflow")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")
    


# ### MLflow ModelRegistry (local) 

# In[ ]:


from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
client=MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


# In[ ]:


client.search_experiments()


# In[ ]:


runs = client.search_runs(
    experiment_ids='1',
    filter_string='metrics.f1 < 0.9',
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=10,
    order_by=['metrics.f1 ASC']
)


# In[ ]:


for run in runs:
    print(f"Run ID: {run.info.run_id}, f1: {run.data.metrics['f1']}")


# ### Loading Model

# In[ ]:


RUN_ID = '956c7d6a75d24c76b06606eeec98fd86'


# In[ ]:


model_uri = f'runs:/{RUN_ID}/models_mlflow'


# In[ ]:


# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(model_uri)


# In[ ]:


loaded_model


# In[ ]:


loaded_model = mlflow.xgboost.load_model(model_uri)


# In[ ]:


y_pred_proba = loaded_model.predict(dvalid)
y_pred = (y_pred_proba > 0.5).astype(int)
f1 = f1_score(y_val, y_pred)
print('F1 Score: ',f1)


# ### Model Registry

# In[ ]:


model_name = 'daniel1'


# In[ ]:


mlflow.register_model(model_uri=model_uri,name=model_name)


# In[ ]:


latest_versions = client.get_latest_versions(name=model_name)


# In[ ]:


for lv in latest_versions:
    print(f"Version: {lv.version}, stage: {lv.current_stage}")


# #### Transition Stages: None -> Staging

# In[ ]:


client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage='Staging',
    archive_existing_versions=False
)


# In[ ]:


latest_versions = client.get_latest_versions(name=model_name)
for lv in latest_versions:
    print(f"Version: {lv.version}, stage: {lv.current_stage}")


# ### Update Model Desc and Version

# In[ ]:


client.update_model_version(
    name=model_name,
    version=1,
    description="The description was added to the model"
)


# In[ ]:


from datetime import datetime

date = datetime.today().date()
client.update_model_version(
    name=model_name,
    version=model_version,
    description=f"The model version {model_version} was transitioned to {new_stage} on {date}"
)


# In[ ]:




