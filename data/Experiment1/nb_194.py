#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# In[2]:


# Reading the dataset
df = pd.read_csv("Datasets/Encoded_data_without_PCA.csv")


# In[3]:


# Removing the orginal non-one-hot-encoding columns
non_ohe_columns = ['Unnamed: 0', 'BoRace', 'BoGender', 'PropType', 'AcqTyp', 'FedGuar', 'BoEth']
BoRace_col = df['BoRace']
df = df.drop(non_ohe_columns, axis = 1)


# In[9]:


def print_classification_results(model, y_train, y_pred_train, y_test, y_pred_test):
    print((model, "Training Results & Evaluation:"))
    print()
    print((confusion_matrix(y_train,y_pred_train)))
    print((classification_report(y_train,y_pred_train)))
    print()
    print((model, "Testing Results & Evaluation:"))
    print()
    print((confusion_matrix(y_test,y_pred_test)))
    print((classification_report(y_test,y_pred_test)))


# In[4]:


# Classification for BoRace

# Creating the dataset for BoEth Class Prediction
df_BoRace = df

df_BoRace['BoRace'] = BoRace_col

# Removing one-hot-encoding values for BoRace
df_BoRace = df_BoRace.drop(['BoRace_2', 'BoRace_5'], axis = 1)

for col in df_BoRace.columns:
    df_BoRace[col] = df_BoRace[col].astype(float)
    
# Encoding (White)5 = > 0, (Non-White)1,2,3,4 => 1    
df_BoRace['BoRace'] = df_BoRace['BoRace'].astype(int)
df_BoRace['BoRace'] = np.where(df_BoRace['BoRace'] == 5, 0, df_BoRace['BoRace'])
df_BoRace['BoRace'] = np.where(df_BoRace['BoRace'] == 2, 1, df_BoRace['BoRace'])
df_BoRace['BoRace'] = np.where(df_BoRace['BoRace'] == 3, 1, df_BoRace['BoRace'])
df_BoRace['BoRace'] = np.where(df_BoRace['BoRace'] == 4, 1, df_BoRace['BoRace'])                             

# Spliting the dataset for training & testing
X = df_BoRace.iloc[:, :-1].values
y = df_BoRace.iloc[:, -1:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[10]:


# Predicting the race, using XG Boosting
BoRace_XGB_Model = XGBClassifier(objective='multi:softmax', num_class = len(df_BoRace['BoRace'].unique()), random_state = 0)
BoRace_XGB_Model.fit(X_train, y_train.ravel())

y_pred_train = BoRace_XGB_Model.predict(X_train)
y_pred_test = BoRace_XGB_Model.predict(X_test)

# Model Evaluations
print_classification_results("XGBoost", y_train, y_pred_train, y_test, y_pred_test)


# In[11]:


# Predicting the race, using Neural Networks
BoRace_NN_Model = MLPClassifier(hidden_layer_sizes=(128, 64, 8), activation='relu', solver='adam', \
                                learning_rate = 'constant', alpha = 0.00001, max_iter = 20000, random_state = 0)
BoRace_NN_Model.out_activation_ = 'softmax'
BoRace_NN_Model.fit(X_train,y_train.ravel())

y_pred_train = BoRace_NN_Model.predict(X_train)
y_pred_test = BoRace_NN_Model.predict(X_test)

# Model Evaluations
print_classification_results("Neural Network", y_train, y_pred_train, y_test, y_pred_test)


# In[12]:


# Reading the dataset
df_pca = pd.read_csv("Datasets/Encoded_data_with_PCA.csv")


# In[13]:


# Classification for BoRace

# Creating the dataset after PCA for BoEth Class Prediction
df_BoRace = df_pca.dropna()

# Encoding (White)5 = > 0, (Non-White)1,2,3,4 => 1 
# Column 12 is BoRace Column
df_BoRace['12'] = np.where(df_BoRace['12'] == 5.0, 0.0, df_BoRace['12'])
df_BoRace['12'] = np.where(df_BoRace['12'] == 2.0, 1.0, df_BoRace['12'])
df_BoRace['12'] = np.where(df_BoRace['12'] == 3.0, 1.0, df_BoRace['12'])
df_BoRace['12'] = np.where(df_BoRace['12'] == 4.0, 1.0, df_BoRace['12']) 
#df_BoRace['BoRace'] = np.where(df_BoRace['BoRace'].isnull(), 1.0, df_BoRace['BoRace']) 

X = df_BoRace.iloc[:, :-1].values
y = df_BoRace.iloc[:, -1:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[14]:


# Predicting the race, using XG Boosting
BoRace_XGB_Model = XGBClassifier(objective='multi:softmax', num_class = len(df_BoRace['12'].unique()), random_state = 0)
BoRace_XGB_Model.fit(X_train, y_train.ravel())

y_pred_train = BoRace_XGB_Model.predict(X_train)
y_pred_test = BoRace_XGB_Model.predict(X_test)

# Model Evaluations
print_classification_results("XGBoost", y_train, y_pred_train, y_test, y_pred_test)


# In[15]:


# Predicting the race, using Neural Networks
BoRace_NN_Model = MLPClassifier(hidden_layer_sizes=(32,4), activation='relu', solver='adam', \
                                learning_rate = 'constant', alpha = 0.00001, max_iter = 20000, random_state = 0)
BoRace_NN_Model.out_activation_ = 'softmax'
BoRace_NN_Model.fit(X_train,y_train.ravel())

y_pred_train = BoRace_NN_Model.predict(X_train)
y_pred_test = BoRace_NN_Model.predict(X_test)

# Model Evaluations
print_classification_results("Neural Network", y_train, y_pred_train, y_test, y_pred_test)


# In[ ]:




