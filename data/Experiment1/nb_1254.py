#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os


# In[2]:


# Load data
diabetes_data_file_path = os.path.join('input', 'Churn_Modelling.csv')
dataset = pd.read_csv(diabetes_data_file_path)

dataset.head()


# In[3]:


features = [
    'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
]
# X is our features, and y our target
X = dataset[features]    # X = dataset.iloc[:, 3:13].values
y = dataset.Exited       # y = dataset.iloc[:, 13].values


# In[4]:


# Encoding categorical data
# in this case : 'Male' => 1, 'Female' => 0
le = LabelEncoder()
X.Gender = le.fit_transform(X.Gender)

ct = ColumnTransformer([('my_OHE', OneHotEncoder(), [1])], remainder='passthrough')
X = ct.fit_transform(X)

X = X[:, 1:] # get rid of CreditScore ?

# Split data into training and validation data, for both features and target.
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.2, random_state = 0)

# apply feature scaling
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
val_X = sc.transform(val_X)


# In[5]:


# Part 2 - Now let's make the ANN!
# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer
#        the second hidden layer
#        the output layer
# + compile ("Configures the model for training")
model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# model.fit(train_X, train_y, batch_size=10, epochs=100)
model.fit(train_X, train_y, batch_size=10, epochs=10)


# In[6]:


# Predicting the Test set results
val_predictions = model.predict(val_X)
val_predictions = (val_predictions > 0.5) # transform float to bool

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(val_y, val_predictions)

print(cm)

val_mae = mean_absolute_error(val_y, val_predictions)
print(f'mean_absolute_error = {val_mae}')


# In[ ]:


person = ['France', 600, 'Male', 40, 3, 60000, 2, 1, 1, 50000]

