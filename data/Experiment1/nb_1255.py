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


# In[3]:


# Load data
file_path = os.path.join('input', 'diabetes.csv')
dataset = pd.read_csv(file_path)

dataset.head()


# In[4]:


features = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]
X = dataset[features]
y = dataset.Outcome


# In[5]:


# Split data into training and validation data, for both features and target.
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.2, random_state = 0)

# apply feature scaling
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
val_X = sc.transform(val_X)


# In[12]:


model = Sequential()

# Adding the input layer and the first hidden layer
#        the second hidden layer
#        the output layer
# + compile ("Configures the model for training")
model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# model.fit(train_X, train_y, batch_size=10, epochs=100)
model.fit(train_X, train_y, batch_size=10, epochs=100)


# In[9]:


# Predicting the Test set results
val_predictions = model.predict(val_X)
val_predictions = (val_predictions > 0.5) # transform float to bool

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(val_y, val_predictions)

print(cm)

val_mae = mean_absolute_error(val_y, val_predictions)
print(f'mean_absolute_error = {val_mae}')

