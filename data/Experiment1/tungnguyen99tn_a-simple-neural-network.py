#!/usr/bin/env python
# coding: utf-8

# # A Simple Neural Network working on Iris flower dataset

# # Load dataset

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


iris_df = pd.read_csv('../input/iris/Iris.csv')
iris_df.head()


# In[3]:


iris_df.shape


# In[4]:


iris_df = pd.get_dummies(iris_df, columns=['Species'])
iris_df.head()


# One-hot labels by get_dummies() method

# In[5]:


X = iris_df.values[:, 1:5]
y = iris_df.values[:, 5:8]


# In[6]:


def normalize(array):
    arr_min = array.min(axis=(0, 1))
    arr_max = array.max(axis=(0, 1))
    return (array - arr_min) / (arr_max - arr_min)


# In[7]:


X = normalize(X)


# normalize dataset between (0, 1) before training

# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# # Build NN with Keras

# In[9]:


from keras.models import Sequential 
from keras.layers import Dense 
from keras.optimizers import Adam 


# In[10]:


def create_network():
    model = Sequential()
    model.add(Dense(7, input_shape=(4,), activation='relu')) 
    model.add(Dense(8, activation='relu')) 
    model.add(Dense(5, activation='relu'))
    model.add(Dense(3, activation='softmax')) #3 classes need 3 nodes at output layer
        
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy']) 
    return model


# In[11]:


model = create_network()


# In[12]:


results = model.fit(X_train,y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))


# In[13]:


plt.figure(figsize=(5, 5))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();


# In[14]:


model.predict(X_test[:5])


# This is results of prediction on 5 samples and the below is real labels

# In[15]:


y_test[:5]


# In[16]:


import keras.backend as K


# In[17]:


weight, biases = model.layers[0].get_weights()


# In[18]:


weight


# In[19]:


biases

