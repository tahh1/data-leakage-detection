#!/usr/bin/env python
# coding: utf-8

# # **Introduction**
# 
# Face Detection Systems have great uses in today’s world which demands security, accessibility or joy! Today, we will be building a model that can plot 15 key points on a face.
# 
# Face Landmark Detection models form various features we see in social media apps. The face filters you find on Instagram are a common use case. The algorithm aligns the mask on the image keeping the face landmarks as base points.
# 
# In this notebook, we'll develop a model which marks keypoints on a given image of a human face. We'll build a Convolutional Neural Network which takes an image and returns a array of keypoints.
# 
# We'll require a GPU Hardware accelerator for training the model. Change the runtime type to GPU by going to Tools > Change Runtime Type > Hardware Accelerator > GPU.
# 
# ![](https://miro.medium.com/max/2000/1*qNNr1hrFoaeAWru7VI0SbQ.png)

# # **Reading data and Preprocessing**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf


# In[2]:


# Ploting images with landmarks
def plot_image_landmarks(img_array, df_landmarks, index):
    plt.imshow(img_array[index, :, :, 0], cmap = 'gray')
    plt.scatter(df_landmarks.iloc[index][0: -1: 2], df_landmarks.iloc[index][1: : 2], c = 'y')
    plt.show()


# In[3]:


features = np.load('../input/face-images-with-marked-landmark-points/face_images.npz')
features = features.get(features.files[0]) # images
features = np.moveaxis(features, -1, 0)
features = features.reshape(features.shape[0], features.shape[1], features.shape[1], 1)


# In[4]:


keypoints = pd.read_csv('../input/face-images-with-marked-landmark-points/facial_keypoints.csv')
keypoints.head()


# In[5]:


# Cleaing data
keypoints = keypoints.fillna(0)
num_missing_keypoints = keypoints.isnull().sum(axis = 1)
num_missing_keypoints


# In[6]:


new_features = features[keypoints.index.values, :, :, :] #Nums of rows,w, H, Channels
new_features = new_features / 255
keypoints.reset_index(inplace = True, drop = True)


# In[7]:


plot_image_landmarks(new_features, keypoints, 3)


# In[8]:


x_train, x_test, y_train, y_test = train_test_split(new_features, keypoints, test_size=0.2)


# # **Our Model**

# In[9]:


from tqdm.keras import TqdmCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam


# **kernel_initializer** in Keras : Initializers define the way to set the initial random weights of Keras layers.
# 
# **glorot_uniform()** : It draws samples from a uniform distribution within [-limit, limit] where limit is sqrt(6 / (fan_in + fan_out)) where fan_in is the number of input units in the weight tensor and fan_out is the number of output units in the weight tensor.
# 
# We are using **the Mean Squared Error** as we are performing a regression task. A small learning rate is always good if you have a good amount of data

# In[10]:


img_size = 96


# In[11]:


model = Sequential()

model.add(Input(shape=(img_size, img_size, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding="same",kernel_initializer=glorot_uniform()))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding="same",kernel_initializer=glorot_uniform()))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding="same",kernel_initializer=glorot_uniform()))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))  

model.add(Flatten())
model.add(Dense(256,kernel_initializer=glorot_uniform()))
model.add(LeakyReLU(alpha=0.1))

model.add(Dropout(0.5)) 

model.add(Dense(64,kernel_initializer=glorot_uniform()))
model.add(LeakyReLU(alpha=0))

model.add(Dense(30,kernel_initializer=glorot_uniform()))

model.summary()
model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['mean_squared_error'])


# In[12]:


BATCH_SIZE = 100
EPOCHS = 150


# # **Training Model**

# In[13]:


history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
    shuffle=True,
    verbose=1,
)


# In[14]:


plt.plot(history.history['mean_squared_error'], label='MSE (training data)')
plt.plot(history.history['val_mean_squared_error'], label='MSE (validation data)')
plt.title('MSE for Facial keypoints')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()


# # **Model Evaluation**

# In[15]:


y_pred = model.predict(x_test)
y_pred


# In[16]:


def plot_img_preds(images, truth, pred, index):
    plt.imshow(images[index, :, :, 0], cmap = 'gray')
    
    t = np.array(truth)[index]
    plt.scatter(t[0::2], t[1::2], c = 'y')
    
    p = pred[index, :]
    plt.scatter(p[0::2], p[1::2], c = 'r')
    
    plt.show()


# In[18]:


plot_img_preds(x_test, y_test, y_pred, 3)


# In[27]:


plot_img_preds(x_test, y_test, y_pred, 18)

