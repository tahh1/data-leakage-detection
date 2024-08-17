#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pickle
import random
import sys
import io
import os
import re
import keras
import urllib.request, urllib.parse, urllib.error
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import LambdaCallback,ModelCheckpoint
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.preprocessing.text import one_hot
from skimage.color import rgb2grey
from sklearn.cluster import KMeans
from sklearn import preprocessing
from keras.datasets import cifar10
import warnings
warnings.filterwarnings("ignore")


# In[2]:


def unpickle(file):
    with open(file, 'rb') as fo:
        dict= pickle.load(fo, encoding ='bytes')
    return dict

# def cifar_10_reshape(batch):
#     output=np.reshape(len(batch),3,32,32).transpose(0,2,3,1)
#     return output


# In[3]:


def load_cifar10_data(data):
    train_data = None
    train_labels = []

    for i in range(1, 6):
        batch = unpickle(data + "/data_batch_{}".format(i))
        print((list(batch.keys())))
        if i == 1:
            train_data = batch[b'data']
        else:
            train_data = np.vstack((train_data, batch[b'data']))
        train_labels += batch[b'labels']

    test_batch = unpickle(data + "/test_batch")
    test_data = test_batch[b'data']
    test_labels = test_batch[b'labels']

    train_data = train_data.reshape((len(train_data), 3, 32, 32)).transpose([0, 2, 3, 1])
#     train_data = np.rollaxis(train_data, 1, 4)
    train_labels = np.array(train_labels)

    test_data = test_data.reshape((len(test_data), 3, 32, 32)).transpose([0, 2, 3, 1])
#     test_data = np.rollaxis(test_data, 1, 4)
    test_labels = np.array(test_labels)

    return train_data, train_labels, test_data, test_labels


# In[4]:


data_dir = '/Users/phuongqn/Desktop/INF552/Homework/Homework 7 Data/cifar-10-batches-py'

train_data, train_labels, test_data, test_labels = load_cifar10_data(data_dir)

print((train_data.shape))
print((train_labels.shape))

print((test_data.shape))
print((test_labels.shape))

plt.imshow(train_data[200])
plt.show()


# In[5]:


bird_train = train_data[train_labels == 2]
bird_test = test_data[test_labels == 2]
birds=np.concatenate([bird_train, bird_test])


# In[6]:


birds.shape


# In[7]:


print('First 10 Images in the dataset:')
fig = plt.figure(figsize=(8, 3))
for i in range(0, 10):
    ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
    plt.imshow(bird_train[i])
plt.show()


# In[8]:


total_px=birds.shape[0]*birds.shape[1]*birds.shape[2]
pct=0.9
rand_len=int(total_px*pct)
img_idx = np.random.randint(6000, size = (rand_len, ))
row_idx = np.random.randint(32, size = (rand_len, ))
col_idx = np.random.randint(32, size = (rand_len, ))


# In[9]:


rand_px = birds[img_idx, row_idx, col_idx, :]


# In[10]:


rand_px.shape


# In[11]:


k=4
kmodel=KMeans(n_clusters=k, random_state= 158).fit(rand_px)


# In[12]:


main_colors=kmodel.cluster_centers_
print(('RGB', k, 'main colors:'))
print(main_colors)


# In[13]:


pred_birds=kmodel.predict(birds.reshape(-1, birds.shape[-1]))
four_color_birds=[]
px_colors=[]
for i in range(len(pred_birds)):
    clusterLabel = pred_birds[i]
    four_color_birds.append(main_colors[clusterLabel])
    oneHotEncoding = np.zeros(k)
    oneHotEncoding[clusterLabel] = 1
    px_colors.append(oneHotEncoding)
    i += 1
four_color_birds = np.array(four_color_birds)
four_color_birds = np.reshape(four_color_birds, (6000,32,32,3)) 

px_colors = np.array(px_colors)
px_colors = np.reshape(px_colors, (6000, 32*32*k))


# In[14]:


pred_birds2=kmodel.predict(rand_px)


# In[15]:


pred_birds2.shape


# In[16]:


four_color_birds2 = kmodel.cluster_centers_[pred_birds2]


# In[17]:


four_color_birds2


# In[18]:


four_color_birds2 = np.uint8(four_color_birds2.reshape((5400, 32, 32, 3)))


# In[20]:


print('Output Images in the dataset:')
fig = plt.figure(figsize=(8, 3))
for i in range(0, 10):
    ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
    plt.imshow(four_color_birds[i].astype(np.uint8))
plt.show()


# In[21]:


plt.imshow(four_color_birds2[3].astype(np.uint8))
plt.show()


# In[22]:


px_colors.shape


# In[34]:


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    
gray_train = rgb2gray(bird_train)
gray_test = rgb2gray(bird_test)
plt.imshow(gray_train[0])
plt.show()
plt.imshow(gray_test[1], cmap='gray')
plt.show()


# In[70]:


CNN=Sequential()
CNN.add(Conv2D(64, kernel_size=(5, 5), strides=(1,1), input_shape=(32, 32, 1), padding='same',activation='relu'))
CNN.add(MaxPooling2D(pool_size = (2, 2),strides=(1,1), padding='same'))
CNN.add(Dropout(0.25))
CNN.add(Conv2D(64, kernel_size=(5, 5), strides=(1,1), padding='same',activation='relu'))
CNN.add(MaxPooling2D(pool_size = (2, 2),strides=(1,1),padding='same'))
CNN.add(Dropout(0.22))
# CNN.add(Flatten())
CNN.add(Dense(32, activation='softmax'))
CNN.add(Dense(4, activation='softmax'))
CNN.summary()


# In[71]:


adam = keras.optimizers.Adam(lr=0.5)
CNN.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


# In[72]:


filepath="/Users/phuongqn/Desktop/INF552/HW-Phuong/LSTMweights/weights-improvement-bird-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
gray_train=gray_train.reshape(5000, 32, 32,1)
gray_test=gray_test.reshape(1000, 32, 32,1)


# In[73]:


from sklearn.model_selection import train_test_split
X_train = gray_train[:5000, :, :, :]
labels = px_colors[:5000, :]
labels= labels.reshape(5000, 32,32,4)
# Xtrain, Xval, ytrain, yval = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


# In[74]:


history = CNN.fit(gray_train, labels, verbose=1, validation_split=0.1, epochs=30, batch_size = 32, callbacks= callbacks_list)


# In[75]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Train vs Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


# In[133]:


file='/Users/phuongqn/Desktop/INF552/HW-Phuong/LSTMweights/weights-improvement-bird-30-1.3611-bigger.hdf5'
CNN.load_weights(file)
CNN.compile(loss='categorical_crossentropy', optimizer='adam')
preds=CNN.predict(gray_test)
preds.shape


# In[134]:


ytest = px_colors[5000:, :]
ytest=ytest.reshape(1000, 32,32,4)


# In[135]:


scores=CNN.evaluate(gray_test, ytest, verbose=1)
print(('Test loss:', scores))
# print('Test accuracy:', scores[1])


# In[137]:


preds = np.argmax(preds, axis=3)
preds= kmodel.cluster_centers_[preds]


# In[138]:


preds.shape


# In[136]:


ytest = np.argmax(ytest, axis=3)
ytest= kmodel.cluster_centers_[ytest]
ytest.shape


# In[140]:


for i in range(10):
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(np.uint8(gray_test[i].reshape((32, 32))), cmap = 'gray')
    axarr[1].imshow(np.uint8(preds[i].reshape((32, 32, 3))))
    axarr[2].imshow(np.uint8(ytest[i]))


# In[ ]:




