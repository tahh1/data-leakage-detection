#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


# In[2]:


(X_train , y_train) , (X_test, y_test) =  tf.keras.datasets.fashion_mnist.load_data()


# In[3]:


plt.imshow(X_train[0] , cmap ='gray')


# In[4]:


X_train.shape


# In[5]:


X_test.shape


# In[6]:


y_train.shape


# In[7]:


y_test.shape


# In[8]:


i = random.randint(1,60000)
plt.imshow(X_train[i], cmap='gray')


# In[9]:


label= y_train[i]
label


# In[10]:


W_grid = 15
L_grid = 15

fig ,axes = plt.subplots(L_grid,W_grid ,  figsize =(17,17))

axes = axes.ravel()
n_training = len(X_train)

for i in np.arange(0,W_grid*L_grid):
    index = np.random.randint(0,n_training)
    axes[i].imshow(X_train[index] , cmap='gray')
    axes[i].set_title(y_train[index] , fontsize = 8)
    axes[i].axis("off")
    


# In[11]:


X_train = X_train / 255
X_test = X_test / 255


# In[12]:


noise_factor = 0.3
noise_dataset_train = []

for img in X_train:
    noisy_image = img + noise_factor*np.random.randn(*img.shape)
    noisy_image = np.clip(noisy_image , 0 ,1)
    noise_dataset_train.append(noisy_image)
    


# In[13]:


plt.imshow(noise_dataset_train[22], cmap =  'gray')


# In[14]:


noise_factor = 0.3
noise_dataset_test = []

for img in X_test:
    noisy_image = img + noise_factor*np.random.randn(*img.shape)
    noisy_image = np.clip(noisy_image , 0 ,1)
    noise_dataset_test.append(noisy_image)
    


# In[15]:


plt.imshow(noise_dataset_test[22], cmap =  'gray')


# In[16]:


noise_dataset_train = np.array(noise_dataset_train)
noise_dataset_test = np.array(noise_dataset_test)


# In[17]:


autoencoder  = tf.keras.models.Sequential()
autoencoder.add(tf.keras.layers.Conv2D(filters = 16 , kernel_size = 3 , strides = 1 , padding='same' , input_shape =(28,28,1)))
autoencoder.add(tf.keras.layers.Conv2D(filters = 8 , kernel_size = 3 , strides = 2 , padding='same' ))
autoencoder.add(tf.keras.layers.Conv2D(filters = 8 , kernel_size = 1 , strides = 2 , padding='same' ))

autoencoder.add(tf.keras.layers.Conv2DTranspose(filters = 16 , kernel_size = 3 , strides = 2 , padding='same' ))
autoencoder.add(tf.keras.layers.Conv2DTranspose(filters = 1 , kernel_size = 3 , strides = 2 , padding='same', activation = 'sigmoid' ))


# In[18]:


autoencoder.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.Adam(lr =0.001))
autoencoder.summary()


# In[19]:


autoencoder.fit(noise_dataset_train.reshape(-1,28,28,1),
               X_train.reshape(-1,28,28,1),
               epochs = 15,
               batch_size = 200,
               validation_data = (noise_dataset_test.reshape(-1,28,28,1), X_test.reshape(-1,28,28,1)))


# In[21]:


evaluation = autoencoder.evaluate(noise_dataset_test.reshape(-1,28,28,1), X_test.reshape(-1,28,28,1))
print(("Test Accuracy : {:.3f}".format(evaluation)))


# In[23]:


pred = autoencoder.predict(noise_dataset_test[:10].reshape(-1,28,28,1))


# In[28]:


fig ,axes = plt.subplots(nrows = 2 ,ncols = 10 , sharex = True , sharey =True , figsize=(20,4))
for images  , row in zip([noise_dataset_test[:10],pred],axes):
    for img , ax in zip(images , row):
        ax.imshow(img.reshape((28,28)),cmap ='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        


# In[ ]:




