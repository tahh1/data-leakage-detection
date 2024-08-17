#!/usr/bin/env python
# coding: utf-8

# In[57]:


import tensorflow as tf
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import seaborn as sns
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
from sklearn.metrics import confusion_matrix


# In[58]:


train_data = pd.read_csv("./mnist_train.csv")


# In[59]:


test_data = pd.read_csv("./mnist_test.csv")


# In[60]:


train_data.head()


# In[61]:


test_data.head()


# In[62]:


train_data.shape


# In[63]:


test_data.shape


# In[64]:


x_train = train_data.drop(labels=["label"],axis=1)/255
y_train = train_data["label"]
x_test = test_data.drop(labels=["label"],axis=1)/255
y_test = test_data["label"]


# In[65]:


print(x_train)


# In[66]:


plt.figure(figsize = (15,7))
g = sns.countplot(y_train)


# In[67]:


#normalize data (x values)
x_train = tf.keras.utils.normalize(x_train, axis=1)


# In[68]:


x_test = tf.keras.utils.normalize(x_test, axis=1)


# In[69]:


img = x_train.iloc[1].values.reshape(28,28)
img.shape
plt.imshow(img)
plt.show()


# In[70]:


#resize image for cNN
x_train = x_train.values.reshape(-1,28,28,1)
x_test = x_test.values.reshape(-1,28,28,1)
print((x_train.shape))
print((x_test.shape))


# In[71]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape = (28, 28)),
  tf.keras.layers.Dense(128,activation = 'relu'),
  tf.keras.layers.Dense(10, activation = 'softmax')
])


# In[72]:


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)


# In[97]:


model.save('model.h5')


# In[98]:


print((model.evaluate(x_test,y_test)))
#gives loss and accuracy


# In[99]:


# Predict the values from the validation dataset
y_pred = model.predict(x_test)
y_pred


# In[100]:


num = eval(input("Enter name of image:"))


# In[101]:


num


# In[102]:


img = cv.imread('{}.png'.format(int(num)))[:,:,0] #read in image
img = np.invert(np.array([img]))
prediction = model.predict(img)
print(("The number is likely: {}".format(np.argmax(prediction))))
plt.imshow(img[0])
plt.show()


# In[ ]:




