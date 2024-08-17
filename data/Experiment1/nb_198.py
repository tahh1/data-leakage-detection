#!/usr/bin/env python
# coding: utf-8

# # Deep Learning Computation using MNIST dataset

# In[1]:


#Importing tensorflow
import tensorflow as tf


# In[2]:


#Check the version of the tensorflow installed 
print((tf.__version__))


# In[3]:


import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')



# Using Tensorflow Keras instead of the original Keras we are importing the required layers 

from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential  #importing sequential layer 
from tensorflow.keras.layers import Dense  #importing the dense layer 

from tensorflow.keras.layers import BatchNormalization


# In[4]:


(xtrain,ytrain),(xtest,ytest)=mnist.load_data()


# In[5]:


xtrain.shape


# In[6]:


plt.imshow(xtrain[1,:,:],cmap='gray')


# In[7]:


ytrain[1]


# In[8]:


plt.imshow(xtest[0,:,:],cmap='gray')


# In[9]:


ytrain[:50]


# In[10]:


L=pd.DataFrame(ytrain)
L[0].value_counts()


# In[11]:


#Represent Training & Testing samples suitable for #tensorflow backend
x_train=xtrain.reshape(xtrain.shape[0],784).astype('float32')
x_test=xtest.reshape(xtest.shape[0],784).astype('float32')


# In[12]:


x_test.shape


# In[13]:


x_train/=255
x_test/=255


# In[14]:


from tensorflow import keras

y_train = keras.utils.to_categorical(ytrain, 10)
y_test = keras.utils.to_categorical(ytest, 10)


# In[15]:


# Initialize the constructor

model = Sequential()

# Define model architecture
#Dense layers are used in this architecture in a sequential manner

model.add(Dense(784,activation='relu'))
model.add(Dense(100, activation ='relu'))
model.add(Dense(10,activation='softmax'))


# In[16]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[17]:


epochs = 20
batch_size = 512


history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=.1, verbose=True)
loss,accuracy  = model.evaluate(x_test, y_test, verbose=False)


# In[18]:


print((history.history['val_accuracy']))

print((history.history['accuracy']))

ta = pd.DataFrame(history.history['accuracy'])
va = pd.DataFrame(history.history['val_accuracy'])

tva = pd.concat([ta,va] , axis=1)

tva.boxplot()


# In[19]:


loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(('Accuracy: %.3f'  % acc))
print(('Loss: %.3f' % loss))


# In[20]:


y_predict = model.predict(x_test)


# In[21]:


y_predict[0]


# In[22]:


np.argmax(y_predict[0])


# In[23]:


y_pred = []
for val in y_predict:
    y_pred.append(np.argmax(val))
#print(y_pred)    
#convert 0 1 to 1 and 1 0 as 0
cm = metrics.confusion_matrix(ytest,y_pred)
print(cm)


# In[24]:


cr=metrics.classification_report(ytest,y_pred)
print(cr)

