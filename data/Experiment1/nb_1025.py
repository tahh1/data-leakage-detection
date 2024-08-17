#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


mnist = tf.keras.datasets.mnist
(training_images,training_labels) , (test_images,test_labels) = mnist.load_data()


# In[20]:


plt.imshow(training_images[0])
print((training_labels[0]))
print((training_images[0]))


# In[32]:


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>=90):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()


# In[26]:


(training_images,test_images) = training_images/255.0 , test_images/255.0


# In[28]:


model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(1024,activation="relu"),
                            tf.keras.layers.Dense(19,activation="softmax")
                            ])


# In[31]:


model.compile(loss = "sparse_categorical_crossentropy",optimizer = "adam", metrics = ["accuracy"])



# In[33]:


model.fit(training_images,training_labels,epochs=10, callbacks=[callbacks])


# In[34]:


model.evaluate(test_images,test_labels)


# In[39]:


predict = model.predict(test_images[1:2])
print((np.argmax(predict)))
plt.imshow(test_images[1])


# In[ ]:




