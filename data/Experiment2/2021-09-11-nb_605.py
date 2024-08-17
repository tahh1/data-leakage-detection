#!/usr/bin/env python
# coding: utf-8

# Task 1:
# When working on this assignment, I used the tensorflow website https://www.tensorflow.org/tutorials/images/classification as a general guide for working with the framework for image classification.  Whenever I came accross topics that I did not understand, I did research on other websites to determine what was happening and what I needed to do to get the framework to function correctly.
# 
# I used the keras and Sequential resources in this assignment.  Keras was used for performing the train-dev-test split and for creating the Sequential model which was used for acutally training the model using the fit() function.  With the Sequential model, I could use it to add layers to the neural network using tf.keras.layers.Conv2D.  Sequential also gives the ability to perform minibatch gradient descent to train the model and save the results of the computations of each epoch.  This was critical in allowing me to create a two layer neural network and was the reason I used the keras and Sequential resources.
# 
# Below are the additional websites I used for understanding the different aspects of creating a two layer neural network:
# 
# Used for learning Sequential model:
# https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# 
# Used for understanding forward propagation:
# https://programmer.group/tensorflow-implements-forward-propagation-of-neural-networks.html
# https://towardsdatascience.com/coding-neural-network-forward-propagation-and-backpropagtion-ccf8cf369f76
# 
# Used for understanding of a two layer neural network:
# https://www.easy-tensorflow.com/tf-tutorials/neural-networks/two-layer-neural-network?view=article&id=124:two-layer-neural-network
# 
# Used to understand what each part of the train-dev-test split was for:
# https://towardsdatascience.com/how-to-split-a-tensorflow-dataset-into-train-validation-and-test-sets-526c8dd29438
# 
# Used to learn more about optimizers and helped me in choosing Adam:
# https://keras.io/api/optimizers/
# 
# Used to learn about using the test set to predict and measure how accurate predicitons were:
# https://www.machinecurve.com/index.php/2020/02/21/how-to-predict-new-samples-with-your-keras-model/
# https://www.tensorflow.org/api_docs/python/tf/math/argmax

# Task 2:
# Exploratory Data Analysis:
# The dataset I used was from https://www.kaggle.com/alessiocorrado99/animals10 and was already used by another person.  All of the images had already been checked and the images were split up into 10 groups depending on which animal they depicted.  This made it perfect to use immediately for training and image classification.  The only changes I did to the dataset iself was change the names of the folders from spanish to english and removed the corresponding translation file provided by the original owner of the dataset.

# In[2]:


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# In[3]:


import pathlib
data_dir = pathlib.Path("/mnt/c/users/bouchc2/downloads/animal_images_archive")
batch_size = 32
img_height = 180
img_width = 180


# In[4]:


train_set = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[5]:


dev_split = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_set = dev_split.take(30)
dev_set = dev_split.skip(30)


# In[6]:


data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)


# In[7]:


num_classes = 10
model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])


# In[8]:


model.compile(optimizer='Adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[20]:


epochs = 3
data = model.fit(
  train_set,
  validation_data=dev_set,
  batch_size = batch_size,
  epochs=epochs
)


# In[10]:


acc = data.history['accuracy']
val_acc = data.history['val_accuracy']

loss = data.history['loss']
val_loss = data.history['val_loss']
epochs_range = list(range(epochs))


# In[19]:


classes = train_set.class_names 
predictions = model.predict(test_set)
prediction_indeces = np.argmax(predictions, axis = 1)

n = list([x[1] for x in test_set])

image_list = []
for x in n:
    for y in x:
        image_list.append(y)   
        
# All predictions are given a confidence value for each category of animal
# The index with the highest confidence is chosen
# The first 10 predictions are printed out to show this
for i in range(0, 10):
    print(("This image most likely belongs to {} with a {:.2f} percent confidence."
          .format(classes[np.argmax(tf.nn.softmax(predictions[i]))], 100 * np.max(tf.nn.softmax(predictions[i])))))
        
correct = 0
for i in range(0, len(image_list)):
    if (image_list[i] == prediction_indeces[i]):
        correct += 1

print(("The accuracy of the model is {:.2f}.".format(correct/len(image_list))))


# Task 3:
# In task 2 I chose the hyperparameters to be as follows: I chose the number of epochs to be 3 and the batch size to be 32.  I chose the number of epochs to be 3 as I did some reasearch online and found that there is no optimum number of epochs to use, and instead the user should stop adding epochs when the accuracy of the training begins to level off.  I chose 3 epochs as I started with 1 and continued to add more, but once I used more than 3 the model took such a long time to train that I could not get results before running out of memory.  I chose a batch size of 32 after doing some searching online and finding that 32 is a good starting point for smaller datasets.  Also, I read https://machinelearningmastery.com/how-to-control-the-speed-and-stability-of-training-neural-networks-with-gradient-descent-batch-size/ to find that it was good number for minibatch since the size is greater than 1 but less than the total number of examples in my training set.
# 
# I did not use regularization since the accuracy of the model was not incredibly high, meaning that the model did not fit the data too well which would be a sign of overfitting.  I did use the optimization algorithm SGD though as I read about when to use optimizers and found from https://www.kdnuggets.com/2020/12/optimization-algorithms-neural-networks.html that it's a good idea to do so when you do not know what weights to use for training.  Since this was the case, I used the well known Adam optimizer mentioned in the same webpage in order to minimize loss.

# In[ ]:




