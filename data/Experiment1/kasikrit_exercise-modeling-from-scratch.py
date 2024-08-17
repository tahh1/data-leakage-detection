#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# You've seen how to build a model from scratch to identify handwritten digits.  You'll now build a model to identify different types of clothing.  To make models that train quickly, we'll work with very small (low-resolution) images. 
# 
# As an example, your model will take an images like this and identify it as a shoe:
# ![Imgur](https://i.imgur.com/GyXOnSB.png)

# # Data Preparation
# This code is supplied, and you don't need to change it. Just run the cell below.

# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python import keras

img_rows, img_cols = 28, 28
num_classes = 10

def prep_data(raw, train_size, val_size):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y

fashion_file = "../input/fashionmnist/fashion-mnist_train.csv"
fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')


# In[ ]:


fashion_data.shape


# In[ ]:


x, y = prep_data(fashion_data, train_size=50000, val_size=5000)


# In[ ]:


x.shape


# In[ ]:


y.shape


# In[ ]:





# # Specify Model
# **STEPS:**
# 1. Create a `Sequential` model. Call it `fashion_model`.
# 2. Add 3 `Conv2D` layers to `fashion_model`.  Make each layer have 12 filters, a kernel_size of 3 and a **relu** activation.  You will need to specify the `input_shape` for the first `Conv2D` layer.  The input shape in this case is `(img_rows, img_cols, 1)`.
# 3. Add a `Flatten` layer to `fashion_model` after the last `Conv2D` layer.
# 4. Add a `Dense` layer with 100 neurons to `fashion_model` after the `Flatten` layer.  
# 5. Add your prediction layer to `fashion_model`.  This is a `Dense` layer.  We alrady have a variable called `num_classes`.  Use this variable when specifying the number of nodes in this layer. The activation should be `softmax` (or you will have problems later).

# In[ ]:


from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D

# Your Code Here
fashion_model = Sequential()
fashion_model.add(Conv2D(12, kernel_size=(3,3),
                        activation='relu',
                        input_shape=(img_rows, img_cols,1)))

fashion_model.add(Conv2D(12, kernel_size=(3,3),
                        activation='relu'))

fashion_model.add(Conv2D(12, kernel_size=(3,3),
                        activation='relu'))

fashion_model.add(Flatten())
fashion_model.add(Dense(100, activation='relu'))
fashion_model.add(Dense(num_classes, activation='softmax'))


# In[ ]:





# # Compile Model
# Run the command `fashion_model.compile`.  Specify the following arguments:
# 1. `loss = keras.losses.categorical_crossentropy`
# 2. `optimizer = 'adam'`
# 3. `metrics = ['accuracy']`

# In[ ]:


# Your code to compile the model in this cell
fashion_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam', #stochis gradient descent
              metrics=['accuracy'])


# In[ ]:





# # Fit Model
# Run the command `fashion_model.fit`. The arguments you will use are
# 1. The first two are arguments are the data used to fit the model, which are `x` and `y` respectively.
# 2. `batch_size = 100`
# 3. `epochs = 4`
# 4. `validation_split = 0.2`
# 
# When you run this command, you can watch your model start improving.  You will see validation accuracies after each epoch.

# In[ ]:


# Your code to fit the model here
fashion_model.fit(x, y,
                 batch_size=100,
                 epochs=4,
                 validation_split=0.2)


# In[ ]:


# evaluate the model
X, Y = x, y
scores = fashion_model.evaluate(X, Y, verbose=0)
print(("%s: %.2f%%" % (fashion_model.metrics_names[1], scores[1]*100)))


# In[ ]:


# evaluate the model
X, Y = x, y
scores = fashion_model.evaluate(X, Y, verbose=0)
print(("%s: %.2f%%" % (fashion_model.metrics_names[1], scores[1]*100)))


# In[ ]:


# serialize model to JSON
model_json = fashion_model.to_json()
with open("fashion_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
fashion_model.save_weights("fashion_model.h5")
print("Saved model to disk")


# In[ ]:


get_ipython().system('ls -l')


# # Keep Going
# You are ready to learn about **[strides and dropout](https://www.kaggle.com/dansbecker/dropout-and-strides-for-larger-models)**, which become important as you start using bigger and more powerful models.
# 
# ---
# **[Deep Learning Track Home Page](https://www.kaggle.com/learn/deep-learning)**
# 
# 
