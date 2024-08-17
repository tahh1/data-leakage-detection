#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[1]:


# CIFAR-10 dataset is pre-existing in Keras.
# Importing the CIFAR dataset

from tensorflow.keras.datasets import cifar10


# ## <font color="red"> Image Data Pre-processing </font>

# In[3]:


# CIFAR dataset already exists in the form of train and test data of (50000,10000) images respectively
# Have to use tuple unpacking to save the data in the data in the respective train and test datasets

(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# In[4]:


# 50000 images of 32*32 pixels with 3 color channels
X_train.shape


# In[5]:


X_test.shape


# In[6]:


# Getting single image dimensions
single_img = X_train[0]
single_img.shape


# In[7]:


# See single 2D image
plt.imshow(single_img)


# In[8]:


# Maximum Value of the single image
single_img.max()


# In[9]:


# Now we need to do the scaling of the values only for the feature data
# and as we already know that the images shall always be channeled between 0-255 range, even for the color images
# so we can divide the feature data by 255 for performing the scaling

X_train = X_train/255


# In[10]:


X_test = X_test/255


# In[11]:


# As we can see the shape is still the same. We are only scaling the features.
X_train.shape


# In[12]:


X_test.max()


# In[13]:


# In the CIFAR-10 dataset the labels are the numbers.They represent the respective category of images
y_train
# We need to change the labels by dummy labels through one hot encoding as it is a multiclass classification problem


# In[14]:


# "to_categorical" method creates the dummy variables and is the inbuilt method in keras
from tensorflow.keras.utils import to_categorical


# In[15]:


# Current shape of y_train
y_train.shape


# In[16]:


# Conversion of existing labels into dummy variables for training as well as test dataset
y_cat_test = to_categorical(y_test, num_classes=10)


# In[17]:


y_cat_train = to_categorical(y_train, num_classes=10)


# In[18]:


y_cat_train.shape


# In[19]:


y_cat_test.shape


# In[20]:


# Unlike MNIST dataset model creation No need of reshaping the images 
# as the data is not required to be converted in grayscale
# We will work on the color pictures in this one


# ## <font color="red"> Creating and Training Model </font>

# In[21]:


# Importing the "sequential Layer". Its basiclly required for the Input Layer
from tensorflow.keras.models import Sequential


# In[22]:


# Importing "Dense" for the input layers
# Importing "Conv2D" for the Convulational Layers
# Importing "MaxPool2D" for the Pooling Later
# Importing "Flatten" for flattning out the images in to order to feed it to the last dense layer and perform the classifiaction

from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten


# In[23]:


# As the data of images is more complex (32*32*3 = 3072) in comparision to the MNIST dataset 
# So we'll make multiple convulational and pooling layers


# In[24]:


model = Sequential()

# First Layer of CNN is "Convulational Layer"

# Generally the "number of Filter" and the "kernel size" is taken in the multiple of 2
# In our case as the CIFAR-10 dataset is a bit more complex than MNIST dataset, so we'll take the 
# multiple convulational and pooling layers having-
    # filters as 32
    # Grid/Kernel_size as 4*4
    # Stride as (1,1)
# Generally filters amount are also expanded with each convulational layer but in our case we'll it as same
# We'll not be needing any padding as the our image size is 32 and hence when we divide the image size with the grid
# 32/4, we get proper whole integer as 8
# If we would have been getting the value in points then we could have added padding as "same"
# The "same" criteria of padding would have dealt with extra values
# In our case the default "valid" is fine as it doesnt apply padding
# and keeps the image as it is

# "Input_shape" is the shape of the image. In our case its (32,32,3) i.e width, Height, One color Channel
# Convulational Layer --------- 1

model.add(Conv2D(filters=32, kernel_size=(4,4), strides=(1,1), padding='valid', input_shape=(32,32,3), activation="relu"))


# Second Layer of CNN is "Pool Layer"

# We have taken Pool size as (2,2). It is also the default size.
# In our case it is also the half of the grid size
# We can also add strides and padding in the pool layer as well

# Pool Layer --------- 1

model.add(MaxPool2D(pool_size=(2, 2)))


# Convulational Layer --------- 2

model.add(Conv2D(filters=32, kernel_size=(4,4), strides=(1,1), padding='valid', input_shape=(32,32,3), activation="relu"))

# Pool Layer --------- 2

model.add(MaxPool2D(pool_size=(2, 2)))


# We can add multiple convulational and pool layers according to the complexity of data


# Third step is to flatten the images


# This means we have to convert the grid of the images into a single array
# i.e. grid of 32*32*3 has to be converted to array of 3072 (32*32*3=3072) in our case

model.add(Flatten())


# Fourth Layer of CNN is "Dense Layer"


# We can add multiple dense layers as per the complexity of the dataset
# Dense layer should generally be equal to flatten array i.e. in our case is 3072 (32*32*3=3072)
# But for now we'll add only one dense layer with 256 neurons which are more than the MNIST dataset
# But it is required as this model is using color images

model.add(Dense(256,activation="relu"))


# Fifth Layer of CNN is "Output Layer". It will be a dense layer


# The layer will have one neuron per class for the classification
# That's why In our case it will be 10 neurons in the final layer
# We'll use the softmax function as the activation function as we have multiclass classification problem

model.add(Dense(10, activation="softmax"))


# Final Step is to compile the model
# Loss parameter is generally taken as "categorical_crossentropy" for the classification problem
# optimizer is "adam"
# We can also add metrics. More documentation in relation to metrics can be found at "keras.io/metrics"

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


# ## In the above cells there are 2 types of Hyperparameters- Changeable and Un-changeable
# 
# #### Un-Changeable
# 
# The parameters which are fixed and should be determine on the basis of your dataset are as follows:
# 
# 1. model.add(Conv2D(filters=32, kernel_size=(4,4), strides=(1,1), padding='valid',<font color="green"> input_shape=(32,32,3) </font>, activation="relu"))
# 
# 2. <font color="green">model.add(Flatten())</font>
# 
# 3. <font color="green">model.add(Dense(10, activation="softmax"))</font>
# 
# These are all <font color="cyan"> HYPERPARAMETERS</font> based on your data. <font color="cyan">There is correct value for them</font>
# 
# 
# #### Changeable
# 
# 1. <font color="green"> model.add(Conv2D(filters=32, kernel_size=(4,4), strides=(1,1)</font>, padding='valid', input_shape=(32,32,3) , <font color="green">activation="relu"</font>))
# 
# 2. <font color="green">model.add(Dense(256,activation="relu"))</font>
# 
# These are <font color="cyan"> HYPERPARAMETERS you can experiment with</font>

# In[25]:


# To see the summary of the model
model.summary()


# In[26]:


# Training the Model with "early stopping call back" so that we dont have to choose the number of epochs

from tensorflow.keras.callbacks import EarlyStopping


# In[27]:


# making the instance of "EarlyStopping"

# By default the monitor value is "Validation loss", we can also take the monitor value as "validation accuracy" 
# as we have given the metrics as "accuracy" during the model compilation

# Setting patience as 1. It will wait for 1 epoch to go up from the lowest validation loss level and
# then will stop the model

early_stop = EarlyStopping(monitor="val_loss", patience=2)


# In[28]:


# Training the model

model.fit(X_train,y_cat_train, epochs = 15, 
          validation_data = (X_test, y_cat_test),
          callbacks = [early_stop])


# ## <font color="red"> Evaluating the Model </font>

# In[29]:


metrics = pd.DataFrame(model.history.history)
metrics


# In[30]:


metrics[["loss","val_loss"]].plot()


# In[31]:


metrics[["accuracy","val_accuracy"]].plot()


# In[32]:


# To know metrices available in the model

model.metrics_names


# In[33]:


# Evaluating the model for the validation test

# First index in the list will be loss and the second index will be the accuracy of the validation test

# It shows[validation loss, validation accuracy]

model.evaluate(X_test, y_cat_test, verbose = 0)


# In[34]:


# Getting the predictions

from sklearn.metrics import classification_report, confusion_matrix


# In[35]:


# "Predict_classes" will be used in our case as this is the multi classification problem
predictions = model.predict_classes(X_test)


# In[36]:


y_cat_test.shape


# In[37]:


# we need to compare the predictions of the model to the actual labels
# i.e. instead of "y_cat_test", we'll compare it to y_test
# We dont have any longer need to use the classification ones as the model is already made

y_test


# In[38]:


# Comparing the true "y_test" values with our predicted values "predictions = model.predict_classes(X_test)"

print((classification_report(y_test, predictions)))


# In[39]:


confusion_matrix(y_test, predictions)


# In[40]:


# To visualize the confusion matrix

import seaborn as sns

plt.figure(figsize=(12,10))
sns.heatmap(confusion_matrix(y_test, predictions), annot=True)


# In[41]:


# To see the prediction

my_img = X_test[0]

plt.imshow(my_img)


# In[42]:


# To know actually which images is this, as we are unable to see clearly

y_test[0]


# In[43]:


# The actual answer is the image is of Cat. As, we can verify the labels of the CIFAR-10 dataset on the google
# And label 3 is of Cat


# In[44]:


X_test[0].shape


# In[45]:


# Converting the image in the below written shape
# number of images, width, height, color channels
# feeding it into the model

model.predict_classes(my_img.reshape(1,32,32,3))


# In[ ]:





# In[46]:


# Predicted the picture as Cat. Hence the model is successful


# In[ ]:





# ## <font color="red"> Trying the Model on External Images </font>
# 

# ### 1. Trying model on External Images with big sizes than 28*28

# In[47]:


from PIL import Image


# In[66]:


# Opening the image
img = Image.open(r"C:\Users\iprak\Desktop\truck.jpg")

# Print the size of the image
print((img.size))


# In[67]:


# resize image and ignore original aspect ratio
img_resized = img.resize((32,32))

# report the size of the thumbnail
print((img_resized.size))

# Saving new image
img_resized.save(r'C:\Users\iprak\Desktop\img_resized.jpg')


# In[68]:


# Loading Compressed Image
img_example = Image.open(r"C:\Users\iprak\Desktop\img_resized.jpg")


# In[69]:


# convert image to numpy array
data_example = np.asarray(img_example)
data_example.shape


# In[70]:


# Reshaping image as taken by the model
data_reshape_example = data_example.reshape(1,32,32,3)

# To see new shape
data_reshape_example.shape


# In[71]:


# Predicting the number
model.predict_classes(data_reshape_example)


# In[ ]:





# In[54]:


# Saving the model
model.save('CNN_on_CIFAR-10.h5')


# In[ ]:





# ### 2. Trying model on images with the size of 32*-32*-3

# In[58]:


# Reading Image

a = plt.imread(r"C:\Users\iprak\Desktop\img_resized.jpg")
print((a.shape))
plt.imshow(a)


# In[59]:


# Converting image into a numpy array
a_a = np.asarray(a)

# reshaping image as taken by the model
a_b = a_a.reshape(1,32,32,3)
print((a_b.shape))

# Predicting the number
model.predict_classes(a_b)


# In[ ]:





# ### <font color="Magenta">Note: The model is working on external pictures. But Not Perfect</font>

# In[ ]:




