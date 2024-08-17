#!/usr/bin/env python
# coding: utf-8

# # Stock Price Prediction

# #### Predict the closing stock price of a corporation using the past 60 day stock price. 
# #### model use RNN LSTM

# In[4]:


#Import Libraries
import math
import pandas_datareader as web
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# ### Get Stock quote

# In[10]:


df = web.DataReader('AAPL', data_source = 'yahoo', start = '2010-01-01', end='2020-08-17')


# In[11]:


# exploaring Data
df


# In[12]:


df.shape


# In[13]:


plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Data', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()


# In[15]:


# Creating new dataframe with only the close column
data = df.filter(['Close'])
# Convert the dataframe to numpy array
dataset = data.values
#get the number of row to train the model on 
training_data_len = math.ceil( len(dataset) * .8)

training_data_len


# In[16]:


#Scale Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data


# In[18]:


#create the training data set
train_data = scaled_data[0:training_data_len, :]
#split data
x_train = []
y_train = []

for i in range (60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i <= 60:
        print (x_train)
        print (y_train)
        print ()


# In[20]:


# Convert the x-train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)


# In[21]:


x_train.shape


# In[22]:


# reshape data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# In[23]:


x_train.shape


# In[25]:


#Build The LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))


# In[26]:


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[28]:


# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=5)


# In[30]:


# Creating the testing data set
test_data = scaled_data[training_data_len - 60: , :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

    


# In[31]:


#convert data to a numpy array
x_test = np.array(x_test)


# In[32]:


#reshape data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[33]:


x_test.shape


# In[34]:


# get the model predicted price value
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[35]:


#get the root mean squared error
rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse


# In[36]:


#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title("Model")
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train','Val', 'Predictions'], loc='lower right')
plt.show()


# In[37]:


# show the valid and the predicted price 
valid


# In[ ]:




