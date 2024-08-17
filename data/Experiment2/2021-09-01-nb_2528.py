#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# 1.9 Kill warnings
import warnings
warnings.filterwarnings("ignore")


# # LSTM BITCOIN

# In[3]:


data = pd.read_csv("bitcoin daily data.csv")
data.set_index('Date', drop=True, inplace=True)
data.head()


# In[4]:


plt.style.use("fivethirtyeight")
plt.figure(figsize=(16,8))
plt.title("Bitcoin Closing Stock Price")
plt.plot(data["Close"])
plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Price USD ($)", fontsize=18)
plt.show()


# In[5]:


data.shape


# In[6]:


btc=data.filter(["Close"])


# In[7]:


dataset=btc.values


# In[8]:


train_data_len=math.ceil(len(dataset) *.8)
train_data_len


# In[9]:


scaler = MinMaxScaler(feature_range= (0,1))
scale_data=scaler.fit_transform(dataset)


# In[10]:


train_data = scale_data[0:train_data_len,:]


# In[11]:


x_train = []
y_train = []
for i in range (80,len(train_data)):
  x_train.append(train_data[i-80:i,0])
  y_train.append (train_data[i,0])
  if i<=81:
    print(x_train)
    print(y_train)
    print()


# In[12]:


x_train, y_train= np.array(x_train), np.array(y_train)


# In[13]:


x_train=np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1 ))
x_train.shape


# In[14]:


model =Sequential()
model.add(LSTM(70, return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(70, return_sequences=False))
model.add(Dense(35))
model.add(Dense(1))


# In[15]:


model.compile(optimizer= 'rmsprop', loss='mean_squared_error')


# In[16]:


model.fit(x_train, y_train, batch_size=1, epochs =20 )


# In[17]:


test_data = scale_data[train_data_len-80: , :]
x_test = []
y_test = dataset[train_data_len:, :]
for i in range (80, len(test_data)):
  x_test.append(test_data[i-80:i,0])


# In[18]:


x_test= np.array(x_test)


# In[19]:


x_test =np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[20]:


predictions = model.predict (x_test)
predictions = scaler.inverse_transform(predictions)


# In[20]:


rmse = np.sqrt(np.mean(predictions -y_test)**2)
rmse


# In[21]:


train = data[:train_data_len]
valid = data[train_data_len:]
valid['Predictions'] = predictions 


# In[22]:


plt.figure(figsize=(20,10))
plt.title('Bitcoin close price predictions')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close price (INR)', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Validation', 'Predictions'], loc = 'lower right')
plt.show()


# # LSTM Ethereum
# 

# In[23]:


data_1 = pd.read_csv("eth daily data.csv")
data_1.set_index('Date', drop=True, inplace=True)
data_1.head()


# In[24]:


plt.style.use("fivethirtyeight")
plt.figure(figsize=(16,8))
plt.title("Ethereum Closing Stock Price")
plt.plot(data_1["Close"])
plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Price USD ($)", fontsize=18)
plt.show()


# In[25]:


data_1.shape


# In[26]:


eth=data_1.filter(["Close"])


# In[27]:


dataset=eth.values


# In[28]:


train_data_len=math.ceil(len(dataset) *.8)
train_data_len


# In[29]:


scaler = MinMaxScaler(feature_range= (0,1))
scale_data=scaler.fit_transform(dataset)


# In[30]:


train_data = scale_data[0:train_data_len,:]


# In[31]:


x_train = []
y_train = []
for i in range (80,len(train_data)):
  x_train.append(train_data[i-80:i,0])
  y_train.append (train_data[i,0])
  if i<=81:
    print(x_train)
    print(y_train)
    print()


# In[32]:


x_train, y_train= np.array(x_train), np.array(y_train)


# In[33]:


x_train=np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1 ))
x_train.shape


# In[34]:


model =Sequential()
model.add(LSTM(70, return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(70, return_sequences=False))
model.add(Dense(35))
model.add(Dense(1))


# In[35]:


model.compile(optimizer= 'rmsprop', loss='mean_squared_error')


# In[36]:


model.fit(x_train, y_train, batch_size=1, epochs =20 )


# In[37]:


test_data = scale_data[train_data_len-80: , :]
x_test = []
y_test = dataset[train_data_len:, :]
for i in range (80, len(test_data)):
  x_test.append(test_data[i-80:i,0])


# In[38]:


x_test= np.array(x_test)


# In[39]:


x_test =np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[40]:


predictions = model.predict (x_test)
predictions = scaler.inverse_transform(predictions)


# In[21]:


rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse


# In[42]:


train = data_1[:train_data_len]
valid = data_1[train_data_len:]
valid['Predictions'] = predictions 


# In[43]:


plt.figure(figsize=(20,10))
plt.title('Ethereum close price predictions')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close price (INR)', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Validation', 'Predictions'], loc = 'lower right')
plt.show()


# In[ ]:




