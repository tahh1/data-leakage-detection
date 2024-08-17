#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
from sklearn.model_selection import train_test_split
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install keras')
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# In[517]:


data = pd.read_csv('converted.csv')
data = data.head(84*30)
#data['Inflation (Can)'] = data['Inflation (Can)'].shift(-30)
#data.drop([84],inplace=True)
data.drop(['Period'],axis=1,inplace=True)


# In[534]:


scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
y = scaled_data[:,0]
x = scaled_data[:,1:]
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
scaled_data = pca.fit_transform(x)


# In[535]:


train_X = scaled_data[:72*30]
train_y = y[:72*30]
test_X = scaled_data[72*30:84*30]
test_y = y[72*30:84*30]

#train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
#test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


# In[551]:


from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# In[541]:


svr = SVR()
svr.fit(train_X,train_y)
svr.score(test_X,test_y)


# In[552]:


para = {
    'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
    'tol' : [0.001,0.01],
    'C' : [0.8,1,1.2]
}
grid_ser = GridSearchCV(SVR(),scoring = 'neg_mean_absolute_error',param_grid=para ,n_jobs =5,cv = 4,verbose=5)
grid_ser.fit(train_X,train_y)


# In[578]:


a = scaler.inverse_transform(
    concatenate(
        (grid_ser.best_estimator_.predict(test_X).reshape(360,-1),pca.inverse_transform(test_X)),
         axis=1
    )
)[:,0]

b = scaler.inverse_transform(
    concatenate(
        (test_y.reshape(360,-1),pca.inverse_transform(test_X)),
         axis=1
    )
)[:,0]

pyplot.plot(a, label='predicted')
pyplot.plot(b, label='actual')
pyplot.legend()
pyplot.show()


# In[585]:


mean_squared_error(a,b)


# In[586]:


mean_absolute_error(a,b)


# In[587]:


from sklearn.metrics import r2_score
r2_score(a,b)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


yhat = model.predict(test_X)
test_X1 = pca.inverse_transform(test_X.reshape((test_X.shape[0], test_X.shape[2])))
# invert scaling for forecast
inv_yhat = concatenate((yhat.reshape(12*30,1), test_X1), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y1 = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y1, test_X1), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
#rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
#print('Test RMSE: %.3f' % rmse)
pyplot.plot(inv_yhat, label='predicted')
pyplot.plot(inv_y, label='actual')
pyplot.legend()
pyplot.show()


# In[ ]:





# In[483]:


model = Sequential()

model.add(LSTM(128,activation='relu', input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
model.add(LSTM(128,return_sequences=True))
model.add(LSTM(64,return_sequences=True))
model.add(LSTM(16,return_sequences=True))
model.add(tf.keras.layers.Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=7, batch_size=100, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# In[484]:


yhat = model.predict(test_X)
test_X1 = pca.inverse_transform(test_X.reshape((test_X.shape[0], test_X.shape[2])))
# invert scaling for forecast
inv_yhat = concatenate((yhat.reshape(12*30,1), test_X1), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y1 = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y1, test_X1), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
#rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
#print('Test RMSE: %.3f' % rmse)
pyplot.plot(inv_yhat, label='predicted')
pyplot.plot(inv_y, label='actual')
pyplot.legend()
pyplot.show()


# In[ ]:





# In[ ]:


yhat = model.predict(pred_X)
test_X1 = pred_X.reshape((pred_X.shape[0], pred_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat.reshape(13,1), test_X1), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y1 = pred_y.reshape((len(pred_y), 1))
inv_y = concatenate((test_y1, test_X1), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
#rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
#print('Test RMSE: %.3f' % rmse)
pyplot.plot(inv_yhat, label='predicted')
pyplot.plot(inv_y, label='actual')
pyplot.legend()
pyplot.show()


# In[364]:


data = pd.read_csv('raw_data.csv')
data = data.drop(['Inflation (Can)','Period'],axis=1)


# In[363]:





# In[374]:





# In[ ]:




