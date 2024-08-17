#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

from matplotlib import pyplot


# In[2]:


with open('/Users/pranav/nba_allNBA_predictor/nn_train_data.pkl', 'rb') as f:
    X_train, y_train = pickle.load(f, encoding='latin1')
with open('/Users/pranav/nba_allNBA_predictor/nn_test_data.pkl', 'rb') as f:
    X_test, y_test = pickle.load(f, encoding='latin1')

print(('Training Sizes:', X_train.shape,'and', y_train.shape))
print(('Testing Sizes:', X_test.shape, 'and', y_test.shape))


# In[6]:


model = Sequential()

model.add(Dense(12, input_dim=28, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(4, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.summary()


# In[7]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[8]:


history = model.fit(X_train, y_train, epochs=35, batch_size=24)


# In[9]:


y_pred = model.predict(X_test)
predictions = list()
for i in range(len(y_pred)):
    predictions.append(np.argmax(y_pred[i]))

tests = list()
for i in range(len(y_test)):
    tests.append(np.argmax(y_test[i]))
    
accuracy = accuracy_score(predictions,tests)
print(('Accuracy is:', accuracy*100))


# In[10]:


history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=35, batch_size=24)


# In[11]:


print("Evaluate on test data")
results = model.evaluate(X_test, y_test, batch_size=128)
print(("test loss, test acc:", results))


# In[12]:


print("Generate predictions for 3 samples")
predictions = model.predict(X_test[10:23])
print(("predictions shape:", predictions.shape))

