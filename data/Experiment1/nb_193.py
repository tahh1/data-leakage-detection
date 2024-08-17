#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('../Data/RSCCASN.csv',index_col=0,parse_dates=True)


# In[3]:


df.head()


# In[4]:


df.index


# In[5]:


df.columns


# In[6]:


plt.plot(df)


# In[7]:


plt.figure(figsize=(10,10),dpi=100)
plt.plot(df,color="purple")


# In[8]:


df.head()


# In[9]:


df.tail()


# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


df.isnull()


# In[13]:


df.isnull().sum()


# In[14]:


sns.heatmap(df.corr())


# In[15]:


sns.heatmap(df.isnull())


# In[16]:


test_percent=0.1


# In[17]:


len(df)


# In[18]:


test_index=int(np.round(len(df)-len(df)*test_percent))


# In[19]:


test_index


# In[20]:


train=df[0:test_index:]
test=df[test_index::1]


# In[21]:


train


# In[22]:


test


# In[23]:


from sklearn.preprocessing import MinMaxScaler


# In[24]:


scaler=MinMaxScaler()


# In[25]:


scaler.fit(train)


# In[26]:


scaled_train=scaler.transform(train)
scaled_test=scaler.transform(test)


# In[27]:


len(scaled_test)


# In[28]:


length=16
batch_size=1


# In[29]:


from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# In[30]:


generator=TimeseriesGenerator(scaled_train,scaled_train,length=length,batch_size=batch_size)


# In[31]:


validation_generator=TimeseriesGenerator(scaled_test,scaled_test,length=length,batch_size=batch_size)


# In[32]:


from tensorflow.keras.models import load_model,Sequential


# In[33]:


from tensorflow.keras.layers import Dense,Activation,Dropout,LSTM


# In[34]:


from tensorflow.keras.optimizers import Adam


# In[35]:


from tensorflow.keras.callbacks import EarlyStopping,TensorBoard


# In[36]:


early_stop=EarlyStopping(monitor="val_loss",mode="min",patience=10,verbose=1)


# In[37]:


n_features=1


# In[38]:


model=Sequential()
model.add(LSTM(100,input_shape=(length,n_features),activation="relu"))
model.add(Dense(1))
model.compile(loss="mse",optimizer="adam")


# In[39]:


model.summary()


# In[40]:


model.fit_generator(generator,validation_data=validation_generator,epochs=600,verbose=1,callbacks=[early_stop])


# In[41]:


losses=pd.DataFrame(model.history.history)


# In[42]:


losses.plot.line()


# In[43]:


prediction=[]


# In[44]:


first_eval_batch=scaled_train[-length::]


# In[45]:


current_eval_batch=first_eval_batch.reshape(1,length,n_features)


# In[46]:


for i in scaled_test:
    current_pred=model.predict(current_eval_batch)[0]
    prediction.append(current_pred)
    current_eval_batch=np.append(current_eval_batch[:,1:,:],[[current_pred]],axis=1)


# In[47]:


true_prediction=scaler.inverse_transform(prediction)


# In[48]:


test


# In[49]:


true_prediction


# In[50]:


test["pred"]=true_prediction


# In[51]:


test


# In[52]:


plt.plot(test)


# In[53]:


model.save("rnnmodel.h5")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




