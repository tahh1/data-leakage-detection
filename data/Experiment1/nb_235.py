#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd #verileri çekmek için kullanacağız
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn


# In[2]:


dataframe = pd.read_excel("veri/merc.xlsx") # verileri okuduk


# In[4]:


dataframe.head(10) # bu kod ile çektiğimiz verinin içindeki ilk beş satırı çekerek, verimizde neler var bakabiliriz


# In[7]:


dataframe.shape


# In[8]:


dataframe.describe()


# In[9]:


dataframe.isnull().sum() # hangi sütun da toplam kaç tane eksiklik var bize gösterir


# In[10]:


plt.figure(figsize=(8,6)) #penceremizi boyutlandırabiliriz
sbn.distplot(dataframe["price"])


# In[11]:


plt.figure(figsize=(8,6)) #penceremizi boyutlandırabiliriz
sbn.countplot(dataframe["year"])


# In[12]:


dataframe.corr() #


# In[15]:


sbn.pairplot(dataframe[['year', 'price', 'mileage', 'tax','mpg','engineSize']], diag_kind='kde')


# In[13]:


dataframe.corr()["price"].sort_values()


# In[14]:


sbn.scatterplot(x="mileage",y="price",data =dataframe) 


# kilometre arttıkça fiyat azalıyor

# In[15]:


dataframe.sort_values("price",ascending=False).head(20) #yüksek fiyattan düşüğe doğru arabaları sırala


# In[16]:


dataframe.sort_values("price",ascending=True).head(20) #yüksek fiyattan düşüğe doğru arabaları sırala


# In[17]:


len(dataframe)


# In[18]:


len(dataframe)*0.01


# In[19]:


yuzdeDoksanDokuzDf=dataframe.sort_values("price",ascending=False).iloc[131:] #data setten yüzde 1lik kısmı atacağız


# In[20]:


yuzdeDoksanDokuzDf.describe()


# In[21]:


plt.figure(figsize=(8,6))
sbn.distplot(yuzdeDoksanDokuzDf["price"])


# In[22]:


dataframe.groupby("year").mean()["price"]


# In[23]:


yuzdeDoksanDokuzDf.groupby("year").mean()["price"]


# In[24]:


dataframe[dataframe.year !=1970].groupby("year").mean()["price"]


# In[25]:


dataframe =yuzdeDoksanDokuzDf


# In[26]:


dataframe.describe()


# In[27]:


dataframe =dataframe[dataframe.year != 1970] #verimizden 1970 yılını çıkardık


# In[28]:


dataframe.groupby("year").mean()["price"]


# In[29]:


dataframe.describe()


# In[30]:


dataframe = dataframe.drop("transmission",axis=1) #vites sütunu da veri setinden çıkarıyoruz


# In[31]:


dataframe.head()


# ## veriyi test/train olarak ikiye ayırmak

# In[32]:


from sklearn.model_selection import train_test_split #ikiye ayrımak için kullanacağımız modülü import ettik


# In[33]:


# y=wx+b
# y-->label
y = dataframe["price"].values 

# x-->feature(özellik)
x = dataframe.drop("price",axis=1).values

#veriyi burada ikiye böldük yüzde 33'ünü test versisi geri kalanını öğrenme verisi olarak ayırdık
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=10) 


# In[34]:


x_train


# In[35]:


x_test


# In[36]:


y_train


# In[37]:


y_test


# In[38]:


x_train.shape


# In[39]:


x_test.shape


# In[40]:


y_train.shape


# In[41]:


from sklearn.preprocessing import MinMaxScaler  


# In[42]:


scaler =MinMaxScaler()


# In[43]:


x_train = scaler.fit_transform(x_train)# burada veriler 0 ile 1 arasında uygun değerlere getirildi


# In[44]:


x_test =scaler.transform(x_test)# burada veriler 0 ile 1 arasında uygun değerlere getirildi


# In[45]:


from tensorflow.keras.models import Sequential #modelimiz oluşturyoruz
from tensorflow.keras.layers import Dense #katmanlarımızı oluşturacağız
import tensorflow as tf


# In[46]:


x_train.shape


# In[47]:


# burada modelimizi belirledik
model = Sequential() 

model.add(Dense(12,activation="relu")) #gizli katmanlarımızı ekliyoruz
model.add(Dense(12,activation="relu")) #ve içinde kaç nöron olacağını belirdik
model.add(Dense(12,activation="relu")) #ve son olarak aktivasyon modelimizi seçtik
model.add(Dense(12,activation="relu")) #ve son olarak aktivasyon modelimizi seçtik
#kısacası 4 gizli katmana sahip her katmanda 12 nöron bulunan ve aktivasyonu RELU olan bir sinir ağı tasarladık

model.add(Dense(1)) #çıktı sayımızı belirledik

model.compile(optimizer="adam",loss ="mse") # gradient descent (optimizasyon) algoritmamızı belirledik.


# In[48]:


model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=250,epochs=300) #burada verilerimizi eğitmeye başlıyoruz


# In[49]:


import pandas as pd


# In[50]:


kayipverisi = pd.DataFrame(model.history.history)


# In[51]:


kayipverisi.head()


# In[52]:


plt.figure(figsize=(9,8))
kayipverisi.plot()


# In[53]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[54]:


tahminDizisi =model.predict(x_test)


# In[55]:


tahminDizisi


# In[56]:


mean_absolute_error(y_test,tahminDizisi)


# In[57]:


dataframe.describe()


# In[58]:


tahminDF =pd.DataFrame(y_test,columns=["Gerçek Y"])


# In[59]:


tahminDF


# In[60]:


testTahminleri = pd.Series(tahminDizisi.reshape(3897,))


# In[61]:


testTahminleri


# In[62]:


tahminDF = pd.concat([tahminDF,testTahminleri],axis=1)


# In[63]:


tahminDF


# In[64]:


tahminDF.columns = ["Gerçek Y","Tahmin Y"]


# In[65]:


tahminDF.head(30)


# In[66]:


plt.scatter(y_test,tahminDizisi)
plt.plot(y_test,y_test,"g-*")


# In[67]:


dataframe.iloc[2]


# In[68]:


newcar = dataframe.drop("price",axis=1).iloc[2]


# In[69]:


type(newcar)


# In[70]:


newcar = scaler.transform(newcar.values.reshape(-1,5))


# In[71]:


model.predict(newcar)


# In[ ]:




