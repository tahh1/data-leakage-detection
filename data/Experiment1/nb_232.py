#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd #verileri çekmek için kullanacağız


# In[19]:


dataframe = pd.read_excel("veri/bisiklet_fiyatlari.xlsx") # verileri okuduk


# In[20]:


dataframe.head() # bu kod ile çektiğimiz verinin içindeki ilk beş satırı çekerek, verimizde neler var bakabiliriz


# In[21]:


import seaborn as sbn
import matplotlib.pyplot as plt


# In[22]:


sbn.pairplot(dataframe) # değerlerin birbine göre dağılımlarını görebiliriz


# ## veriyi test/train olarak ikiye ayırmak

# In[23]:


from sklearn.model_selection import train_test_split #ikiye ayrımak için kullanacağımız modülü import ettik


# In[39]:


# y=wx+b
# y-->label
y = dataframe["Fiyat"].values 

# x-->feature(özellik)
x = dataframe[["BisikletOzellik1","BisikletOzellik2"]].values

#veriyi burada ikiye böldük yüzde 33'ünü test versisi geri kalanını öğrenme verisi olarak ayırdık
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=15) 


# In[40]:


x_train


# In[26]:


x_train.shape # %70


# In[27]:


x_test.shape  # %30


# In[31]:


y_train.shape


# In[32]:


y_test.shape


# In[30]:


# scaling (boyutlandırmak)


# In[41]:


from sklearn.preprocessing import MinMaxScaler  


# In[42]:


scaler =MinMaxScaler()


# In[43]:


scaler.fit(x_train) # burada veriler 0 ile 1 arasında uygun değerlere getirildi


# In[44]:


x_train = scaler.transform(x_train)
x_test =scaler.transform(x_test)


# In[45]:


x_train


# In[46]:


import tensorflow as tf


# In[49]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[79]:


# burada modelimizi belirledik
model = Sequential() 

model.add(Dense(5,activation="relu")) #gizli katmanlarımızı ekliyoruz
model.add(Dense(5,activation="relu")) #ve içinde kaç nöron olacağını belirdik
model.add(Dense(5,activation="relu")) #ve son olarak aktivasyon modelimizi seçtik
#kısacası 3 gizli katmana sahip her katmanda 5 nöron bulunan ve aktivasyonu RELU olan bir sinir ağı tasarladık

model.add(Dense(1)) #çıktı sayımızı belirledik

model.compile(optimizer="rmsprop",loss ="mse") # gradient descent (optimizasyon) algoritmamızı belirledik.


# In[80]:


model.fit(x_train,y_train,epochs=250) #burada verilerimizi eğitmeye başlıyoruz


# In[81]:


loss=model.history.history["loss"]
print(loss)


# In[82]:


sbn.lineplot(x=list(range(len(loss))),y=loss) # kayıplarımızı görebiliyoruz


# In[83]:


trainLoss = model.evaluate(x_train,y_train,verbose=0)


# In[84]:


testLoss = model.evaluate(x_test,y_test,verbose=0)


# In[85]:


trainLoss


# In[86]:


testLoss


# In[87]:


# kayıpların birbirine yakın olması iyi denilebilir


# In[88]:


testTahminleri = model.predict(x_test) # özelliklerden y testi çıkardı


# In[89]:


testTahminleri


# In[90]:


tahminDF =pd.DataFrame(y_test,columns=["Gerçek Y"])


# In[91]:


tahminDF


# In[92]:


testTahminleri = pd.Series(testTahminleri.reshape(330,))


# In[93]:


testTahminleri


# In[94]:


tahminDF = pd.concat([tahminDF,testTahminleri],axis=1)


# In[95]:


tahminDF


# In[96]:


tahminDF.columns = ["Gerçek Y","Tahmin Y"]


# In[97]:


tahminDF


# In[98]:


sbn.scatterplot(x="Gerçek Y",y="Tahmin Y",data =tahminDF)


# In[99]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[100]:


mean_absolute_error(tahminDF["Gerçek Y"],tahminDF["Tahmin Y"]) # fiyatlarda 7 lira sapma olduğunu gösterir


# In[101]:


mean_squared_error(tahminDF["Gerçek Y"],tahminDF["Tahmin Y"])


# In[102]:


dataframe.describe()


# In[121]:


yeniBisikletOzellikleri =[[1745,1749]] # eklenen yeni verinin tahmini fiyatı ne olur ona bakalım


# In[122]:


yeniBisikletOzellikleri = scaler.transform(yeniBisikletOzellikleri)


# In[123]:


model.predict (yeniBisikletOzellikleri)


# In[124]:


from tensorflow.keras.models import load_model


# In[125]:


model.save("bisiklet_modeli.h5") #modelimi kaydedebiliriz


# In[126]:


sonradanCagirilanModel = load_model("bisiklet_modeli.h5") # ve sonradan çağırıp tekrar kullanabiliriz


# In[127]:


sonradanCagirilanModel.predict(yeniBisikletOzellikleri)


# In[ ]:




