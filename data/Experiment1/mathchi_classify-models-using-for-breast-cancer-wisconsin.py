#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from warnings import filterwarnings
filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# Any results you write to the current directory are saved as output.


# In[2]:


import pandas as pd
df = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
data = df.copy()
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)                  # Unnamed: 32 sutunu veriye baktigimizda nan lardan olusuyor ondan drop edelim
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]   # binary yani 0 ile 1 degerlerden olusturmamiz gerekiyor. object lerden olusuyor bunun yerine 0 ile 1 lerden olurmali. cunku bize int veya float lazim
data.head()


# In[3]:


data.describe()


# In[4]:


y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)


# In[5]:


# x degerlerimiz baktigimizda degerlerin cok buyuk oldugu gorulur. Dolayisiyla verimizi normallestirmemiz gerekiyor

#*** Normalize ***#
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.30, 
                                                    random_state=42)


# # Logistic Regresyon
# 
# * Amac henuz gozlenmemis bir x deger seti geldiginde bunun sonucunda olusacak olan sinifi ortaya cikarmak tahmin etmek bir siniflandirici cikarmaktir.
# * Siniflandirma problemi icin bagimli ve bagimsiz degiskenler arasindaki iliskiyi tanimlayan linear bir model kurmaktir.
# * Bagimli degiskenin 1 yada 0 olmasi durumuyla ilgilenir yada evet veya hayir durumu
# * Bize int veya float degerlerle is yapar

# ## MODEL
# 

# In[7]:


# statsmodels araciligiyla model kurup fit yapalim. Burda bize modelin anlamliligi ve hangi degiskenin ne kadar etki ettigi bu tablodan cikiyor

loj = sm.Logit(y, x)
loj_model= loj.fit()
loj_model.summary()


# In[8]:


from sklearn.linear_model import LogisticRegression
loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(x,y)
loj_model


# In[9]:


# sabit degeri
loj_model.intercept_


# In[10]:


# butun bagimsiz degiskenlerin katsayi degerleri
loj_model.coef_


# ## PREDICT and MODEL TUNNING

# In[11]:


# tahmini yapalim
y_pred = loj_model.predict(x)


# In[12]:


# Gercekte 1 iken 1(PP) olanlar 1 iken 0(PN) olanlar, gercekte 0 iken 1(NP) olanlar 0 iken 0(NN) olanlar
confusion_matrix(y, y_pred)


# In[13]:


# accuracy degerine bakalim
accuracy_score(y, y_pred)


# In[14]:


# en detayli bir siniflandirma algoritmasinin sonuclarini degerlendirecek ciktilardan biri
print((classification_report(y, y_pred)))


# In[15]:


# ilk 10 model tahmini
loj_model.predict(x)[0:10]


# In[16]:


# yukarda 1 ve 0 verdigi degerlerden ziyade asil degerlerini versin istiyorsak 'predict_proba' modulunu kullanarak gercek degerleri
# matriste 0. indexinde veya sol tarafi 0 a ait degerleri, 1. indexinde veya sag tarafi 1 e ait degerleri verir 
loj_model.predict_proba(x)[0:10][:,0:2]                # ilk 10


# In[17]:


# simdi yukardaki 'predict_proba' on tahmin olasilik degerlerini model haline getirmeye calisalim
y_probs = loj_model.predict_proba(x)
y_probs = y_probs[:,1]


# In[18]:


y_probs[0:10]               # ilk 10


# In[19]:


# burdaki tahmin degerlerimizi donguye sokup 0.5 ten buyuklere 1 ve kucuk olanlara 0 versin
y_pred = [1 if i > 0.5 else 0 for i in y_probs]


# In[20]:


# yukardaki degere baktigimizda degisikligi farketmis oluruz ama burda degisiklik yok cunku dogrulanmasi gereken cok bir deger yokmus demekki. Bunu yapma amacimiz modelimizi dogrulamaktir.
y_pred[0:10]


# In[21]:


confusion_matrix(y, y_pred)


# In[22]:


accuracy_score(y, y_pred)


# In[23]:


print((classification_report(y, y_pred)))


# In[24]:


# bunu yukarda yaptik ilk 5 eleman gorunsun
loj_model.predict_proba(x)[:,1][0:5]


# In[25]:


logit_roc_auc = roc_auc_score(y, loj_model.predict(x))


# In[26]:


fpr, tpr, thresholds = roc_curve(y, loj_model.predict_proba(x)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Oranı')
plt.ylabel('True Positive Oranı')
plt.title('ROC')
plt.show()
# mavi cizgi kurmus oldugumuz model ile ilgili basarimizin grafigi
# kirmizi cizgi hicbirsey yapmasak modelimiz bu sekilde olacak


# Sekilde goruldugu gibi cok degistirilmesi veya dogrulanmasi gereken deger bulamadi bu veride.



# In[27]:


# test train ayirma islemine tabi tutalim
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)


# In[28]:


# Modelimizi olusturup fit edelim
loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(X_train,y_train)
loj_model


# In[29]:


# dogrulanma skorunu bulalim
accuracy_score(y_test, loj_model.predict(X_test))


# In[30]:


# dogrulanmis modelin CV skoru bulalim
cross_val_score(loj_model, X_test, y_test, cv = 10).mean()


# # KNN (K-Nearst Neigbourhood)
# 

# * Tahminler gozlem benzerligine gore yapilir. Bana arkadasini soyle sana kim oldugunu soyleyeyeyim mantigi ile calisir.
# 
# * Bagimsiz degiskenler ile diger degiskenler arasindaki uzaklik hesaplanir. en yakin k adet gozlemi bulup bunun icin en yakin gozlenen sinif model sinifidir.

# In[31]:


# model kurma
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)
knn_model


# In[32]:


# tahmin degeri
y_pred = knn_model.predict(X_test)


# In[33]:


accuracy_score(y_test, y_pred)


# In[34]:


# detayli ciktimizida alalim. 
print((classification_report(y_test, y_pred)))


# ##  MODEL TUNNING 

# In[35]:


# KNN parametrelerini bulma
knn_params = {"n_neighbors": np.arange(1,50)}


# In[36]:


# siniflandirmasi ve CV ile fit yapalim
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, knn_params, cv=10)
knn_cv.fit(X_train, y_train)


# In[37]:


# bunu sadece gozlemlemek icin yapiyoruz. Final modeli onemli bizim icin
print(("En iyi skor:" + str(knn_cv.best_score_)))
print(("En iyi parametreler: " + str(knn_cv.best_params_)))


# In[38]:


# yukarida ciktida ortaya cikan n_neighbors 11 cikmisti bunu kullanarak KNN olusturulup tuned edelim
knn = KNeighborsClassifier(11)
knn_tuned = knn.fit(X_train, y_train)


# In[39]:


# simdide test in tuned score una bakalim
knn_tuned.score(X_test, y_test)


# In[40]:


# tahmin degeri
y_pred = knn_tuned.predict(X_test)


# In[41]:


accuracy_score(y_test, y_pred)


# # SVC (Support Vector for Classification)

# * Amac iki sinif arasindaki ayrimin(marjinin) optimum olmasini saglayacak hiper-duzlemi bulmaktir
# 
# * Linear ve NonLinear SVM ler mevcut.

# In[42]:


# model ve nesne olusturma fit ile beraber yapalim
from sklearn.svm import SVC

svm_model = SVC(kernel = "linear").fit(X_train, y_train)


# In[43]:


svm_model


# In[44]:


y_pred = svm_model.predict(X_test)


# In[45]:


accuracy_score(y_test, y_pred)


# ## MODEL TUNNING

# In[46]:


# C parametresi olusturulacak olan dogrunun veya ayrimin olusturulmasiyla ilgili bir kontrol etme imkani saglayan parametredir
# C degeri 0 olamaz hata verir ondan 1 den baslasin

svc_params = {"C": np.arange(1,10)}


# In[47]:


svc = SVC(kernel = "linear")


# In[48]:


svc_cv_model = GridSearchCV(svc,svc_params, 
                            cv = 10, 
                            n_jobs = -1, 
                            verbose = 2 )

svc_cv_model.fit(X_train, y_train)


# In[49]:


# en iyi parametre degerleri
print(("En iyi parametreler: " + str(svc_cv_model.best_params_)))


# In[50]:


# tuned edip fit leyelim
svc_tuned = SVC(kernel = "linear", C = 5).fit(X_train, y_train)


# In[51]:


# simdi gercek deger ile tahmin edilen degerin karsilastirma islemini yapalim
y_pred = svc_tuned.predict(X_test)
accuracy_score(y_test, y_pred)


# # Naive Bayes Model
# 
# * Olasilik temelli bir modelleme teknigidir. Amac belirli bir ornegin her bir sinifa ait olma olasiliginin kosullu olasilik temelli hesaplanmasidir.
# 
# * e-ticaret veya cok sinifli veri setlerinde gayet iyi calistigi gorulmustur. 
# 
# *Ornek aylik geliri 2000 olan bu kisi krediyi odeyebilir mi?
# bu tarz orneklerde gayet uygun bir modeldir.

# ## MODEL, TAHMIN VE MODEL TUNNING

# In[52]:


from sklearn.naive_bayes import GaussianNB


# In[53]:


nb = GaussianNB()
nb_model = nb.fit(X_train, y_train)
nb_model


# In[54]:


# tahmin islemini yapalim
nb_model.predict(X_test)[0:10]


# In[55]:


y_pred = nb_model.predict(X_test)


# In[56]:


accuracy_score(y_test, y_pred)


# In[57]:


cross_val_score(nb_model, X_test, y_test, cv = 10).mean()


# ### As we can see between 4 models(Logistic Regresyon, KNN, SVC and Naive Bayes) SVC is most suitable model in Breast Cancer Wisconsin data. SVC model can explain accuracy score 98% of this data.
# 
# 
