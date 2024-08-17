#!/usr/bin/env python
# coding: utf-8

# Class the attacker is success or fail
# ,and find the most import feature in the successfull or fail when the attacker
# ,and predict some attacker can be successfull?

# In[1]:


import numpy as np 
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import linear_model, datasets
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn


# In[2]:


df=pd.read_csv("../input/globalterrorismdb_0616dist.csv", encoding='ISO-8859-1',low_memory=False)
df.head()


# extract usefull feature

# In[3]:


df1=df[['iyear','imonth','iday','country','region','latitude','longitude','specificity'
        ,'vicinity','crit1','crit2','crit3','doubtterr','multiple','success','suicide'
        ,'attacktype1','targtype1','targsubtype1','ingroup','guncertain1','weaptype1']]


# In[4]:


dfs=df1[df1['success']==1].dropna()
dff=df1[df1['success']==0].dropna()
dfs=dfs.sample(len(dff))


yf=dff['success']
xf=dff.drop(['success'],axis=1)


ys=dfs['success']
xs=dfs.drop(['success'],axis=1)


# In[5]:


train1,test1=train_test_split(dfs,test_size=0.3)
train2,test2=train_test_split(dff,test_size=0.3)

train=train1.append(train2)
test=test1.append(test2)


# In[6]:


Y=train['success'].values
X = train.drop(['success'],axis=1).values

y=test['success'].values
x = test.drop(['success'],axis=1).values

#X=MinMaxScaler().fit_transform(X)
#x=MinMaxScaler().fit_transform(x)

#pca=PCA(n_components=5)
#Xd=pca.fit_transform(X)
#xd=pca.fit_transform(x)


# In[7]:


#transformer=SelectKBest(score_func=chi2,k=5)
#Xt=transformer.fit_transform(abs(X),Y)
#xt=transformer.fit_transform(abs(x),y)


# In[8]:


reg_model = RandomForestClassifier()
reg_model = reg_model.fit(X,Y)

pred = reg_model.predict(x)

print((len(pred[pred==y])/float(len(y))))


# list the importance feature

# In[9]:


X = train.drop(['success'],axis=1)

w11=pd.Series(np.sort(reg_model.feature_importances_),X.columns[np.argsort(reg_model.feature_importances_)])
w11.sort_values(inplace=True,ascending=False)
print (w11)


# we find the top 5 importanc feature is :
# 
#  1. 'longitude' :12.5% 
#  2. 'latitude'    :11.76%
#  3. 'attacktype1'     :10.19%
#  4. 'iyear'        :10.07%
#  5. 'iday'         :9.37

# In[10]:


print((metrics.classification_report(y, pred)))
print((metrics.confusion_matrix(y, pred)))


# In[11]:


false_positive_rate, true_positive_rate, thresholds = roc_curve(y, pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print(('AUC = %0.4f'% roc_auc))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[12]:


chi=df[df['country_txt']=='China']
bj=chi[chi['provstate']=='Beijing']
bj1=bj[['iyear','imonth','iday','country','region','latitude','longitude','specificity'
        ,'vicinity','crit1','crit2','crit3','doubtterr','multiple','success','suicide'
        ,'attacktype1','targtype1','targsubtype1','ingroup','guncertain1','weaptype1']]

bj1['iyear']=2017
bj1=bj1.dropna()
bjx = bj1.drop(['success'],axis=1).values



# In[13]:





# In[13]:


bjy=reg_model.predict_proba(bjx)
pd.DataFrame(bjy,columns=['fail_prob','success_prob'])


# In[14]:


bj1[-2:]


# In[15]:


#bj[bj['targtype1']==14]['targtype1_txt']
#bj[bj['weaptype1']==8]['weaptype1_txt']
#bj[bj['targsubtype1']==74]['targsubtype1_txt']
#bj[bj['attacktype1']==2]['attacktype1_txt']


# beijing security is so bad ,2 new attack (Armed Assault) have 70% successed

# so 2017-03-19 attack the Military(Military Personnel (soldiers, troops, officers...) with  Firearms,
# 2017-10-28 attack the Private Citizens & Property (Marketplace/Plaza/Square) with Incendiary, can success

# **Just for fun !!!Don,t do that!!!**
