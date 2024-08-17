#!/usr/bin/env python
# coding: utf-8

# # Project On Thyroid Detection Using Machine Learning Algorithms

# In[1]:


#import required libraries


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


data = pd.read_csv("hypothyroid.data")


# In[4]:


data.columns = ["class","age","sex","on_thyroxine","query_on_thyroxine","on_antithyroid_medication","thyroid_surgery","query_hypothyroid","query_hyperthyroid","pregnant","sick","tumor","lithium","goitre","TSH_measured","TSH","T3_measured","T3","TT4_measured","TT4","T4U_measured","T4U","FTI_measured","FTI","TBG_measured","TBG"]


# In[5]:


data.head()


# In[6]:


data.shape
# rows 3162 columns 26


# In[7]:


data.describe()


# In[8]:


data.info()


# In[9]:


# lets check null values
data.isnull().sum()


# In[10]:


# data does not have null values in it but has ?.... which need to be replaced by NaN
data["TBG"].value_counts().head(10)


# In[11]:


data.replace("?",np.nan, inplace=True)


# In[12]:


# Now we can see the all ? are getting replaced by NaN values
data.head()


# In[13]:


#  check number of nulls in dataset
data.isnull().sum()


# In[14]:


data.describe()
# we can see the number of elements, total unique values, top val, and frequency of top val


# In[15]:


#we found that there are maximum null values in column TBG, it is better to remove it
data.drop(["TBG"], axis = 1, inplace= True)


# In[16]:


data.head()


# In[17]:


# import label_encoder, and apply label encoding on all columns which have categorical 
# values, , also as all our columns have dtype as "object", so label encoder needs a str or numeric input,
# we converted object type to string type
# Six columns are having continuous value but their dtype is "object" so converted it to 
# float type
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cols = ["class","sex","on_thyroxine","query_on_thyroxine","on_antithyroid_medication","thyroid_surgery","query_hypothyroid","query_hyperthyroid","pregnant","sick","tumor","lithium","goitre","TSH_measured","T3_measured","TT4_measured","T4U_measured","FTI_measured","TBG_measured"]
for i in cols:
    data[i] = data[i].astype(str)
    data[i] = le.fit_transform(data[i])

    
cols1 = ["age","TSH","T3","TT4","T4U","FTI"]

for j in cols1:
    data[j] = data[j].astype(float)



# In[28]:


data['age'].isnull().sum()
pd.set_option("display.max_columns", 26)


# In[20]:


data.info()


# In[23]:


data['age'].isnull().sum()


# In[26]:


data.fillna(data.mean(), inplace=True)


# In[27]:


data['age'].isnull().sum()


# In[32]:


data.info()


# In[34]:


data.corr().head()


# In[40]:


#class has strong correlation with Strong Negative Corr with TSH, Positie with T3, TT4

import seaborn as sns
ax = sns.pairplot(data, vars= ["class","TSH","T3","TT4"])


# In[43]:


col2 = ["class","TSH","T3","TT4"]
data[col2].corr()

# we found that these columns have good correlation


# In[47]:


sns.heatmap(data[col2])


# Lets get started with data spiliting

# In[85]:


# split input and output data
y = data.iloc[:,0:1]
x = data.iloc[:,1:]


# In[86]:


x.head()


# In[87]:


y.head()


# In[88]:


print(("x shape:", x.shape, "y shape:",y.shape))


# In[93]:


from sklearn.model_selection import train_test_split
# data spitting for trainig and testing
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.20, random_state = 42)


# In[98]:


# Model imlementation

from sklearn.linear_model import LogisticRegressionCV
model = LogisticRegressionCV(max_iter=1000, cv = 10)
model.fit(xtrain, ytrain)
pred = model.predict(xtest)


# In[112]:


n_ytest = np.array(ytest)
model.score(xtest,pred)


# In[116]:


# calculate accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

acc = accuracy_score(ytest, pred) * 100
cm = confusion_matrix(ytest, pred)
cr = classification_report(ytest, pred)

print(("Accuracy = ",acc))
print(("Confuion Matrics = \n", cm))
print("Classification_Report = ")
print(cr)


# In[ ]:




