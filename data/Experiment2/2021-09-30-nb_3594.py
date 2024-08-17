#!/usr/bin/env python
# coding: utf-8

# In[80]:


#Let's start with importing necessary libraries
import os
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler , label_binarize
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,auc, confusion_matrix, roc_curve, roc_auc_score,classification_report
from pandas_profiling import ProfileReport
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# Initially I have corrected 1 file- dataset4.csv from bending2 folder as was separated with both ',' &
# ' ' space and then I have created the file with merging all the data

# In[19]:


all_dat = pd.DataFrame()
for dirpath, dirnames, files in os.walk('.'):
    print(f'Found directory: {dirpath}')
    for file_name in files:
        if file_name.endswith('.csv'):
            current_data = pd.read_csv(dirpath+"/"+file_name , encoding = "ISO-8859-1", skiprows=4,error_bad_lines=False)  
            current_data['label'] = dirpath[2:]                
            all_dat = pd.concat([all_dat,current_data])
            print(file_name)


# In[20]:


#renaming colmn
all_dat.rename(columns={'# Columns: time':'time'}, inplace=True)


# In[21]:


all_dat.to_csv("final_data.csv",index = False)


# In[2]:


df = pd.read_csv('final_data.csv')


# In[3]:


df


# In[4]:


# converting time column into category
df['time'] = df.time.astype('object')


# In[5]:


df.dtypes


# In[6]:


df.describe()


# In[40]:


ProfileReport(df)


# From Pearson graph, we can see that there is no multicollinearity between feature columns.
# 
# Later on we will check with the help of VIF.

# In[31]:


# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in df:
    if plotnumber<=7 :     # as there are 9 columns in the data
        ax = plt.subplot(3,3,plotnumber)
        sns.distplot(df[column])
        plt.xlabel(column,fontsize=20)
        #plt.ylabel('Salary',fontsize=20)
    plotnumber+=1
plt.show()


# In[21]:


fig, ax = plt.subplots(figsize=(15,10))
sns.boxplot(data=df.drop(columns = 'time'), width= 0.5,ax=ax,  fliersize=3)


# As we can see there are skewness in the columns of our dataset so now we have to handle particular data having skewness -> -> 

# In[8]:


q = df['avg_rss12'].quantile(0.02)
# we are removing the bottom 1% data from the avg_rss12 column
data_cleaned = df[df['avg_rss12']>q]
q = df['var_rss12'].quantile(0.95)
# we are removing the top 5% data from the var_rss12 column
data_cleaned = data_cleaned[data_cleaned['var_rss12']<q]

q = df['avg_rss13'].quantile(0.99)
# we are removing the top 5% data from the avg_rss13 column
data_cleaned = data_cleaned[data_cleaned['avg_rss13']<q]
q = df['avg_rss13'].quantile(0.99)
# we are removing the top 5% data from the avg_rss23 column
data_cleaned = data_cleaned[data_cleaned['avg_rss23']<q]

q = df['var_rss13'].quantile(0.95)
# we are removing the top 5% data from the var_rss13 column
data_cleaned = data_cleaned[data_cleaned['var_rss13']<q]
q = df['var_rss23'].quantile(0.95)
# we are removing the top 5% data from the var_rss23 column
data_cleaned = data_cleaned[data_cleaned['var_rss23']<q]



# In[105]:


# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in data_cleaned:
    if plotnumber<=7 :     # as there are 9 columns in the data
        ax = plt.subplot(3,3,plotnumber)
        sns.distplot(data_cleaned[column])
        plt.xlabel(column,fontsize=20)
        #plt.ylabel('Salary',fontsize=20)
    plotnumber+=1
plt.show()


# In[106]:


fig, ax = plt.subplots(figsize=(15,10))
sns.boxplot(data=data_cleaned.drop(columns = 'time'), width= 0.5,ax=ax,  fliersize=3)


# Here we have to tried to handle skewness in our data

# In[9]:


x = data_cleaned.drop(columns= ['label','time'])
y = data_cleaned['label']


# In[10]:


x


# In[11]:


y


# In[12]:


scalar = StandardScaler()
x_scaled = scalar.fit_transform(x)


# In[12]:


vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(x_scaled,i) for i in range(x_scaled.shape[1])]
vif["Features"] = x.columns

#let's check the values
vif


# As we can see from VIF, all the values are less than 10 i.e no multicollinearity is present amoung feature columns

# In[13]:


x_train,x_test,y_train,y_test = train_test_split(x_scaled,y, test_size=0.2, random_state =144)


# In[254]:


met = pd.DataFrame()
met['recall'] = recall_score(y_test, y_predicted_multinomial,average=None)
met['precision'] = precision_score(y_test, y_predicted_multinomial,average=None)
met['f1 score'] = f1_score(y_test, y_predicted_multinomial,average=None)
met['columns'] = ['bending1', 'bending2', 'cycling', 'lying', 'sitting', 'standing', 'walking']
met.set_index('columns',inplace=True)
met


# In[327]:


y_binary = label_binarize(y_test, classes=['bending1', 'bending2', 'cycling', 'lying','sitting', 'standing', 'walking'])
auc_macro_dict = {}
accuracy = {}
f1_s = {}
solver = ["lbfgs", "sag", "saga", "newton-cg"]
multiclass = ['ovr','multinomial']
for k in range(len(multiclass)):
    for i in range(len(solver)):
        lr = LogisticRegression(multi_class=multiclass[k], solver=solver[i],random_state= 40)
        lr.fit(x_train, y_train)
        y_pred = lr.predict(x_test)
        y_pred_prob = lr.predict_proba(x_test)
        val = roc_auc_score(y_binary, y_pred_prob, average='macro')
        
        auc_macro_dict[solver[i],multiclass[k]] = val
        accuracy[solver[i],multiclass[k]] = accuracy_score(y_test, y_pred)
        f1_s[solver[i],multiclass[k]] = f1_score(y_test, y_pred, average='macro')
for k in range(len(multiclass)):
    for i in range(len(solver)):
        print(f"For {solver[i],multiclass[k]} followings are the matrics ")
        print(f'AUC values after macro average = {round(auc_macro_dict[solver[i],multiclass[k]]*100, 6)}')
        print(f'accuracy = {round(accuracy[solver[i],multiclass[k]]*100, 6)}')
        print(f'f1 score is {round(f1_s[solver[i],multiclass[k]]*100, 6)}')
        print("*********************************************")


# In[332]:


max_value = max(auc_macro_dict.values())  # maximum value
max_keys = [k for k, v in list(auc_macro_dict.items()) if v == max_value] # getting all keys containing the `maximum`
print(f'{max_keys} is a model we should choose as it is getting highest {max_value} AUC_ROC Score')


# In[ ]:




