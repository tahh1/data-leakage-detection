#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing required packages.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Loading dataset
wine = pd.read_csv('../input/winequality-red.csv')


# In[ ]:


#Let's check how the data is distributed
wine.head()


# In[ ]:


#Information about the data columns
wine.info()


# **Plot**

# In[ ]:


plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("Correlation Heatmap")
corr = wine.corr()
sns.heatmap(corr, cmap='RdBu',
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[ ]:


def plot(feature_x,target='quality'):
    sns.set_palette("Paired")
    sns.factorplot(x=target,y=feature_x,data=wine,kind='swarm',size=5,aspect=1)


# In[ ]:


plot('volatile acidity','quality',)


# In[ ]:


plot('alcohol','quality')


# In[ ]:


wine_subset_1 = wine[["quality","fixed acidity","volatile acidity","citric acid","residual sugar","chlorides"]]


# In[ ]:


sns.set(style="ticks", color_codes=True)
g = sns.pairplot(wine_subset_1,hue="quality")


# In[ ]:


wine_subset_2 = wine[["quality","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]


# In[ ]:


g = sns.pairplot(wine_subset_2, hue="quality")


# ## Preprocessing Data for performing Machine learning algorithms

# In[ ]:


#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)


# In[ ]:


#Now lets assign a labels to our quality variable
label_quality = LabelEncoder()


# In[ ]:


#Bad becomes 0 and good becomes 1 
wine['quality'] = label_quality.fit_transform(wine['quality'])


# In[ ]:


wine['quality'].value_counts()


# In[ ]:


sns.countplot(wine['quality'])


# In[ ]:


#Now seperate the dataset as response variable and feature variabes
X = wine.drop('quality', axis = 1)
y = wine['quality']


# In[ ]:


#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[ ]:


#Applying Standard scaling to get optimized result
sc = StandardScaler()


# In[ ]:


X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# ## Our training and testing data is ready now to perform machine learning algorithm

# ### Random Forest Classifier

# In[ ]:


rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)


# In[ ]:


#Let's see how our model performed
print((classification_report(y_test, pred_rfc)))


# #### Random forest gives the accuracy of 88%

# In[ ]:


#Confusion matrix for the random forest classification
print((confusion_matrix(y_test, pred_rfc)))


# ## Let's try to increase our accuracy of models
# ## Grid Search CV

# In[ ]:


#using grid sesarch to find out best parameter in the model
#1.n_estimators  (114)
param_test1 = {'n_estimators':list(range(100,130,2))}  
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
                                  min_samples_leaf=20,max_depth=8,max_features='sqrt' ,oob_score = True ,n_jobs = -1,random_state=10), 
                       param_grid = param_test1, scoring='accuracy',cv=10)
gsearch1.fit(X_train, y_train)
gsearch1.best_params_, gsearch1.best_score_


# In[ ]:


#2.max_depth & min_samples_split      (5,2)     
param_test2 = {'max_depth':list(range(2,10,1)), 'min_samples_split':list(range(2,10,2))}
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 114, 
                                  min_samples_leaf=20,max_features='sqrt' ,oob_score=True, n_jobs = -1, random_state=10),
   param_grid = param_test2, scoring='accuracy',iid=False, cv=10)
gsearch2.fit(X_train, y_train)
gsearch2.best_params_, gsearch2.best_score_


# In[ ]:


rf1 = RandomForestClassifier(n_estimators= 114, max_depth=5, min_samples_split=2,
                                  oob_score=True, random_state=10)
model1 = rf1.fit(X_train, y_train)
print((rf1.oob_score_))


# In[ ]:


#apply model to test data
PRED = rf1.predict(X_test)
print((classification_report(y_test, PRED)))


# 

# In[ ]:


#Now lets try to do some evaluation for random forest model using cross validation.
rf1_eval = cross_val_score(estimator = rf1, X = X_train, y = y_train, cv = 10)
rf1_eval.mean()


# ### Random forest accuracy increases from 88% to 89% using cross validation score
