#!/usr/bin/env python
# coding: utf-8

# # SUV Car Purchase Prediction

# In this project, we will build a model and on the basis of the model we will predict whether a Person will buy the SUV car or not.

# First, we will import all the library a

# In[482]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


# With the help of this dataset, we will build a model that will help us to make prediction on further related datasets.

# In[483]:


data=pd.read_csv(r'C:\Users\Shloka Daga SD03\dataset\suv_dataset.csv')
data


# In[484]:


data.columns


# In[485]:


data.describe()


# In[486]:


gender_map={'Male':1,'Female':0}
data['Gender']=data['Gender'].map(gender_map)
data


# In[487]:


plt.figure(figsize=(15,8))
plt.scatter(x='Age',y='EstimatedSalary',data=data)
plt.xlabel('Age')
plt.ylabel('EstimatedSalary')
plt.title('Relation between Age and Salary')
plt.plot()


# ## Heatmap

# In[488]:


plt.figure(figsize=(15,8))
sns.heatmap(data.corr())
plt.show()


# In[489]:


X=data.iloc[:,[2,3]]
y=data.iloc[:,4].values


# In[490]:


X


# In[491]:


y


# In[492]:


from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)


# ## Feature Scaling

# In[493]:


from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
X_train=SS.fit_transform(X_train)
X_test=SS.fit_transform(X_test)


# ## Logistic Regression

# In[494]:


from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(X_train,y_train)
y_pred=LR.predict(X_test)
LR_accuracy=accuracy_score(y_pred,y_test)*100
print(('Logistic Regression Accuracy : ',LR_accuracy))


# ## DecisionTreeClassifier

# In[495]:


from sklearn.tree import DecisionTreeClassifier
DTC=DecisionTreeClassifier()
DTC.fit(X_train,y_train)
y_pred=DTC.predict(X_test)
DTC_accuracy=accuracy_score(y_pred,y_test)*100
print(('Decision Tree Classifier Accuracy : ',DTC_accuracy))


# ## RandomForestClassifier

# In[496]:


from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(n_estimators=100)
RF.fit(X_train,y_train)
y_pred=RF.predict(X_test)
RF_accuracy=accuracy_score(y_pred,y_test)*100
print(('Random Forest Accuracy : ',RF_accuracy))


# ## SVC

# In[497]:


from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
svc_accuracy=accuracy_score(y_pred,y_test)*100
print(('SVC Accuracy : ',svc_accuracy))


# ## GradientBoostingClassifier

# In[498]:


from sklearn.ensemble import GradientBoostingClassifier
GBC=GradientBoostingClassifier()
GBC.fit(X_train,y_train)
y_pred=GBC.predict(X_test)
GBC_accuracy=accuracy_score(y_pred,y_test)*100
print(('Gradient Boosting Classifier Accuracy : ',GBC_accuracy))


# ## Model Comparison

# In[499]:


model_df=pd.DataFrame({'Model':['Logistic Regression','Decision Tree Classifier','Random Forest Classifier','SVC',
                               'Gradient Boostring Classifier'],
                      'Accuracy Score':[LR_accuracy,DTC_accuracy,RF_accuracy,svc_accuracy,GBC_accuracy]})
model_df


# In[500]:


model_df=model_df.sort_values(by='Accuracy Score',ascending=False)
model_df


# Since, Random Forest Classifier gives us higher accuracy than other model. So we will use Random Forest Classifier model to predict whether a Person will buy a SUV car or will not buy the SUV car.
# The model gives a 93.18% accuracy, which means that the model data is 93.18% accurate.

# So, by providing new dataset to the model, we can now predict whether the Person can but the SUV car or not.
