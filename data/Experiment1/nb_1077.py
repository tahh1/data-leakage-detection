#!/usr/bin/env python
# coding: utf-8

# Github Link - https://github.com/mohansameer1983/bank-campaign-for-selling-loans-aiml

# In[158]:


#Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[159]:


# Load Data from csv
df = pd.read_csv('Bank_Personal_Loan_Modelling.csv')


# In[160]:


# Print data
df.head()


# In[128]:


# Print columns datatype and non-null count
df.info()


# In[129]:


# Print Statistical Summary
df.describe()


# In[130]:


# Print Shape
df.shape


# In[131]:


# Check null values sum per column
df.isnull().values.any()


# In[132]:


# No of unique in each column
df.nunique()


# In[133]:


# No of people with zero mortgage
df[(df['Mortgage']>0)]['Mortgage'].count()


# In[134]:


# Number of people with zero credit card spending per month
df[(df['CCAvg']==0)]['CCAvg'].count()


# In[135]:


# Value counts of all categorical columns
df[{'Family','Education'}].count()


# In[136]:


# Univariate Analysis
for i in df.columns:
    plt.figure(figsize=(5,5))
    sns.distplot(df[i],)
    plt.show()


# In[149]:


# From above analysis we can see some negative values in 'Experience'
df[df['Experience'] < 0]['Experience'].count()


# In[150]:


# Replace negative Experience with column mean() in dataset
df[df['Experience'] < 0] = df['Experience'].mean()

# Check negative values again
df[df['Experience'] < 0]['Experience'].count()


# In[139]:


# Lets check graph again just for Experience column
plt.figure(figsize=(5,5))
sns.distplot(df['Experience'])
plt.show()


# In[140]:


# Bivariate Analysis with Personal Loan as target variable
for i in df.columns:
    plt.figure(figsize = (5,5))
    sns.lineplot(x = df[i], y = df['Personal Loan'])
    plt.show() 


# In[190]:


# Get data model ready, where 'Personal Loan' column is target variable.
y = df['Personal Loan']
X = df.drop(['Personal Loan','ZIP Code'], axis=1)


# In[191]:


#splitting the data in 70:30 ratio of train to test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)


# In[193]:


##### Import sklearn libraries
# Fit a model
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

log_reg = LogisticRegression(random_state=7,solver='lbfgs', max_iter=800)
log_reg.fit(X_train, y_train)


# In[194]:


# Publish metrics evaluating model performance
y_predicted = log_reg.predict(X_test)
print(("Test Data Accuracy:",metrics.accuracy_score(y_test, y_predicted)))
print(("Train Data Accuracy:",log_reg.score(X_train, y_train)))
print(("Precision:",metrics.precision_score(y_test, y_predicted)))
print(("Recall:",metrics.recall_score(y_test, y_predicted)))
print(("f1_Score:",metrics.f1_score(y_test, y_predicted)))


# In[203]:


# Publish 'roc_auc_score'
logit_roc_auc = metrics.roc_auc_score(y_test, y_predicted)
print(("roc_auc_score:",logit_roc_auc))


# In[205]:


#AUC ROC curve
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, log_reg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[206]:


# Draw heatmap to display confusion matrix
cm = metrics.confusion_matrix( y_test, y_predicted)
sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["Accept", "Not Accept"] , yticklabels = ["Accept", "Not Accept"] )
plt.ylabel('Observed')
plt.xlabel('Predicted')
plt.show()


# In[207]:


# Calculating Coefficients against each attribute
coef_table = pd.DataFrame(list(X_train.columns)).copy()
coef_table.insert(len(coef_table.columns),"Coeff",log_reg.coef_.transpose())
coef_table.sort_values(by='Coeff',ascending=False)


# In[168]:


# Publish rows from test data, where actual class not equal to predicted.
df_compare = pd.DataFrame(X_test, columns=["ID"])
df_compare["Observed Class"] = y_test
df_compare["Predicted Class"] = y_predicted

df_incorrect = df_compare[df_compare["Observed Class"] != df_compare["Predicted Class"]]
df_incorrect.head(20)


# **Conclusion**
# 
# Thera bank executives want to find ways to convert its liability customers to personal loan customers.
# We can say test data and train data accuracy is almost same, which indicated that data is not under or overfit.
# There is high probability that person with higher family size/income/education have higher chances of taking up personal loan.
# 
# ##### `Important Features`
# 
# "Education", "Family", "CCAvg", "CDAccount", "Age"
# 
# seems to be top 5 features which influence the model's output. Based on the coefficients value.
# 
