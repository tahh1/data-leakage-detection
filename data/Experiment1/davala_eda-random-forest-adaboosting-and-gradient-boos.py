#!/usr/bin/env python
# coding: utf-8

# **CUSTOMER CHURN IN TELECOM**

# **DATA QUESTIONS**

# As you probably guessed from the title of the dataset, this model aims to predict churn — a very common problem businesses face.
# Thinking about which metrics we want to use to evaluate our model, let’s think about what we want our model to predict and what is worse: a false negative prediction or a false positive prediction.
# Our model should predict whether a customer in our dataset will stay with the company(False) or if they will leave (True).
# In this scenario, we have:
# False negative: the model predicts that a customer stays with the company (False), when in fact that customer churns (True).
# 
# False positive: the model predicts that a customer will churn (True), when in fact they will stay(False).
# Given this, we would probably argue that false negatives are more costly to the company as it would be a missed opportunity to market towards keeping those customers. For this reason, we will use accuracy and recall scores to evaluate our model performance.

# In[35]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Load Classification Packages and Accuracy Packages for the analysis

# In[36]:


# Load the necessary libraries
import pandas as pd
import seaborn as sns
import numpy as np
import pandas_profiling as pf
pd.set_option('display.max_column', 60)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
print('complete')


# In[37]:


# load the dataset
data = pd.read_csv('/kaggle/input/churn-in-telecoms-dataset/bigml_59c28831336c6604c800002a.csv')
data.head()


# In[38]:


# checking the dimension of the dataset
rows, cols = data.shape
print(f'The number of rows in our dataset are {rows} \nWhile the number of columns are {cols}')


# In[39]:


# display the columns
data.columns


# let's replace empty space in our columns with underscore 

# In[40]:


# replace space in the columns with underscore
data.columns = data.columns.str.replace(' ', '_')
data.columns


# In[41]:


# descriptive statistics
data.describe()


# In[42]:


data.select_dtypes('number').head(3)


# In[43]:


# lets chech how these attributes correlated to our target variable
data.corr()['churn']


# In[44]:


# Lets visualiza our target variable
churn = data['churn'].value_counts()
sns.barplot(churn.index, churn.values)


# we can see that the number of custumners that leaved were lesser than customers that stay

# **Dealing with Categorical columns**

# In[45]:


# dealing with categorical columns 
data.select_dtypes('object').head(4)


# 
# you may have noticed from the above data(info), that we have 4 object types, and 3 of these object types are useful for our analysis. The other object column is phone_number which, you guessed it, is a customer’s phone number. A person’s phone number shouldn’t have any great bearing on whether they decide to stick with a phone company, so for this reason, I choose to simply drop this column from our feature set.

# In[46]:


# checking the number of values in 'international_plan' columns
data['international_plan'].value_counts()


# In[47]:


# checking the number of values in 'voice_mail_plan' columns
data['voice_mail_plan'].value_counts()


# In[48]:


# number of unique values in state
data['state'].nunique()


# In[49]:


# lets drop phone number columns
data = data.drop(['phone_number'], axis = 1)


# In[50]:


# create dummy variables for categorical columns
data = pd.get_dummies(data)
#data


# **Create Target and Features Variables:**

# In[51]:


# Target column:
y = data.churn

# features columns
x = data.drop('churn', axis = 1)


# **Train-Test Split:**

# In[52]:


# lets train our data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)


# **Model Evaluation using accuracy, precison and recal**

# **Model 1: DecisionTreeClassifier**

# In[53]:


# instantiate decision tree object with default params
dtc = DecisionTreeClassifier(max_depth = 5)

dtc.fit(x_train, y_train)
pred = dtc.predict(x_test)
accur = accuracy_score(pred, y_test)
print(accur)

# calculate recall_score for test data:
print((precision_score(y_test, dtc.predict(x_test))))

# calculate recall_score for test data:
print((recall_score(y_test, dtc.predict(x_test))))


# Well, you got a classification rate of 94%, considered as good accuracy.
# 
# Precision: Precision is about being precise, i.e., how accurate your model is. In other words, you can say, when a model makes a prediction, how often it is correct. In your prediction case, when your Decision Tree Classifier model predicted customer churn, that means customers will churn 89% of the time.
# 
# Recall: If there are customers who churned in the test set and your Decision Tree Classifier model can identify it 65% of the time.

# 
# Remember, for this problem, we care more about recall score since we want to catch false negatives. Recall is what we want to try to maximise from this model.

# **Model 2: Random Forest**

# In[54]:


RF  = RandomForestClassifier(n_estimators = 100, max_depth= 5)
RF.fit(x_train, y_train)

# prediction
pred =RF.predict(x_test)

# model eveluation
print(("Accuracy:",accuracy_score(y_test, pred)))
print(("Precision:",precision_score(y_test, pred)))
print(("Recall:",recall_score(y_test, pred)))


# 
# Our accuracy score has actually gone down from our first model. Let’s check recall:
# 
# our recall score has gone way down for our random forrest model! 

# **Model 3: AdaBoosting**

# In[55]:


# instantiate adaboost classifier object
ABC = AdaBoostClassifier(random_state = 15)

# fit the model to the training data:
ABC.fit(x_train, y_train)

# prediction
pred =ABC.predict(x_test)

# model eveluation
print(("Accuracy:",accuracy_score(y_test, pred)))
print(("Precision:",precision_score(y_test, pred)))
print(("Recall:",recall_score(y_test, pred)))


# Our recall score has improved somewhat significantly from the random forest model.

# **Model 4: Gradient Boosting**

# In[56]:


# instantiate gradient boost classifier object
GBC = GradientBoostingClassifier(random_state = 15)

# fit the model to the training data:
GBC.fit(x_train, y_train)

# prediction
pred =GBC.predict(x_test)

# model eveluation
print(("Accuracy:",accuracy_score(y_test, pred)))
print(("Precision:",precision_score(y_test, pred)))
print(("Recall:",recall_score(y_test, pred)))


# 
# Our highest accuracy scores so far. They’re not too far away from our first decision tree model. There’s also no significant evidence of overfitting on preliminary inspection.

# Once again, our highest recall score so far and this does outperform our first model significantly.
# Given these 4 models, we would choose the gradient boosting model as our best.
