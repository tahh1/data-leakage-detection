#!/usr/bin/env python
# coding: utf-8

# # **The PUBG CHALLENGE**

# In[ ]:


import pandas as pd
import numpy as np

import warnings
warnings.simplefilter("ignore")




train_data_path = "../input/train_V2.csv"#getting the path of the file to be inputed and assigned it
train_data = pd.read_csv(train_data_path)#now reading the csv file to continue the proceaa
train_data.head()#to show the first 5 Columns


# #### Now looking into the head of the file we can see that the row 'Id', 'groupId' and 'matchId are features which doent affect the result.So we need to remove it from the test data to get better accuracy.

# In[ ]:


unwanted_features = ['Id', 'groupId', 'matchId']#making a list of unwanted features
train = train_data.drop(unwanted_features, axis=1)#dropping the unwanted features from the data
train.head()#showing the new head()


# #### After checking our new dataset we could understand we have an important feature 'matchType' but is in object form, which is a problem as models can't train object type files.So we use "ONE HOT ENCODING".

# In[ ]:


new_train = pd.get_dummies(train, columns=['matchType'], drop_first = True)#implementing one hot encoding on matchType and storing it on new train data
new_train.head()#displaying its head


# #### Now that we have coverted every column of our data to int/float values  lets check wheater the dataset is clean / it contain any NaN values

# In[ ]:


new_train.isnull().sum()#sum of number of NaN values in each column


# In[ ]:


new_train.shape#check the number of rows and columns


# ####  Only one column has missing value out of 4446966 colums.So I'm going to drop the one column. 

# In[ ]:


new_train.dropna(inplace=True)#dropping all the columns with NaN values
new_train.shape#Now again checking the number of columns and rows.


# In[ ]:


new_train.isnull().sum()#YaaaY......!Sum of NaN values became zero


# # TRAINING
# ### Now we have our clean dataset we can now start training our model.First step is to split tge data to X, y and train and test dat.

# In[ ]:


from sklearn.model_selection import train_test_split


#Splitting my data into features and Label
X = new_train.drop(['winPlacePerc'], axis=1)
y = new_train.winPlacePerc

#Splitting my features and label into train and test set so that I could check the accuracy and improve the model

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state= 1, test_size=0.4)
print("Done◇")


# #### Now its time to choose an algorithm

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor



models = [ RandomForestRegressor , AdaBoostRegressor, XGBRegressor]


# #### Now we should define a function that could give the mae of each models by itrating through each algorithm

# In[ ]:


#def best_model(n):
 #   model = n()
  #  model.fit(train_X, train_y)
   # pred = model.predict(test_X)
    #Error = mae(pred, test_y)
    #return Error



#for i in models:
    #print("MAE of ", i, best_model(i))


# In[ ]:


test_data_path = "../input/test_V2.csv"
test_data = pd.read_csv(test_data_path)
test_data = test_data.drop(unwanted_features, axis=1)
test = pd.get_dummies(test_data, columns=["matchType"], drop_first = True)
test.head()


# In[ ]:


model = RandomForestRegressor()
model.fit(X, y)
pred = model.predict(test)
print("Done♡")


# In[ ]:


test_data_id = pd.read_csv(test_data_path)
output = pd.DataFrame({'Id' : test_data_id.Id , 'winPlacePerc' : pred})
output.to_csv("output.csv", index=False)


# 

# 

# 

# 

# 

# 

# 

# 

# 
