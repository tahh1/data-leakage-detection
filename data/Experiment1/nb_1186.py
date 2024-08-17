#!/usr/bin/env python
# coding: utf-8

# In[63]:


#all the imports
#not all might be needed -- e.g. I am using Google colab and importing files from my Google drive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ###Question 1 (1 point) ###
#  Load the English and Italian data into 2 separate data frames.
# 

# In[64]:


#load the data
#your code goes here.
#%cd /content/drive/My Drive/COURSES/2020/DATA/
import pandas as pd
en_data = pd.read_csv('https://raw.githubusercontent.com/smumudin/NLP---Mini-Language-Models/master/Datasets/CONcreTEXT_trial_EN.tsv',sep='\t')
it_data = pd.read_csv('https://raw.githubusercontent.com/smumudin/NLP---Mini-Language-Models/master/Datasets/CONcreTEXT_trial_IT.tsv',sep='\t')


# In[65]:


#check to see what the data look like
en_data.head(100)


# In[66]:


it_data.head()


# ###Question 2 (2 points): ###
# Next we will create three columns in each data frame. 
# 
# One column will be called 'CONCRETE' and it will be a boolean. If the mean value of the word is >=4 then this value is 0, otherwise 1. 
# 
# Basically, we are assuming that all words with a MEAN score greater than or equal to 4 are concrete words, and the rest are abstract.
# 
# 
# Next column will be another boolean 'IS_NOUN'. If the column POS == N (noun), then the 'IS_NOUN' column will be 0 otherwise it will be 1. 
# 
# The third column is 'IS_EARLY'. We are setting another boolean variable column, where, if the word appears early in the sentence, then this value is 0 otherwise 1. We will assume that any word which appears at INDEX < 5 appears early in this sentence. 
# 
# Your dataframe after you write this code should look like the example shown below. 
# 
# You may find it useful to use numpy and the 'where' condition for this question. 
# 
# Repeat the same process for the Italian data dataframe.
# 

# In[67]:


# Create 3 new columns for english dataset based on the above conditions
import numpy as np
en_data['CONCRETE'] = np.where(en_data['MEAN']>=4,0,1)
en_data['IS_NOUN'] = np.where(en_data['POS']=='N',0,1)
en_data['IS_EARLY'] = np.where(en_data['INDEX']<5,0,1)

en_data.head()


# In[68]:


#create three new columns using conditions on the main data frame

#your code goes here
it_data['CONCRETE'] = np.where(it_data['MEAN']>=4,0,1)
it_data['IS_NOUN'] = np.where(it_data['POS']=='N',0,1)
it_data['IS_EARLY'] = np.where(it_data['INDEX']<5,0,1)

#below is what the dataframe should look like after you have inserted the new columns and set the appropriate values
it_data.head()                                


# 
# ###Question 3(3 points):###
# 
# Using sklearn implementation of the Perceptron algorithm. 
# 
# First, create X and y. 
# 
# X are your explanatory variables -- in this case, we will use the newly created 'IS_NOUN' and 'IS_EARLY' columns are our X.
# 
# y is the target variable -- the 'CONCRETE' column you created above is what we need to predict.
# 
# Create a train, test split using sklearn and fit the perceptron algorithm.
# 
# 

# In[69]:


#your code goes here.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# creation of X and Y for English dataset 
X = en_data[['IS_NOUN','IS_EARLY']]
y = en_data['CONCRETE']

# create a train and test data for English dataset
X_train_eng, X_test_eng, y_train_eng,y_test_eng = train_test_split(X,y,
                                        test_size = 0.2,random_state=42)



# In[70]:


# creation of X and Y for Italian dataset 
X = it_data[['IS_NOUN','IS_EARLY']]
y = it_data['CONCRETE']

# create a train and test data for English dataset
X_train_it, X_test_it, y_train_it,y_test_it = train_test_split(X,y,
                                        test_size = 0.2,random_state=42)


# ###Question 4(1 points):###
# 
# Predict the values on the test set and print the result accuracy value of the model. 

# In[71]:


# Apply the trained perceptron on the X data to make predicts for the y test data
# View the accuracy of the model, which is: 1 - (observations predicted wrong / total observations)
#sample output shown below

#your code goes here
from sklearn.linear_model import Perceptron
perceptron_eng_model = Perceptron(eta0=0.02,max_iter=20)
perceptron_eng_model.fit(X_train_eng,y_train_eng)

#print(perceptron_eng_model.get_params())


# In[72]:


predictions_english = perceptron_eng_model.predict(X_test_eng)
accuracy_english = accuracy_score(y_test_eng,predictions_english)
print(('Accuracy : {:.2f}'.format(accuracy_english)))


# 
# ###Question 5 (3 points):###
# 
# Repeat the Questions 3 and 4 for the Italian data. 
# 

# In[73]:


#Repeat the process for the italian dataset
#your code goes here.
#sample output shown below

perceptron_it_model = Perceptron(eta0=0.01,max_iter=20)
perceptron_it_model.fit(X_train_it,y_train_it)

#print(perceptron_it_model.get_params())


# In[74]:


# Apply the trained perceptron on the X data to make predicts for the y test data
# View the accuracy of the model, which is: 1 - (observations predicted wrong / total observations)
#sample output shown below
#your code goes here

predictions_italian = perceptron_it_model.predict(X_test_it)
accuracy_italian = accuracy_score(y_test_it,predictions_italian)
print(('Accuracy : {:.2f}'.format(accuracy_italian)))

