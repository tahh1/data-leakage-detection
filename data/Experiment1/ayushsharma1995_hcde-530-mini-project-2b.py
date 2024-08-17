#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction
# Building upon our cleaned up and organized data from Mini Project 1b, we are using the information from United States Department of Labour to model it into a classifier. Let's import the final data from our previous notebook here and explore it further.
# 
# ## 1.1 Data Profile
# The data consists of 18 columns, that describe various employer and employee characteristics of people applying for an H1-B Visa in 2020. Here is how the visa process in real life looks like:
# 1. An person on an immigrant visa applies for H1-B approval
# 2. If approved, the person is eligible to apply in the lottery, where there is a randomized selection of people for visa
# 3. Once approved, the person gets 3 more years of visa to stay in United States.
# 
# Here is a list of some of the preprocessing that was performed on the data set:
# 1. Replacing missing values
# 2. Extracting useful columns
# 3. Making the data types consistent
# 4. Transforming 'ANNUAL_WAGE' into a single yearly column (from hour, week, bi-week, month, and year)
# 5. Finding duplicates
# 
# For more information, visualization, and preprocessing, refer to the earlier notebook below:
# [Link to previous notebook](https://www.kaggle.com/ayushsharma1995/hcde-530-mini-project-1b)

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

#Import Scikit Learn Library
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from statistics import mode

#Used for running SMOTE
import re
from xgboost import XGBClassifier
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## 1.2 Importing and Viewing the data
# As we grab the data from our earlier notebook into our current one, we find that some of the pre-processing is lost in translation. Let's view some features of our data:

# In[2]:


df = pd.read_csv('/kaggle/input/final.csv')
df.head()


# In[3]:


#Checking the datatypes of our data before modeling
df.dtypes


# As we can see, the datatypes for columns are reset. Let's assign the right datatype to each feature before performing operations on it.

# ## 1.3 Correcting feature datatypes
# ### 1.3.1 Organizing the Independent Variable(s)

# In[4]:


#Converting all possible dates from object to datetime format
df['RECEIVED_DATE'] =  pd.to_datetime(df['RECEIVED_DATE'])
df['DECISION_DATE'] =  pd.to_datetime(df['DECISION_DATE'])
df['BEGIN_DATE'] =  pd.to_datetime(df['BEGIN_DATE'])
df['END_DATE'] =  pd.to_datetime(df['END_DATE'])

#Changing binary valued columns to categorical data (unordered)
df['FULL_TIME_POSITION'] = df['FULL_TIME_POSITION'].astype('category')
df['AGENT_REPRESENTING_EMPLOYER'] = df['AGENT_REPRESENTING_EMPLOYER'].astype('category')
df['H-1B_DEPENDENT'] = df['H-1B_DEPENDENT'].astype('category')
df['VISA_CLASS'] = df['VISA_CLASS'].astype('category')


# In[5]:


#Examining all other remaining object dtypes
df.select_dtypes(object)


# All of the above object datatype columns have too many categories in them to just change their data type. Therefore, we are going to bin this into finer categories, starting with:
# 1. EMPLOYER_STATE: Categorizing all the states based on region (East, West, South, and Midwest)
# 2. EMPLOYER_NAME: Categorizing into a University Employer and Private Employer
# 3. SOC_TITLE: Since SOC_TITLE is more generic, we'll get rid of this column entirely
# 4. JOB_TITLE: Categorizing into generalized categories (Computer, Mathematics, Arts and Design, Business, Medical, Teaching, Engineering and Hardware, Sales, and Legal)
# 
# Apart from this initial categorizing, we'd be using one-hot encoding in the end to convert them to all numbers.

# ### Feature 1: Employer State
# Categorizing into west coast, east coast, mid-west, and south region

# In[6]:


#Viewing all unique state values
df['EMPLOYER_STATE'].unique()


# In[7]:


#Create a list of states by region
region_east = ['CONNECTICUT', 'MAINE', 'MASSACHUSETTS', 'NEW HAMPSHIRE', 'RHOSE ISLAND', 'VERMONT' 'NEW JERSEY', 'NEW YORK', 'PENNSYLVANIA']
region_midwest = ['ILLINOIS', 'INDIANA', 'MICHIGAN', 'OHIO', 'WISCONSIN', 'IOWA', 'KANSAS', 'MINNESOTA', 'MISSOURI', 'NEBRASKA', 'NORTH DAKOTA', 'SOUTH DAKOTA']
region_south = ['DELAWARE', 'FLORIDA', 'GEORGIA', 'MARYLAND', 'NORTH CAROLINA', 'SOUTH CAROLINA', 'VIRGINIA', 'DISTRICT OF COLUMBIA', 'WEST VIRGINIA', 'ALABAMA', 'KENTUCKY', 'MISSISSIPPI', 'TENNESSEE', 'ARKANSAS', 'LOUISIANA', 'OKLAHOMA', 'TEXAS']
region_west = ['ARIZONA', 'COLORADO', 'IDAHO', 'MONTANA', 'NEVADA', 'NEW MEXICO', 'UTAH', 'WYOMING', 'ALASKA', 'CALIFORNIA', 'HAWAII', 'OREGON', 'WASHINGTON']

#Create a new column EMPLOYER_REGION and select all the values from EMPLOYER_STATE column based on region
df['EMPLOYER_REGION'] = (
    np.select(
        condlist=[df['EMPLOYER_STATE'].isin(region_east), df['EMPLOYER_STATE'].isin(region_west), df['EMPLOYER_STATE'].isin(region_midwest), df['EMPLOYER_STATE'].isin(region_south)], 
        choicelist=['East Coast', 'West Coast', 'Mid-West Region', 'South Region']))

#Dropping all the other values that do not belong to these four regions
dropRegion = df[df['EMPLOYER_REGION'] == '0'].index
df.drop(dropRegion, inplace=True)


# In[8]:


#Changing the data type of EMPLOYER_REGION to category
df['EMPLOYER_REGION'].astype('category')


# ## Feature 2: Employer Name
# For the employer name, we're going to characterize it into whether the employer is an academic institution, or a private company.

# In[9]:


#All the terms that might be related to an academic institution
terms = ['UNIVERSITY', 'university', 'CITY', 'STATE', 'COLLEGE']
q = r'\b(?:{})\b'.format('|'.join(map(re.escape, terms)))

#Create an empty column by the name of 'EMPLOYER_TYPE'
df['EMPLOYER_TYPE'] = np.nan
df.EMPLOYER_TYPE[df['EMPLOYER_NAME'].str.contains(q)] = 'University/College'

#Replacing all other as Private Company Category
df['EMPLOYER_TYPE']= df.EMPLOYER_TYPE.replace(np.nan, 'Private Company', regex=True)

#Changing the data type of this column 
df['EMPLOYER_TYPE'] = df['EMPLOYER_TYPE'].astype('category')


# ## Feature 3: Job Title
# This categorization is a little complex. Upon examining this feature, we found out that there are over 700 unique values (including some redundant ones and some inconsistence case errors). We cannot have these many categories in this columns, therefore, we are going to proceed in the following manner:
# 1. We are going to extract a list all possible job titles for a given domain (e.g. engineering, arts etc.) from an external website
# 2. We are doing the same using beautiful soup and putting it into a list, and then using it as a filter for categorizing our new column
# 
# For more information about the job titles: refer to this [link](https://www.careerbuilder.com/browse)

# In[10]:


#Importing beautifulsoup
import requests
from bs4 import BeautifulSoup

#Creating a function that extracts a list of job titles by domain into a list
def extractJobs(link):
    html = requests.get(link).text
    bs = BeautifulSoup(html)
    possible_links = bs.find_all('a')
    title_list = []
    for link in possible_links[11:-28]:
        if link.string != 'Select Location':
            title_list.append(link.string.upper())
    return title_list


# In[11]:


#Create a new column named JOB_CATEGORY
df['JOB_CATEGORY'] = np.nan


# ### Software/Computer Jobs
# Inlcudes terms like software developer, back-end engineer, DevOps, .NET developer etc.

# In[12]:


computer_link = 'https://www.careerbuilder.com/browse/category/computer-occupations'
computer_terms = extractJobs(computer_link)
computer_filter = r'\b(?:{})\b'.format('|'.join(map(re.escape, computer_terms)))

df.JOB_CATEGORY[df['JOB_TITLE'].str.contains(computer_filter)] = 'Software/Computer'


# ### Arts and Design Jobs
# Includes terms like animator, interaction designer, graphic designer etc.

# In[13]:


design_link = 'https://www.careerbuilder.com/browse/category/art-and-design-workers'
design_terms = extractJobs(design_link)
design_filter = r'\b(?:{})\b'.format('|'.join(map(re.escape, design_terms)))

df.JOB_CATEGORY[df['JOB_TITLE'].str.contains(design_filter)] = 'Arts and Design'


# ### Mathematical Sciences Jobs
# Includes terms like acturial scientist, statistician, etc.

# In[14]:


math_link = 'https://www.careerbuilder.com/browse/category/mathematical-science-occupations'
math_terms = extractJobs(math_link)
math_filter = r'\b(?:{})\b'.format('|'.join(map(re.escape, math_terms)))

df.JOB_CATEGORY[df['JOB_TITLE'].str.contains(math_filter)] = 'Mathematical Sciences'


# ### Teaching Jobs
# Includes terms like assitant professors, junior primary teacher, etc.

# In[15]:


teaching_link = 'https://www.careerbuilder.com/browse/category/postsecondary-teachers'
teaching_terms = extractJobs(teaching_link)
teaching_filter = r'\b(?:{})\b'.format('|'.join(map(re.escape, teaching_terms)))

df.JOB_CATEGORY[df['JOB_TITLE'].str.contains(teaching_filter)] = 'Teaching'


# ### Sales
# Includes terms like Customer service associate, brand ambassador, etc.

# In[16]:


sales_link = 'https://www.careerbuilder.com/browse/category/sales-representatives-services'
sales_terms = extractJobs(sales_link)
sales_filter = r'\b(?:{})\b'.format('|'.join(map(re.escape, sales_terms)))

df.JOB_CATEGORY[df['JOB_TITLE'].str.contains(sales_filter)] = 'Sales'


# ### Engineering and Hardware
# Includes terms like CAD Engineer, Civil Engineer, Mechanical Engineer etc.

# In[17]:


eng_link = 'https://www.careerbuilder.com/browse/category/engineers'
eng_terms = extractJobs(eng_link)
eng_filter = r'\b(?:{})\b'.format('|'.join(map(re.escape, eng_terms)))

df.JOB_CATEGORY[df['JOB_TITLE'].str.contains(eng_filter)] = 'Engineering and Hardware'


# ### Community and Social Service
# Includes terms like counselors, therapists, volunteers, etc.

# In[18]:


comm_link = 'https://www.careerbuilder.com/browse/category/counselors-social-workers-and-other-community-and-social-service-specialists'
comm_terms = extractJobs(comm_link)
comm_filter = r'\b(?:{})\b'.format('|'.join(map(re.escape, comm_terms)))

df.JOB_CATEGORY[df['JOB_TITLE'].str.contains(comm_filter)] = 'Community and Social Services'


# ### Healthcare
# Includes terms like nurses, doctors, surgeons, etc.

# In[19]:


health_link = 'https://www.careerbuilder.com/browse/category/health-diagnosing-and-treating-practitioners'
health_terms = extractJobs(health_link)
health_filter = r'\b(?:{})\b'.format('|'.join(map(re.escape, health_terms)))

df.JOB_CATEGORY[df['JOB_TITLE'].str.contains(health_filter)] = 'Healthcare'


# ### Business
# Includes terms like agent, advisor, coordinator, etc.

# In[20]:


biz_link = 'https://www.careerbuilder.com/browse/category/business-operations-specialists'
biz_terms = extractJobs(biz_link)
biz_filter = r'\b(?:{})\b'.format('|'.join(map(re.escape, biz_terms)))

df.JOB_CATEGORY[df['JOB_TITLE'].str.contains(biz_filter)] = 'Business/Management'


# ### Legal
# Includes terms like judges, lawyers, etc.

# In[21]:


law_link = 'https://www.careerbuilder.com/browse/category/lawyers-judges-and-related-workers'
law_terms = extractJobs(law_link)
law_filter = r'\b(?:{})\b'.format('|'.join(map(re.escape, law_terms)))

df.JOB_CATEGORY[df['JOB_TITLE'].str.contains(law_filter)] = 'Legal'


# ### Other Jobs
# All other jobs that don't belong to a category go in here

# In[22]:


df['JOB_CATEGORY']= df.JOB_CATEGORY.replace(np.nan, 'Other', regex=True)


# Finally, we are changing the datatypes of our newly created columns

# In[23]:


df['JOB_CATEGORY'] = df['JOB_CATEGORY'].astype('category')
df['EMPLOYER_REGION'] =df['EMPLOYER_REGION'].astype('category')

#Drop older columns that are no longer pertinent
df = df.drop(['SOC_TITLE', 'EMPLOYER_NAME', 'EMPLOYER_STATE', 'JOB_TITLE'], axis=1)


# ### Feature 4: Decision Date and Received date
# Combining the above two columns into a single 'DECISION_TIME' column, that subtracts the dates in two columns into a final days column

# In[24]:


import datetime as dt
df['DECISION_TIME'] = df['DECISION_DATE'] - df['RECEIVED_DATE']
df['DECISION_TIME'] = df['DECISION_TIME'].dt.days


# ### Feature 5: Begin Date and End Date
# Doing the above procedure on these columns too

# In[25]:


df['TIME_EMPLOYED'] = df['END_DATE'] - df['BEGIN_DATE']
df['TIME_EMPLOYED'] = df['TIME_EMPLOYED'].dt.days


# In[26]:


#Dropping our old columns since we created new ones 
df = df.drop(['BEGIN_DATE', 'END_DATE', 'DECISION_DATE', 'RECEIVED_DATE'], axis=1)


# Converting all string categories to numerical categories

# In[27]:


df.select_dtypes('category')


# #### One Hot Encoding:
# Since Scikit Learn doesn't allow string categories, we are using one-hot encoding to convert them to numbers.

# In[28]:


#Visa Class
df = pd.get_dummies(df, columns=['VISA_CLASS', 'EMPLOYER_REGION', 'JOB_CATEGORY', 'EMPLOYER_TYPE'])

#Fulltime Position
df['FULL_TIME_POSITION'].replace({'Y': 1, 'N': 0}, inplace=True)
df['FULL_TIME_POSITION'] = df['FULL_TIME_POSITION'].astype(int)

#AGENT_REPRESENTING_EMPLOYER
df['AGENT_REPRESENTING_EMPLOYER'].replace({'Y': 1, 'N': 0}, inplace=True)
df['AGENT_REPRESENTING_EMPLOYER'] = df['AGENT_REPRESENTING_EMPLOYER'].astype(int)

#H-1B_DEPENDENT
df['H-1B_DEPENDENT'].replace({'Y': 1, 'N': 0}, inplace=True)
df['H-1B_DEPENDENT'] = df['H-1B_DEPENDENT'].astype(int)


# ## 1.3.2 Organizing Dependent Variable
# For this dataset, our dependent variable is the 'CASE_STATUS', which currently has different values:
# * Certified: A person applied for H1-b and was approved
# * Denied: A person applied for H1-b and was approved
# * Withdrawn: A person applied for H1-b but decided to withdraw their application
# * Certified-Withdrawn: A person applied for H1-b and approved, but decided to withdraw their application
# 
# We are going to map this into a binary categorization (i.e. 'CERTIFIED' and 'DENIED' only, further into 1 and 0 respectively)

# In[29]:


df = df.drop(df[df.CASE_STATUS == 'Certified - Withdrawn'].index)
df = df.drop(df[df.CASE_STATUS == 'Withdrawn'].index)
# df['CASE_STATUS'] = df['CASE_STATUS'].replace({'Certified': 1, 'Denied': 0}, inplace=True)
df['CASE_STATUS']


# In[30]:


df['CASE_STATUS'] = df['CASE_STATUS'].astype('category')
df['CASE_STATUS'].replace({'Certified': 1, 'Denied': 0}, inplace=True)


# In[31]:


#Viewing the distribution of our target variable
sns.countplot(x='CASE_STATUS', data=df, palette='hls')
plt.show()


# In[32]:


df['CASE_STATUS'].value_counts()


# From the graph, it is evident that there is a class imbalance: CERTIFIED cases are significantly higher than the DENIED cases. Let's view their actual percentage:

# In[33]:


#Finding the class ratio to detect imbalance:
count_certified = len(df[df['CASE_STATUS']== 1])
count_denied = len(df[df['CASE_STATUS']== 0])

#Calculating percentage of certified
pct_certified = count_certified/(count_certified+count_denied)
print(("percentage of no certified is", pct_certified*100))

#Calculating percentage of denied
pct_denied = count_denied/(count_certified+count_denied)
print(("percentage of denied", pct_denied*100))


# ### Handling the class imbalance
# The above class imbalance may affect the model accuracy in the end. Therefore, we are going to oversample the minority class (i.e. DENIED) using SMOTE

# In[34]:


df.info()


# In[35]:


#Traning-test split before oversampling with SMOTE

#All columns except target
X = df.loc[:, df.columns != 'CASE_STATUS']

#Target variable
y = df.loc[:, df.columns == 'CASE_STATUS']

#Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Describe info about train and test set 
print(("Number transactions X_train dataset: ", X_train.shape)) 
print(("Number transactions y_train dataset: ", y_train.shape)) 
print(("Number transactions X_test dataset: ", X_test.shape)) 
print(("Number transactions y_test dataset: ", y_test.shape)) 


# We're going to fit a logistic regression model before and after the oversampling procedure, to compare how things like accuracy, recall etc. are affected.

# In[36]:


import numpy as np
#Now train the model without handling the imbalanced class distribution
# logistic regression object 
lr = LogisticRegression() 
  
# train the model on train set 
lr.fit(X_train, y_train) 
  
predictions = lr.predict(X_test) 
  
# print classification report 
print((classification_report(y_test, predictions))) 


# As we can see, we have 99% accuracy, however, the recall is 50%, which means the model is biased towards the majority class. Let's use the SMOTE algorithm to balance the minority class.

# In[37]:


print(("Before OverSampling, counts of label '1': {}".format(y_train['CASE_STATUS'].value_counts()[1]))) 
print(("Before OverSampling, counts of label '0': {} \n".format(y_train['CASE_STATUS'].value_counts()[0]))) 

# Importing SMOTE module from imblearn library 
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(X_train, y_train) 
  
print(('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))) 
print(('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))) 
  
print(("After OverSampling, counts of label '1': {}".format(y_train_res['CASE_STATUS'].value_counts()[1]))) 
print(("After OverSampling, counts of label '0': {}".format(y_train_res['CASE_STATUS'].value_counts()[0])))


# As we can see here, the minority class (CASE_STATUS ==0) has been reshaped and new has the same frequency as the majority class. Let's fit a logistic regression model again to view the change in accuracy and recall.

# In[38]:


lr1 = LogisticRegression() 
lr1.fit(X_train_res, y_train_res) 
predictions = lr1.predict(X_test) 
  
# print classification report 
print((classification_report(y_test, predictions)))


# Upon fitting the model again on our new training data, we can see that the recall has changed from 50% to 88%, while the accuracy has only dipped from 99% to 96%.

# ### ROC Curve
# The receiver operating characteristic (ROC) curve is another common tool used with binary classifiers. The dotted line represents the ROC curve of a purely random classifier; a good classifier stays as far away from that line as possible (toward the top-left corner).
# 
# Source: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

# In[39]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, predictions)
fpr, tpr, thresholds = roc_curve(y_test, lr1.predict_proba(X_test)[:,1])
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


# ## 3. Next Steps
# 1. The next logical step is to comapre different classifier models: like Decision Trees, SVMs, Naive Bayes etc. and which one performs the best for this kind of data.
# 2. While I did go at great lengths in cleaning the data, the data currently is far from being perfect: the categories could've been better, maybe the feature selection could've been more thought through etc.
