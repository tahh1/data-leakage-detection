#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string


# In[3]:


df = pd.read_csv('../input/twitter-sample-dataset/twitter dataset.csv',encoding='iso-8859-1')
df


# In[4]:


df.info()


# # Feature Selection

# In[5]:


df = df.drop(['gender_gold','profile_yn_gold','tweet_coord','tweet_id','tweet_location','user_timezone'],axis = 1)
df.head()


# In[6]:


df.info()


# In[7]:


df['gender'].unique()


# In[8]:


df = df[df['gender'] != 'nan']
df.info()


# In[9]:


df = df.dropna(subset = ['gender'])
df.info()


# In[10]:


df = df.replace(np.nan, '', regex = True)
df.info()


# In[11]:


df = df[df['gender'] != 'unknown']
df['gender'].unique()


# In[12]:


df.shape


# In[13]:


df = df.drop(['_unit_state', 'created', 'tweet_created', 'sidebar_color', 'link_color', 'profileimage', '_last_judgment_at', '_trusted_judgments'],axis = 1)
df.head()


# In[14]:


df.shape


# I have done the modelling in two ways. One, using numerical values only- gender:confidence, fav_number, retweet_count, and tweet_count, and the second, using tweet text and NLP.****

# # Label Encoding

# In[15]:


from sklearn.preprocessing import LabelEncoder
labelen = LabelEncoder()


# In[16]:


df['gender1'] = labelen.fit_transform(df['gender'])
df['profile_yn1'] = labelen.fit_transform(df['profile_yn'])
df.head()


# In[17]:


df['gender1'].value_counts()


# #  Balancing the Dataset

# In[18]:


df.describe()


# In[19]:


df = df.sort_values('gender:confidence', ascending = True)
df


# In[20]:


fig, ax = plt.subplots(figsize=(15,6))
ax.scatter(df['gender:confidence'], df['gender1'])
ax.set_xlabel('Gender Confidence')
ax.set_ylabel('gender')
plt.show()


# In[21]:


df = df[df['gender:confidence']>=0.6]
df


# In[22]:


df = df.sort_values('tweet_count', ascending = False)
df


# In[23]:


df = df[df['tweet_count']>=1000]
df


# In[24]:


df.describe()


# In[25]:


import seaborn as sb


# In[26]:


sb.boxplot(x=df['tweet_count'])


# In[27]:


fig, ax = plt.subplots(figsize=(15,6))
ax.scatter(df['tweet_count'], df['gender1'])
ax.set_xlabel('Number of Tweets')
ax.set_ylabel('gender')
plt.show()


# In[28]:


df = df[df['tweet_count']<=500000]
df


# In[29]:


df['gender1'].value_counts()


# In[30]:


male_df = df[df['gender1'] == 1][:4903]
female_df = df[df['gender1'] == 2]
female_df.shape,male_df.shape

df = male_df
df = df.append(female_df)


# In[31]:


df.shape


# In[32]:


df['gender1'].value_counts()


# In[33]:


df.describe()


# ## Cleaning the tweet text by Natural Language Processing using the Natural Language Toolkit or NLTK 

# In[34]:


tweets = list(df['text'])
tweets[5]


# In[35]:


def strip_all_entities(text):
    words = []
    entity_prefixes = ['@','#','\\']

    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)


for i in range(0,len(tweets)):
    tweets[i] = tweets[i].lower()
    tweets[i] = strip_all_entities(tweets[i])

tweets[5]


# In[36]:


def remove_links(text):
    words = []
    for word in text.split():
        if not 'https' in word:
            words.append(word)
    return ' '.join(words)


for i in range(0,len(tweets)):
    tweets[i] = remove_links(tweets[i])
    tweets[i] = tweets[i].replace("[^a-zA-Z#]"," ")
tweets[0:5]


# In[37]:


def remove_punc(text):
    words = nltk.word_tokenize(text)
    words=[word for word in words if word.isalpha()]
    return ' '.join(words)

for i in range(0,len(tweets)):
    tweets[i] = remove_punc(tweets[i])

tweets[0:10]


# In[38]:


stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

for i in range(0,len(tweets)):
    tweets[i] = remove_stopwords(tweets[i])

tweets[0:10]


# In[39]:


from nltk.stem import WordNetLemmatizer

lemm = WordNetLemmatizer()

def get_root_words(text):
    words = nltk.word_tokenize(text)
    words = [lemm.lemmatize(word) for word in words]
    return " ".join(words)

for i in range(0,len(tweets)):
    tweets[i] = get_root_words(tweets[i])

tweets[0:10]


# In[40]:


df['tweets'] = tweets

df.head()


# In[41]:


df.info()


# # Data Visualization

# In[42]:


import seaborn as sb

plt.subplots(figsize=(20,15))
sb.heatmap(df.corr(), annot=True)


# In[43]:


df.to_csv('twitter dataset_final.csv')
df


# In[44]:


df.columns


# In[45]:


df.info()


# # Building a Model which predicts the gender based on the numerical values given in the data
# ## Here we use the following algorithms:
# ### 1. Logistic Regression
# ### 2. Decision Tree Classifier
# ### 3. Gaussian Naive-Bayes
# ### 4. Random Forest Classifier
# ### 5. K-Nearest Neighbours
# ### 6. Voting Classifier for ensemble modelling

# In[46]:


X = df[['gender:confidence','fav_number', 'retweet_count', 'tweet_count']].values
Y = df[['gender1']].values


# In[47]:


from sklearn.model_selection import train_test_split


# In[48]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[49]:


X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


#  ## Logistic Regression

# In[50]:


from sklearn.linear_model import LogisticRegression


# In[51]:


lr = LogisticRegression()


# In[52]:


lr.fit(X_train, Y_train)


# In[53]:


lr.score(X_test, Y_test)


# ## DTC

# In[54]:


from sklearn.tree import DecisionTreeClassifier 


# In[55]:


dtc=DecisionTreeClassifier(random_state = 0)


# In[56]:


dtc.fit(X_train,Y_train)


# In[57]:


dtc.score(X_test,Y_test)


# ## GaussianNB

# In[58]:


from sklearn.naive_bayes import GaussianNB


# In[59]:


gnb = GaussianNB()
gnb.fit(X_train,Y_train)


# In[60]:


gnb.score(X_test,Y_test)


# ## Random Forest

# In[61]:


from sklearn.ensemble import RandomForestClassifier


# In[62]:


rf = RandomForestClassifier(random_state = 0)
rf.fit(X_train,Y_train)


# In[63]:


rf.score(X_test,Y_test)


# ## KNN

# In[64]:


from sklearn.neighbors import KNeighborsClassifier


# In[65]:


knn = KNeighborsClassifier()


# In[66]:


knn.fit(X_train,Y_train)


# In[67]:


knn.score(X_test,Y_test)


# ## Voting Classifier for Ensemble Learning

# In[68]:


from sklearn.ensemble import VotingClassifier


# In[69]:


clf1 = LogisticRegression(multi_class = 'multinomial', random_state = 0)
clf2 = RandomForestClassifier(n_estimators = 50, random_state = 0)
clf3 = DecisionTreeClassifier(random_state = 0)


# In[70]:


vc = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('dtc', clf3)], voting='hard')
vc.fit(X_train,Y_train)


# In[71]:


vc.score(X_test,Y_test)


# # Building a Model which predicts the gender based on the tweet text
# ## Here I used the following algorithms:
# ### 1. Multinomial Naive-Bayes
# ### 2. Decision Tree Classifier
# ### 3. Random Forest Classifier
# ### 4. Gaussian Naive-Bayes
# ### 5. Logistic Regression
# ### 6. Voting Classifier for ensemble modelling

# In[72]:


from sklearn.feature_extraction.text import CountVectorizer


# In[73]:


cv = CountVectorizer(ngram_range = (1,3),max_features = 10000)

X = cv.fit_transform(df['tweets'])

df['gender'].unique()


# In[74]:


y = df['gender1']


# In[75]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)


# ### Multinomial Naive-Bayes 

# In[76]:


from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(X_train,y_train)
pred1 = nb.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(pred1,y_test)


# ### Decision Tree Classifier

# In[77]:


dtc.fit(X_train,y_train)
pred2 = dtc.predict(X_test)

accuracy_score(pred2,y_test)


# ### Random Forest Classifier

# In[78]:


rf.fit(X_train,y_train)
pred3 = rf.predict(X_test)

accuracy_score(pred3,y_test)


# ### Logistic Regression

# In[79]:


X_test = X_test.toarray()
X_train = X_train.toarray()
lr.fit(X_train,y_train)
pred5 = lr.predict(X_test)

accuracy_score(pred5,y_test)


# ### Voting Classifier for Ensemble Modelling
# 

# In[80]:


clf1 = LogisticRegression(multi_class='multinomial', random_state=0)
clf2 = RandomForestClassifier(n_estimators=50, random_state=0)
clf3 = MultinomialNB()

vc = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('mnb', clf3)], voting='hard')
vc.fit(X_train,y_train)
pred6 = vc.predict(X_test)

accuracy_score(pred6,y_test)


# # Q.1:What are the most common emotions/words used by Males and Females?

# In[81]:


from sklearn.feature_extraction.text import CountVectorizer

from yellowbrick.text import FreqDistVisualizer


# In[82]:


male_df = df[df['gender'] == 'male']
female_df = df[df['gender'] == 'female']


# In[83]:


cvm = CountVectorizer()
cvf = CountVectorizer()

Xm = cvm.fit_transform(male_df['tweets'])
Xf = cvf.fit_transform(female_df['tweets'])


# In[84]:


featuresm   = cvm.get_feature_names()

visualizerm = FreqDistVisualizer(features=featuresm, orient='v',size=(1080, 720),n = 100)
visualizerm.fit(Xm)
visualizerm.show()


# In[85]:


featuresf   = cvf.get_feature_names()

visualizerf = FreqDistVisualizer(features=featuresf, orient='v',size=(1080, 720),n = 100)
visualizerf.fit(Xf)
visualizerm.show()


# ### The 5 most common words used by females are 'like', 'get', 'one', 'love', 'day'.
# ### The 5 most common words used by males are 'like', 'get', 'one', 'time', 'go'.

# # Q.2: Which gender prefers prime numbers as their favourite number?

# In[86]:


df['fav_number']


# In[87]:


import sympy as sp


# In[88]:


lst=[]
for i in df['fav_number']:
    if sp.isprime(i)==True:
        lst.append('1')
    else:
        lst.append('0')


# In[89]:


lst


# In[90]:


arr=np.asarray(lst, dtype=np.int64)
arr


# ## Adding a new Column indicating whether fav_number is Prime or not

# #### 1 indicates prime number 

# #### 0 indicates not a prime number

# In[91]:


df['prime_numbers']=arr
df


# In[92]:


df.info()


# ## Considering only Prime numbers 

# In[93]:


df=df[df['prime_numbers']==1]
df


# In[94]:


df['gender'].value_counts()


# #### Since number of males is greater than number of females, males prefer prime number as their favourite number more than females.
