#!/usr/bin/env python
# coding: utf-8

# In[1]:


#We would work on a small dataset of 5000 or so emails with spam and not spam(ham)
#dataset can be found on kaggle here : https://www.kaggle.com/balakishan77/spam-or-ham-email-classification
#Although there are very big datasets too in kaggle with 500000 emails, 
#but for ease of training and understanind, we are using 5600


# ## Loading libraries

# In[1]:


import numpy as np 
import pandas as pd

import nltk
from nltk.corpus import stopwords


# ## Data Loading and checking for nulls and other

# In[2]:


data = pd.read_csv("spamham.csv")
data.head()


# In[3]:


print((data.columns))
data.shape


# In[4]:


data.drop_duplicates(inplace = True)
data.isnull().sum()


# In[5]:


data.shape


# ## Tokenizing our dataset and creating a function for that

# In[6]:


from nltk.tokenize import RegexpTokenizer

def clean_text(text):
    tokenizer = RegexpTokenizer(r'\w+')

    cleaned_text = tokenizer.tokenize(text)
    return " ".join(cleaned_text).lower()           #We are using join to ceate clean_text into sentence rather than list

#unlike previously, we will download onl;y stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

def clean_with_stopwords(clean_text):

    sr = stopwords.words('english')

    summary_words = []
    for word in clean_text.split():

        if word.lower() not in sr:
            summary_words.append(word.lower())

    return summary_words


# In[7]:


def process_text(text):

    cleaned_text             = clean_text(text)
    cleaned_without_stopword = clean_with_stopwords(cleaned_text) 

    return cleaned_without_stopword


# In[9]:


data['text'].head().apply(process_text)


# ## Vectorizing our data

# In[10]:


from sklearn.feature_extraction.text import CountVectorizer

messages_in_vector = CountVectorizer(analyzer=process_text).fit_transform(data['text'])


# In[11]:


messages_in_vector.shape


# ## Train-Test Split

# In[12]:


from sklearn.model_selection import train_test_split
X = messages_in_vector
y = data["spam"]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.20, random_state = 7)


# In[13]:


for i in (X_train, X_test, y_train, y_test):
    print((i.shape))


# ## Creating our data

# In[14]:


from sklearn.naive_bayes import MultinomialNB

spam_filter = MultinomialNB()
spam_filter.fit(X_train, y_train)


# In[15]:


from sklearn.metrics import confusion_matrix, classification_report

predictions = spam_filter.predict(X_test)
actual      = y_test

confusion_matrix(actual, predictions)


# In[16]:


print(( classification_report(actual, predictions) ))


# In[ ]:




