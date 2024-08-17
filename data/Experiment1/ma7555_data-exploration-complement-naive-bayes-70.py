#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print((os.listdir("../input")))

# Any results you write to the current directory are saved as output.


# In[2]:


data = pd.read_json('../input/News_Category_Dataset_v2.json', lines=True)
data.head()


# In[3]:


# 2. Remove punctiation & convert to lowercase:
data['headline_processed'] = data.headline.str.replace('[{}]'.format(string.punctuation), '').str.lower()
data['short_description_processed'] = data.short_description.str.replace('[{}]'.format(string.punctuation), '').str.lower()


# In[4]:


data.category.unique()


# As you might have realized, some categories are pretty much the same:
# * 'WORLDPOST' || 'THE WORLDPOST'
# * 'PARENTS' || 'PARENTING'
# * 'ARTS' || 'ARTS & CULTURE' || 'CULTURE & ARTS'
# * 'STYLE & BEAUTY' || 'STYLE'
# * 'EDUCATION' || 'COLLEGE'
# * 'TASTE' || 'FOOD & DRINK'

# In[5]:


def category_cleaner(x):
    
    if x == 'THE WORLDPOST':
        return 'WORLDPOST'
    elif x == 'PARENTING':
        return 'PARENTS'
    elif x == 'ARTS' or x == 'CULTURE & ARTS':
        return 'ARTS & CULTURE'
    elif x == 'STYLE':
        return 'STYLE & BEAUTY'
    elif x == 'COLLEGE':
        return 'EDUCATION'
    elif x == 'TASTE':
        return 'FOOD & DRINK'
    else:
        return x
    
data['category'] = data.category.apply(category_cleaner)


le = LabelEncoder()
data_labels = le.fit_transform(data.category)
list(le.classes_)


# In[6]:


data.authors.unique()


# In[7]:


data['authors'] = data.authors.apply(lambda x: x.split(',')[0])
data['authors'] = data.authors.str.replace(' ', '', regex=False)
data.authors.unique()


# In[8]:


plt.figure(figsize=(15,15))
sizes = data.category.value_counts().values
labels = data.category.value_counts().index
plt.pie(sizes, labels=labels, autopct='%.1f%%',
        shadow=True, pctdistance=0.85, labeldistance=1.05, startangle=20, 
        explode = [0 if i > 0 else 0.2 for i in range(len(sizes))])
plt.axis('equal')
plt.show()


# In[9]:


# Splitting into train sets and test sets
x_train, x_test, y_train, y_test = train_test_split(data, data_labels, 
                                                    train_size=0.8, test_size=0.2,
                                                    random_state=555)


# In[10]:


# 3. Text Counters & Classifiers..

counter = CountVectorizer(stop_words='english')

train_headline_counts = counter.fit_transform(x_train.headline_processed)
test_headline_counts = counter.transform(x_test.headline_processed)

train_short_description_counts = counter.fit_transform(x_train.short_description_processed)
test_short_description_counts = counter.transform(x_test.short_description_processed)


# In[11]:


# 4. Classify with Naive Bayes..
classifier = ComplementNB()

classifier.fit(train_headline_counts, y_train)
headline_soft_predictions = classifier.predict_proba(test_headline_counts)
headline_predictions = classifier.predict(test_headline_counts)

classifier.fit(train_short_description_counts, y_train)
short_description_soft_predictions = classifier.predict_proba(test_short_description_counts)
short_description_predictions = classifier.predict(test_short_description_counts)


# In[12]:


soft_predictions = (headline_soft_predictions + short_description_soft_predictions) / 2
predictions = np.argmax(soft_predictions, axis = 1)
print(('Accuracy without tokenization: {}'.format(accuracy_score(predictions, y_test))))


# A question that came into my mind, would there be any difference in accuracy if we merged both headlines and short description into one attribute?
# I will try it out and see.

# In[13]:


# Combining both text columns into one
train_counts = counter.fit_transform(x_train.headline_processed.str.cat(x_train.short_description_processed))
test_counts = counter.transform(x_test.headline_processed.str.cat(x_test.short_description_processed))
classifier.fit(train_counts, y_train)
predictions_combined =  classifier.predict(test_counts)
text_combined_soft_predictions = classifier.predict_proba(test_counts)
print(('Accuracy with combined text and without tokenization: {}'.format(accuracy_score(predictions, y_test))))


# Exactly the same..

# In[14]:


# stemming...
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

porter_stemmer = PorterStemmer()
stopwords = set(stopwords.words('english'))

def stemming_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [porter_stemmer.stem(word) if word not in stopwords else word for word in words]
    return words


def count_classify_with_tokenizer(tokenizer):
    counter_words = CountVectorizer(stop_words=stopwords, tokenizer=tokenizer)
    
    train_headline_counts = counter_words.fit_transform(x_train.headline_processed)
    test_headline_counts = counter_words.transform(x_test.headline_processed)

    train_short_description_counts = counter_words.fit_transform(x_train.short_description_processed)
    test_short_description_counts = counter_words.transform(x_test.short_description_processed)

    classifier.fit(train_headline_counts, y_train) # Train on headlines
    headline_soft_predictions = classifier.predict_proba(test_headline_counts)

    classifier.fit(train_short_description_counts, y_train) # Train on description
    short_description_soft_predictions = classifier.predict_proba(test_short_description_counts)

    soft_predictions = (headline_soft_predictions + short_description_soft_predictions) / 2
    predictions = np.argmax(soft_predictions, axis = 1)
    return accuracy_score(predictions, y_test)

print(('Accuracy with Stemmer: {}'.format(count_classify_with_tokenizer(stemming_tokenizer))))


# In[15]:


# lemmatizing...
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def lemmatizer_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [wordnet_lemmatizer.lemmatize(word) if word not in stopwords else word for word in words]
    return words

print(('Accuracy with Lemmatizer: {}'.format(count_classify_with_tokenizer(lemmatizer_tokenizer))))


# Remember that we still have a valuable piece of information, the author of the article which should be included in our decision as well.

# In[16]:


train_authors_counts = counter.fit_transform(x_train.authors)
test_authors_counts = counter.transform(x_test.authors)

classifier.fit(train_authors_counts, y_train)
authors_prediction = classifier.predict(test_authors_counts)
authors_soft_prediction = classifier.predict_proba(test_authors_counts)
accuracy_score(authors_prediction, y_test)


# In[17]:


# Add both probabilities together..
final_prediction = np.argmax((text_combined_soft_predictions + authors_soft_prediction) / 2, axis=1)
accuracy_score(final_prediction, y_test)


# That is it for now, best accuracy I could achieve using Naive Bayes is around 70%
