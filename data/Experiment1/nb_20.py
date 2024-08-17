#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#We would work on a small dataset of 5000 or so emails with spam and not spam(ham)
#dataset can be found on kaggle here : https://www.kaggle.com/balakishan77/spam-or-ham-email-classification
#Although there are very big datasets too in kaggle with 500000 emails, 
#but for ease of training and understanind, we are using 5600


# In[ ]:


#Upload our data set and keep in same folder as this notebook
import numpy as np 
import pandas as pd


import nltk
from nltk.corpus import stopwords


# In[ ]:


## Data Loading and checking for nulls and other


# In[ ]:


data = pd.read_csv("spamham.csv")
data.head()


# In[ ]:


print((data.columns))
data.shape


# In[ ]:


data.drop_duplicates(inplace = True)
data.isnull().sum()


# In[ ]:


data.shape


# In[ ]:


#Tokenizing our dataset and creating a function for that

from nltk.tokenize import RegexpTokenizer

def clean_text(text):
    tokenizer = RegexpTokenizer(r'\w+')

    clean_text = tokenizer.tokenize(text)
    return " ".join(clean_text).lower()           #We aqre using join to ceate clean_text into sentence rather than list

#unlike previously, we will download only stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

def clean_with_stopwords(clean_text):

    sr = stopwords.words('english')
    len(sr)

    summary_words = []
    for word in clean_text.split():

        if word.lower() not in sr:
            summary_words.append(word)

    return summary_words


# In[ ]:


data['text'].head().apply(clean_text).apply(clean_with_stopwords)


# In[ ]:


def process_text(text):

    cleaned_text             = clean_text(text)
    cleaned_without_stopword = clean_with_stopwords(cleaned_text) 

    return cleaned_without_stopword


# In[ ]:


#vectorizing our data
from sklearn.feature_extraction.text import CountVectorizer

messages_in_vector = CountVectorizer(analyzer=process_text).fit_transform(data['text'])


# In[ ]:


messages_in_vector.shape


# In[ ]:


#Train test split

from sklearn.model_selection import train_test_split
X = messages_in_vector
y = data["spam"]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.20, random_state = 7)


# In[ ]:


for i in (X_train, X_test, y_train, y_test):
    print((i.shape))


# In[ ]:


#Creating our model
from sklearn.naive_bayes import MultinomialNB

spam_filter = MultinomialNB()
spam_filter.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import confusion_matrix

predictions = spam_filter.predict(X_test)
actual      = y_test

c_matrix = confusion_matrix(actual, predictions)


# In[ ]:


recall_without_avg = 0

for i in range(2):
    
    recall_sum = 0

    c_1 = c_matrix[i][i]
    
    for j in range(2):
    
        recall_sum += c_matrix[i][j]
    
    recall_without_avg += c_1/recall_sum       

recall = recall_without_avg/2

print(("Recall is", recall))


# In[ ]:


positive = 0

for i in range(2):
    
    for j in range(2):
    
        if i==j:
        
            positive += c_matrix[i][j]
         
        
Accuracy = positive /  c_matrix.sum()

print(("Accuracy is", Accuracy))


# In[ ]:


from sklearn.metrics import classification_report
print((classification_report(actual, predictions)))

