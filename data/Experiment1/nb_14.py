#!/usr/bin/env python
# coding: utf-8

# ## Practical 6 - Implementation text classification using Naïve Bayes, SVM.

# In[51]:


import pandas as pd 
data =  pd.read_csv('SMSSpamCollection.csv', sep = '\t', names = ['label', 'message'])
data.head()


# In[52]:


text = data['message']
label = data['label']


# In[53]:


#Number of Words
#x = lambda a : a + 10
#print(x(5))
data['word_count'] = data['message'].apply(lambda x: len(str(x).split(" ")))
data[['message','word_count']].head()


# In[54]:


#Number of characters
data['char_count'] = data['message'].str.len() ## this also includes spaces
data[['message','char_count']].head()


# In[55]:


#Average Word Length
def avg_word(sentence):
  words = sentence.split()
  #print(words)
  return (sum(len(word) for word in words)/len(words))

data['avg_word'] = data['message'].apply(lambda x: avg_word(x))
data[['message','avg_word']].head()


# In[56]:


#Number of stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

data['stopwords'] = data['message'].apply(lambda x: len([x for x in x.split() if x in stop]))
data[['message','stopwords']].head()


# In[57]:


#Number of special characters
data['hastags'] = data['message'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
data[['message','hastags']].head()


# In[58]:


#Number of numerics
data['numerics'] = data['message'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
data[['message','numerics']].head()


# In[59]:


#Number of Uppercase words
data['upper'] = data['message'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
data[['message','upper']].head()


# In[60]:


pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
from textblob import TextBlob, Word, Blobber
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

data['noun_count'] = data['message'].apply(lambda x: check_pos_tag(x, 'noun'))
data['verb_count'] = data['message'].apply(lambda x: check_pos_tag(x, 'verb'))
data['adj_count'] = data['message'].apply(lambda x: check_pos_tag(x, 'adj'))
data['adv_count'] = data['message'].apply(lambda x: check_pos_tag(x, 'adv'))
data['pron_count'] = data['message'].apply(lambda x: check_pos_tag(x, 'pron'))
data[['message','noun_count','verb_count','adj_count', 'adv_count', 'pron_count' ]].head()


# In[61]:


data[['message','word_count','char_count','avg_word','stopwords','hastags','numerics','upper','noun_count','verb_count','adj_count', 'adv_count', 'pron_count','label' ]].head()


# In[62]:


features = data[['word_count','char_count','avg_word','stopwords','hastags','numerics','upper','noun_count','verb_count','adj_count', 'adv_count', 'pron_count']]


# In[63]:


#label = data['label']

import numpy as np
classes_list = ["ham","spam"]
label_index = data['label'].apply(classes_list.index)
label = np.asarray(label_index)


# In[64]:


import numpy as np
features_array = np.asarray(features)


# In[65]:


features_array.shape


# In[66]:


# data split into train and text
import numpy as np
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features_array, label, test_size=0.33, random_state=42)


# In[67]:


from sklearn.metrics import accuracy_score
from sklearn import metrics

from sklearn.svm import SVC
model_SVM = SVC()
model_SVM.fit(x_train, y_train)
y_pred_SVM = model_SVM.predict(x_test)
print("SVM")
print(("Accuracy score =", accuracy_score(y_test, y_pred_SVM)))
print((metrics.classification_report(y_test, y_pred_SVM)))



from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()
naive.fit(x_train,y_train)
y_pred_naive = naive.predict(x_test)
print("Naive Bayes")
print(("Accuracy score =", accuracy_score(y_test, y_pred_naive)))
print((metrics.classification_report(y_test, y_pred_naive )))


# In[68]:


# data split into train and text
import numpy as np
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features_array, label, test_size=0.33, random_state=42)


# In[69]:


x_train.shape


# In[70]:


data = pd.read_csv('SMSSpamCollection.csv', sep = '\t', names = ['label','message'])


# In[71]:


text = data['message']
class_label = data['label']


# In[72]:


import numpy as np
classes_list = ["ham","spam"]
label_index = class_label.apply(classes_list.index)
label = np.asarray(label_index)


# In[73]:


import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=0.33, random_state=42)


# In[74]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range = (1,1))
x_train = vectorizer.fit_transform(X_train)
x_test = vectorizer.transform(X_test)


# In[75]:


x_train.shape


# In[76]:


vectorizer.get_feature_names()


# In[77]:


from sklearn.metrics import accuracy_score
from sklearn import metrics

from sklearn.svm import SVC
model_SVM = SVC()
model_SVM.fit(x_train, y_train)
y_pred_SVM = model_SVM.predict(x_test)
print("SVM")
print(("Accuracy score =", accuracy_score(y_test, y_pred_SVM)))
print((metrics.classification_report(y_test, y_pred_SVM)))


from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()
naive.fit(x_train.toarray(),y_train)
y_pred_naive = naive.predict(x_test.toarray())
print("Naive Bayes")
print(("Accuracy score =", accuracy_score(y_test, y_pred_naive)))
print((metrics.classification_report(y_test, y_pred_naive )))

