#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('F:\kutty\severeinjury1.csv',encoding='cp1252')


# In[3]:


import nltk


# In[4]:


from nltk.tokenize import WhitespaceTokenizer


# In[5]:


from nltk.stem.snowball import SnowballStemmer
stemming = SnowballStemmer("english")


# In[6]:


data["narrative"]= data["title.new"]+data["summary.new"]


# In[7]:


def identify_tokens(row):
    summary_title = row["narrative"]
    tokens = nltk.word_tokenize(summary_title)
    token_words = [ w for w in tokens if w.isalpha()]
    return token_words


# In[8]:


data['words']= data.apply(identify_tokens,axis=1)


# In[9]:


data['words']


# In[10]:


def stem_list(row):
    my_list = row['words']
    stemmed_list = [stemming.stem(word) for word in my_list]
    return(stemmed_list)

data['stemmed_words'] = data.apply(stem_list, axis=1)


# In[11]:


data['stemmed_words']


# In[12]:


from nltk.corpus import stopwords
stops = set(stopwords.words("english"))                  

def remove_stops(row):
    my_list = row['stemmed_words']
    meaningful_words = [w for w in my_list if not w in stops]
    return (meaningful_words)

data['stem_meaningful'] = data.apply(remove_stops, axis=1)


# In[13]:


data['stem_meaningful']


# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer
import math


# In[15]:


data['stem_meaningful'] = ["  ".join(review) for review in data['stem_meaningful'].values]


# In[16]:


data['stem_meaningful']


# In[17]:


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
v=TfidfVectorizer()


# In[18]:


x=v.fit_transform(data['stem_meaningful'])


# In[19]:


x


# In[20]:


first_vector_v = x[0]


# In[21]:


df = pd.DataFrame(first_vector_v.T.todense(),index = v.get_feature_names(),columns = ["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)


# In[22]:


from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score


# In[23]:


import numpy as np


# In[48]:


Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(data['summary.new'],data['Tagged2'],test_size=0.2)


# In[49]:


Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)


# In[50]:


Train_X_Tfidf = v.transform(Train_X)
Test_X_Tfidf = v.transform(Test_X)


# In[51]:


print((v.vocabulary_))


# In[52]:


print(Train_X_Tfidf)


# In[53]:


SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)


# In[54]:


predictions_SVM = SVM.predict(Test_X_Tfidf)


# In[55]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print((confusion_matrix(Test_Y,predictions_SVM)))
print((classification_report(Test_Y,predictions_SVM)))
print((accuracy_score(Test_Y, predictions_SVM)))


# In[56]:


Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)


# In[57]:


predictions_NB = Naive.predict(Test_X_Tfidf)


# In[58]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print((confusion_matrix(Test_Y,predictions_NB)))
print((classification_report(Test_Y,predictions_NB)))
print((accuracy_score(Test_Y, predictions_NB)))


# In[59]:


from sklearn.tree import DecisionTreeClassifier


# In[60]:


clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(Train_X_Tfidf,Train_Y)

#Predict the response for test dataset
y_pred = clf.predict((Test_X_Tfidf))


# In[61]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print((confusion_matrix(Test_Y,y_pred)))
print((classification_report(Test_Y,y_pred)))
print((accuracy_score(Test_Y, y_pred)))


# In[62]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=15)


# In[63]:


classifier.fit(Train_X_Tfidf,Train_Y)


# In[64]:


y_pred = classifier.predict((Test_X_Tfidf))


# In[65]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print((confusion_matrix(Test_Y,y_pred)))
print((classification_report(Test_Y,y_pred)))
print((accuracy_score(Test_Y, y_pred)))


# In[66]:


from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
import pickle


# In[69]:


from sklearn.linear_model import LogisticRegression
X= data['summary.new']
Y = data['Tagged2']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2)


# In[70]:


pipeline = Pipeline([('vect', v),
                     ('chi',  SelectKBest(chi2, k=1200)),
                     ('clf', LogisticRegression(random_state=0))])



# In[71]:


model = pipeline.fit(X_train, y_train)
with open('LogisticRegression.pickle', 'wb') as f:
    pickle.dump(model, f)


# In[72]:


ytest = np.array(y_test)


# In[73]:


print((classification_report(ytest, model.predict(X_test))))
print((confusion_matrix(ytest, model.predict(X_test))))



# In[ ]:




