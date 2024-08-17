#!/usr/bin/env python
# coding: utf-8

# This notebook is written to extracted several features from the text and then the extracted features are used to classify them into the given classes. The implementation is performed using the spam dataset. The participants have to use different datasets given in the Dataset folder and perform the classification. 

# You have to specify your own path where the code and datasets are present

# In[ ]:


import pandas as pd 
data =  pd.read_csv('SMSSpamCollection.csv', sep = '\t', names = ['label', 'message'])
data.head()


# In[ ]:


text = data['message']
class_label = data['label']


# In[ ]:


import numpy as np
classes_list = ["ham","spam"]
label_index = class_label.apply(classes_list.index)
label = np.asarray(label_index)


# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=0.33, random_state=42)


# In[ ]:


#!wget http://nlp.stanford.edu/data/glove.840B.300d.zip
get_ipython().system('wget http://nlp.stanford.edu/data/glove.6B.zip')


# In[ ]:


get_ipython().system('unzip glove*.zip')


# In[ ]:


from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


# In[ ]:


glove_file = datapath('/content/glove.6B.300d.txt')
word2vec_glove_file = get_tmpfile("glove.6B.300d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)


# In[ ]:


model = KeyedVectors.load_word2vec_format(word2vec_glove_file)


# In[ ]:


def get_w2v_general(tweet, size, vectors, aggregation='mean'):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tweet.split():
        try:
            vec += vectors[word].reshape((1, size)) #* tfidf[word]
            count += 1.
        except KeyError:
            continue
    if aggregation == 'mean':
        if count != 0:
            vec /= count
        return vec
    elif aggregation == 'sum':
        return vec


# In[ ]:


from sklearn.preprocessing import scale

train_vecs_glove_mean = scale(np.concatenate([get_w2v_general(z, 300, model,'mean') for z in X_train]))
test_vecs_glove_mean = scale(np.concatenate([get_w2v_general(z, 300, model,'mean') for z in X_test]))


# In[ ]:


train_vecs_glove_mean


# In[ ]:


from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression




names = ["Logistic Regression", "KNN","Linear SVC","DC","SGD","RF","LinearSVC with L1-based feature selection", 
         "Bernoulli NB", "Ridge Classifier", "AdaBoost", "Perceptron","Passive-Aggresive", "Nearest C4entroid"]
classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(),
    LinearSVC(),
    DecisionTreeClassifier(),
    SGDClassifier(),
    RandomForestClassifier(),
    Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
  ('classification', LinearSVC(penalty="l2"))]),
    BernoulliNB(),
    RidgeClassifier(),
    AdaBoostClassifier(),
    Perceptron(),
    PassiveAggressiveClassifier(),
    NearestCentroid()
    ]
zipped_clf = list(zip(names,classifiers))


# In[ ]:


from time import time
from sklearn import metrics


def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
    t0 = time()
    sentiment_fit = pipeline.fit(x_train, y_train)
    y_pred = sentiment_fit.predict(x_test)
    train_test_time = time() - t0
    #accuracy = accuracy_score(y_test, y_pred)
    accuracy= (metrics.accuracy_score(y_test, y_pred))
    #print ("accuracy score: {0:.2f}%".format(accuracy*100))
    print(("train and test time: {0:.2f}s".format(train_test_time)))
    print(("-"*80))
    
  
    p= metrics.precision_score(y_test, y_pred,average= 'weighted')
    r = metrics.recall_score(y_test, y_pred,average= 'weighted')
    f=metrics.f1_score(y_test, y_pred, average= 'weighted')
    
    
    return accuracy, p,r,f,train_test_time


# In[ ]:


def classifier_comparator():
    result = []
    classifier=zipped_clf
    
    for n,c in classifier:
        checker_pipeline = Pipeline([
                     ('classifier', c)
        ])
        print(("Validation result for {}".format(n)))
        print (c)
        #clf_accuracy,p,r,f,tt_time = accuracy_summary(checker_pipeline, train_vecs_dbow, y_train, validation_vecs_dbow, y_valid)
        #clf_accuracy,p,r,f,tt_time = accuracy_summary(checker_pipeline, train_vecs_dmc, y_train, validation_vecs_dmc, y_valid)
        clf_accuracy,p,r,f,tt_time = accuracy_summary(checker_pipeline, train_vecs_glove_mean, y_train,test_vecs_glove_mean,y_test)

        result.append((n,clf_accuracy,p,r,f,tt_time))
    return result

trigram_result = classifier_comparator()
from pandas import ExcelWriter
from pandas import ExcelFile

df = pd.DataFrame(trigram_result)
df.columns=['classifier','acc','p','r','f1','time']

 
writer = ExcelWriter('FB_hindi_test_fasttext_sum.xlsx',engine='openpyxl')
df.to_excel(writer,'Sheet1',index=False)
writer.save()


# In[ ]:


df.head()


# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn import metrics

from sklearn.svm import SVC
model_SVM = SVC()
model_SVM.fit(x_train, y_train)
y_pred_SVM = model_SVM.predict(x_test)
print("SVM")
print(("Accuracy score =", accuracy_score(y_test, y_pred_SVM)))
print((metrics.classification_report(y_test, y_pred_SVM)))

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_split=2, random_state=0)
rf.fit(x_train,y_train)
y_pred_rf = rf.predict(x_test)
print("random")
print(("Accuracy score =", accuracy_score(y_test, y_pred_rf)))
print((metrics.classification_report(y_test, y_pred_rf)))

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(x_train,y_train)
y_pred_LR = LR.predict(x_test)
print("Logistic Regression")
print(("Accuracy score =", accuracy_score(y_test, y_pred_LR)))
print((metrics.classification_report(y_test, y_pred_LR )))

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors = 5)
neigh.fit(x_train,y_train)
y_pred_KNN = neigh.predict(x_test)
print("KNN")
print(("Accuracy score =", accuracy_score(y_test, y_pred_KNN)))
print((metrics.classification_report(y_test, y_pred_KNN )))

from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()
naive.fit(x_train.toarray(),y_train)
y_pred_naive = naive.predict(x_test.toarray())
print("Naive Bayes")
print(("Accuracy score =", accuracy_score(y_test, y_pred_naive)))
print((metrics.classification_report(y_test, y_pred_naive )))

from sklearn.ensemble import GradientBoostingClassifier
gradient = GradientBoostingClassifier(n_estimators=100,max_depth=None,min_samples_split=2, random_state=0)
gradient.fit(x_train,y_train)
y_pred_gradient = gradient.predict(x_test)
print("Gradient Boosting")
print(("Accuracy score =", accuracy_score(y_test, y_pred_gradient)))
print((metrics.classification_report(y_test, y_pred_gradient )))

    
from sklearn.tree import DecisionTreeClassifier
decision = DecisionTreeClassifier()
decision.fit(x_train,y_train)
y_pred_decision = decision.predict(x_test)
print("Decision Tree")
print(("Accuracy score =", accuracy_score(y_test, y_pred_decision)))
print((metrics.classification_report(y_test, y_pred_decision )))
    


# In[ ]:




