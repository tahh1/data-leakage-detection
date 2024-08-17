#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

msg=pd.read_csv('data6.csv',names=['message','label'])#tabular form data
print(('Total instances in the dataset:',msg.shape[0]))
msg['labelnum']=msg.label.map({'pos':1,'neg':0})
X=msg.message
Y=msg.labelnum    
    

xtrain,xtest,ytrain,ytest=train_test_split(X,Y)
count_vect=CountVectorizer()
xtrain_dtm=count_vect.fit_transform(xtrain)#Sparse matrix
xtest_dtm=count_vect.transform(xtest)
print(('\n Total features extracted using CountVectorizor:',xtrain_dtm.shape[1]))
print('\n Features for first 5 training instances are listed below')
df=pd.DataFrame(xtrain_dtm.toarray(),columns=count_vect.get_feature_names())

print((df[0:5]))


from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB().fit(xtrain_dtm,ytrain)
predicted=clf.predict(xtest_dtm)
print('\n Classification results of testing samples are given below')
for doc,p in zip(xtest,predicted):
    pred='pos' if p==1 else 'neg'
    print((doc,"=",pred))

    
from sklearn import metrics
print('\n Accuracy metrics')
print(('Accuracy of the classifier is',metrics.accuracy_score(ytest,predicted)))
print(('Recall:',metrics.recall_score(ytest,predicted)))
print(('Precision:',metrics.precision_score(ytest,predicted)))
print('Confusion matrix')
print((metrics.confusion_matrix(ytest,predicted)))


# In[ ]:





# In[ ]:




