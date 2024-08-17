#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
iris_dataset=load_iris()
print(("\n IRIS FEATURES \ TARGET NAMES:\n",iris_dataset.target_names))
for i in range(len(iris_dataset.target_names)):
    print(("\n [{0}]:[{1}]".format(i,iris_dataset.target_names[i])))
print(("\n IRIS DATA:\n",iris_dataset["data"]))
x_train,x_test,y_train,y_test=train_test_split(iris_dataset["data"],iris_dataset["target"],random_state=0)
classifier=KNeighborsClassifier(n_neighbors=8,p=3,metric='euclidean')
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
m=cm(y_test,y_pred)
print(('Confusion matrix is as follows\n',cm(y_test,y_pred)))
print('Accuracy metrics')
print((classification_report(y_test,y_pred)))
print(("correct prediction",accuracy_score(y_test,y_pred)))
print(("Wrong prediction",(1-accuracy_score(y_test,y_pred))))


# In[ ]:




