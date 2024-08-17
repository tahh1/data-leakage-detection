#!/usr/bin/env python
# coding: utf-8

# ### We need to up sampling the data taken from kaggale source  https://www.kaggle.com/anish9167473766/churndata to create synthetic samples upto ~1 million. Here we used Weka tool for oversampling with class balanced synthesized data. Weka is an open source  software collection of machine learning algorithms for data mining tasks. 

# In[22]:


from collections import Counter
from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE

from sklearn.datasets import make_hastie_10_2

from xgboost import XGBClassifier
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import itertools


# In[23]:


def print_results(headline, true_value, pred):
    print(headline)
    print(("accuracy: {}".format(accuracy_score(true_value, pred))))
    print(("precision: {}".format(precision_score(true_value, pred))))
    print(("recall: {}".format(recall_score(true_value, pred))))
    print(("f1: {}".format(f1_score(true_value, pred))))


# In[24]:


#from google.colab import drive 
#from google.colab import files
#import io

#uploaded = files.upload()

#data = io.BytesIO(uploaded['telecom_churn.csv']) 
#drive.mount('/content/gdrive')
#df=pd.read_csv('gdrive/my-drive/telecom_churn.csv.csv')
#churn_data = pd.read_csv(data)

churn_data = pd.read_csv("telecom_churn_resample_2.csv")
#churn_data = pd.read_csv("telecom_churn_org.csv")
#churn_data


# In[25]:


churn_data.head()


# In[13]:


#Dropping the 'categorical' columns as of now for "Oversampleing" Excercise
churn_data.drop(['state', '\'phone number\'', '\'international plan\'', '\'voice mail plan\'' ], axis=1, inplace=True)
#churn_data.drop(['state', 'phone number', 'international plan', 'voice mail plan' ], axis=1, inplace=True)
churn_data['churn'] = churn_data['churn'].apply(lambda x: 1 if x == False else -1)
#spilt the target varibale and Predictors
X = churn_data.drop("churn", axis = 1)
y = churn_data.churn

#X = churn_data[:20000].drop("churn", axis = 1)
#y = churn_data[:20000].churn
#print(churn_data['churn'])

churn_count = churn_data.churn.value_counts()
print(("churn Class: ", len(churn_count)))

#print('Class 0:', churn_count[0])
#print('Class 1:', churn_count[1])
#print('\n')

#print('Proportion:', round(churn_count[0] / churn_count[1], 2), ': 1')
#churn_data
#print(y)

#churn_data['churn'].head(1000)


# In[14]:


churn_data.head()


# In[9]:


# Percentage of missing values in each column
missingDataCount = round((churn_data.isnull().sum()/len(churn_data))*100,2)
print(("Percentage of missing data \n", missingDataCount))


# In[10]:


# look at data statistics
churn_data.describe(include='all')


# In[11]:


churn_data.dropna(how='any', inplace = True)
#churn_data


# In[12]:


#plotting the correlation matrix, removing the multi colnearity
import matplotlib.pyplot as plt 
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize = (20,10))
sns.heatmap(churn_data.corr(),annot = True)


# In[13]:


#Normal Distribution
print(("NORMAL data distribution: {}".format(Counter(churn_data['churn']))))
churn_count.plot(kind='bar', title='Count (churn)');



# In[14]:


#Evaluating with RandomForestClassifier
print(("Normal disrtibution: {}" .format("subrata")))
#Normal Distribution
print(("NORMAL data distribution: {}".format(Counter(churn_data['churn']))))

#Split the  data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

#using RandomForestClassifier to predict the accuracy BEFORE up-sampling
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(("Accuracy: %.2f%%" % (accuracy * 100.0)))

#Summary of the accuracy
print_results("\n Data distribution:",y_test, y_pred )



# In[15]:


# base estimator: a weak learner with max_depth=2
shallow_tree = DecisionTreeClassifier(max_depth=3, random_state = 100)


# In[16]:


# fit the shallow decision tree 
shallow_tree.fit(X_train, y_train)

# test error
y_pred = shallow_tree.predict(X_test)
score = accuracy_score(y_test, y_pred)
score


# In[17]:


# adaboost with the tree as base estimator

estimators = list(range(1, 50, 10))

abc_scores = []
for n_est in estimators:
    ABC = AdaBoostClassifier( base_estimator=shallow_tree,  n_estimators = n_est)
    ABC.fit(X_train, y_train)
    y_pred = ABC.predict(X_test)
   # score = accuracy_score(y_test, y_pred)
   # score = print_results("\nData distribution:",y_test, y_pred )
    #print("\n")
    results = confusion_matrix(y_test, y_pred)
    #print('Confusion Matrix :')
    #print(results)
    #print("*****************")
    #abc_scores.append(score)
    


# In[18]:


# plot test scores and n_estimators
# plot
#plt.plot(estimators, abc_scores)
#plt.xlabel('n_estimators')
#plt.ylabel('accuracy')
#plt.ylim([0.85, 1])
#plt.show()


# In[19]:


X_test.isnull().any().any()


# In[20]:


""" HELPER FUNCTION: GET ERROR RATE ========================================="""
def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))

""" HELPER FUNCTION: PRINT ERROR RATE ======================================="""
def print_error_rate(err_train, err_test):
    #print ('Error rate: Training: %.4f - Test: %.4f False Negative: %.4f' % (err_train, err_test, err_miss_fn)) 
    print(('Error rate: Training: %.4f - Test: %.4f' % (err_train, err_test)))

def  get_miss_fn(pred, Y) :
    s = 0
    for (x,y) in zip(pred, Y): 
        if (x == -1 and y == 1) :
            s += 1 
    return s


# In[21]:


""" PLOT FUNCTION ==========================================================="""
def plot_error_rate(er_train, er_test):
    df_error = pd.DataFrame([er_train, er_test]).T
    df_error.columns = ['Training', 'Test']
    plot1 = df_error.plot(linewidth = 3, figsize = (8,6),
            color = ['lightblue', 'darkblue'], grid = True)
    plot1.set_xlabel('Number of iterations', fontsize = 12)
    plot1.set_xticklabels(list(range(0,450,50)))
    plot1.set_ylabel('Error rate', fontsize = 12)
    plot1.set_title('Error rate vs number of iterations', fontsize = 16)
    #plt.axhline(y=er_test[0], linewidth=1, color = 'red', ls = 'dashed')


# In[22]:


#Split the  data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)


# In[23]:


def generic_clf(Y_train, X_train, Y_test, X_test, clf):
    clf.fit(X_train,Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)
           


# In[24]:


import math

def adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    miss_fn = []
    #print("w={} ".format(w))
    for i in range(M):
        # Fit a classifier with the specific weights
        clf.fit(X_train, Y_train, sample_weight = w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)
        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        #miss_fn = [int(x) for x in (pred_train_i == -1 and Y_train == 1)]
        miss_fn.clear()
        
       # for (x,y) in itertools.izip_longest(pred_train_i, Y_train): 
        for (x,y) in zip(pred_train_i, Y_train): 
            if (x == -1 and y == 1) :
                #print(x,y)
                miss_fn.append(1) 
            else :
                miss_fn.append(0)         
          
        # Equivalent with 1/-1 to update weights
        miss2 = [x if x==1 else -1 for x in miss]
        
        # Error
        err_m = np.dot(w,miss) / sum(w)
        
        # Alpha
        alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))       
        alpha =  [ float(x) * alpha_m for x in miss2]
        beta  =   [ 0.2 * float(x) * alpha_m for x in miss_fn]
        alpha_beta = np.add(alpha,beta)
        exp_alpha_beta = np.exp(alpha_beta)
        w = np.multiply(w, exp_alpha_beta)
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train, 
                                          [ x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test, 
                                         [ x * alpha_m for x in pred_test_i])]
        
    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    
    #print("sum(pred_train != Y)= {}".format(sum(pred_train != Y_train)))
    #print("Error: False Negative:{}".format(get_miss_fn(pred_train,Y_train )))
    
    # Return error rate in train and test set
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)
           


# In[25]:


#Check the misclassification error ( test/train ) over each boosting rounds, it can be noticed
# the errors get reduced sharply over each boosting rounds

# Fit a simple decision tree first
clf_tree = DecisionTreeClassifier(max_depth = 3, random_state = 1)
er_tree = generic_clf(y_train, X_train, y_test, X_test, clf_tree)

# Fit Adaboost classifier using a decision tree as base estimator
# Test with different number of iterations
er_train, er_test = [er_tree[0]], [er_tree[1]]

for i in [10, 20, 40]:    
    er_i = adaboost_clf(y_train, X_train, y_test, X_test, i, clf_tree)
    er_train.append(er_i[0])
    er_test.append(er_i[1])
    #print_error_rate(er_i[0], er_i[1])  
    
print("Using proposed cost sensitive Adaboost :")
# Compare error rate vs number of iterations
plot_error_rate(er_train, er_test)

