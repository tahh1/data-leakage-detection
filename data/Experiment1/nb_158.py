#!/usr/bin/env python
# coding: utf-8

# In[223]:


import numpy as np
from collections import Counter
import json
import math
import os.path

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn import preprocessing

import datetime
import time
import nltk


# In[3]:


def getExistence(source_tweets, filePath):
#@@ Param:  Source_tweets: a list of source tweets ID
#           filePath: source tweets  parent document directory
#@@ Return: an n x 2 array, n is the amount/number of source tweets' IDs    
    
    existence = np.zeros((len(source_tweets),2))
    for i in range(len(source_tweets)):
        data = json.load(open(filePath+'/'+source_tweets[i]+'/source-tweet/'+source_tweets[i]+'.json','r'))
        if len(data['entities']['hashtags']) != 0:
            existence[i][0] = 1
        else:
            existence[i][0] = 0
        
        if len(data['entities']['urls']) != 0:
            existence[i][1] = 1
        else:
            existence[i][1] = 0
            
    return existence


# In[4]:


# def getText(source_tweets, filePath):
# #@@ Param:  Source_tweets: a list of source tweets ID
# #           filePath: source tweets  parent document directory
# #@@ Return: tweets text coresspoding to source tweets' IDs   

#     tweet_text = []
#     for i in range(len(source_tweets)):
#         data = json.load(open(filePath+'/'+source_tweets[i]+'/source-tweet/'+source_tweets[i]+'.json','r'))
#         tweet_text += [[data['text']]]
    
#     return tweet_text


# In[5]:


def getUserInfo(source_tweets,featureList,filePath):
    
    userFeatures = []
    for i in range(len(source_tweets)):
        data = json.load(open(filePath+'/'+source_tweets[i]+'/source-tweet/'+source_tweets[i]+'.json','r'))
        userData = [data['user'][feature] for feature in featureList]
        userData += [data['created_at']]
        userFeatures += [userData]
    
    return userFeatures


# In[6]:


# extract features, Percentage of replying tweets classified as queries, denies or supports
def getPercentage(source_tweets, tweets_stances, filePath):
#@@ Param: source_tweets: a list of source tweets ID
#          tweets_stances：a list of stances, which contains all tweets' stance including reply tweets and source tweets
#          filePath: source tweets  parent document directory
#@@ Return: an n x 3 array, n is the amount/number of source tweets' IDs
   
    stance_list = []
    for tweet_ID in source_tweets:
        replies_json_list = os.listdir(filePath+'/'+tweet_ID+'/replies') # make a list of reply tweets' IDs .json
        replies_list = [dot_json.split('.')[0] for dot_json in replies_json_list] # remove filename suffixes '.json'
        tmp = []
        for reply_ID in replies_list:
            tmp_stance = tweets_stances[reply_ID]
            tmp += [tmp_stance]
        stance_list += [tmp]
    
    
    stance_percentage = np.zeros((len(stance_list),3)) # initialise an array, column 0,1,2 are percentages of query, deny, support
    
    for i in range(len(stance_list)):
        count = Counter(stance_list[i])
        l = len(stance_list[i])
        if 'query' in count:
            stance_percentage[i][0] = count['query']/l
        else:
            stance_percentage[i][0] = 0 
        
        if 'deny' in count:
            stance_percentage[i][1] = count['deny']/l
        else:
            stance_percentage[i][1] = 0 
        
        if 'support' in count:
            stance_percentage[i][2] = count['support']/l
        else:
            stance_percentage[i][2] = 0 
    
    return stance_percentage


# In[7]:


#transfer label
def str2no(y_string):
#@@ Param: a list of rumour veracity labels, each element is a string
#@@ Return: a list of label, each element is an int; 0,1,2 represent 'unverified','false','true' respectively
    
    y = []
    for cls in y_string:
        if cls=='unverified':
            y += [0]
        elif cls=='false':
            y += [1]
        elif cls=='true':
            y += [2]
    return y


# In[55]:


def transferUserInfo(userInfo):
    results = np.zeros((len(userInfo),len(userInfo[0])-1),dtype=int)
    for i in range(len(userInfo)):
        #if it has been verified
        if userInfo[i][0]==False:
            results[i][0] = 0
        else:
            results[i][0] = 1
        
        #if it has location?               
        if userInfo[i][1]=='' or userInfo[i][1]==None:
            results[i][1] = 0
        else:
            results[i][1] = 1
            
        #if it has description?
        if userInfo[i][2]=='' or userInfo[i][2]==None:
            results[i][2] = 0
        else:
            results[i][2] = 1

        #how many followers? followers_count
        results[i][3] = userInfo[i][3]               
        #how many people it follows? friends_count
        results[i][4] = userInfo[i][4]
        #how many tweets it posted? statuses_count
        results[i][5] = userInfo[i][5]    
        # favourites_count
        results[i][6] = userInfo[i][6]
        #how many days, after creating this account, when he/she posted this tweet
        tp = time.strptime(userInfo[i][-1],"%a %b %d %H:%M:%S %z %Y")
        tc = time.strptime(userInfo[i][-2],"%a %b %d %H:%M:%S %z %Y")
        diff = (datetime.datetime(tp.tm_year, tp.tm_mon, tp.tm_mday) - datetime.datetime(tc.tm_year, tc.tm_mon, tc.tm_mday)).days
        results[i][7] = diff
        
    return results


# In[269]:


def scorer(y_truth, y_hat, confidence): # rumourEval 2019 version of scorer
#@ Param:  y_truth: a list of true labels
#          y_hat: a list of predicted y values
#          confidence: a list of confidence values related to y_hat
# Return: accuracy score, RMSE and Macro averaged F1 score
    
    correct = 0
    total = len(y_hat)
    errors = []
    y_pred = []
    
    for i in range(total):
        if confidence[i]>0.5:
                y_pred += [y_hat[i]]
        else:
                y_pred += [0]   
        
        if y_pred[i] == y_truth[i] and y_truth[i]!=0:
            correct += 1
            errors += [(1-confidence[i])**2]

        elif y_truth[i] == 0:
            errors += [ (confidence[i])**2 ]

        else:
            errors += [1.0]
            
    score = correct / total
    rmse = math.sqrt( sum(errors) / len(errors) )
    macroF = f1_score(y_truth, y_pred, average='macro')

    return score,rmse,macroF


# In[218]:


# the U,T,F of the source post
train_file = './rumoureval-2019-training-data/train-key.json'
f = json.load(open(train_file, 'r'))

# extract features, hashtag existence and URL existence
filePath = './rumoureval-2019-training-data/twitter-english'


# In[219]:


#extract train data

source_tweets_train = list(f['subtaskbenglish'].keys()) # make a list of source tweets' ID

# a dictionary of tweets' stances
tweets_stances_train = f['subtaskaenglish'] ################## train and dev data are using same datasets


features = ['verified', 'location', 'description', 'followers_count',  ####################user feature we need
                                  'friends_count', 'statuses_count', "favourites_count",'created_at']

userInfoString = getUserInfo(source_tweets_train,features,filePath)
#  hashtag existence, URL existence, percentage of queries, denies, supports
existence = getExistence(source_tweets_train, filePath)
qds_percentage = getPercentage(source_tweets_train, tweets_stances_train, filePath)
ue_train = transferUserInfo(userInfoString)

y_train_string = list(f['subtaskbenglish'].values())


# In[220]:


# preprocessing tweets' texts
# train_text = getText(source_tweets_train, filePath)
# train_text


# In[221]:


X_train = np.hstack((np.hstack((existence,qds_percentage)),ue_train)) ################# X_train : existence + qds_percent + userInfo
y_train = str2no(y_train_string) #################### y_train


# In[59]:


userInfoString[0]


# In[60]:


ue_train[0]


# In[224]:


############################normalize the training data
scaler = preprocessing.StandardScaler().fit(X_train) 
X_train = scaler.transform(X_train)


# In[225]:


#extract dev data

dev_file = './rumoureval-2019-training-data/dev-key.json'
f = json.load(open(dev_file, 'r'))

source_tweets_dev = list(f['subtaskbenglish'].keys())
y_dev_string = list(f['subtaskbenglish'].values())
tweets_stances_dev = f['subtaskaenglish']

userInfoString = getUserInfo(source_tweets_dev,features,filePath)

existence = getExistence(source_tweets_dev, filePath)
qds_percentage = getPercentage(source_tweets_dev, tweets_stances_dev, filePath)
ue_dev = transferUserInfo(userInfoString)


# In[226]:


X_dev = np.hstack((np.hstack((existence,qds_percentage)),ue_dev)) ################# X_dev
X_dev = scaler.transform(X_dev) ######################## normalize dev data
y_dev = np.array(str2no(y_dev_string)) ################# y_dev


# In[227]:


#extract test data

te_Path = './rumoureval-2019-test-data/twitter-en-test-data'

f = json.load(open('./final-eval-key.json', 'r'))

source_tweets_te = list(f['subtaskbenglish'].keys())
y_te_string = list(f['subtaskbenglish'].values())
tweets_stances_te = f['subtaskaenglish']

userInfoString = getUserInfo(source_tweets_te,features,te_Path)

existence = getExistence(source_tweets_te, te_Path)
qds_percentage = getPercentage(source_tweets_te, tweets_stances_te, te_Path)
ue_test = transferUserInfo(userInfoString)


# In[228]:


X_te = np.hstack((np.hstack((existence,qds_percentage)),ue_test)) ################# X_te
X_te = scaler.transform(X_te)####################normalize test data
y_te = np.array(str2no(y_te_string))


# In[66]:


source_tweets_te[2],X_te[2]


# In[80]:


clf = RandomForestClassifier(n_estimators=200)
clf.fit(X=X_train, y=y_train)
sig_clf = CalibratedClassifierCV(clf,method='isotonic', cv='prefit')
sig_clf.fit(X_dev,y_dev)


# In[81]:


sig_clf_probs = sig_clf.predict_proba(X_te) 

y_hat = sig_clf.predict(X_te) ######################### predicted label as 3 classes


# In[82]:


y_hat


# In[83]:


clf_confidence = np.zeros(y_hat.shape) ##################### confidence value for each piece of prediction

for i in range(len(y_hat)):
    if y_hat[i]!=0:
        clf_confidence[i] = sig_clf_probs[i][y_hat[i]]
    else:
        if sig_clf_probs[i][1]>=sig_clf_probs[i][2]:
            clf_confidence[i] = sig_clf_probs[i][1]
            y_hat[i] = 1
        else:
            clf_confidence[i] = sig_clf_probs[i][2]
            y_hat[i] = 2


# In[84]:


y_hat ###################### after processing, y_hat should only contains 2 classes


# In[85]:


clf_confidence


# In[272]:


def scorer(y_truth, y_hat, confidence): # rumourEval 2019 version of scorer
#@ Param:  y_truth: a list of true labels
#          y_hat: a list of predicted y values
#          confidence: a list of confidence values related to y_hat
# Return: accuracy score, RMSE and Macro averaged F1 score
    
    correct = 0
    total = len(y_hat)
    errors = []
    y_pred = []
    
    for i in range(total):
        if confidence[i]>0.5:
                y_pred += [y_hat[i]]
        else:
                y_pred += [0]   
        
        if y_pred[i] == y_truth[i] and y_truth[i]!=0:
            correct += 1
            errors += [(1-confidence[i])**2]

        elif y_truth[i] == 0:
            errors += [ (confidence[i])**2 ]

        else:
            errors += [1.0]
    
    score = correct / total
    rmse = math.sqrt( sum(errors) / len(errors) )
    macroF = f1_score(y_truth, y_pred, average='macro')

    return score,rmse,macroF,y_pred


# In[86]:


score,rmse,macroF = scorer(y_te,y_hat,clf_confidence) # F-score is ill-defined and being set to 0.0 in labels with no predicted samples
print(('accuracy:', score))
print(('RMSE:', rmse))
print(('Macro averaged F1 socre:', macroF))


# In[97]:


rmse_list = []
macroF_list = []
esnum_list = []

for estimator_num in range(100,5000,100):
    
    clf = RandomForestClassifier(n_estimators=estimator_num)
    clf.fit(X=X_train, y=y_train)
    sig_clf = CalibratedClassifierCV(clf,method='isotonic', cv='prefit')
    sig_clf.fit(X_dev,y_dev)
    
    sig_clf_probs = sig_clf.predict_proba(X_dev) 

    y_hat = sig_clf.predict(X_dev) ######################### predicted label as 3 classes

    clf_confidence = np.zeros(y_hat.shape) ##################### confidence value for each piece of prediction

    for i in range(len(y_hat)):
        if y_hat[i]!=0:
            clf_confidence[i] = sig_clf_probs[i][y_hat[i]]
        else:
            if sig_clf_probs[i][1]>=sig_clf_probs[i][2]:
                clf_confidence[i] = sig_clf_probs[i][1]
                y_hat[i] = 1
            else:
                clf_confidence[i] = sig_clf_probs[i][2]
                y_hat[i] = 2
    
    score,rmse,macroF = scorer(y_dev,y_hat,clf_confidence)
    
    esnum_list += [estimator_num]
    rmse_list += [rmse]
    macroF_list += [macroF]


# In[99]:


import matplotlib.pyplot as plt
plt.title("I'm a scatter diagram.") 
# plt.xlim(xmax=7,xmin=0)
# plt.ylim(ymax=7,ymin=0)
plt.plot(macroF_list,rmse_list,'ro')
plt.xlabel("macroF1")
plt.ylabel("rmse")
plt.show()


# In[100]:


rmse_list.index(min(rmse_list))


# In[101]:


esnum_list[18]


# In[281]:


clf = RandomForestClassifier(n_estimators=1900)
clf.fit(X=X_train, y=y_train)
sig_clf = CalibratedClassifierCV(clf,method='isotonic', cv='prefit')
sig_clf.fit(X_dev,y_dev)

sig_clf_probs = sig_clf.predict_proba(X_te) 

y_hat = sig_clf.predict(X_te) ######################### predicted label as 3 classes

clf_confidence = np.zeros(y_hat.shape) ##################### confidence value for each piece of prediction

for i in range(len(y_hat)):
    if y_hat[i]!=0:
        clf_confidence[i] = sig_clf_probs[i][y_hat[i]]
    else:
        if sig_clf_probs[i][1]>=sig_clf_probs[i][2]:
            clf_confidence[i] = sig_clf_probs[i][1]
            y_hat[i] = 1
        else:
            clf_confidence[i] = sig_clf_probs[i][2]
            y_hat[i] = 2




score,rmse,macroF, y_pred = scorer(y_te,y_hat,clf_confidence) # F-score is ill-defined and being set to 0.0 in labels with no predicted samples
print(('accuracy:', score))
print(('RMSE:', rmse))
print(('Macro averaged F1 socre:', macroF))


# In[70]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.figure


# In[109]:


print(y_pred)


# In[110]:


print(y_te)


# In[280]:


get_ipython().run_line_magic('pinfo', 'classification_report')


# In[282]:


print((confusion_matrix(y_te,y_pred)))
print((classification_report(y_te,y_pred)))


# In[26]:


import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# In[30]:


X_train[10]


# In[ ]:


categorical_columns = ['verified', 'location', 'description']
numerical_columns = ['query%','deny%','support%','followers_count','friends_count', 'statuses_count', "favourites_count",'created_at']


# In[234]:


clf.feature_importances_


# In[239]:


columns = ['hashtag existence', 'URL existence','query%','deny%','support%','verified', 'location', 'description',
                       'followers_count','friends_count', 'statuses_count', "favourites_count",'time']
feat_importances = pd.Series(clf.feature_importances_, index= columns)
feat_importances.nlargest(13).plot(kind='barh')



# In[244]:


def transferUserInfo(userInfo): ###########第二版
    results = np.zeros((len(userInfo),5),dtype=int)
    for i in range(len(userInfo)):
        #if it has been verified
#         if userInfo[i][0]==False:
#             results[i][0] = 0
#         else:
#             results[i][0] = 1
        
#         #if it has location?               
#         if userInfo[i][1]=='' or userInfo[i][1]==None:
#             results[i][1] = 0
#         else:
#             results[i][1] = 1
            
#         #if it has description?
#         if userInfo[i][2]=='' or userInfo[i][2]==None:
#             results[i][2] = 0
#         else:
#             results[i][2] = 1

        #how many followers? followers_count
        results[i][0] = userInfo[i][3]               
        #how many people it follows? friends_count
        results[i][1] = userInfo[i][4]
        #how many tweets it posted? statuses_count
        results[i][2] = userInfo[i][5]    
        # favourites_count
        results[i][3] = userInfo[i][6]
        #how many days, after creating this account, when he/she posted this tweet
        tp = time.strptime(userInfo[i][-1],"%a %b %d %H:%M:%S %z %Y")
        tc = time.strptime(userInfo[i][-2],"%a %b %d %H:%M:%S %z %Y")
        diff = (datetime.datetime(tp.tm_year, tp.tm_mon, tp.tm_mday) - datetime.datetime(tc.tm_year, tc.tm_mon, tc.tm_mday)).days
        results[i][4] = diff
        
    return results


# In[245]:


# the U,T,F of the source post
train_file = './rumoureval-2019-training-data/train-key.json'
f = json.load(open(train_file, 'r'))

# extract features, hashtag existence and URL existence
filePath = './rumoureval-2019-training-data/twitter-english'

#extract train data

source_tweets_train = list(f['subtaskbenglish'].keys()) # make a list of source tweets' ID

# a dictionary of tweets' stances
tweets_stances_train = f['subtaskaenglish'] ################## train and dev data are using same datasets


features = ['verified', 'location', 'description', 'followers_count',  ####################user feature we need
                                  'friends_count', 'statuses_count', "favourites_count",'created_at']

userInfoString = getUserInfo(source_tweets_train,features,filePath)
#  hashtag existence, URL existence, percentage of queries, denies, supports
existence = getExistence(source_tweets_train, filePath)
qds_percentage = getPercentage(source_tweets_train, tweets_stances_train, filePath)
ue_train = transferUserInfo(userInfoString)

y_train_string = list(f['subtaskbenglish'].values())

X_train = np.hstack((qds_percentage,ue_train)) ################# X_train : existence + qds_percent + userInfo
y_train = str2no(y_train_string) #################### y_train

scaler = preprocessing.StandardScaler().fit(X_train) 
X_train = scaler.transform(X_train)


# In[246]:


len(X_train[0])


# In[247]:


dev_file = './rumoureval-2019-training-data/dev-key.json'
f = json.load(open(dev_file, 'r'))

source_tweets_dev = list(f['subtaskbenglish'].keys())
y_dev_string = list(f['subtaskbenglish'].values())
tweets_stances_dev = f['subtaskaenglish']

userInfoString = getUserInfo(source_tweets_dev,features,filePath)

existence = getExistence(source_tweets_dev, filePath)
qds_percentage = getPercentage(source_tweets_dev, tweets_stances_dev, filePath)
ue_dev = transferUserInfo(userInfoString)

X_dev =  np.hstack((qds_percentage,ue_dev)) ################# X_dev
X_dev = scaler.transform(X_dev) ######################## normalize dev data
y_dev = np.array(str2no(y_dev_string)) ################# y_dev


# In[248]:


te_Path = './rumoureval-2019-test-data/twitter-en-test-data'

f = json.load(open('./final-eval-key.json', 'r'))

source_tweets_te = list(f['subtaskbenglish'].keys())
y_te_string = list(f['subtaskbenglish'].values())
tweets_stances_te = f['subtaskaenglish']

userInfoString = getUserInfo(source_tweets_te,features,te_Path)

existence = getExistence(source_tweets_te, te_Path)
qds_percentage = getPercentage(source_tweets_te, tweets_stances_te, te_Path)
ue_test = transferUserInfo(userInfoString)

X_te =  np.hstack((qds_percentage,ue_test)) ################# X_te
X_te = scaler.transform(X_te)####################normalize test data
y_te = np.array(str2no(y_te_string))


# In[270]:


rmse_list = []
macroF_list = []
esnum_list = []

for estimator_num in range(100,8000,100):
    
    clf = RandomForestClassifier(n_estimators=estimator_num)
    clf.fit(X=X_train, y=y_train)
    sig_clf = CalibratedClassifierCV(clf,method='sigmoid', cv='prefit')
    sig_clf.fit(X_dev,y_dev)
    
    sig_clf_probs = sig_clf.predict_proba(X_dev) 

    y_hat = sig_clf.predict(X_dev) ######################### predicted label as 3 classes

    clf_confidence = np.zeros(y_hat.shape) ##################### confidence value for each piece of prediction

    for i in range(len(y_hat)):
        if y_hat[i]!=0:
            clf_confidence[i] = sig_clf_probs[i][y_hat[i]]
        else:
            if sig_clf_probs[i][1]>=sig_clf_probs[i][2]:
                clf_confidence[i] = sig_clf_probs[i][1]
                y_hat[i] = 1
            else:
                clf_confidence[i] = sig_clf_probs[i][2]
                y_hat[i] = 2
    
    score,rmse,macroF = scorer(y_dev,y_hat,clf_confidence)
    
    esnum_list += [estimator_num]
    rmse_list += [rmse]
    macroF_list += [macroF]


# In[271]:


import matplotlib.pyplot as plt
plt.title("I'm a scatter diagram.") 
# plt.xlim(xmax=7,xmin=0)
# plt.ylim(ymax=7,ymin=0)
plt.plot(macroF_list,rmse_list,'ro')
plt.xlabel("macroF1")
plt.ylabel("rmse")
plt.show()


# In[273]:


np.argmin(rmse_list)


# In[275]:


esnum_list[1]


# In[278]:


clf = RandomForestClassifier(n_estimators=200)
clf.fit(X=X_train, y=y_train)
sig_clf = CalibratedClassifierCV(clf,method='sigmoid', cv='prefit')
sig_clf.fit(X_dev,y_dev)

sig_clf_probs = sig_clf.predict_proba(X_te) 

y_hat = sig_clf.predict(X_te) ######################### predicted label as 3 classes

clf_confidence = np.zeros(y_hat.shape) ##################### confidence value for each piece of prediction

for i in range(len(y_hat)):
    if y_hat[i]!=0:
        clf_confidence[i] = sig_clf_probs[i][y_hat[i]]
    else:
        if sig_clf_probs[i][1]>=sig_clf_probs[i][2]:
            clf_confidence[i] = sig_clf_probs[i][1]
            y_hat[i] = 1
        else:
            clf_confidence[i] = sig_clf_probs[i][2]
            y_hat[i] = 2




score,rmse,macroF, y_pred = scorer(y_te,y_hat,clf_confidence) # F-score is ill-defined and being set to 0.0 in labels with no predicted samples
print(('accuracy:', score))
print(('RMSE:', rmse))
print(('Macro averaged F1 socre:', macroF))


# In[279]:


print((confusion_matrix(y_te,y_pred)))
print((classification_report(y_te,y_pred)))


# In[284]:


from sklearn.naive_bayes import GaussianNB


# In[296]:


clf = GaussianNB()
clf.fit(X=X_train, y=y_train)
sig_clf = CalibratedClassifierCV(clf,method='sigmoid', cv='prefit')
sig_clf.fit(X_dev,y_dev)

sig_clf_probs = sig_clf.predict_proba(X_te) 

y_hat = sig_clf.predict(X_te) ######################### predicted label as 3 classes

clf_confidence = np.zeros(y_hat.shape) ##################### confidence value for each piece of prediction

for i in range(len(y_hat)):
    if y_hat[i]!=0:
        clf_confidence[i] = sig_clf_probs[i][y_hat[i]]
    else:
        if sig_clf_probs[i][1]>=sig_clf_probs[i][2]:
            clf_confidence[i] = sig_clf_probs[i][1]
            y_hat[i] = 1
        else:
            clf_confidence[i] = sig_clf_probs[i][2]
            y_hat[i] = 2




score,rmse,macroF, y_pred = scorer(y_te,y_hat,clf_confidence) # F-score is ill-defined and being set to 0.0 in labels with no predicted samples
print(('accuracy:', score))
print(('RMSE:', rmse))
print(('Macro averaged F1 socre:', macroF))


# In[297]:


print((confusion_matrix(y_te,y_pred))) ######NB
print((classification_report(y_te,y_pred)))


# In[298]:


clf = svm.LinearSVC(multi_class='ovr', C=50, max_iter=1000000)
clf.fit(X=X_train, y=y_train)
sig_clf = CalibratedClassifierCV(clf,method='sigmoid', cv='prefit')
sig_clf.fit(X_dev,y_dev)

sig_clf_probs = sig_clf.predict_proba(X_te) 

y_hat = sig_clf.predict(X_te) ######################### predicted label as 3 classes

clf_confidence = np.zeros(y_hat.shape) ##################### confidence value for each piece of prediction

for i in range(len(y_hat)):
    if y_hat[i]!=0:
        clf_confidence[i] = sig_clf_probs[i][y_hat[i]]
    else:
        if sig_clf_probs[i][1]>=sig_clf_probs[i][2]:
            clf_confidence[i] = sig_clf_probs[i][1]
            y_hat[i] = 1
        else:
            clf_confidence[i] = sig_clf_probs[i][2]
            y_hat[i] = 2




score,rmse,macroF, y_pred = scorer(y_te,y_hat,clf_confidence) # F-score is ill-defined and being set to 0.0 in labels with no predicted samples
print(('accuracy:', score))
print(('RMSE:', rmse))
print(('Macro averaged F1 socre:', macroF))


# In[299]:


print((confusion_matrix(y_te,y_pred))) ##########svm
print((classification_report(y_te,y_pred)))


# In[3]:


# [1,24,-2,0,0]
def func(arr):
    positive = 0
    negative = 0
    zeros = 0
    for num in arr:
        if num > 0:
            positive += 1
        elif num == 0:
            zeros += 1
        else:
            negative += 1
            
    return positive, negative, zeros


# In[5]:


a = [1,24,-2,0,0]
func(a)


# In[30]:


tmp = -6 / 132
-int(-tmp)


# In[32]:


-6 / float(132)


# In[ ]:


51.5


# In[ ]:





# In[ ]:




