#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import gensim
from sklearn.naive_bayes import GaussianNB

import datetime
import time
import nltk


# In[2]:


emb = gensim.models.KeyedVectors.load_word2vec_format('Set8_TweetDataWithoutSpam_GeneralData_Word_Phrase.bin', binary=True)


# In[7]:





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


# In[3]:


def getText(source_tweets, filePath):
#@@ Param:  Source_tweets: a list of source tweets ID
#           filePath: source tweets  parent document directory
#@@ Return: tweets text coresspoding to source tweets' IDs   

    tweet_text = []
    for i in range(len(source_tweets)):
        data = json.load(open(filePath+'/'+source_tweets[i]+'/source-tweet/'+source_tweets[i]+'.json','r'))
        tweet_text += [data['text']]
    
    return tweet_text


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


# In[22]:


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


# In[8]:


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

        #how many followers?
        results[i][3] = userInfo[i][3]               
        #how many people it follows?
        results[i][4] = userInfo[i][4]
        #how many tweets it posted?
        results[i][5] = userInfo[i][5]               
        #how many days, after creating this account, when he/she posted this tweet
        tp = time.strptime(userInfo[i][-1],"%a %b %d %H:%M:%S %z %Y")
        tc = time.strptime(userInfo[i][-2],"%a %b %d %H:%M:%S %z %Y")
        diff = (datetime.datetime(tp.tm_year, tp.tm_mon, tp.tm_mday) - datetime.datetime(tc.tm_year, tc.tm_mon, tc.tm_mday)).days
        results[i][6] = diff
        
    return results


# In[4]:


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


# In[10]:


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


# In[5]:


# the U,T,F of the source post
train_file = './rumoureval-2019-training-data/train-key.json'
f = json.load(open(train_file, 'r'))

# extract features, hashtag existence and URL existence
filePath = './rumoureval-2019-training-data/twitter-english'


# In[6]:


source_tweets_train = list(f['subtaskbenglish'].keys()) # make a list of source tweets' ID
y_train_string = list(f['subtaskbenglish'].values())
text = getText(source_tweets_train, filePath)


# In[7]:


def readFile(filePath):
    with open(filePath,'r') as f:
        s = f.readlines()
  
    for i in range(len(s)):
        s[i] = s[i].replace('\n','')

    return s


# In[8]:


X_tr_string = readFile('./files/X_tr_string.txt')
X_dev_string = readFile('./files/X_dev_string.txt')
X_te_string = readFile('./files/X_te_string.txt')


# In[9]:


def tokenizeText(x_raw):

    x = x_raw.split(' ')
    x.remove('')

    return x


# In[10]:


X_tr = [tokenizeText(x) for x in X_tr_string] ###############################
X_dev = [tokenizeText(x) for x in X_dev_string] ################################ into token
X_te = [tokenizeText(x) for x in X_te_string] ################################


# In[11]:


f = open('./files/dict.txt','r')
f = f.read()
word2id = eval(f) ###############################
id2word = dict([val, key] for key, val in list(word2id.items())) ##############################################


# In[12]:


def toNumSeq(X_raw,vocab):
  
    doc = []
  
    for text in X_raw:
        numSeq = []
        for wrd in text:
            if wrd in vocab:
                numSeq += [vocab[wrd]]
            else:
                numSeq += [vocab['<OOV>']]
        doc += [numSeq]
  
    return doc


# In[13]:


X_tr_num = toNumSeq(X_tr,word2id)############## num of sequence
X_dev_num = toNumSeq(X_dev,word2id)############## num of sequence
X_te_num = toNumSeq(X_te,word2id)############## num of sequence


# In[14]:


embedding_matrix = np.zeros((len(word2id)+1,300))
for word, i in list(word2id.items()):
    if word in emb:
        embedding_vector = emb[word]
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = np.zeros(300)


# In[15]:


def num2vec(X_num, embedding):
  
    vec_list = []
    for x in X_num:
        vec = []
        for num in x:
            vec += [embedding[num]]
        vec_list += [vec]
  
    return vec_list


# In[16]:


X_tr_vec = num2vec(X_tr_num, embedding_matrix)
X_dev_vec = num2vec(X_dev_num, embedding_matrix)
X_te_vec = num2vec(X_te_num, embedding_matrix)


# In[17]:


def avgZeroVec(X_tmp):
    
    X_vec = X_tmp
    # this is the code to turn all-zeros vectors in X_vec into average vectors 
    for i in range(len(X_vec)):
        for j in range(len(X_vec[i])):
            if np.all(X_vec[i][j]==0):
                X_vec[i][j] = sum(X_vec[i])/len(X_vec[i])
    return X_vec


# In[18]:


X_tr_vec = avgZeroVec(X_tr_vec)
X_dev_vec = avgZeroVec(X_dev_vec)
X_te_vec = avgZeroVec(X_te_vec)


# In[19]:


def getRep(X_vec):
    
    X_rep = np.zeros((len(X_vec),300)) ####################################

    for i in range(len(X_vec)): 
        #if X_vec[i]!=[]: # some tweets only contains mentions and url that were removed, so in X_vec, there are some [] array
        X_rep[i] = sum(X_vec[i])/len(X_vec[i])
    return X_rep


# In[20]:


X_tr_rep = getRep(X_tr_vec)
X_dev_rep = getRep(X_dev_vec)
X_te_rep = getRep(X_te_vec)


# In[10]:


# #extract train data

# source_tweets_train = list(f['subtaskbenglish'].keys()) # make a list of source tweets' ID

# # a dictionary of tweets' stances
# tweets_stances_train = f['subtaskaenglish'] ################## train and dev data are using same datasets


# features = ['verified', 'location', 'description', 'followers_count',  ####################user feature we need
#                                   'friends_count', 'statuses_count', "favourites_count",'created_at']

# userInfoString = getUserInfo(source_tweets_train,features,filePath)
# #  hashtag existence, URL existence, percentage of queries, denies, supports
# existence = getExistence(source_tweets_train, filePath)
# qds_percentage = getPercentage(source_tweets_train, tweets_stances_train, filePath)
# ue_train = transferUserInfo(userInfoString)

# y_train_string = list(f['subtaskbenglish'].values())


# In[11]:


# preprocessing tweets' texts
# train_text = getText(source_tweets_train, filePath)
# train_text


# In[23]:


#X_train = np.hstack((np.hstack((existence,qds_percentage)),ue_train)) ################# X_train : existence + qds_percent + userInfo
y_train = np.array(str2no(y_train_string)) #################### y_train


# In[13]:


############################normalize the training data
# scaler = preprocessing.StandardScaler().fit(X_train) 
# X_train = scaler.transform(X_train)


# In[24]:


#extract dev data

dev_file = './rumoureval-2019-training-data/dev-key.json'
f = json.load(open(dev_file, 'r'))

source_tweets_dev = list(f['subtaskbenglish'].keys())
y_dev_string = list(f['subtaskbenglish'].values())
tweets_stances_dev = f['subtaskaenglish']

# userInfoString = getUserInfo(source_tweets_dev,features,filePath)

# existence = getExistence(source_tweets_dev, filePath)
# qds_percentage = getPercentage(source_tweets_dev, tweets_stances_dev, filePath)
# ue_dev = transferUserInfo(userInfoString)


# In[25]:


# X_dev = np.hstack((np.hstack((existence,qds_percentage)),ue_dev)) ################# X_dev
# X_dev = scaler.transform(X_dev) ######################## normalize dev data
y_dev = np.array(str2no(y_dev_string)) ################# y_dev


# In[26]:


#extract test data

te_Path = './rumoureval-2019-test-data/twitter-en-test-data'

f = json.load(open('./final-eval-key.json', 'r'))

source_tweets_te = list(f['subtaskbenglish'].keys())
y_te_string = list(f['subtaskbenglish'].values())
tweets_stances_te = f['subtaskaenglish']

# userInfoString = getUserInfo(source_tweets_te,features,te_Path)

# existence = getExistence(source_tweets_te, te_Path)
# qds_percentage = getPercentage(source_tweets_te, tweets_stances_te, te_Path)
# ue_test = transferUserInfo(userInfoString)


# In[27]:


# X_te = np.hstack((np.hstack((existence,qds_percentage)),ue_test)) ################# X_te
# X_te = scaler.transform(X_te)####################normalize test data
y_te = np.array(str2no(y_te_string))


# In[28]:


X_train = X_tr_rep
X_dev = X_dev_rep
X_te = X_te_rep


# In[29]:


clf = svm.LinearSVC(multi_class='ovr', C=10000, max_iter=100000)
clf.fit(X=X_train, y=y_train)
sig_clf = CalibratedClassifierCV(clf,method='sigmoid', cv='prefit')
sig_clf.fit(X_dev,y_dev)


# In[30]:


sig_clf_probs = sig_clf.predict_proba(X_te) 

y_hat = sig_clf.predict(X_te) ######################### predicted label as 3 classes


# In[31]:


y_hat


# In[32]:


y_te


# In[33]:


clf_confidence = [sig_clf_probs[i][y_hat[i]] for i in range(len(y_hat))]


# In[35]:


score,rmse,macroF,y_pred = scorer(y_te,y_hat,clf_confidence) # F-score is ill-defined and being set to 0.0 in labels with no predicted samples
print(('accuracy:', score))
print(('RMSE:', rmse))
print(('Macro averaged F1 socre:', macroF))


# In[37]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.figure
print((confusion_matrix(y_te,y_pred)))
print((classification_report(y_te,y_pred)))


# In[180]:


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


# In[181]:


y_hat ###################### after processing, y_hat should only contains 2 classes


# In[182]:


y_te


# In[183]:


def scorer2017(y_truth, y_hat, confidence): # rumourEval 2019 version of scorer
#@ Param:  y_truth: a list of true labels
#          y_hat: a list of predicted y values
#          confidence: a list of confidence values related to y_hat
# Return: accuracy score, RMSE and Macro averaged F1 score
    
    correct = 0
    total = len(y_hat)
    errors = []
    
    for i in range(total):
        
        if y_hat[i] == y_truth[i] and y_truth[i]!=0:
            correct += 1
            errors += [(1-confidence[i])**2]

        elif y_hat[i] == 0:
            errors += [ (confidence[i])**2 ]

        else:
            errors += [1.0]
    
    score = correct / total
    rmse = math.sqrt( sum(errors) / len(errors) )
    macroF = f1_score(y_truth, y_hat, average='macro')

    return score,rmse,macroF


# In[184]:


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


# In[185]:


score,rmse,macroF = scorer(y_te,y_hat,clf_confidence) # F-score is ill-defined and being set to 0.0 in labels with no predicted samples
print(('accuracy:', score))
print(('RMSE:', rmse))
print(('Macro averaged F1 socre:', macroF))


# In[124]:


get_ipython().run_line_magic('pinfo', 'GaussianNB')


# In[137]:


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

score,rmse,macroF = scorer(y_te,y_hat,clf_confidence) # F-score is ill-defined and being set to 0.0 in labels with no predicted samples
print(('accuracy:', score))
print(('RMSE:', rmse))
print(('Macro averaged F1 socre:', macroF))


# In[136]:


clf = RandomForestClassifier(n_estimators=3200)
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

score,rmse,macroF = scorer(y_te,y_hat,clf_confidence) # F-score is ill-defined and being set to 0.0 in labels with no predicted samples
print(('accuracy:', score))
print(('RMSE:', rmse))
print(('Macro averaged F1 socre:', macroF))


# In[128]:


clf = svm.LinearSVC(multi_class='ovr', C=500, max_iter=100000)
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

score,rmse,macroF = scorer(y_te,y_hat,clf_confidence) # F-score is ill-defined and being set to 0.0 in labels with no predicted samples
print(('accuracy:', score))
print(('RMSE:', rmse))
print(('Macro averaged F1 socre:', macroF))


# In[ ]:




