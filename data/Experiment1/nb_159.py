#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import regex as re

FLAGS = re.MULTILINE | re.DOTALL

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = "<hashtag> {} <allcaps>".format(hashtag_body.lower())
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps> " # amackcrane added trailing space


def tokenize(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\w+", hashtag)  # amackcrane edit
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
    

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    #text = re_sub(r"([A-Z]){2,}", allcaps)  # moved below -amackcrane

    # amackcrane additions
    text = re_sub(r"([a-zA-Z<>()])([?!.:;,])", r"\1 \2")
    text = re_sub(r"\(([a-zA-Z<>]+)\)", r"( \1 )")
    text = re_sub(r"  ", r" ")
    text = re_sub(r" ([A-Z]){2,} ", allcaps)
    
    return text.lower()


    


# In[3]:


text = "I TEST alllll kinds of #hashtags and #HASHTAGS, @mentions and 3000 (http://t.co/dkfjkdf). w/ <3 :) haha!!!!!"
text2 = "TEStiNg some *tough* #CASES" # couple extra tests -amackcrane
tokens = tokenize(text)
print(tokens)
print((tokenize(text2)))


# In[25]:


'sss'.gr


# In[69]:


def get_glove_embeddings(f_zip, f_txt, word2id, emb_size=200):
    
    w_emb = np.zeros((len(word2id), emb_size))
    
    with zipfile.ZipFile(f_zip) as z:
        with z.open(f_txt) as f:
            for line in f:
                line = line.decode('utf-8')
                word = line.split()[0]
                     
                if word in word2id:
                    emb = np.array(line.strip('\n').split()[1:]).astype(np.float32)
                    w_emb[word2id[word]] += emb
    return w_emb


# In[5]:


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


# In[12]:


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


# In[133]:


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


# In[13]:


# the U,T,F of the source post
train_file = './rumoureval-2019-training-data/train-key.json'
f = json.load(open(train_file, 'r'))

# extract features, hashtag existence and URL existence
filePath = './rumoureval-2019-training-data/twitter-english'

source_tweets_train = list(f['subtaskbenglish'].keys()) # make a list of source tweets' ID
y_train_string = list(f['subtaskbenglish'].values())
text_tr = getText(source_tweets_train, filePath)


# In[14]:


dev_file = './rumoureval-2019-training-data/dev-key.json'
f = json.load(open(dev_file, 'r'))

source_tweets_dev = list(f['subtaskbenglish'].keys())
y_dev_string = list(f['subtaskbenglish'].values())
text_dev = getText(source_tweets_dev, filePath)


# In[16]:


te_Path = './rumoureval-2019-test-data/twitter-en-test-data'

f = json.load(open('./final-eval-key.json', 'r'))

source_tweets_te = list(f['subtaskbenglish'].keys())
y_te_string = list(f['subtaskbenglish'].values())
text_te = getText(source_tweets_te, te_Path)


# In[31]:


X_tr_string = [tokenize(x) for x in text_tr] ###############################
X_dev_string = [tokenize(x) for x in text_dev] ################################ into token
X_te_string = [tokenize(x) for x in text_te] ################################


# In[34]:


X_tr_string


# In[40]:


def tokenizeText(x_raw):

    x = x_raw.split(' ')
    #x.remove('')

    return x


# In[41]:


X_tr = [tokenizeText(x) for x in X_tr_string] ###############################
X_dev = [tokenizeText(x) for x in X_dev_string] ################################ into token
X_te = [tokenizeText(x) for x in X_te_string] ################################


# In[96]:


X_all = X_tr + X_dev + X_te

def getVocab(X_all):
    
    count = 1
    word2id = {}
    for line in X_all:
        for i in range(len(line)):
            if line[i] not in word2id:
                word2id[line[i]] = count
                count += 1
    word2id['<OOV>'] = count
    return word2id


# In[97]:


word2id = getVocab(X_all)

id2word = dict(list(zip(list(word2id.values()),list(word2id.keys()))))


# In[98]:


import zipfile
w_glove = get_glove_embeddings("glove.twitter.27B.zip","glove.twitter.27B.200d.txt",word2id)

w_glove.shape


# In[99]:


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


# In[100]:


X_tr_num = toNumSeq(X_tr,word2id)############## num of sequence
X_dev_num = toNumSeq(X_dev,word2id)############## num of sequence
X_te_num = toNumSeq(X_te,word2id)############## num of sequence


# In[109]:


def num2vec(X_num, embedding):
  
    vec_list = []
    for x in X_num:
        vec = []
        for num in x:
            vec += [embedding[num]]
        vec_list += [vec]
  
    return vec_list


# In[110]:


embedding_matrix = w_glove
X_tr_vec = num2vec(X_tr_num, embedding_matrix)
X_dev_vec = num2vec(X_dev_num, embedding_matrix)
X_te_vec = num2vec(X_te_num, embedding_matrix)


# In[111]:


def avgZeroVec(X_tmp):
    
    X_vec = X_tmp
    # this is the code to turn all-zeros vectors in X_vec into average vectors 
    for i in range(len(X_vec)):
        for j in range(len(X_vec[i])):
            if np.all(X_vec[i][j]==0):
                X_vec[i][j] = sum(X_vec[i])/len(X_vec[i])
    return X_vec


# In[112]:


X_tr_vec = avgZeroVec(X_tr_vec)
X_dev_vec = avgZeroVec(X_dev_vec)
X_te_vec = avgZeroVec(X_te_vec)


# In[119]:


def getRep(X_vec):
    
    X_rep = np.zeros((len(X_vec),200)) ####################################

    for i in range(len(X_vec)): 
        #if X_vec[i]!=[]: # some tweets only contains mentions and url that were removed, so in X_vec, there are some [] array
        X_rep[i] = sum(X_vec[i])/len(X_vec[i])
    return X_rep


# In[120]:


X_tr_rep = getRep(X_tr_vec)
X_dev_rep = getRep(X_dev_vec)
X_te_rep = getRep(X_te_vec)


# In[121]:


#X_train = np.hstack((np.hstack((existence,qds_percentage)),ue_train)) ################# X_train : existence + qds_percent + userInfo
y_train = np.array(str2no(y_train_string)) #################### y_train


# In[122]:


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


# In[123]:


# X_dev = np.hstack((np.hstack((existence,qds_percentage)),ue_dev)) ################# X_dev
# X_dev = scaler.transform(X_dev) ######################## normalize dev data
y_dev = np.array(str2no(y_dev_string)) ################# y_dev


# In[124]:


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


# In[125]:


# X_te = np.hstack((np.hstack((existence,qds_percentage)),ue_test)) ################# X_te
# X_te = scaler.transform(X_te)####################normalize test data
y_te = np.array(str2no(y_te_string))


# In[126]:


X_train = X_tr_rep
X_dev = X_dev_rep
X_te = X_te_rep


# In[127]:


clf = svm.LinearSVC(multi_class='ovr', C=10000, max_iter=100000)
clf.fit(X=X_train, y=y_train)
sig_clf = CalibratedClassifierCV(clf,method='sigmoid', cv='prefit')
sig_clf.fit(X_dev,y_dev)


# In[128]:


sig_clf_probs = sig_clf.predict_proba(X_te) 

y_hat = sig_clf.predict(X_te) ######################### predicted label as 3 classes


# In[129]:


y_hat


# In[130]:


y_te


# In[131]:


clf_confidence = [sig_clf_probs[i][y_hat[i]] for i in range(len(y_hat))]


# In[134]:


score,rmse,macroF,y_pred = scorer(y_te,y_hat,clf_confidence) # F-score is ill-defined and being set to 0.0 in labels with no predicted samples
print(('accuracy:', score))
print(('RMSE:', rmse))
print(('Macro averaged F1 socre:', macroF))


# In[135]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.figure
print((confusion_matrix(y_te,y_pred)))
print((classification_report(y_te,y_pred)))


# In[136]:


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


# In[137]:


y_hat ###################### after processing, y_hat should only contains 2 classes


# In[138]:


y_te


# In[139]:


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


# In[140]:


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


# In[141]:


score,rmse,macroF = scorer(y_te,y_hat,clf_confidence) # F-score is ill-defined and being set to 0.0 in labels with no predicted samples
print(('accuracy:', score))
print(('RMSE:', rmse))
print(('Macro averaged F1 socre:', macroF))


# In[124]:


get_ipython().run_line_magic('pinfo', 'GaussianNB')


# In[142]:


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


# In[150]:


clf = RandomForestClassifier(n_estimators=2000)
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




