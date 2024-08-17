#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import numpy as np
import sys
from keras.preprocessing.text import Tokenizer
import string
import re


# In[2]:


file_names=['AIIMAT.txt', 'MLOE.txt','OKEWFSMP.txt','TAM.txt','TAMatter.txt', 'THWP.txt','TPP.txt']
corpus=[]
def get_corpus(path):
    with open(path,'r', errors= 'ignore') as data:
        corpus=data.read()
        corpus=corpus.lower()
    return corpus

for i in range(len(file_names)):
    path='/Users/phuongqn/Desktop/INF552/Homework/Homework 7 Data/Book Files/books/'+file_names[i]+''
    corpus.append(get_corpus(path))


# In[3]:


def clean_text(text):
    text = text.translate(string.punctuation)
    
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=:;<]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ", text)
    text = re.sub(r"\+", " ", text)
    text = re.sub(r"\-", " ", text)
    text = re.sub(r"\=", " ", text)
    text = re.sub(r"<", " ", text)
    text = re.sub(r";", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[0123456789]", " ", text)
    
    return text


# In[ ]:


c0 = clean_text(corpus[0])
c1 = clean_text(corpus[1])
c2 = clean_text(corpus[2])
c3 = clean_text(corpus[3])
c4 = clean_text(corpus[4])
c5 = clean_text(corpus[5])
c6 = clean_text(corpus[6])
corpus_c=[c0,c1,c2,c3,c4,c5,c6]
t = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=False, split=' ', char_level=True, oov_token=None, document_count=0)
t.fit_on_texts(corpus_c)
t.word_index


# In[5]:


chars=[]
for i in corpus_c:
    chars.extend(list(set(set(i))))
    
chars = sorted(list(set(chars)))
char_int = dict((c, i) for i, c in enumerate(chars))
int_char = dict((i, c) for i, c in enumerate(chars))


# In[6]:


char_int


# In[7]:


def i_o(file, size):
    ip=[]
    op=[]
    for i in range(0, len(file) - size+1, 1):
        seq_in = file[i:i + size - 1]
        seq_out = file[i + size-1]
        ip.append([char_int[c] for c in seq_in])
        op.append(char_int[seq_out])
    return ip, op


# In[8]:


ip_seq=[]
op_char=[]
for c in corpus_c:
    w=100
    ip, op = i_o(c, w)
    ip_seq.extend(ip)
    op_char.extend(op)


# In[9]:


# reshape X
X = np.reshape(ip_seq, (len(ip_seq), 99,1))
# normalize
X = X / float(len(chars))
# one hot encode the output variable
y = np_utils.to_categorical(op_char)
print((X.shape))


# In[10]:


y


# Here I noticed that using an LSTM with N=256 will typically yield better results, but the run time is long (4hrs per epoch). I have attached that version, which I stopped prematurely, for comparison!

# In[14]:


LSTMmodel = Sequential()
LSTMmodel.add(LSTM(28, input_shape=(X.shape[1], X.shape[2])))
# LSTMmodel.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), dropout=0.2))
LSTMmodel.add(Dropout(0.2))
LSTMmodel.add(Dense(y.shape[1], activation='softmax'))
print((LSTMmodel.summary()))
LSTMmodel.compile(loss='categorical_crossentropy', optimizer='adam')


# In[15]:


filepath= '/Users/phuongqn/Desktop/INF552/HW-Phuong/LSTMweights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


# In[16]:


LSTMmodel.fit(X, y, epochs=30, batch_size=64, callbacks=callbacks_list)


# In[17]:


weight='/Users/phuongqn/Desktop/INF552/HW-Phuong/LSTMweights/weights-improvement-30-2.1433.hdf5'
LSTMmodel.load_weights(weight)
LSTMmodel.compile(loss='categorical_crossentropy', optimizer='adam')


# In[20]:


seed = 'There are those who take mental phenomena naively, just as they would physical phenomena This school of psychologists tends not to emphasize the object'

pattern = [char_int[c] for c in seed[-99:].lower()]

for i in range(1000):
    seq = np.reshape(pattern, (1, len(pattern), 1))
    seq = seq / float(len(chars))
    charpredict = LSTMmodel.predict(seq, verbose=0)
    ind = np.argmax(charpredict)
    seed+=int_char[ind]
    pattern.append(ind)
    pattern = pattern[1:len(pattern)]

print(seed)


# 
