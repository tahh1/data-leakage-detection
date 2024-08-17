#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.datasets import load_files
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import KeyedVectors
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SimpleRNN
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.engine import Input


# In[3]:


nltk.download('punkt')


# In[4]:


#Reading the data
data = pd.read_csv('/content/drive/MyDrive/Classroom/NLP/competition_1/malayalam_news_train.csv')


# In[5]:


data.head()


# In[6]:


print((data['headings']))


# Removing punctuations

# In[7]:


punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''


# In[8]:


for ele in data:
  if ele in punc:
    data = data.replace(ele, "")


# In[9]:


#Tokenization
max_features = 1200 #number of words to keep. 
#1200 is the number of unique words in the corpus.
tokenizer = Tokenizer(nb_words=max_features, split=' ')
tokenizer.fit_on_texts(data['headings'].values)


# In[10]:


#Unique words and and their count
tokenizer.word_counts


# In[11]:


X = tokenizer.texts_to_sequences(data['headings'].values)
X = pad_sequences(X, padding = 'post') #Zero padding at the end of the sequence


# In[12]:


labels = data['label'].tolist()
le = preprocessing.LabelEncoder()
Y = le.fit_transform(labels)


# In[13]:


Y = to_categorical(Y)
print(Y)


# In[14]:


from sklearn.model_selection import KFold


# In[15]:


kf = KFold(n_splits=4)


# In[16]:


for training, testing in kf.split(X):
  print(("TRAIN:", X[training], "TESTING:",X[testing]))
  X_train, X_test = X[training], X[testing]
  Y_train, Y_test = Y[training], Y[testing]


# In[17]:


#Printing the size of the train data, train label, test data and test label
print(("Shape train data = ",np.shape(X_train)))
print(("Shape of train label = ",np.shape(Y_train)))
print(("Shape of test data = ",np.shape(X_test)))
print(("Shape of test label = ",np.shape(Y_test)))


# In[18]:


embed_dim = 500
hidden_layer = 100
model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
model.add(Dropout(0.2))
model.add(SimpleRNN(hidden_layer))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print((model.summary()))


# In[19]:


model.fit(X_train,Y_train,epochs = 20, batch_size = 32)


# In[20]:


score = model.evaluate(X_train, Y_train, verbose = 1, batch_size = 32)
print(("Accuracy: %.2f" % (score[1]*100)))


# In[21]:


score = model.evaluate(X_test, Y_test, verbose = 1, batch_size = 32)
print(("Accuracy: %.2f" % (score[1]*100)))


# In[22]:


test = pd.read_csv('/content/drive/MyDrive/Classroom/NLP/competition_1/malayalam_news_test.csv')
test.head()


# In[23]:


print((test['headings']))


# In[24]:


test = tokenizer.texts_to_sequences(test['headings'].values)
test = pad_sequences(test, padding = 'post')


# In[25]:


class_label = model.predict_classes(test)
print((le.inverse_transform(class_label)))


# In[26]:


pred_labels=le.inverse_transform(class_label)
print(pred_labels)


# In[27]:


df = pd.read_csv('/content/drive/MyDrive/Classroom/NLP/competition_1/ROLLNO_Task_1_submission.csv')
df.head()


# In[28]:


df['Predicted labels']=pred_labels


# In[29]:


pred_labels


# In[30]:


df.to_csv(r'/content/drive/MyDrive/Classroom/NLP/competition_1/pred_5.csv', index=False)


# In[ ]:




