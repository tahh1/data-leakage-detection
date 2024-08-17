#!/usr/bin/env python
# coding: utf-8

# In[85]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import sklearn
import re
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from datetime import date
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
from nltk.tokenize import word_tokenize
import scipy as sp
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, auc,\
                            roc_auc_score
from keras import backend as K
from keras.models import load_model
import tensorflow as keras
from tensorflow import keras
from keras import layers
import warnings
warnings.filterwarnings('ignore')


# In[2]:


books = pd.read_csv('./books.csv', index_col=1)
books = books.iloc[:, 1:]
ratings = pd.read_csv('./ratings.csv')
book_tags = pd.read_csv('./book_tags.csv')
tags = pd.read_csv('./tags.csv')
to_read = pd.read_csv('to_read.csv')


# In[4]:


Xtrain, Xtest = train_test_split(ratings, test_size=0.2, random_state=1)
print(f"Shape of train data: {Xtrain.shape}")
print(f"Shape of test data: {Xtest.shape}")


# In[9]:


users_id, books_id, _ = ratings.nunique()


# In[ ]:


keras.layer


# In[41]:


input_books = keras.layers.Input(shape=[1])
embed_books = keras.layers.Embedding(books_id + 1,15)(input_books)
books_out = keras.layers.Flatten()(embed_books)

#user input network
input_users = keras.layers.Input(shape=[1])
embed_users = keras.layers.Embedding(users_id + 1,15)(input_users)
users_out = keras.layers.Flatten()(embed_users)

conc_layer = keras.layers.Concatenate()([books_out, users_out])
x = keras.layers.Dense(128, activation='relu')(conc_layer)
x_out = x = keras.layers.Dense(1, activation='relu')(x)
model = keras.Model([input_books, input_users], x_out)


# In[51]:


opt = tf.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mean_squared_error')


# In[52]:


model.summary()


# In[53]:


hist = model.fit([Xtrain.book_id, Xtrain.user_id], Xtrain.rating, 
                 batch_size=64, 
                 epochs=5, 
                 verbose=1,
                 validation_data=([Xtest.book_id, Xtest.user_id], Xtest.rating))


# In[54]:


train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
hist.history


# In[56]:


plt.plot(train_loss, color='r', label='Train Loss')
plt.plot(val_loss, color='b', label='Validation Loss')
plt.title("Train and Validation Loss Curve")
plt.legend()
plt.show()


# In[78]:


model.save('model')


# In[58]:


# Extract embeddings
book_em = model.get_layer('embedding_2')
book_em_weights = book_em.get_weights()[0]
book_em_weights.shape


# - predict

# In[69]:


book_arr = np.array(b_id) #get all book IDs
user = np.array([100 for i in range(len(b_id))])
pred = model.predict([book_arr, user])
pred


# JSON format

# In[72]:


web_book_data = books.reset_index()[["book_id", "title", "image_url", "authors"]]
web_book_data = web_book_data.sort_values('book_id')
web_book_data.to_json(r'web_book_data.json', orient='records')


# In[87]:


get_ipython().system(' saved_model_cli show --dir 0/ --all')


# In[ ]:




