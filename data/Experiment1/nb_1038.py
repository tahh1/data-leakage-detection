#!/usr/bin/env python
# coding: utf-8

# ### News Category Classification Assignnment 
# 
#                                                                                         -- Submitted By Rinki Chatterjee

# #### Importing Necessary Libraries

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report,confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM,Bidirectional,SpatialDropout1D
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import Embedding,Flatten,Dense,Dropout
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import asarray,zeros
from wordcloud import WordCloud

import warnings
warnings.filterwarnings('ignore')


# In[2]:


final_df = pd.read_csv("final_data.csv")


# In[3]:


final_df.head()


# In[4]:


final_df.category.unique()


# * WORLDPOST and THE WORLDPOST were given as two separate categories in the dataset. Here I change the category THE WORLDPOST to WORLDPOST 

# In[6]:


final_df.category = final_df.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)


# In[7]:


final_df.category.nunique() # number of unique categories


# ### Deep Learning Models

# In[40]:


import matplotlib.pyplot as plt

text_length = []

for text in range(len(final_df['MergedColumn'])):
    try:
        text_length.append(len(final_df['MergedColumn'][text]))

    except Exception as e:
        pass

print(("Maximum length of  Data", np.max(text_length)))
print(("Minimum length of  Data", np.min(text_length)))
print(("Median length of  Data", np.median(text_length)))
print(("Average length of  Data",np.mean(text_length)))
print(("Standard Deviation of  Data",np.std(text_length)))
plt.boxplot(text_length)
plt.show()


# In[8]:


max_features = 1000
maxlen = 150
embedding_size = 100
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(final_df['MergedColumn'])
X = tokenizer.texts_to_sequences(final_df['MergedColumn'])
X = pad_sequences(X, maxlen = maxlen)
y = np.asarray(final_df['category'])
y = pd.get_dummies(final_df['category']).values


# In[9]:


print((X.shape))
print((y.shape))


# In[10]:


(tokenizer.num_words)


# In[11]:


del final_df


# ### Train test split

# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# #### Extract the GloVe embedding file

# In[13]:


embeddings_dictionary = dict()

#glove_file = open('/content/drive/My Drive/glove.6B.200d.txt', encoding="utf8")
#glove_file = open('../input/glove6b100d/glove.6B.100d.txt', encoding="utf8")
glove_file = open('glove.6B.100d.txt', encoding="utf8")
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions

glove_file.close()
num_words = len(tokenizer.word_index) + 1
embedding_matrix = zeros((num_words, 100))
for word, index in list(tokenizer.word_index.items()):
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


print((len(list(embeddings_dictionary.values()))))
print(("Num words",num_words))
print(("matrix size ",embedding_matrix.shape))
print(("embeddings size ",embedding_size))
print(("Max len",maxlen))


# ### Building the model

# In[15]:


model = Sequential()
model.add(Embedding(num_words, embedding_size, weights = [embedding_matrix]))
model.add(SpatialDropout1D(0.2))
#model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(40, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print((model.summary()))


# In[16]:


model.save("new_category_lstm.h5")


# In[ ]:


history= model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=5,batch_size=64)


# In[ ]:


loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(('Accuracy: %f' % (accuracy*100)))


# In[ ]:


plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show();


# In[ ]:


plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();


# In[ ]:




