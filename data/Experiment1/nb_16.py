#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.initializers import Constant
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[3]:


df = pd.read_json('News_Category_Dataset_v2.json', lines = True)
df.head()


# In[4]:


cat = df.groupby('category')
print(("Total categories", cat.ngroups))
print((cat.size()))


# In[5]:


df.category = df.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)


# In[6]:


df['text'] = df.headline + " " + df.short_description

token = Tokenizer()
token.fit_on_texts(df.text)
X = token.texts_to_sequences(df.text)
df['words'] = X

df['word_len'] = df.words.apply(lambda i: len(i))
df = df[df.word_len >= 5]

df.head()


# In[7]:


maxlen = 50
X = list(sequence.pad_sequences(df.words, maxlen = maxlen))


# In[8]:


categories = df.groupby('category').size().index.tolist()
cat_int = {}
int_cat = {}
for i, k in enumerate(categories):
    cat_int.update({k : i})
    int_cat.update({i : K})
df['c2id'] = df['category'].apply(lambda x: cat_int[x])


# In[9]:


word_index = token.word_index

EmbD = 100

embedding_index = {}
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    embedding_index[word] = coefs
f.close()

print(('Unique tokens', len(word_index)))
print(('Total word vectors', len(embedding_index)))


# In[10]:


embedding_matrix = np.zeros((len(word_index) + 1, EmbD))
for word, i in list(word_index.items()):
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[11]:


X = np.array(X)
Y = np_utils.to_categorical(list(df.c2id))

seed = 30
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size = 0.2, random_state = seed)


# In[12]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(word_index)+1,
                            EmbD,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=maxlen,
                            trainable=False),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)),
    tf.keras.layers.Conv1D(64, kernel_size = 3),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(len(int_cat), activation = 'softmax')
])

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()


# In[13]:


history = model.fit(x_train, y_train, batch_size = 128, epochs = 10, validation_data = (x_val, y_val))


# In[15]:


plt.figure(figsize = (6,6))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = list(range(1, len(acc) + 1))

plt.title("Train and Val Accuracy")
plt.plot(epochs, acc, 'red', label = 'Training acc')
plt.plot(epochs, val_acc, 'blue', label = 'Validation loss')
plt.legend()

plt.show()


# In[16]:


#Have to make Predictions
Test_string = 'IPS officer suspended after being caught on camera for beating wife daughter backs him'
token_list = token.texts_to_sequences([Test_string])
pred = model.predict(token_list)
print((np.argmax(pred)))
a = np.argmax(pred)


# In[17]:


print((y_train[39]))


# In[18]:


df.head()


# In[19]:


b = df[df.c2id == a]['category']
b = b.iloc[0]
print(b)

