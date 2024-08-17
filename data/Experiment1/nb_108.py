#!/usr/bin/env python
# coding: utf-8

# # Lab 4 - Neural Network Classifier Using Simple Word Embeddings
# 
# 
# 
# ---
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In this session, we demonstrate how to solve a text classification task using simple 
# feedforward neural network classifier. We will use IMDB Large Movie Review Dataset to train a binary classification model, able to predict whether a review is positive or negative. First, our network takes one-hot word vectors as input, averages them to make one vector and trains a 
# fully-connected layer to predict the output. In the second part, we replace the one-hot vectors with the word embeddings and add a layer to see how much that improves the performance.
# 
# We are going to use Keras Sequential API in this session. The Sequential API allows you to make models layer-by-layer. But it is not straightforward to define models where layers connect to more than just the previous and next layers. 

# In[ ]:


import keras
import numpy as np
from keras.layers import Lambda, GlobalAveragePooling1D, Dense, Embedding
from keras import backend as K
from keras.models import Sequential


# ### Downloading data

# The dataset we will be using is the IMDB Large Movie Review Dataset, which consists of 50000 labeled movie reviews. These are split into 25,000 reviews for training and 25,000 reviews for testing. The  dataset contains an even number of positive and negative reviews, so randomly guessing yields 50% accuracy. The data is preprocessed. For text classification, it is ususal to limit the size of the vocabulary to stop the dataset from becoming too sparse, creating possible overfitting. We keep the top 10,000 most frequently occurring words in the training data.
# 

# In[ ]:


imdb = keras.datasets.imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)


# We now can start playing around with the data, let’s first see the length:

# In[57]:


print(("Training entries: {}, labels: {}".format(len(X_train), len(y_train))))


# The  reviews have been converted to integers and each integer represents a  word in a dictionary. 

# In[58]:


X_train[0]


# We can convert integers back to words by querying a dictionary object that contains the integer to string mapping:

# In[ ]:


word_index = imdb.get_word_index()


# Index 1 represents the beginning of the sentence and the index 2 is assigned to all unknown tokens. Index 0 will be used for padding.

# In[ ]:


word_index = {k:(v+3) for k,v in list(word_index.items())}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  
word_index["<UNUSED>"] = 3


# To reverse key and values in a dictionary:

# In[ ]:


reverse_word_index = dict([(value, key) for (key, value) in list(word_index.items())])


# To view a word:

# In[62]:


reverse_word_index[25]


# And to recreate the whole sentence from our training data we define decode_review:
# 

# In[ ]:


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# In[64]:


decode_review(X_train[10])


# ### Creating One-hot word vectors

# It is  common to use one-hot representation as input in Natural Language Processing tasks. In Keras, the Embedding layer takes an index as an input and convert it to one-hot vector with the length of the vocabulary size. Then multiplies these vectors by a normal weight matrix. But there is no way to only get a one-hot vector as the output of a layer in Keras. To solve this we use Lambda() layer and a function that creates the one-hot layer. 

# In[ ]:


def OneHot(input_dim=None, input_length=None):
    
    if input_dim is None or input_length is None:
        raise TypeError("input_dim or input_length is not set")

    
    def _one_hot(x, num_classes):
        return K.one_hot(K.cast(x, 'uint8'),
                          num_classes=num_classes)

    return Lambda(_one_hot,
                  arguments={'num_classes': input_dim},
                  input_shape=(input_length,))


# input_dim refers to the length of the one-hot vector and input_length refers to the length of the input sequence. Since the input to K.one_hot should be an integer tensor, we cast x to one (Keras passes around float tensors by default).
# 
# 

#  Each text sequence has in most cases different length of words. Here, we fill sequences with a pad token (0) to fit the size. This special tokens is then masked not to be accounted in averaging, loss calculation etc. We set the maximum length to 256.

# ### Preparing input data

# In[ ]:


VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 256

X_train_enc = keras.preprocessing.sequence.pad_sequences(X_train,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

X_test_enc = keras.preprocessing.sequence.pad_sequences(X_test,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)


# And to view a padded review:

# In[67]:


print((X_train_enc[1]))


# Now we want to build the neural network model. We  are going to have a hidden layer with 16 hidden units. 
# 
# First, we want to transform each index to an embedded vector and then average all vectors to a single one. It has been showed that unweighted average of word vectors outperforms many complicated networks that model semantic and syntactic compositionality. As an example you can take a look at this: (http://anthology.aclweb.org/P/P15/P15-1162.pdf)
# 
# To average we need to ignore padded zeros:

# In[ ]:


class GlobalAveragePooling1DMasked(GlobalAveragePooling1D):
    def call(self, x, mask=None):
        if mask != None:
            return K.sum(x, axis=1) / K.sum(mask, axis=1)
        else:
            return super().call(x)


# ### Neural Network model using one-hot vectors

# The first layer is an one-hot layer. The second layer is to compute average on all word vectors in a sentence without considering padding. The  output vector is piped through a fully-connected layer. The last layer is connected with a single output node with the sigmoid activation function. The final value is a float between 0 and 1. 
# The vocabulary count of the movie reviews (10000) is used as the input shape. At the end we visualize the model summary.

# In[ ]:


# put your code here

model = Sequential()
model.add(OneHot(input_dim = VOCAB_SIZE, input_length = MAX_SEQUENCE_LENGTH))
model.add(GlobalAveragePooling1DMasked())
model.add(Dense(1, activation='sigmoid'))


# ### Training the model

# To compile the model we need a loss function and an optimizer. We use binary_crossentropy loss function which is just a special case of categorical cross entropy. We also use Adam optimizer that can be used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data. You can read more about it here:
# (https://arxiv.org/abs/1412.6980v8
# )
# 

# In[70]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()


# When training, we want to check the accuracy of the model on data it hasn't seen before. So we create a validation set:

# In[ ]:


X_val = np.array(X_train_enc[:10000])
partial_X_train = np.array(X_train_enc[10000:])

y_val = np.array(y_train[:10000])
partial_y_train = np.array(y_train[10000:])


# Then we start to train the model for 40 epochs in mini-batches of 512 samples and monitor the model's loss and accuracy on the validation set.

# In[72]:


history = model.fit(partial_X_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(X_val, y_val),
                    verbose=1)


# ### Evaluating the model

# To evaulate the model on test data:

# In[46]:


results = model.evaluate(X_test_enc, y_test)


# In[47]:


print(results)
# loss, accuracay 


# Our first model accuracy using one-hot vectors is 
# 
# 
# 
# 

# ### Plotting the accuracy graph

# To plot a graph of accuracy and loss over time we can use Matplotlib:

# In[48]:


import matplotlib.pyplot as plt

history_dict = history.history

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = list(range(1, len(acc) + 1))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# ### Neural Network model using word embeddings

# Now instead of one-hot vectors, we want to use embedding. We change our first layer in model1 to an Embedding layer. This layer takes the integer-encoded vocabulary and looks up the embedding vector for each word-index. These vectors are learned as the model trains. The vectors add a dimension to the output array. The resulting dimensions are: (batch, sequence, embedding).

# In[49]:


VOCAB_SIZE= 10000

# put the code here

model2 = Sequential()
model2.add(Embedding(VOCAB_SIZE , MAX_SEQUENCE_LENGTH))
model2.add(GlobalAveragePooling1DMasked())
model2.add(Dense(1, activation='sigmoid'))


model2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# model2 summary
model2.summary()


X_val = np.array(X_train_enc[:10000])
partial_X_train = np.array(X_train_enc[10000:])

history2 = model2.fit(partial_X_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(X_val, y_val),
                    verbose=1)

results = model2.evaluate(X_test_enc, y_test)
print(results)


# In[73]:


results = model2.evaluate(X_test_enc, y_test)


# In[74]:


print (results)


# In[75]:


import matplotlib.pyplot as plt

history_dict = history2.history

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = list(range(1, len(acc) + 1))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# The accuracy of model2 is 87%. Using Embedding layer instead of one-hot layer improved the performance.

# ### Using pre-trained word embeddings

# The Embedding layer can be used to load a pre-trained word embedding model. We are going to use GloVe embeddings, which you can read about it here (https://nlp.stanford.edu/projects/glove/). GloVe stands for "Global Vectors for Word Representation". It's a somewhat popular embedding technique based on factorizing a matrix of word co-occurence statistics. You can download GloVe and we can seed the Keras Embedding layer with weights from the pre-trained embedding for the words in your dataset.
# First, we need to read GloVe and map words to GloVe:
# 

# In[ ]:


from keras.initializers import Constant

def readGloveFile(gloveFile):
    with open(gloveFile, 'r') as f:
        wordToGlove = {}  
        wordToIndex = {}  
        indexToWord = {}  

        for line in f:
            record = line.strip().split()
            token = record[0] 
            wordToGlove[token] = np.array(record[1:], dtype=np.float64) 
            
        tokens = sorted(wordToGlove.keys())
        for idx, tok in enumerate(tokens):
            kerasIdx = idx + 1  
            wordToIndex[tok] = kerasIdx 
            indexToWord[kerasIdx] = tok 

    return wordToIndex, indexToWord, wordToGlove


# Now, we create our pre-trained Embedding layer:

# In[ ]:


def createPretrainedEmbeddingLayer(wordToGlove, wordToIndex, isTrainable):
    vocabLen = len(wordToIndex) + 1  
    embDim = next(iter(list(wordToGlove.values()))).shape[0]  
   
    embeddingMatrix = np.zeros((vocabLen, embDim))  
    for word, index in list(wordToIndex.items()):
        embeddingMatrix[index, :] = wordToGlove[word] 

    embeddingLayer = Embedding(vocabLen, embDim, embeddings_initializer=Constant(embeddingMatrix), trainable=isTrainable)
    return embeddingLayer


# We freeze the weights. To create the model:

# In[80]:


# put the code here

wordToIndex, indexToWord, wordToGlove = readGloveFile('./glove.6B.50d.txt')
model3 = Sequential()
model3.add(createPretrainedEmbeddingLayer(wordToGlove, wordToIndex, False))
model3.add(GlobalAveragePooling1DMasked())
model3.add(Dense(1, activation='sigmoid'))

model3.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# model3 summary
model3.summary()


# In[81]:


X_val = np.array(X_train_enc[:10000])
partial_X_train = np.array(X_train_enc[10000:])

y_val = np.array(y_train[:10000])
partial_y_train = np.array(y_train[10000:])

history = model3.fit(partial_X_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(X_val, y_val),
                    verbose=1)


# In[83]:


import matplotlib.pyplot as plt

history_dict = history.history

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = list(range(1, len(acc) + 1))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# ### Adding another hidden layer to the network

# In model3, we only add another dense layer to see if that improves the performance.

# In[87]:


# put your code here
wordToIndex, indexToWord, wordToGlove = readGloveFile('./glove.6B.50d.txt')
model3 = Sequential()
model3.add(createPretrainedEmbeddingLayer(wordToGlove, wordToIndex, False))
model3.add(GlobalAveragePooling1DMasked())
model3.add(Dense(50, activation='relu'))
model3.add(Dense(1, activation='sigmoid'))

model3.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# model3 summary
model3.summary()


# In[88]:


X_val = np.array(X_train_enc[:10000])
partial_X_train = np.array(X_train_enc[10000:])

y_val = np.array(y_train[:10000])
partial_y_train = np.array(y_train[10000:])

history3 = model3.fit(partial_X_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(X_val, y_val),
                    verbose=1)


# In[89]:


import matplotlib.pyplot as plt

history_dict = history3.history

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = list(range(1, len(acc) + 1))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# The accuracy of model3 with an additional layer is 85%. Adding more layers can help you to extract more features. But we can do that upto a certain extent. After some point, instead of extracting features, we tend to overfit the data. Overfitting can lead to errors in some or the other form like false positives. It is not easy to choose the number of units in a hidden layer or the number of hidden layers in a neural network. For many applications, one hidden layer is enough. As a general rule, the number of units in that hidden layer is between the number of inputs and the number of outputs.
#  The best way to decide on the number of units and hidden layers is to try various parameters. Train several neural networks with different numbers of hidden layers and neurons, and monitor the performance of them. You will have to experiment using a series of different architectures. 
# 
# 
# 
# 

# 
# 
# ---
# 
# 

# In[ ]:





# This tutorial is substantially based on this document:
# https://www.tensorflow.org/tutorials/keras/basic_text_classification
# 
# To read more about Sequential APIs you can go to: https://keras.io/getting-started/sequential-model-guide/
# 
# The one-hot word vector layer is taken from:
# https://fdalvi.github.io/blog/2018-04-07-keras-sequential-onehot/
# 

# In[ ]:




