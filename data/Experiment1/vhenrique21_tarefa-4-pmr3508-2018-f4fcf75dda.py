#!/usr/bin/env python
# coding: utf-8

# ### Import of TensorFlow: 

# In[ ]:


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print((tf.__version__))


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D as MaxPool2D
from keras.callbacks import EarlyStopping


#    

#    

# ## Exploração dos DataSets: 

# <table>
#   <tr>
#     <th>Label</th>
#     <th>Class</th> 
#   </tr>
#   <tr>
#     <td>0</td>
#     <td>T-shirt/top</td> 
#   </tr>
#   <tr>
#     <td>1</td>
#     <td>Trouser</td> 
#   </tr>
#     <tr>
#     <td>2</td>
#     <td>Pullover</td> 
#   </tr>
#     <tr>
#     <td>3</td>
#     <td>Dress</td> 
#   </tr>
#     <tr>
#     <td>4</td>
#     <td>Coat</td> 
#   </tr>
#     <tr>
#     <td>5</td>
#     <td>Sandal</td> 
#   </tr>
#     <tr>
#     <td>6</td>
#     <td>Shirt</td> 
#   </tr>
#     <tr>
#     <td>7</td>
#     <td>Sneaker</td> 
#   </tr>
#     <tr>
#     <td>8</td>
#     <td>Bag</td> 
#   </tr>
#     <tr>
#     <td>9</td>
#     <td>Ankle boot</td> 
#   </tr>
# </table>

# In[ ]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[ ]:


labels = pd.read_csv("../input/train_labels.csv",index_col=0)
labels.shape


# In[ ]:


labels_categorical = np_utils.to_categorical(labels)


# ###### DataSet Original (sem ruídos):  

# In[ ]:


original = np.load('../input/train_images_pure.npy')
original.shape


# In[ ]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(original[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[labels.label[i]])


# ###### DataSet Rotacionado (entre 0 e 45 graus):  

# In[ ]:


rotated = np.load('../input/train_images_rotated.npy')
rotated.shape


# In[ ]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(rotated[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[labels.label[i]])


# ###### DataSet com ruído do tipo salt-and-pepper:

# In[ ]:


noisy = np.load('../input/train_images_noisy.npy')
noisy.shape


# In[ ]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(noisy[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[labels.label[i]])


# ###### DataSet com os dois tipos de ruído:

# In[ ]:


both = np.load('../input/train_images_both.npy')
both.shape


# In[ ]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(both[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[labels.label[i]])


# ###### Diferenças entre os DataSets:
#     - Original: Imagens limpas e sem qualquer ruído
#     - Rotated: Imagens iguais as originais, porém, rotacionadas de forma aleatória
#     - Noisy: Imagens com pontos aleatórios como ruído
#     - Both: Imagens rotacionadas e com o ruído acima    

#    

# ###### DataSet de Teste:

# In[ ]:


test = np.load('../input/Test_images.npy')
test.shape


# In[ ]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test[i], cmap=plt.cm.binary)


#    

#    

# ## Preprocess dos Dados:

# - Read image
# - Resize image
# - Remove noise(Denoise)
# - Segmentation
# - Morphology(smoothing edges)

# In[ ]:


plt.figure()
plt.imshow(both[0])
plt.colorbar()
plt.grid(False)


# In[ ]:


original = original / 255.0
rotated = rotated / 255.0
noisy = noisy / 255.0
both = both / 255.0

test = test / 255.0


# In[ ]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(both[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[labels.label[i]])


#    

# # Parte I: Rede Neural sem Pooling 

# In[ ]:


seed = 7
np.random.seed(seed)


# In[ ]:


Xoriginal = original.reshape(original.shape[0], 28, 28, 1).astype('float32')
Xnoisy = noisy.reshape(noisy.shape[0], 28, 28, 1).astype('float32')
Xrotated = rotated.reshape(rotated.shape[0], 28, 28, 1).astype('float32')
Xboth = both.reshape(both.shape[0], 28, 28, 1).astype('float32')

XtestFinal = test.reshape(test.shape[0], 28, 28, 1).astype('float32')


# ## Modelo da Rede:

# In[ ]:


def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    #model.add(Conv2D(64, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[ ]:


modelo = baseline_model()
modelo.summary()


# In[ ]:


callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)]


# ### Criação do DataSet de Treino e de Teste:
#     - Nesse primeiro momento iremos usar o DataSet Original

# In[ ]:


Xtrain,Xtest,Ytrain,Ytest = train_test_split(Xoriginal,labels_categorical, test_size = 0.25)


# In[ ]:


modelo.fit(Xtrain, Ytrain, validation_data=(Xtest,Ytest), epochs=40, 
          batch_size = 200, verbose=1, callbacks = callbacks)


# In[ ]:


#Dados de Validação:
scores = modelo.evaluate(Xtest, Ytest, verbose=0)
print(("Validação:", scores[1]))

#Dados Originais:
scores = modelo.evaluate(Xoriginal, labels_categorical, verbose=0)
print(("Original:", scores[1]))

#Dados Rotated:
scores = modelo.evaluate(Xrotated, labels_categorical, verbose=0)
print(("Rotated:", scores[1]))

#Dados Noisy:
scores = modelo.evaluate(Xnoisy, labels_categorical, verbose=0)
print(("Noisy:", scores[1]))

#Dados Both:
scores = modelo.evaluate(Xboth, labels_categorical, verbose=0)
print(("Both:", scores[1]))


# Após o treino usando o DataSet Original notamos um desempenho ruim quando usamos tal modelo nos DataSets com ruído. Principalmente no DataSet rotacionado

# Vamos então refazer o modelo, utilizando o DataSet misto, que possui os dois tipo de ruídos:

# ### Criação do DataSet de Treino e de Teste:
#     - Usando o DataSet com ambos ruídos

# In[ ]:


Xtrain,Xtest,Ytrain,Ytest = train_test_split(Xboth,labels_categorical, test_size = 0.25)


# In[ ]:


modelo.fit(Xtrain, Ytrain, validation_data=(Xtest,Ytest), epochs=40, 
          batch_size = 200, verbose=1, callbacks = callbacks)


# In[ ]:


#Dados de Validação:
scores = modelo.evaluate(Xtest, Ytest, verbose=0)
print(("Validação:", scores[1]))

#Dados Originais:
scores = modelo.evaluate(Xoriginal, labels_categorical, verbose=0)
print(("Original:", scores[1]))

#Dados Rotated:
scores = modelo.evaluate(Xrotated, labels_categorical, verbose=0)
print(("Rotated:", scores[1]))

#Dados Noisy:
scores = modelo.evaluate(Xnoisy, labels_categorical, verbose=0)
print(("Noisy:", scores[1]))

#Dados Both:
scores = modelo.evaluate(Xboth, labels_categorical, verbose=0)
print(("Both:", scores[1]))


# Temos uma grande melhora no caso dos DataSets Rotacionados, portanto o ideal é usar uma rede alimentada pelo DataSet Misto

#     
#     

#     
#     

# # Parte II: Rede Neural com Pooling 

# In[ ]:


seed = 7
np.random.seed(seed)


# In[ ]:


Xoriginal = original.reshape(original.shape[0], 28, 28, 1).astype('float32')
Xnoisy = noisy.reshape(noisy.shape[0], 28, 28, 1).astype('float32')
Xrotated = rotated.reshape(rotated.shape[0], 28, 28, 1).astype('float32')
Xboth = both.reshape(both.shape[0], 28, 28, 1).astype('float32')

XtestFinal = test.reshape(test.shape[0], 28, 28, 1).astype('float32')


# ## Modelo da Rede:

# In[ ]:


def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding="same", input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPool2D((2,2)))
    model.add(Conv2D(64, (5, 5), padding="same", activation='relu'))
    model.add(MaxPool2D((2,2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[ ]:


modeloPool = baseline_model()
modeloPool.summary()


# In[ ]:


callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)]


# ### Criação do DataSet de Treino e de Teste:
#     - Nesse primeiro momento iremos usar o DataSet Original

# In[ ]:


Xtrain,Xtest,Ytrain,Ytest = train_test_split(Xoriginal,labels_categorical, test_size = 0.25)


# In[ ]:


modeloPool.fit(Xtrain, Ytrain, validation_data=(Xtest,Ytest), epochs=40, 
          batch_size = 200, verbose=1, callbacks = callbacks)


# In[ ]:


#Dados de Validação:
scores = modeloPool.evaluate(Xtest, Ytest, verbose=0)
print(("Validação:", scores[1]))

#Dados Originais:
scores = modeloPool.evaluate(Xoriginal, labels_categorical, verbose=0)
print(("Original:", scores[1]))

#Dados Rotated:
scores = modeloPool.evaluate(Xrotated, labels_categorical, verbose=0)
print(("Rotated:", scores[1]))

#Dados Noisy:
scores = modeloPool.evaluate(Xnoisy, labels_categorical, verbose=0)
print(("Noisy:", scores[1]))

#Dados Both:
scores = modeloPool.evaluate(Xboth, labels_categorical, verbose=0)
print(("Both:", scores[1]))


# Novamente percebemos uma acurácia bastante baixa ao avaliar os DataSets rotacionados, entretanto notamos uma pequena melhora em relação ao primeiro teste sem pooling.

# Vamos então testar usando o DataSet com os dois tipos de ruído como alimentação da rede.

# ### Criação do DataSet de Treino e de Teste:
#     - Usando o DataSet com ambos ruídos

# In[ ]:


Xtrain,Xtest,Ytrain,Ytest = train_test_split(Xboth,labels_categorical, test_size = 0.25)


# In[ ]:


modeloPool.fit(Xtrain, Ytrain, validation_data=(Xtest,Ytest), epochs=40, 
          batch_size = 200, verbose=1, callbacks = callbacks)


# In[ ]:


#Dados de Validação:
scores = modeloPool.evaluate(Xtest, Ytest, verbose=0)
print(("Validação:", scores[1]))

#Dados Originais:
scores = modeloPool.evaluate(Xoriginal, labels_categorical, verbose=0)
print(("Original:", scores[1]))

#Dados Rotated:
scores = modeloPool.evaluate(Xrotated, labels_categorical, verbose=0)
print(("Rotated:", scores[1]))

#Dados Noisy:
scores = modeloPool.evaluate(Xnoisy, labels_categorical, verbose=0)
print(("Noisy:", scores[1]))

#Dados Both:
scores = modeloPool.evaluate(Xboth, labels_categorical, verbose=0)
print(("Both:", scores[1]))


# Temos uma grande melhora no caso dos DataSets Rotacionados, e uma pequena melhora geral em relação a Rede sem Pooling

#     
#     

# # DataSet de Teste, export: 

# In[ ]:


Pred = modeloPool.predict_classes(XtestFinal)

result = pd.DataFrame(columns = ['Id','label'])
result.label = Pred
result.Id = list(range(len(test)))
result.to_csv("result.csv",index=False)


#    

# # Single Pixel Attack
# 
# Como vimos ao longo dos testes, ao adicionarmos ruído às imagens temos um aumento na texa de erro. O Single Pixel Attack se baseia na ocorrencia de erros de classificação alterando apenas um pixel da imagem.
# Uma forma de evitar ou reduzir esse problema é realizando um preprocessamento das imagens antes de coloca-las na Rede Neural. Podemos por exemplo normalizar os valores, nessa atividade fizemos isso dividindo todas as entradas por 255, adicionando blur na imagens ou utilizando de inúmeros outros filtros.
