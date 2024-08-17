#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy as np
import matplotlib.pyplot as plt
import utils_img_rec as ut

import pathlib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import layers, models, utils

import pickle
inicio = time.time()


# ## Definindo Variaveis

# In[2]:


BATCH_SIZE = 5000
EPOCHS = 50
IMG_SIZE = 32
numero_de_canais = 3
teste_treino = True

DATA_DIR = '../bases/pickle/imagens/cifar10/cifar10-train-pickle.pickle'

TEST_DIR = '../bases/pickle/imagens/cifar10/cifar10-test-pickle.pickle'

caminho_modelo = '../modelos_salvos/tensorflow/'
nome = 'modelo_cifar10' +'-aug-'+'EPOCHS='+str(EPOCHS)

DATA_DIR = pathlib.Path(DATA_DIR)
TEST_DIR = pathlib.Path(TEST_DIR)


# ## Lendo e preparando os dados

# In[3]:


pickle_in = open(DATA_DIR,"rb")
data_train = pickle.load(pickle_in)


# In[4]:


if teste_treino:
    pickle_in = open(TEST_DIR,"rb")
    data_test = pickle.load(pickle_in)


# In[5]:


CATEGORIES = ut.get_classes(data_train)
CATEGORIES.sort()
try:
    CATEGORIES.remove('.ipynb_checkpoints')
except:
    pass
print(CATEGORIES)


# ## Exibindo amostra dos dados de treino

# In[6]:


plt.figure(figsize=(10,10))
m = 70
if( len(data_train) < 70 ):
    m = len(data_train)
for i in range(m):
    plt.subplot(7,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(data_train[i][0], cmap='gray')
    plt.xlabel("{}\n({})".format( data_train[i][1] , data_train[i][2] ), color='white')    
plt.show()


# ## Processo de aumento da base

# In[7]:


import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../Image-Augmentation/')

import Aug
import utils


# In[8]:


pipe = Aug.Pipe()

pipe.add(Aug.Invert(prob=0.5))
pipe.add(Aug.Color(prob=0.6, min_factor=-3, max_factor=3))
pipe.add(Aug.Random_Erasing(prob=0.3, rectangle_area=0.3, repetitions=4))
pipe.add(Aug.Rotacao(prob=0.5, max_left_rotation=88, max_right_rotation=88, fill='edge'))
pipe.add(Aug.Shift(prob=0.5, horizontal_max=0.4, vertical_max=0.4, randomise=True, fill='nearest'))
pipe.add(Aug.Random_Noise(prob=0.5) )

pipe.print_pipe()


# In[9]:


data_train = utils.call_thread(data = data_train, pipe_instance = pipe, img_per_thread=5000, image_per_image = 2,
                               salvar_imagens_gerada=False)


# ### OBS: O codigo usado no aumento é parte de uma biblioteca ainda não finalizada
# #### Para mais informações sobre a biblioteca
# * https://github.com/Birunda3000/Image-Augmentation

# ## Exibindo amostra dos dados de treino aumentados

# In[10]:


utils.print_list_img(data_train, limite=40)


# ## Separando imagem e label e misturando dados

# In[11]:


train_X, train_y = ut.prep_data(data_train, CATEGORIES, IMG_SIZE, numero_de_canais)
#print("Number of training images: ",len(data_train)
print(('Entradas de treino - {} - ({}x{})'.format( train_X.shape[0], train_X.shape[1], train_X.shape[2] )))


# In[12]:


if teste_treino:
    test_X, test_y = ut.prep_data(data_test, CATEGORIES, IMG_SIZE, numero_de_canais)
    #print("Number of test images: ",len(data_test))
    print(('Entradas de teste - {} - ({}x{})'.format( test_X.shape[0], test_X.shape[1], test_X.shape[2] )))


# ## Exibindo amostra dos dados de teste

# In[13]:


plt.figure(figsize=(10,10))
m = 70
if( len(data_test) < 70 ):
    m = len(data_test)
for i in range(m):
    plt.subplot(7,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(data_test[i][0], cmap='gray')
    plt.xlabel("{}\n({})".format( data_test[i][1] , data_test[i][2] ), color='white')    
plt.show()


# ## Normalizando dados

# In[14]:


train_X=np.array(train_X/255.0)
train_y=np.array(train_y)
if teste_treino:
    test_X=np.array(test_X/255.0)
    test_y=np.array(test_y)


# ## Modelo

# In[15]:


model = models.Sequential()
model.add( layers.Conv2D(filters=32, kernel_size=(8, 8), activation='relu', input_shape=(test_X.shape[1:])))
model.add( layers.MaxPooling2D((2, 2)))

model.add( layers.Dropout(rate=0.2) )

model.add( layers.Conv2D(filters=16, kernel_size=(4, 4), activation='relu'))
model.add( layers.MaxPooling2D((2, 2)))

model.add( layers.Dropout(rate=0.2) )

#model.add( layers.Conv2D(filters=32, kernel_size=(2, 2), activation='relu'))
#model.add( layers.MaxPooling2D((2, 2)))

#model.add( layers.Dropout(rate=0.2) )

model.add( layers.Flatten( ) )
model.add( layers.Dense(60000, activation='relu') )
model.add( layers.Dropout(rate=0.4) )

model.add( layers.Dense(2000, activation='relu') )
model.add( layers.Dropout(rate=0.1) )

model.add( layers.Dense(128, activation='relu') )
model.add( layers.Dense(len(CATEGORIES) , activation='softmax') )


# ## Realizando o treino

# In[16]:


#opt = tf.keras.optimizers.SGD(
#    learning_rate=0.01, momentum=0.1, nesterov=False, name="SGD")
model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=['accuracy'])
if  teste_treino:
    history = model.fit(train_X, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(test_X, test_y))
else:
    history = model.fit(train_X, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)


# ## Grafico do treino

# In[17]:


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.45, 1])
plt.legend(loc='lower right')


# In[18]:


test_loss, test_acc = model.evaluate(test_X, test_y, verbose=1)
print(('Tempo de execução: ', time.time() - inicio))


# In[19]:


nome_saida = caminho_modelo+nome+' - val_acc = '+   str(round(test_acc, 6))  + ' exec_time - '+ str(time.time() - inicio) + '.h5'
#nome_saida = caminho_modelo+nome+'.h5'

model.save(nome_saida)
print(('Salvo como: ',nome_saida))


# In[ ]:




