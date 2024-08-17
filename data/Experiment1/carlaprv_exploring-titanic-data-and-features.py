#!/usr/bin/env python
# coding: utf-8

# ## 1. Instalação das principais bibliotecas que iremos utilizar

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import re as re

from matplotlib import pyplot as plt

# blibliotecas para plotarmos os dados
import seaborn as sns
sns.set_style("whitegrid")


# ## 2. Leitura e entendimento dos datasets do titanic

# In[2]:


# leitura dos datasets do titanic
# train = leitura do conjunto de dados de treinamento
# test = leitura do conjunto de dados de teste
#genderSubmissionsubmission = vamos utilizar no final da análise
train = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})
submission = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})

# full_data é uma estrutura que armazena os nossos 2 datasets, isso irá facilitar nosso trabalho mais para frente
full_data = [train, test]


# In[3]:


#vamos dar uma olhada nos dados de treinamento
train.head()


# In[4]:


#vamos dar uma olhada nos dados de teste
test.head()


# ## 3. Conhecendo das variaveís do dataset

# In[5]:


# vamos verificar os dados (colunas do nosso dataset)
print((train.info()))


# Das informações acima, podemos ver que algumas colunas estão incompletas (faltando dados) e teremos que tratar isso de alguma maneira.
# 
# Iremos passar por cada variável do dataset para entendê-la e, se necessário, tratar os dados nulos.

# ### 3.1 Pclass - classe de viagem do passageiro

# Podemos observar se a classe de viagem do passageiro tem influência na taxa de sobrevivência.

# In[6]:


print((train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()))


# ### 3.2 Sexo do passageiro
# Distribuição de sobreviventes de acordo com o sexo

# In[7]:


# utilizando o conjunto de Treinamento vamos observar os dados
p = sns.countplot(data=train, x = 'Survived', hue = 'Sex')
plt.title("Distribuição de sobreviventes de acordo com o sexo")
plt.show()

# variáveis para exibir
total_survived_females = train[train.Sex == "female"]["Survived"].sum()
total_survived_males = train[train.Sex == "male"]["Survived"].sum()

print(("Total de sobreviventes: " + str((total_survived_females + total_survived_males))))
print((train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()))


# ### 3.3 SibSp and Parch - tamanho da família
# 

# In[8]:


# Mas e o tamanho da família, seria importante?
# podemos criar uma nova variável chamada FamilySize de acordo com 
# número de irmãos por conjuge e numero de pais por filhos abordo

# criamos uma nova variável para cada dataset em full_data chamada FamilySize
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# informações obtidas do conjunto de treinamento
print((train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()))


# Se olharmos novamento os dados dos **dataset de treinamento e de teste**, podemos ver que uma nova coluna foi adicionada: **FamilySize**.

# In[9]:


# dataset 0 de full_data = dataset de treinamento (train)
full_data[0].head()


# In[10]:


# dataset 1 de full_data = dataset de teste (test)
full_data[1].head()


# O tamanho da família parece ser uma variável influente na taxa de sobrevivência.

# ### 3.4 Embarked
# Essa variável possui alguns valores nulos. Vamos tentar preencher os valores nulos com o valor que mais ocorre: S.

# In[11]:


# dentro de cada dataset (train e teste) corrigimos os valores
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    
print((train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()))


# ### 3.5 Age 
# Essa variável possui alguns valores nulos também. Existem diversar formas de tratar valores nulos em um dataset.
# * substituir os valores nulos pela **média da variável**
# * substituir os valores nulos de forma aleatória dentro de um intervalo **[média - desvio, média + desvio]**

# In[12]:


# observando a média de idade no conjunto de Treinamento
media = train['Age'].mean()
desvio = train['Age'].std()
print(("Média da idade:",media))
print(("Desvio padrão da idade:",desvio))


# In[13]:


# dentro de cada dataset (train e teste) corrigimos o campo Age (idade)
for dataset in full_data:
    age_avg = dataset['Age'].mean() #retorna a média da idade
    age_std = dataset['Age'].std()  # retorna o desvio padrão
    age_null_count = dataset['Age'].isnull().sum() #conta a quantidade de campos nulos
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
        
train['CategoricalAge'] = pd.cut(train['Age'], 5)
print((train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean()))


# ## Limpeza dos dados
# Legal, já conhecemos todas as variáveis do nosso dataset.
# 
# Agora, vamos limpar nossos dados e transformá-los em valores numéricos. 

# In[14]:


# dentro de cada dataset (train e teste) transformamos nossas variáveis em valores numéricos
for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int) 
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4

# Seleção de variáveis que não iremos utilizar
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',\
                 'Parch','Fare']

train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge'], axis = 1)

test  = test.drop(drop_elements, axis = 1)

print((train.head(10)))

train = train.values
test  = test.values


# Ótimo! Agora temos um dataset limpo e pronto para montarmos nosso modelo de predição.

# ## Predição
# Agora vamos usar o SVC como nosso classificador.
# 
# Mais informações sobre o SVC. 
# https://scikit-learn.org/stable/modules/svm.html

# <b>a) Importamos as bibliotecas necessárias</b>

# In[15]:


from sklearn.svm import SVC
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


# <b> b) Criamos o nosso classificador SVC do tipo SVM (Máquinas de Vetores de Suporte) </b>

# In[16]:


classifiers = [SVC(probability=True)]
candidate_classifier = SVC()


# **c) Treinamos o nosso classificador com o conjunto de treinamento.**
# 
# A função ```.fit(X, y) ``` recebe 2 arrays de entrada:
# * **X** em forma de matriz ```[elementos, variáveis]``` com todos os elementos do conjunto de treinamento
# * **y** em forma de array com o nome das variáveis
# 
# Mais informações sobre SVM em python: https://scikit-learn.org/stable/modules/svm.html

# In[17]:


candidate_classifier.fit(train[0::, 1::], train[0::, 0])


# ### Testando nosso classificador
# Uma forma de validar o classificador é usar a técnica chamada **Cross Validation/Validação Cruzada**.
# 
# Abaixo, vamos criar uma função que faz o Cross Validation e nos retorna a acurácia média. A Acurácia, basicamente é o percentual de acertos que modelo teve.

# In[18]:


def acuracia(clf,X,y):
    scores = cross_val_score(clf, X, y, cv=5)
    resultados = cross_val_predict(clf, X, y, cv=5)
    print(("Cross-validated Scores: ",scores))
    acuracia_media = metrics.accuracy_score(y,resultados)
    print(("Acurácia média: ", acuracia_media))
    return None


# Como estamos testando nosso classificador usando os dados de treinamento, podemos:
# 1. executar nosso classificador sobre os **dados de teste**
# 2. comparar os resultados do classificador com os resultados esperados (armazenados na coluna survived)

# In[19]:


# armazenamos os resutados esperados
classes = candidate_classifier.predict(train[0::,1::])


# In[20]:


# executamos a função acuracia
acuracia(candidate_classifier,train,classes)


# Como podemos ver a acurácia ficou na faixa de 97%! O SVM está funcionando muito bem para esses dados.
# 
# Mas precisamos ter em mente que a base de dados do Titanic é uma base pequena e simples, por isso, conseguimos facilmente um bom trabalho do SVM. O objetivo foi apenas mostrar como podemos testar nosso algoritmo de forma prática.
# 
# Agora, vamos seguir com a predição dos nossos dados de teste, que, afinal, é o nosso objetivo!

# <b>d) Após o treinamento, realizamos a predição com os dados de teste</b>

# In[21]:


result = candidate_classifier.predict(test)


# Abaixo o print do resultado, uma array de 0's e 1's. Cada valor representa um dos passageiros. 

# In[22]:


print (result)


# Lembra-se da nossa variável **submission** que guardou os dados de teste? Agora iremos utilizá-la para saber quem são os passageiros que sobreviveram ou não.

# In[23]:


submission.head()


# ## Criamos um Dataframe para armazenar nossos dados
# 
# Um <a href="https://www.tutorialspoint.com/python_pandas/python_pandas_dataframe.htm" targeg="_blank">DataFrame</a> é uma estrutura bidimensional (como uma matriz ou tabela). <br>
# 
# Iremos juntar os dados armazenados em <b>submission</b> com os dados gerados pelo nosso classificador (<b>results</b>) e criar uma nova coluna chamada <b>Survived</b> que terá apenas os valores binários (1 - sobreviveu, 0 - não sobreviveu).
# 

# In[24]:


final = pd.DataFrame({
        # dados armazenados em submission
        "PassengerId": submission["PassengerId"],
        "Pclass": submission["Pclass"],
        "Pclass": submission["Name"],
        "Sex": submission["Sex"],
        "Age": submission["Age"],
        "FamilySize": submission['SibSp'] + submission['Parch'] + 1,
        # dados armazenados em result
        "Survived": result
    })


# #### Após isso, iremos transformar nosso dados em um arquivo csv

# In[25]:


final.to_csv("titanic.csv", index=False)
print((final.shape))


# #### Mas, já vamos dar uma olhada no resultado final da nossa predição

# In[26]:


final


# ## Visualizando os resultados da nossa predição
# Podemos plotar os dados da nossa estrutura <b>final</b> para visualizar e analisar o resultado da nossa predição

# In[27]:


# utilizando o resultado que obtivemos vamos observar os dados
p = sns.countplot(data=final, x = 'Survived', hue = 'Sex')
plt.title("Distribuição de sobreviventes de acordo com o sexo")
plt.show()


# In[28]:


print((final[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()))


# ## Criação do arquivo de submissão

# In[29]:


genderSubmission = pd.DataFrame({
        # dados armazenados em submission
        "PassengerId": submission["PassengerId"],
        # dados armazenados em result
        "Survived": result
    })

genderSubmission.to_csv("gender_submission.csv", index=False)

print(genderSubmission)

