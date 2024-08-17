#!/usr/bin/env python
# coding: utf-8

# ## Pacotes necessários
# 
# pickle <br>
# sklearn <br>
# yellowbrick <br>

# In[23]:


import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from yellowbrick.classifier import ConfusionMatrix


# ## Carregar Dados Salvos - Base Crédito
# 
# X_treinamento, X_teste, Y_treinamento, Y_teste:

# In[24]:


with open('content/credit.pkl', 'rb') as f:
    X_credit_train, X_credit_test, Y_credit_train, Y_credit_test = pickle.load(f)

X_credit_train.shape, X_credit_test.shape, Y_credit_train.shape, Y_credit_test.shape


# ## Treinamento algoritmo logistic Classifier - Base Crédito

# In[25]:


logistic_credit_data = LogisticRegression(random_state=1)
logistic_credit_data.fit(X_credit_train, Y_credit_train)


# ## Previsões algoritmo logistic Classifier - Base Crédito

# In[26]:


predict_credit = logistic_credit_data.predict(X_credit_test)


# ## Estatísticas de acerto - Base Crédito - 94.6%
# 
# Acurácia:

# In[27]:


cm = ConfusionMatrix(logistic_credit_data)
cm.fit(X_credit_train, Y_credit_train)
cm.score(X_credit_test, Y_credit_test)


# Relatório de Classificação:

# In[28]:


print((classification_report(Y_credit_test, predict_credit)))


# ## Carregar Dados Salvos - Base Census
# 
# X_treinamento, X_teste, Y_treinamento, Y_teste:

# In[29]:


with open('content/census.pkl', 'rb') as f:
    X_census_train, X_census_test, Y_census_train, Y_census_test = pickle.load(f)

X_census_train.shape, X_census_test.shape, Y_census_train.shape, Y_census_test.shape


# ## Treinamento algoritmo logistic Classifier - Base Census

# In[30]:


logistic_census_data = LogisticRegression(random_state=1)
logistic_census_data.fit(X_census_train, Y_census_train)


# ## Previsões algoritmo logistic Classifier - Base census

# In[31]:


predict_census = logistic_census_data.predict(X_census_test)


# ## Estatísticas de acerto - Base census - 84.95%
# 
# Acurácia:

# In[32]:


cm = ConfusionMatrix(logistic_census_data)
cm.fit(X_census_train, Y_census_train)
cm.score(X_census_test, Y_census_test)


# Relatório de Classificação:

# In[33]:


print((classification_report(Y_census_test, predict_census)))


