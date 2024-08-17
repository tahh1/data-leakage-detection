#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from numpy import vectorize as vec
import scipy as sp
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

"""
!wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
!chmod +x Miniconda3-latest-Linux-x86_64.sh
!bash ./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local
!conda install -q -y -c rdkit rdkit python=3.7
"""

import sys
sys.path.append('/usr/local/lib/python3.7/site-packages/')
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors,PandasTools
from rdkit.ML.Descriptors import MoleculeDescriptors


# In[3]:


mols = pd.read_csv('3CL_enzymatic_activity-clean.tsv', sep='\t', index_col=0)
mols.head(3)


# In[4]:


ccrf = mols[['washed_SMILES', 'Class']]
ccrf['Class'] = ccrf['Class'] == 'Active'
ccrf.head(3)


# In[5]:


Chem.PandasTools.AddMoleculeColumnToFrame(ccrf, smilesCol='washed_SMILES',
molCol='ROMol')


# In[7]:


names = [x[0] for x in Descriptors._descList]
print(("Number of descriptors in the rdkit: ", len(names)))
np.array(names)


# In[8]:


calculator = MoleculeDescriptors.MolecularDescriptorCalculator(names)
from collections import OrderedDict
desc = OrderedDict()
for mol in ccrf.index:
    desc[mol] = calculator.CalcDescriptors(ccrf.loc[mol, 'ROMol'])
desc_mols = pd.DataFrame.from_dict(desc, orient='index', columns=names)


# In[10]:


desc_mols.to_csv('3CL_descriptors.tsv', sep=' ')


# In[11]:


desc_mols = (desc_mols - desc_mols.mean())/ desc_mols.std()


# In[12]:


desc_mols = desc_mols.dropna(axis=1)


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(desc_mols, ccrf.Class,
train_size=0.9, test_size=0.1, random_state=0)
print("Training Data")
print(("Number of active molecules: ", list(y_train).count(1)))
print(("Number of inactive molecules: ", list(y_train).count(0)))
print("Test Data")
print(("Number of active molecules: ", list(y_test).count(1)))
print(("Number of inactive molecules: ", list(y_test).count(0)))


# In[14]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
from sklearn.metrics import classification_report
print((classification_report(y_test, model.predict(X_test))))


# In[25]:


train_data = pd.concat([X_train, y_train], axis=1)
active_data = pd.concat([train_data[train_data['Class'] == 1]]*(8244//274-1), ignore_index=True)
train_data  = pd.concat([train_data, active_data], ignore_index=True)
train_data =train_data.sample(frac=1).reset_index(drop=True)
len(train_data[train_data['Class'] == 1])


# In[28]:


train_data.head()


# In[26]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=0)
model.fit(train_data.iloc[:, :-1], train_data.Class)


# In[27]:


from sklearn.metrics import classification_report
print((classification_report(y_test, model.predict(X_test))))


# In[23]:


from sklearn.neural_network import MLPClassifier
nn_model = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', 
                         learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, 
                         random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


# In[24]:


nn_model.fit(train_data.iloc[:, :-1], train_data.Class)
print((classification_report(y_test, nn_model.predict(X_test))))


# In[ ]:




