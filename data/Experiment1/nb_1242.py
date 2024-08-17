#!/usr/bin/env python
# coding: utf-8

# # Studying Confidence Intervals in Binary Classification
# 
# The following study was done using Wisconsin Breast Cancer Data

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from fastai.tabular import *


# Let's read the data.
# 
# I'm not sure what 'Unnamed: 32' means and it appears to be all NaNs so for the purpuse of this study I removed it all together since I'm not really interesting in building the most accurate model.

# In[2]:


df = pd.read_csv('breast.csv')
df = df.drop('Unnamed: 32', axis=1)
df.head()


# ## Let's create a random dataset to compare results

# In[3]:


x1 = np.random.rand(1, len(df))[0]
x2 = np.random.rand(1, len(df))[0]
y_aux = np.random.rand(1, len(df))[0]
y = y_aux > 0.5
rand_df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
rand_df[:5]


# ### Split between test and train sets

# In[4]:


y = df['diagnosis']
X = df.drop('diagnosis', axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

df_train = X_train.copy()
df_valid = X_valid.copy()
df_train.insert(loc=len(X_train.columns), column='diagnosis', value=y_train)
df_valid.insert(loc=len(X_valid.columns), column='diagnosis', value=y_valid)


# In[5]:


rand_y = rand_df['y']
rand_X = rand_df.drop('y', axis=1)
rand_X_train, rand_X_valid, rand_y_train, rand_y_valid = train_test_split(rand_X, rand_y, test_size=0.2, random_state=42)


# Let's take advantage of fastai's Categorify to turn our only categorical variable into one: 'diagnosis'

# In[6]:


cont, cat = cont_cat_split(df)
tfm = Categorify(cat, cont)
tfm(df_train)
tfm(df_valid, test=True)
df_train['diagnosis'].cat.categories


# 'M', the label we want to classify, is coded as 1
# 
# Let's also fill in the missing using fastai's brilliant FillMissing class

# In[7]:


tfm = FillMissing(cat, cont)
tfm(df_train)
tfm(df_valid, test=True)


# In[8]:


X_train = df_train.drop('diagnosis', axis=1)
y_train = df_train.diagnosis

X_valid = df_valid.drop('diagnosis', axis=1)
y_valid = df_valid.diagnosis


# ### We train a simple model with no feature engineering
# 
# Getting the best possible model is not the goal here. We simply want a good enough starting point to assert the model's confidence

# In[10]:


m = RandomForestClassifier(n_estimators=200, min_samples_leaf=1, max_features='sqrt', n_jobs=7, oob_score=True)

m.fit(X_train, y_train)

m.score(X_train,y_train), m.score(X_valid, y_valid), m.oob_score_


# Let's train another model on the random data

# In[11]:


rand_m = RandomForestClassifier(n_estimators=200, min_samples_leaf=1, max_features='sqrt', n_jobs=7, oob_score=True)

rand_m.fit(rand_X_train, rand_y_train)

rand_m.score(rand_X_train,rand_y_train), rand_m.score(rand_X_valid, rand_y_valid), rand_m.oob_score_


# After checking the validation set and the oob score making sure we are not overfitting, let's take a look at a more interesting metric for binary classification: F1 score

# In[12]:


from sklearn.metrics import f1_score, recall_score, precision_score

preds = m.predict(X_valid)
f1_score(y_valid, preds, pos_label='M', average='binary')


# In[13]:


rand_preds = rand_m.predict(rand_X_valid)
f1_score(rand_y_valid, rand_preds, pos_label=1, average='binary')


# The random model clearly has no good way of predicting the output, but the model using breast data is sorprisingly good considering no feature engineering was done. However let's see how confident we should be of each prediction.
# 
# For each input we calculate de prediction of each individual tree and we stack them

# In[15]:


preds = np.stack([t.predict(X_valid) for t in m.estimators_])

len(preds[:, 0]), len(preds[0]), len(X_valid), len(y_valid)


# In[16]:


rand_preds = np.stack([t.predict(rand_X_valid) for t in rand_m.estimators_])

len(rand_preds[:, 0]), len(rand_preds[0]), len(rand_X_valid), len(rand_y_valid)


# ## The Bootstrap Method
# 
# We set a number of iterations and define the sample size so we can resample each array of predictions.

# In[17]:


n_iterations = 300
n_size = int(len(preds) * 0.60)


# In[18]:


from sklearn.utils import resample

Lowers = []
Uppers = []

for i in range(len(preds[0])):
    
    means = []
    
    for _ in range(n_iterations):
        rs = resample(preds[:, i], n_samples=n_size, replace=True)
        means.append(np.mean(rs))
    
    alpha = 0.99
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(means, p))
    Lowers.append(lower)
    
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(means, p))
    Uppers.append(upper)


# And again, we do the same with our random model.

# In[19]:


rand_Lowers = []
rand_Uppers = []

for i in range(len(rand_preds[0])):
    
    means = []
    
    for _ in range(n_iterations):
        rs = resample(rand_preds[:, i], n_samples=n_size, replace=True)
        means.append(np.mean(rs))
    
    alpha = 0.99
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(means, p))
    rand_Lowers.append(lower)
    
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(means, p))
    rand_Uppers.append(upper)


# ### Now we have the upper and lower percentiles we can create a dataframe with statistical information about our predictions

# In[20]:


X = pd.DataFrame({'actuals': y_valid.cat.codes,
                  'preds': np.mean(preds, axis=0),
                  'std': np.std(preds, axis=0),
                  'var': np.var(preds, axis=0),
                  'upper': Uppers - np.mean(preds, axis=0),
                  'lower': np.mean(preds, axis=0) - Lowers
                 })
X.reset_index(inplace=True)
X = X.drop('index', axis=1)
print("Breast Data Model")
X[:10]


# In[21]:


rand_X = pd.DataFrame({'actuals': rand_y_valid.astype(int),
                  'preds': np.mean(rand_preds, axis=0),
                  'std': np.std(rand_preds, axis=0),
                  'var': np.var(rand_preds, axis=0),
                  'upper': rand_Uppers - np.mean(rand_preds, axis=0),
                  'lower': np.mean(rand_preds, axis=0) - rand_Lowers
                 })
rand_X.reset_index(inplace=True)
rand_X = rand_X.drop('index', axis=1)
print("Random Data Model")
rand_X[:10]


# Let's plot individual predictions using the average of the N trees and the percentiles as error bars.

# ### Let's plot individual predictions using the average of the N trees and the percentiles as error bars.
# 
# #### Breast Data Model

# In[22]:


import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Bar(
    name='Control',
    y=X['preds'][:50],
    error_y=dict(
            type='data',
            symmetric=False,
            array=X['upper'][:50],
            arrayminus=X['lower'][:50])))

fig.update_layout(shapes=[
    dict(type= 'line', yref='y', y0= 0.5, y1= 0.5, xref= 'x', x0= -1, x1= 50)])
fig.show()


# #### Random Data Model

# In[23]:


import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Bar(
    name='Control',
    y=rand_X['preds'][:50],
    error_y=dict(
            type='data',
            symmetric=False,
            array=rand_X['upper'][:50],
            arrayminus=rand_X['lower'][:50])))

fig.update_layout(shapes=[
    dict(type= 'line', yref='y', y0= 0.5, y1= 0.5, xref= 'x', x0= -1, x1= 50)])
fig.show()


# ## Uncertainty in our Models
# ### Just as a reminder, this was the F1 score for the hole validation dataset

# In[25]:


preds = m.predict(X_valid)
f1_score(y_valid, preds, pos_label='M', average='binary')


# ### And this was the F1 for our random data model

# In[26]:


rand_preds = rand_m.predict(rand_X_valid)
f1_score(rand_y_valid, rand_preds, pos_label=1, average='binary')


# If we now keep only those predictions whose average minus their lower percentile is greater than 0.6, and those whose average plus their upper percentile is smaller than 0.4 we can see the F1 score changes, going now to 98%

# In[29]:


aux = X.copy()
aux = aux.loc[(aux['preds'] - aux['lower'] >= 0.6) | (aux['preds'] + aux['upper'] <= 0.4), :]
a = np.array(aux.preds > 0.5)

aux['prediction'] = 0

aux.loc[aux.preds > 0.5, 'prediction'] = 1


f1_score(aux.actuals, aux.prediction, pos_label=1, average='binary'), len(aux), len(X), len(aux)/len(X)


# It's important to note that the 98% corresponds to 96% of the validation dataset, but the important part is that, **that 96% is not random**.
# 
# If we do the same but we adjust the lower and greater limits to 0.25 and 0.75 respectively we now get a perfect F1 score in our validation set

# In[30]:


aux = X.copy()
aux = aux.loc[(aux['preds'] - aux['lower'] >= 0.75) | (aux['preds'] + aux['upper'] <= 0.25), :]
a = np.array(aux.preds > 0.5)

aux['prediction'] = 0

aux.loc[aux.preds > 0.5, 'prediction'] = 1


f1_score(aux.actuals, aux.prediction, pos_label=1, average='binary'), len(aux), len(X), len(aux)/len(X)


# Instead we now see that this perfect F1 score represents only 87% of the validation set, but again, this 87% is not random. We know exactly which inputs represent this 87% since are the ones that our model trees classify with certain confidence. 
# So you may ask, what about the remaining 13% of our data? Well, simply put, our model is not entirely sure.
# 
# The questions we should now be asking ourselves are: Is this validation set big enough? Is this validation set representative of what the model is going to see in production?
# 
# ## Let's visualize the F1 in terms of a delta for lower and upper percentiles

# In[56]:


deltas = np.arange(0, 0.5, 0.05)
i = 0.5
f1s = []
per = []

for delta in deltas:    
    aux = X.copy()
    aux = aux.loc[(aux['preds'] - aux['lower'] >= i + delta) | (aux['preds'] + aux['upper'] <=  i - delta), :]
    a = np.array(aux.preds > 0.5)

    aux['prediction'] = 0

    aux.loc[aux.preds > 0.5, 'prediction'] = 1
    
    f1s.append(f1_score(aux.actuals, aux.prediction, pos_label=1, average='binary'))
    per.append(len(aux)/len(X))


# In[57]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=deltas, y=f1s, name="f1 data"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=deltas, y=per, name="percentage"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="F1 score and Validation set Percentage"
)

# Set x-axis title
fig.update_xaxes(title_text="delta")

# Set y-axes titles
fig.update_yaxes(title_text="<b>F1 Score</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>Validation Set Percentage</b>", secondary_y=True)

fig.show()


# ## And now we do the same for the random model

# In[52]:


deltas = np.arange(0, 0.5, 0.05)
i = 0.5
f1s = []
per = []

for delta in deltas:    
    aux = rand_X.copy()
    aux = aux.loc[(aux['preds'] - aux['lower'] >= i + delta) | (aux['preds'] + aux['upper'] <=  i - delta), :]
    a = np.array(aux.preds > 0.5)

    aux['prediction'] = 0

    aux.loc[aux.preds > 0.5, 'prediction'] = 1
    
    f1s.append(f1_score(aux.actuals, aux.prediction, pos_label=1, average='binary'))
    per.append(len(aux)/len(X))


# In[54]:


fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Scatter(x=deltas, y=f1s, name="f1 data"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=deltas, y=per, name="percentage"),
    secondary_y=True,
)

fig.update_layout(
    title_text="Double Y Axis Example"
)

fig.update_xaxes(title_text="delta")

fig.update_yaxes(title_text="<b>F1 Score</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>Validation Set Percentage</b>", secondary_y=True)

fig.show()


# And finally, is this enough to determine if our model is making a mistake? Certainly not, but I think this analysis is a step in the right direction.
# 
# What do you think?
# 
# Any insights are welcome, you can reach me at tomi.ambro94@gmail.com
