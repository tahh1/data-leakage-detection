#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('CreditScoring.csv')
df.columns = df.columns.str.lower()


# In[4]:


status_values = {
    1: 'ok',
    2: 'default',
    0: 'unk'
}

df.status = df.status.map(status_values)


home_values = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unk'
}

df.home = df.home.map(home_values)

marital_values = {
    1: 'single',
    2: 'married',
    3: 'widow',
    4: 'separated',
    5: 'divorced',
    0: 'unk'
}

df.marital = df.marital.map(marital_values)

records_values = {
    1: 'no',
    2: 'yes',
    0: 'unk'
}

df.records = df.records.map(records_values)

job_values = {
    1: 'fixed',
    2: 'partime',
    3: 'freelance',
    4: 'others',
    0: 'unk'
}

df.job = df.job.map(job_values)


# In[5]:


for c in ['income', 'assets', 'debt']:
    df[c] = df[c].replace(to_replace=99999999, value=0)


# In[6]:


df = df[df.status != 'unk'].reset_index(drop=True)


# In[7]:


df['default'] = (df.status == 'default').astype(int)
del df['status']


# In[8]:


df.columns


# In[13]:


df.dtypes


# In[20]:


numerical = ['seniority', 'time', 'age','expenses','income','assets','debt','amount','price']


# In[14]:


df.head()


# In[16]:


categorical = ['home', 'marital', 'records','job']


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)


# In[22]:


y_train = df_train.default.values
y_val = df_val.default.values
y_test = df_test.default.values

del df_train['default']
del df_val['default']
del df_test['default']


# In[24]:


from sklearn.feature_extraction import DictVectorizer


# In[26]:


from sklearn.linear_model import LogisticRegression


# In[67]:


dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

model = LogisticRegression()
model.fit(X_train, y_train)


# In[68]:


X_train.shape


# In[28]:


val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)[:, 1]
default_decision = (y_pred >= 0.5)
(y_val == default_decision).mean()


# In[29]:


len(y_val)


# In[30]:


(y_val == default_decision).mean()


# In[31]:


actual_positive = (y_val == 1)
actual_negative = (y_val == 0)


# In[32]:


t = 0.5
predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)


# In[33]:


tp = (predict_positive & actual_positive).sum()
tn = (predict_negative & actual_negative).sum()

fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()


# In[34]:


confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])
confusion_matrix


# In[35]:


(confusion_matrix / confusion_matrix.sum()).round(2)


# In[36]:


from sklearn.metrics import roc_curve


# In[37]:


fpr, tpr, thresholds = roc_curve(y_val, y_pred)


# In[38]:


plt.figure(figsize=(5, 5))

plt.plot(fpr, tpr, label='Model')
plt.plot([0, 1], [0, 1], label='Random', linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()


# In[39]:


from sklearn.metrics import auc


# In[40]:


auc(fpr, tpr)


# In[106]:


bnumerical = ['seniority']


# In[107]:


dv = DictVectorizer(sparse=False)

train_dict = df_train[bnumerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

model = LogisticRegression()
model.fit(X_train, y_train)


# In[108]:


val_dict = df_val[bnumerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)[:, 1]
default_decision = (y_pred >= 0.5)
(y_val == default_decision).mean()


# In[109]:


actual_positive = (y_val == 1)
actual_negative = (y_val == 0)
t = 0.5
predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)


# In[110]:


tp = (predict_positive & actual_positive).sum()
tn = (predict_negative & actual_negative).sum()

fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()


# In[111]:


confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])
confusion_matrix


# In[112]:


fpr, tpr, thresholds = roc_curve(y_val, y_pred)


# In[113]:


plt.figure(figsize=(5, 5))

plt.plot(fpr, tpr, label='Model')
plt.plot([0, 1], [0, 1], label='Random', linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()


# In[114]:


auc(fpr, tpr)


# In[115]:


numerical2 = ['seniority', 'income','assets']


# In[116]:


categorical2 = ['home', 'records','job']


# In[117]:


dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical2 + numerical2].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)


# In[118]:


val_dict = df_val[categorical2 + numerical2].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)[:, 1]
default_decision = (y_pred >= 0.5)
(y_val == default_decision).mean()


# In[119]:


actual_positive = (y_val == 1)
actual_negative = (y_val == 0)
t = 0.5
predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)


# In[120]:


tp = (predict_positive & actual_positive).sum()
tn = (predict_negative & actual_negative).sum()

fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()


# In[121]:


confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])
confusion_matrix


# In[122]:


fpr, tpr, thresholds = roc_curve(y_val, y_pred)


# In[123]:


plt.figure(figsize=(5, 5))

plt.plot(fpr, tpr, label='Model')
plt.plot([0, 1], [0, 1], label='Random', linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()


# In[124]:


auc(fpr, tpr)


# In[125]:


#Precision and Recall


# In[126]:


p = tp / (tp + fp)
p


# In[127]:


r = tp / (tp + fn)
r


# In[133]:


scores = []

thresholds = [0,0.1,0.2,0.3,0.4]

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)
    
    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()
    
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    
    scores.append((t, tp, fp, fn, tn, p, r))


# In[134]:


scores


# In[142]:


scores = []

thresholds = [0.09, 0.1, 0.11, 0.29, 0.3, 0.31, 0.49, 0.5, 0.51, 0.69, 0.7, 0.71]

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)
    
    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()
    
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p+r)
    
    scores.append((t, tp, fp, fn, tn, p, r,f1))


# In[143]:


scores


# In[144]:


from sklearn.model_selection import KFold


# In[145]:


get_ipython().system('pip install tqdm')


# In[146]:


from tqdm.auto import tqdm


# In[148]:


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical2 + numerical2].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model


# In[149]:


def predict(df, dv, model):
    dicts = df[categorical2 + numerical2].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# In[151]:


y_pred = predict(df_val, dv, model)


# In[152]:


from sklearn.metrics import roc_auc_score


# In[153]:


n_splits = 5

for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.default.values
        y_val = df_val.default.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print(('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores))))


# In[154]:


scores


# In[ ]:




