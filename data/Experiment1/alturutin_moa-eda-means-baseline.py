#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os 
from collections import Counter
from tqdm import tqdm
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
data_path = "../input/lish-moa"
get_ipython().system('ls -l --block-size=M $data_path')


# In[2]:


get_ipython().run_cell_magic('time', '', 'train_features = pd.read_csv(os.path.join(data_path, "train_features.csv"))\ntrain_targets_s= pd.read_csv(os.path.join(data_path, "train_targets_scored.csv"))\ntrain_targets_n= pd.read_csv(os.path.join(data_path, "train_targets_nonscored.csv"))\ntest_features  = pd.read_csv(os.path.join(data_path, "test_features.csv"))\nsample_submission = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))\n\ntrain_features.set_index(\'sig_id\', inplace=True)\ntrain_targets_s.set_index(\'sig_id\', inplace=True)\ntest_features.set_index(\'sig_id\', inplace=True)\nsample_submission.set_index(\'sig_id\', inplace=True)\n\nassert np.all(test_features.index == sample_submission.index)\n')


# In[3]:


print(f"train shape {train_features.shape}")
print(f"test  shape {test_features.shape}")
print(f"train targets scored shape {train_targets_s.shape}")
print(f"train targets nonscored shape {train_targets_n.shape}")


# ## Data exploration

# ### As it mentioned in competition data overview, features consist of 2 huge groups (related to genes & cell)

# In[4]:


# features 
features = list(train_features.columns)
short_features = Counter([f[:2] for f in features])
short_features


# ### It's a multilabel task

# In[5]:


train_targets_s.sum(axis=1).value_counts().sort_index().plot(kind="bar", title="targets per record", figsize=(7,4));


# ### Pairs "agonist-antagonist" are not always mutually exclusive

# In[6]:


# detect pairs "agonist-antagonist"
pairs = Counter([t.replace("_antagonist", "").replace("_agonist", "") for t in train_targets_s.columns])
pairs = [t for t, count in list(pairs.items()) if count == 2]

# check if pairs "agonist-antagonist" is crossing
def filter_columns(df, s):
    columns = [c for c in df.columns if s in c]
    return df[columns]

for p in pairs:
    df = filter_columns(train_targets_s, p)
    boolean = np.all(df.sum(axis=1) < 2)
    if not boolean:
        print((p, len(df[df.sum(axis=1) == 2])))


# ### Labels are very unbalanced

# In[7]:


train_targets_s.mean(axis=0)[1:].plot(kind="hist", bins=50, title="classes distribution", figsize=(8,5));


# ### Inside control groups all targets is 0

# In[8]:


display(train_features.cp_type.value_counts(normalize=True))

train_features['targets_total'] = train_targets_s.sum(axis=1)
train_features.groupby('cp_type')['targets_total'].sum()


# ### Experiment features have no obvious connection with targets

# In[9]:


display(train_features.groupby('cp_time')['targets_total'].sum())
train_features.groupby('cp_dose')['targets_total'].sum()


# ### Continuous features exploration

# In[10]:


# 1. Features are continuous
assert np.all(filter_columns(train_features, "g-").dtypes == "float64")
assert np.all(filter_columns(train_features, "c-").dtypes == "float64")
# 2. Gene features belong range [-10, 10]
assert filter_columns(train_features, "g-").max().max() == 10
assert filter_columns(train_features, "g-").min().min() == -10
# 3. Cell features belong range [-10, <10]
assert filter_columns(train_features, "c-").max().max() < 10
assert filter_columns(train_features, "c-").min().min() == -10


# In[11]:


# means distribution
sns.distplot(filter_columns(train_features, "c-").mean(), label="cell")
sns.distplot(filter_columns(train_features, "g-").mean(), label="gene")
plt.legend();


# In[12]:


def distplots(df, columns, nrows=1, ncols=5, title="", figsize=(15,3)):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()
    fig.suptitle(title)
    for i, column in enumerate(columns):
        sns.distplot(df[column], ax=axes[i])
        axes[i].axes.yaxis.set_visible(False)
    # sns.despine(left=True)
    plt.show()

columns = filter_columns(train_features, "c-").iloc[:, [0,10,20,30,42,45,77]].columns.tolist()
distplots(train_features, columns, ncols=len(columns), figsize=(len(columns)*3, 3), title="Cell features example")
columns = filter_columns(train_features, "g-").iloc[:, [0,12,42,101,202,303,424]].columns.tolist()
distplots(train_features, columns, ncols=len(columns), figsize=(len(columns)*3, 3), title="Gene features example")


# * Looks like distributions of features close to normal (with small left and right tails)
# * Cell features skewed to left a little bit; perhaps, it would be useful to create the bucket of left tail for most important cell features

# ## Feature interactions

# In[13]:


def columnwise_corrcoef(m, v):
    """ 
    Vectorized columnwise pearson correlation
    Means of all columns should be equal 0
    m: matrix 
    v: vector
    """
    return (v @ m) / np.sqrt(np.sum(m ** 2, 0) * np.sum(v ** 2))

corr_features = train_features.columns[3:]
df_corr = train_features.copy().loc[train_features.cp_type=='trt_cp', corr_features].astype(np.float64)
df_corr = df_corr - df_corr.mean()
df_corr = df_corr.values


# In[14]:


corr = np.eye(len(corr_features), dtype=np.float32)
n = len(corr_features)
for i in tqdm(list(range(n))):
    corr[i] = columnwise_corrcoef(df_corr, df_corr[:, i])
assert np.all(np.diag(corr) == 1)
assert np.min(corr) >= -1
assert np.max(corr) <= 1
np.fill_diagonal(corr, 0)


# ### There's not much linear correlation between targets and features

# In[15]:


sns.distplot(corr[:, -1], bins=40)
sns.despine()
plt.xlim(-0.5, 0.5)
plt.title("target correlation");


# ### Cell features have huge correlation between each other

# In[16]:


# help(sns.heatmap)
plt.figure(figsize=(20, 15))
sns.heatmap(corr, 
            xticklabels=[],
            yticklabels=[],
            cmap="cividis"
           );


# In[17]:


cell_corr = corr[-100:, -100:].copy()
np.fill_diagonal(cell_corr, 1)
print(("cell features R =", cell_corr.mean()))


# ### Rare targets (<5 records) will always set to 0 in prediction

# In[18]:


targets_sum = train_targets_s.sum(axis=0)
rare_targets = targets_sum[targets_sum < 5].index.tolist()
train_targets_s.drop(rare_targets, axis=1, inplace=True)
rare_targets, train_targets_s.shape


# ## Validation split

# In[19]:


train_features.drop('targets_total', axis=1, inplace=True) # remove targets from train set


# ### Test and train `cp_type` distributions are similar: no need in upsampling/downsampling by experiment type

# In[20]:


assert np.all(train_features.index == train_targets_s.index)
test_features['cp_type'].value_counts(normalize=True)


# ### Preprocessing

# In[21]:


# onehot encoding experiment features
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
categorical_features = ['cp_type', 'cp_time', 'cp_dose']
categories = encoder.fit_transform(train_features[categorical_features])
categories = categories[:, 1:] # keeps only 1 column for cp_type

def join_columns(c1,c2,sep='__'):
    return sep.join([str(c1),str(c2)])
oh_features = [join_columns(f, v) for f, c in zip(categorical_features, encoder.categories_) for v in c][1:]

train_features.drop(categorical_features, axis=1, inplace=True)
train_features = pd.concat([
    pd.DataFrame(categories, index=train_features.index, columns=oh_features), 
    train_features
], axis=1)


# In[22]:


# make holdout fold
from sklearn.model_selection import train_test_split
X_train, X_hold, y_train, y_hold = train_test_split(
    train_features, train_targets_s, 
    test_size=0.25, 
    random_state=171, shuffle=True
)
assert y_hold.loc[:, y_hold.sum(axis=0)<1].shape[1] == 0

ctype_train, ctype_hold = X_train.iloc[:, 0].values, X_hold.iloc[:, 0].values
ctype_train= ctype_train.astype(bool)
ctype_hold = ctype_hold.astype(bool)
assert y_hold[~ctype_hold].sum().sum() == 0

y_hold = y_hold.values


# ## Baseline models

# In[23]:


X_train.drop('cp_type__trt_cp', axis=1, inplace=True)
X_hold.drop('cp_type__trt_cp', axis=1, inplace=True)


# In[24]:


# metrics 
from sklearn.metrics import log_loss

def sklearn_multilabel_logloss(y_true, y_pred):
    """ Sanity check """
    return np.mean([log_loss(y_true[:, i], y_pred[:, i]) for i in range(y_pred.shape[1])])

def logloss(y_true, y_pred, eps=1e-15):
    y_pred[y_pred==0] = eps
    y_pred[y_pred==1] = 1 - eps
    return - (y_true * np.log(y_pred) + (1 - y_true) * (np.log(1 - y_pred))).mean()


# ### Constant

# In[25]:


for c in np.arange(0.1, 0.51, 0.2):
    y_pred = np.full_like(y_hold, c, dtype=np.float64)
    y_pred[~ctype_hold] = 0
    print(f"constant={c:.1f} logloss={logloss(y_hold, y_pred):.5f}")

assert np.allclose(sklearn_multilabel_logloss(y_hold, y_pred) - logloss(y_hold, y_pred), 0)


# ### Means

# In[26]:


class MultilabelMeanEstimator:
    def __init__(self):
        self._means = None
    def fit(self, y_train):
        self._means = y_train.mean(axis=0).to_dict()
    def predict(self, test, zero_mask=None):
        assert self._means, "estimator isn't fitted"
        pred = np.zeros((len(test), len(self._means)), dtype=np.float64)
        for i, c in enumerate(test.columns):
            pred[:, i] = self._means[c]
        if zero_mask is not None:
            pred[zero_mask] = 0
        return pred
    


# In[27]:


estimator = MultilabelMeanEstimator()
estimator.fit(y_train)
test = pd.DataFrame(y_hold, columns=y_train.columns)
y_pred = estimator.predict(test, ~ctype_hold)
print(f"Means logloss = {logloss(y_hold, y_pred): .5f}")


# In[28]:


# return rare targets
train_targets_s[rare_targets] = 0

# fit on full train data
estimator = MultilabelMeanEstimator()
estimator.fit(train_targets_s)


# ## Prediction

# In[29]:


zero_mask = (test_features.cp_type == 'ctl_vehicle').values # set control group to 0
prediction = estimator.predict(sample_submission, zero_mask)
submission = pd.DataFrame(prediction, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv("submission.csv")

