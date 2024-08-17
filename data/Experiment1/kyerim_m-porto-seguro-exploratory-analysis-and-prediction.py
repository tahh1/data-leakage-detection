#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns', 100)


# # Load the data

# In[ ]:


trainset = pd.read_csv('../input/train.csv')
testset = pd.read_csv('../input/test.csv')


# 데이터 관찰
# - 몇 개의 그룹이 정의되고 그 그룹에 속하는 feature의 이름 패턴 ind - individual, reg - registration, car - self-explantory, calc – calculated 이 있다.
# - bin – binary features
# - cat – categorical features
# - bin, cat이 없는 features – 정수(서수 값)의 실수(연속적인 값)
# - -1은 누락 값
# - 예측 대상 값은 target 열에 있다. 이 값은 피보험자에게 천구가 제기되었는지 여부를 나타낸다.
# - id는 데이터 입력 서수
# 

# In[ ]:


trainset.head(20)


# In[ ]:


print(("Train dataset (rows, cols):",trainset.shape, "\nTest dataset (rows, cols):",testset.shape))


# training dataset = testing dataset + 'target'
# 
# 확인 ↓

# In[ ]:


print(("Columns in train and not in test dataset:",set(trainset.columns)-set(testset.columns)))


# # Introduction of metadata
# 
# To make easier the manipulation of data, we will associate few meta-information to the variables in the trainset. This will facilitate the selection of various types of features for analysis, inspection or modeling. We are using as well a **category** field for the `car`, `ind`, `reg` and `calc` types of features.
# 
# What metadata will be used:
# 
# * **use**: input, ID, target
# * **type**: nominal, interval, ordinal, binary
# * **preserve**: True or False
# * **dataType**: int, float, char
# * **category**: ind, reg, car, calc   
# 

# In[ ]:


# uses code from https://www.kaggle.com/bertcarremans/data-preparation-exploration (see references)
data = []
for feature in trainset.columns:
    # Defining the role
    if feature == 'target':
        use = 'target'
    elif feature == 'id':
        use = 'id'
    else:
        use = 'input'
         
    # Defining the type
    if 'bin' in feature or feature == 'target':
        type = 'binary'
    elif 'cat' in feature or feature == 'id':
        type = 'categorical'
    elif trainset[feature].dtype == float or isinstance(trainset[feature].dtype, float):
        type = 'real'
    elif trainset[feature].dtype == int:
        type = 'integer'
        
    # Initialize preserve to True for all variables except for id => ???
    preserve = True
    if feature == 'id':
        preserve = False
    
    # Defining the data type 
    dtype = trainset[feature].dtype
    
    category = 'none'
    # Defining the category
    if 'ind' in feature:
        category = 'individual'
    elif 'reg' in feature:
        category = 'registration'
    elif 'car' in feature:
        category = 'car'
    elif 'calc' in feature:
        category = 'calculated'
    
    
    # Creating a Dict that contains all the metadata for the variable
    feature_dictionary = {
        'varname': feature,
        'use': use,
        'type': type,
        'preserve': preserve,
        'dtype': dtype,
        'category' : category
    }
    data.append(feature_dictionary)
    
metadata = pd.DataFrame(data, columns=['varname', 'use', 'type', 'preserve', 'dtype', 'category'])
metadata.set_index('varname', inplace=True)
metadata


# We can extract, for example, all categorical values:

# In[ ]:


metadata[(metadata.type == 'categorical') & (metadata.preserve)].index


# Let's inspect all features, to see how many category distinct values do we have:

# In[ ]:


pd.DataFrame({'count' : metadata.groupby(['category'])['category'].size()}).reset_index()


# We have 20 *calculated* features, 16 *car*, 18 *individual* and 3 *registration*.
# 
# Let's inspect now all features, to see how many use and type distinct values do we have:

# In[ ]:


pd.DataFrame({'count' : metadata.groupby(['use', 'type'])['use'].size()}).reset_index()


# There are one nominal feature (the **id**), 20 binary values, 21 real (or float numbers), 16 categorical features - all these being as well **input** values and one **target** value, which is as well **binary**, the **target**. => 먼소리양

# # Data analysis and statistics
# 
# 

# ## Target variable

# In[ ]:


plt.figure()
fig, ax = plt.subplots(figsize=(6,6))
x = trainset['target'].value_counts().index.values
y = trainset["target"].value_counts().values
# Bar plot
# Order the bars descending on target mean
sns.barplot(ax=ax, x=x, y=y)
plt.ylabel('Number of values', fontsize=12)
plt.xlabel('Target value', fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


# Only 3.64% of the target data have 1 value. => training dataset 불균형
# 
# target = 0 으로 언더샘플링 or target = 1 로 오버샘플을 언더 샘플링 한다.
# 
# dataset이 크므로 target = 0 언더 샘플링을 한다.

# ## Real features

# In[ ]:


variable = metadata[(metadata.type == 'real') & (metadata.preserve)].index
trainset[variable].describe()


# In[ ]:


(pow(trainset['ps_car_12']*10,2)).head(10)


# In[ ]:


(pow(trainset['ps_car_15'],2)).head(10)


# ### 누락된 값이 있는 features
# 
# ps_reg_o3, ps_car_12, ps_car_14 의 최소값은 -1이므로 값이 누락되었다.
# 
# 
# ### Registration features
# 
# ps_reg_01, ps_reg_02는 분모가 10인 분수이다. (값 0.1, 0.2, 0.3)
# 
# ### Car features
# 
# **ps_car_12** 은 (10으로 나눈 값의) 자연수의 제곱근이고, **ps_car_15**는 자연수의 제곱근이다. *pairplot*을 사용해 값을 나타낸다.
# 
# 

# In[ ]:


sample = trainset.sample(frac=0.05)
var = ['ps_car_12', 'ps_car_15', 'target']
sample = sample[var]
sns.pairplot(sample,  hue='target', palette = 'Set1', diag_kind='kde')
plt.show()


# 무엇???
# 
# ### Calculated features
# 
# The features **ps_calc_01**, **ps_calc_02** and **ps_calc_03** have very similar distributions and could be some kind of ratio, since the maximum value is for all three 0.9. The other calculated values have maximum value an integer value (5,6,7, 10,12). 

# Let's visualize the real features distribution using density plot.

# In[ ]:


var = metadata[(metadata.type == 'real') & (metadata.preserve)].index
i = 0
t1 = trainset.loc[trainset['target'] != 0]
t0 = trainset.loc[trainset['target'] == 0]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(3,4,figsize=(16,12))

for feature in var:
    i += 1
    plt.subplot(3,4,i)
    sns.kdeplot(t1[feature], bw=0.5,label="target = 1")
    sns.kdeplot(t0[feature], bw=0.5,label="target = 0")
    plt.ylabel('Density plot', fontsize=12)
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


# **ps_reg_02**, **ps_car_13**, **ps_car_15** shows the most different distributions between sets of values associated with `target=0` and `target=1`.

# Let's visualize the correlation between the real features

# In[ ]:


def corr_heatmap(var):
    correlations = trainset[var].corr()

    # Create color map ranging between two colors
    cmap = sns.diverging_palette(50, 10, as_cmap=True)

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .75})
    plt.show();
    
var = metadata[(metadata.type == 'real') & (metadata.preserve)].index
corr_heatmap(var)


# Let's visualize the plots of the variables with strong correlations. These are:
# 
# * ps_reg_01 with ps_reg_02 (0.47);  
# * ps_reg_01 with ps_reg_03 (0.64);  
# * ps_reg_02 with ps_reg_03 (0.52);  
# * ps_car_12 with ps_car_13 (0.67);  
# * ps_car_13 with ps_car_15 (0.53);  
# 
# 
# 상관 관계가있는 값 쌍을 표시 pairplot을 사용,
# 샘플에서 2 % 만 사용하여 데이터를 서브 샘플링합니다.
# 
# 

# In[ ]:


sample = trainset.sample(frac=0.05)
var = ['ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_12', 'ps_car_13', 'ps_car_15', 'target']
sample = sample[var]
sns.pairplot(sample,  hue='target', palette = 'Set1', diag_kind='kde')
plt.show()


# # Integer features

# In[ ]:


variable = metadata[(metadata.type == 'integer') & (metadata.preserve)].index
trainset[variable].describe()

def corr_heatmap(var):
    correlations = trainset[var].corr()

    # Create color map ranging between two colors
    cmap = sns.diverging_palette(50, 10, as_cmap=True)

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .75})
    plt.show();
    
corr_heatmap(variable)


# # Binary features
# 
# 

# In[ ]:


v = metadata[(metadata.type == 'binary') & (metadata.preserve)].index
trainset[v].describe()


# Let's plot the distribution of the binary data in the training dataset. 
# 
# 파란색 - 0의 비율, 빨간색 - 1의 비율

# In[ ]:


bin_col = [col for col in trainset.columns if '_bin' in col]
zero_list = []
one_list = []
for col in bin_col:
    zero_list.append((trainset[col]==0).sum()/trainset.shape[0]*100) # 0의 비율
    one_list.append((trainset[col]==1).sum()/trainset.shape[0]*100) # 1의 비율
plt.figure()
fig, ax = plt.subplots(figsize=(6,6))
# Bar plot
p1 = sns.barplot(ax=ax, x=bin_col, y=zero_list, color="blue")
p2 = sns.barplot(ax=ax, x=bin_col, y=one_list, bottom= zero_list, color="red")
plt.ylabel('Percent of zero/one [%]', fontsize=12)
plt.xlabel('Binary features', fontsize=12)
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend((p1, p2), ('Zero', 'One'))
plt.show();


# **ps_ind_10_bin**, **ps_ind_11_bin**, **ps_ind_12_bin** and **ps_ind_13_bin** have very small number of  values `1` (lesss than 0.5%) whilst the number of  value `1` is very large for **ps_ind_16_bin** and **ps_cals_16_bin** (more than 60%).
# 
# Let's see now the distribution of binary data and the corresponding values of **target** variable.
# 

# In[ ]:


var = metadata[(metadata.type == 'binary') & (metadata.preserve)].index
var = [col for col in trainset.columns if '_bin' in col]
i = 0
t1 = trainset.loc[trainset['target'] != 0]
t0 = trainset.loc[trainset['target'] == 0]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(6,3,figsize=(12,24))

for feature in var:
    i += 1
    plt.subplot(6,3,i)
    sns.kdeplot(t1[feature], bw=0.5,label="target = 1")
    sns.kdeplot(t0[feature], bw=0.5,label="target = 0")
    plt.ylabel('Density plot', fontsize=12)
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


# **ps_ind_06_bin**, **ps_ind_07_bin**, **ps_ind_16_bin**, **ps_ind_17_bin**  높은 불균형, **ps_ind_08_bin** 작은 불균형

# ## Categorical features

# 두 가지 방법으로 범주형 데이터 분표 표현
# 
# 먼저, 카테고리 당 target =1의 비율을 계산하고 막대그래프를 이용해 표현

# In[ ]:


var = metadata[(metadata.type == 'categorical') & (metadata.preserve)].index

for feature in var:
    fig, ax = plt.subplots(figsize=(6,6))
    # Calculate the percentage of target=1 per category value
    cat_perc = trainset[[feature, 'target']].groupby([feature],as_index=False).mean()
    cat_perc.sort_values(by='target', ascending=False, inplace=True)
    # Bar plot
    # Order the bars descending on target mean
    sns.barplot(ax=ax,x=feature, y='target', data=cat_perc, order=cat_perc[feature])
    plt.ylabel('Percent of target with value 1 [%]', fontsize=12)
    plt.xlabel(feature, fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show();


# Alternativelly we represent the `categorical` features using density plot. We select values with `target=0` and `target=1` and represent both density plots on the same graphic.

# In[ ]:


var = metadata[(metadata.type == 'categorical') & (metadata.preserve)].index
i = 0
t1 = trainset.loc[trainset['target'] != 0]
t0 = trainset.loc[trainset['target'] == 0]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(4,4,figsize=(16,16))

for feature in var:
    i += 1
    plt.subplot(4,4,i)
    sns.kdeplot(t1[feature], bw=0.5,label="target = 1")
    sns.kdeplot(t0[feature], bw=0.5,label="target = 0")
    plt.ylabel('Density plot', fontsize=12)
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


# **ps_car_03_cat**, **ps_car_05_cat** 이 불균형 젤 심함

# ## Data unbalance between train and test data 

# Let's compare the distribution of the features in the train and test datasets. 
# 
# We start with the `reg` or `registration` features.

# In[ ]:


var = metadata[(metadata.category == 'registration') & (metadata.preserve)].index

# Bar plot
sns.set_style('whitegrid')

plt.figure()
fig, ax = plt.subplots(1,3,figsize=(12,4))
i = 0
for feature in var:
    i = i + 1
    plt.subplot(1,3,i)
    sns.kdeplot(trainset[feature], bw=0.5, label="train")
    sns.kdeplot(testset[feature], bw=0.5, label="test")
    plt.ylabel('Distribution', fontsize=12)
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    #plt.setp(labels, rotation=90)
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


# All `reg` features shows well balanced train and test sets.
# 
# Let's continue with `car` features.

# In[ ]:


var = metadata[(metadata.category == 'car') & (metadata.preserve)].index

# Bar plot
sns.set_style('whitegrid')

plt.figure()
fig, ax = plt.subplots(4,4,figsize=(20,16))
i = 0
for feature in var:
    i = i + 1
    plt.subplot(4,4,i)
    sns.kdeplot(trainset[feature], bw=0.5, label="train")
    sns.kdeplot(testset[feature], bw=0.5, label="test")
    plt.ylabel('Distribution', fontsize=12)
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    #plt.setp(labels, rotation=90)
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


# From the `car` features, all variables looks well balanced between `train` and `test` set.
# 
# Let's look now to the `ind` (`individual`) values.

# In[ ]:


var = metadata[(metadata.category == 'individual') & (metadata.preserve)].index

# Bar plot
sns.set_style('whitegrid')

plt.figure()
fig, ax = plt.subplots(5,4,figsize=(20,16))
i = 0
for feature in var:
    i = i + 1
    plt.subplot(5,4,i)
    sns.kdeplot(trainset[feature], bw=0.5, label="train")
    sns.kdeplot(testset[feature], bw=0.5, label="test")
    plt.ylabel('Distribution', fontsize=12)
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    #plt.setp(labels, rotation=90)
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


# All `ind` features are well balanced between `train` and `test` sets.
# 
# Let's check now `calc` features.

# In[ ]:


var = metadata[(metadata.category == 'calculated') & (metadata.preserve)].index

# Bar plot
sns.set_style('whitegrid')

plt.figure()
fig, ax = plt.subplots(5,4,figsize=(20,16))
i = 0
for feature in var:
    i = i + 1
    plt.subplot(5,4,i)
    sns.kdeplot(trainset[feature], bw=0.5, label="train")
    sns.kdeplot(testset[feature], bw=0.5, label="test")
    plt.ylabel('Distribution', fontsize=12)
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    #plt.setp(labels, rotation=90)
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


# All `calc` features are well balanced between `train` and `test` sets. 
# 
# In reference [5] it is also noticed the well balancing between `train` and `test` sets. It is also suggested that `calc` features might be all engineered and actually not relevant. This can only be assesed by careful succesive elimination using `CV` score using one or more predictive models.
# 
# 

# # Check data quality

# 누락된 값이 있는 feature 검사

# In[ ]:


vars_with_missing = []

for feature in trainset.columns:
    missings = trainset[trainset[feature] == -1][feature].count()
    if missings > 0:
        vars_with_missing.append(feature)
        missings_perc = missings/trainset.shape[0]
        
        print(('Variable {} has {} records ({:.2%}) with missing values'.format(feature, missings, missings_perc)))
        
print(('In total, there are {} variables with missing values'.format(len(vars_with_missing))))


# # Prepare the data for model
# 
# 
# 
# 

# ### Drop **calc** columns
# 
# We also drop the **calc** columns, as recommended in [5]. These seems to be all engineered and, according to Dmitry Altukhov, he was able to improve his CV score while succesivelly removing all of them.
# 

# In[ ]:


col_to_drop = trainset.columns[trainset.columns.str.startswith('ps_calc_')]
trainset = trainset.drop(col_to_drop, axis=1)  
testset = testset.drop(col_to_drop, axis=1)  


# ### Drop variables with too many missing values
# 
# We select from the variables with missing values two, **ps_car_03_cat** and **ps_car_05_cat** to drop.

# In[ ]:


# Dropping the variables with too many missing values
vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
trainset.drop(vars_to_drop, inplace=True, axis=1)
testset.drop(vars_to_drop, inplace=True, axis=1)
metadata.loc[(vars_to_drop),'keep'] = False  # Updating the meta


# In[ ]:


# Script by https://www.kaggle.com/ogrellier
# Code: https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


# ### Replace ps_car_11_cat with encoded value
# 
# Using the **target_encode** function, we replace the **ps_car_11_cat** with an encoded value in both **train** and **test** datasets.
# 
# 

# In[ ]:


train_encoded, test_encoded = target_encode(trainset["ps_car_11_cat"], 
                             testset["ps_car_11_cat"], 
                             target=trainset.target, 
                             min_samples_leaf=100,
                             smoothing=10,
                             noise_level=0.01)
    
trainset['ps_car_11_cat_te'] = train_encoded
trainset.drop('ps_car_11_cat', axis=1, inplace=True)
metadata.loc['ps_car_11_cat','keep'] = False  # Updating the metadata
testset['ps_car_11_cat_te'] = test_encoded
testset.drop('ps_car_11_cat', axis=1, inplace=True)


# ### Balance target variable
# 
# The target variable is highly unbalanced. This can be improved by either undersampling values with **target = 0** or oversampling values with **target = 1**.  Because there is a rather large training set, we opt for the **undersampling**.

# In[ ]:


desired_apriori=0.10

# Get the indices per target value
idx_0 = trainset[trainset.target == 0].index
idx_1 = trainset[trainset.target == 1].index

# Get original number of records per target value
nb_0 = len(trainset.loc[idx_0])
nb_1 = len(trainset.loc[idx_1])

# Calculate the undersampling rate and resulting number of records with target=0
undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)
undersampled_nb_0 = int(undersampling_rate*nb_0)
print(('Rate to undersample records with target=0: {}'.format(undersampling_rate)))
print(('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0)))

# Randomly select records with target=0 to get at the desired a priori
undersampled_idx = shuffle(idx_0, random_state=314, n_samples=undersampled_nb_0)

# Construct list with remaining indices
idx_list = list(undersampled_idx) + list(idx_1)

# Return undersample data frame
trainset = trainset.loc[idx_list].reset_index(drop=True)


# ### Replace **-1** values with NaN
# 
# Most of the classifiers we would use have preety good strategies to manage missing (or NaN) values.
# 

# In[ ]:


trainset = trainset.replace(-1, np.nan)
testset = testset.replace(-1, np.nan)


# ### Dummify **cat** values
# 
# We will create dummy variables for the **categorical** (**cat**) features
# 

# In[ ]:


cat_features = [a for a in trainset.columns if a.endswith('cat')]

for column in cat_features:
    temp = pd.get_dummies(pd.Series(trainset[column]))
    trainset = pd.concat([trainset,temp],axis=1)
    trainset = trainset.drop([column],axis=1)
    
for column in cat_features:
    temp = pd.get_dummies(pd.Series(testset[column]))
    testset = pd.concat([testset,temp],axis=1)
    testset = testset.drop([column],axis=1)


# ### Drop unused and **target** columns
# 
# We separate the **id** and **target** (drop these columns)

# In[ ]:


id_test = testset['id'].values
target_train = trainset['target'].values

trainset = trainset.drop(['target','id'], axis = 1)
testset = testset.drop(['id'], axis = 1)


# Let's inspect the training and test sets:

# In[ ]:


print(("Train dataset (rows, cols):",trainset.values.shape, "\nTest dataset (rows, cols):",testset.values.shape))


# 
# # Prepare the model
# 
# ### Ensable class for cross validation and ensamble
# 
# Prepare an **Ensamble** class to split the data in KFolds, train the models and ensamble the results.
# 
# The class has an **init** method (called when an Ensamble object is created) that accepts 4 parameters:
# 
# * **self** - the object to be initialized  
# * **n_splits** - the number of cross-validation splits to be used  
# * **stacker** - the model used for stacking the prediction results from the trained base models    
# * **base_models** - the list of base models used in training  
# 
# A second method, **fit_predict** has four functions:
# * split the training data in **n_splits** folds;  
# * run the **base models** for each fold;  
# * perform prediction using each model;  
# * ensamble the resuls using the **stacker**;  
# 
# 
# 

# In[ ]:


class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=314).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]


                print(("Base model %d: fit %s model | fold %d" % (i+1, str(clf).split('(')[0], j+1)))
                clf.fit(X_train, y_train)
                cross_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
                print(("cross_score [roc-auc]: %.5f [gini]: %.5f" % (cross_score.mean(), 2*cross_score.mean()-1)))
                y_pred = clf.predict_proba(X_holdout)[:,1]                

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')
        # Calculate gini factor as 2 * AUC - 1
        print(("Stacker score [gini]: %.5f" % (2 * results.mean() - 1)))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:,1]
        return res


# ### Parameters for the base models
# 
# For the base models, we prepare three different LightGBM models and one XGB model. 
# 
# Each model is used to train the data (using as well cross-validation, with 3 folds).
# 

# In[ ]:


# LightGBM params
# lgb_1
lgb_params1 = {}
lgb_params1['learning_rate'] = 0.02
lgb_params1['n_estimators'] = 650
lgb_params1['max_bin'] = 10
lgb_params1['subsample'] = 0.8
lgb_params1['subsample_freq'] = 10
lgb_params1['colsample_bytree'] = 0.8   
lgb_params1['min_child_samples'] = 500
lgb_params1['seed'] = 314
lgb_params1['num_threads'] = 4

# lgb2
lgb_params2 = {}
lgb_params2['n_estimators'] = 1090
lgb_params2['learning_rate'] = 0.02
lgb_params2['colsample_bytree'] = 0.3   
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2
lgb_params2['num_leaves'] = 16
lgb_params2['seed'] = 314
lgb_params2['num_threads'] = 4

# lgb3
lgb_params3 = {}
lgb_params3['n_estimators'] = 1100
lgb_params3['max_depth'] = 4
lgb_params3['learning_rate'] = 0.02
lgb_params3['seed'] = 314
lgb_params3['num_threads'] = 4

# XGBoost params
xgb_params = {}
xgb_params['objective'] = 'binary:logistic'
xgb_params['learning_rate'] = 0.04
xgb_params['n_estimators'] = 490
xgb_params['max_depth'] = 4
xgb_params['subsample'] = 0.9
xgb_params['colsample_bytree'] = 0.9  
xgb_params['min_child_weight'] = 10
xgb_params['num_threads'] = 4


# ### Initialize the models with the parameters
# 
# We init the 3 base models and the stacking model. For the base models we are using the predefined parameters initialized above.
# 
# 

# In[ ]:


# Base models
lgb_model1 = LGBMClassifier(**lgb_params1)

lgb_model2 = LGBMClassifier(**lgb_params2)
       
lgb_model3 = LGBMClassifier(**lgb_params3)

xgb_model = XGBClassifier(**xgb_params)

# Stacking model
log_model = LogisticRegression()


# ### Initialize the ensambling object
# 
# Using Ensamble.init we init the stacking object
# 

# In[ ]:


stack = Ensemble(n_splits=3,
        stacker = log_model,
        base_models = (lgb_model1, lgb_model2, lgb_model3, xgb_model))  


# # Run the predictive models
# 
# 
# Calling the **fit_predict** method of **stack** object, we run the training of the base models, predict the **target** with each model, ensamble the results using the **stacker** model and output the stacked result.
# 

# In[ ]:


y_prediction = stack.fit_predict(trainset, target_train, testset)        


# # Prepare the submission
# 

# In[ ]:


submission = pd.DataFrame()
submission['id'] = id_test
submission['target'] = y_prediction
submission.to_csv('stacked.csv', index=False)


# # References

# [1] Porto Seguro Safe Driver Prediction, Kaggle Competition, https://www.kaggle.com/c/porto-seguro-safe-driver-prediction   
# [2] Bert Carremans, Data Preparation and Exploration, Kaggle Kernel, https://www.kaggle.com/bertcarremans/data-preparation-exploration   
# [3] Head or Tails, Steering Whell of Fortune - Porto Seguro EDA, Kaggle Kernel, https://www.kaggle.com/headsortails/steering-wheel-of-fortune-porto-seguro-eda   
# [4] Anisotropic, Interactive Porto Insights - A Plot.ly Tutorial, Kaggle Kernel, https://www.kaggle.com/arthurtok/interactive-porto-insights-a-plot-ly-tutorial  
# [5] Dmitry Altukhov, Kaggle Porto Seguro's Safe Driver Prediction (3rd place solution),  https://www.youtube.com/watch?v=mbxZ_zqHV9c  
# [6] Vladimir Demidov, Simple Staker LB 0.284, https://www.kaggle.com/yekenot/simple-stacker-lb-0-284  
# [7] Anisotropic, Introduction to Ensembling/Stacking in Python, https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python  
# 
# 
# 

# # Feedback
# 
# I will appreciate your suggestions and observations.
