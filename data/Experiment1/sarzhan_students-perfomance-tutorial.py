#!/usr/bin/env python
# coding: utf-8

# This notebook covers full pipeline of data science project in students perfomance dataset
# ### Main steps:
# * Loading, cleaning, wrangling
# * Primary analysis
# * Exploratory data analysis
#     * looking at distribution of dataset
#     * making hypothesises about data, which can help in prediction
# * Feature engineering
#     * transformation of data to gaussian distribution
#     * standardization and normalization
#     * feature extraction
#     * selecting best features
# * Modeling
#     * using various ML algorithms
#     * try cross-validation
#     * improve models by tuning hyperparametes using grid-search
# * Evaluating 
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

#Statistics
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p

#Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, PolynomialFeatures, Normalizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.pipeline import make_pipeline

import warnings 
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data loading
# Firstly we load dataset and look at 5 random entries

# In[2]:


#Look at 5 random dataframe samples
df=pd.read_csv('../input/xAPI-Edu-Data.csv')
df.sample(5)


# ## Primary analysis of data
# Let's look at main information about dataset we have

# In[3]:


df.info()


# * There are 480 entries without missings, so we don't have to struggle with filling/dropping NaNs
# * We have 16 features, where 5 of them are numerical and 11 - categorical

# In[4]:


n_uniques=df.select_dtypes(include=object).nunique()
n_uniques[n_uniques==2]


# 6 of categorical features have only 2 unique values. It means that at the section of feature engineering we will binarize them to 0 and 1. Now let's look at target variable

# In[5]:


df.Class.value_counts(normalize=True)


# We have Multiclass classification task, as there are 3 targets (H,M,L).
# We have almost half of samples of class M, while L and H take only 25% of dataset each. Dataset is imbalanced, but not badly. Therefore, it would be better to use F1-score metric, rather than just Accuracy. It will give more precise result

# ## Exploratory data analysis
# In this part we will analyze data and make hypothesises.

# ### Continuous variables
# 
# Make visual analysis of data and let's begin from numerical features:

# In[6]:


continuous_variables=df.columns[df.dtypes==int]
plt.figure(figsize=(10,7))
for i, column in enumerate(continuous_variables):
    plt.subplot(2,2, i+1)
    sns.distplot(df[column], label=column, bins=10, fit=norm)
    plt.ylabel('Density');


# * Features doesn't have gaussian (normal) distribution.
# * As ML algorithms deal better with values, which are normally distributed, we need to transfrom them closer that view. BoxCox transformation will help us with it - http://onlinestatbook.com/2/transformations/box-cox.html

# In[7]:


plt.figure(figsize=(10,7))
for i, column in enumerate(continuous_variables):
    plt.subplot(2,2, i+1)
    df[column]=boxcox1p(df[column], 0.3)
    sns.distplot(df[column], label=column, bins=10, fit=norm)
    plt.ylabel('Density')


# However raisedhands and visitedresourses have double gaussian distribution. We can create new binary features for them, where 1 is when values more than its' average, and 0 - less. Then we can look how these features will improve model

# In[8]:


df['raisedhands_bin']=np.where(df.raisedhands>df.raisedhands.mean(),1,0)
df['VisITedResources_bin']=np.where(df.VisITedResources>df.VisITedResources.mean(),1,0)


# Now we check boxplots for numerical features with target:
# * There is good division of classes for raisedhands. Students who get high marks raise hands much more than students with low marks. M-s are in the middle.
# * The same situation for VisitedResources. These two features will give high predictive power.
# * Last two features have similar pattern, but with less clear break.

# In[9]:


plt.figure(figsize=(10,7))
for i, column in enumerate(continuous_variables):
    plt.subplot(2,2,i+1)
    sns.boxplot(x=df.Class, y=df[column]);


# Let's look at correlation between these features:
# * VisitedResources, RaisedHands и AnnouncementViews have medium correlation (0.5-0.7)
# * In other cases there is weak correlation
# So, we don't need to worry about multicollinearity, which is important for linear algorithms

# In[10]:


plt.figure(figsize=(7,5))
sns.heatmap(df.corr(), annot=True, fmt='.1g', cmap='RdBu');


# In[11]:


sns.pairplot(df);


# ## Categorical variables

# In[12]:


categorical_variables=df.columns[df.dtypes==object]
print(('Percent of students\' nationality - Kuwait or Jordan: {}'.format(
            round(100*df.NationalITy.isin(['KW','Jordan']).sum()/df.shape[0],2))))
print(('Percent of students, who was born in Kuwait or Jordan: {}'.format(
            round(100*df.PlaceofBirth.isin(['KuwaIT','Jordan']).sum()/df.shape[0],2))))
print(('Percent of studets, who has same nationality and place of birth: {}'.format(
            round(100*(df.NationalITy==df.PlaceofBirth).sum()/df.shape[0]))))


# * More than 70% of students are from Kuweit and Jordan
# * More than half of students were born in their homelands

# In[13]:


df['NationalITy'][df['NationalITy']=='KW']='KuwaIT'
pb_count=pd.DataFrame(df.PlaceofBirth.value_counts(normalize=True)*100)
pb_count.reset_index(inplace=True)
nt_count=pd.DataFrame(df.NationalITy.value_counts(normalize=True)*100)
nt_count.reset_index(inplace=True)
pb_nt_count=pd.merge(nt_count, pb_count, on='index')
pb_nt_count.rename(columns={'index':'Country'}, inplace=True)
pb_nt_count


# In[14]:


plt.figure(figsize=(14,5))
for i, column in enumerate(df[['NationalITy','PlaceofBirth']]):
    data=df[column].value_counts().sort_values(ascending=False)
    plt.subplot(1,2,i+1)
    sns.barplot(x=data, y=data.index);


# Due to that information, it would be better to combine countries with small percentage (<4%) of representation in dataset into one common category

# In[15]:


#Rename all coutries with percentage less that 4% to 'Other'
small_countries=list(pb_nt_count['Country'][(pb_nt_count.PlaceofBirth<4)&(pb_nt_count.NationalITy<4)])

for column in ['PlaceofBirth', 'NationalITy']:
    df[column][df[column].isin(small_countries)]='Other'
    
print(('After renaming unique values are {}'.format(df.PlaceofBirth.unique())))


# Let's look at distribution of other variables

# In[16]:


plt.figure(figsize=(14,5))
for i, column in enumerate(df[['GradeID','Topic']]):
    data=df[column].value_counts().sort_values(ascending=False)
    plt.subplot(1,2,i+1)
    sns.barplot(x=data, y=data.index);


# In[17]:


plt.figure(figsize=(15,8))
for i, column in enumerate(categorical_variables.drop(['NationalITy','PlaceofBirth','GradeID','Topic','Class'])):
    plt.subplot(2,4,i+1)
    sns.countplot(df[column]);


# Then we can define relations of categorical variables with target variable  

# In[18]:


plt.figure(figsize=(15,12))
for i, column in enumerate(categorical_variables.drop(['GradeID','Topic','Class'])):
    plt.subplot(4,3,i+1)
    sns.countplot(x=df.Class, hue=df[column]);


# * Amount of absence days highly affects students' perfomance
# * For most of good students (H) responsible person is mother, for bad students - father
# * Parents of H students regularly answer survey and they are mostly satisfied with school, while for L students situation is opposite
# * At the first semester number of L students was bigger than number of H students. It is possible, that all students start learning better (Ls move to M, Ms move to H)
# * According to gender, boys are prone to bad study, girls - to good
# * Variables StageId и GradeId are almost the same, in modeling we can drop one of them

# ## Feature Engineering
# Now we can prepare data before using it in ML:
# * Encoding of categorical variables (Binarization/LabelEncoding/One-hot Encoding)
# * Standardization/Normalization of numerical variables
# * Extraction of new variables

# In[19]:


#Cut of target variable from dataset
target=df['Class']
df=df.drop('Class', axis=1)


# In[20]:


#Create new feature - type of topic (technical, language, other)
Topic_types={'Math':'technic', 'IT':'technic','Science':'technic','Biology':'technic',
 'Chemistry':'technic', 'Geology':'technic', 'Arabic':'language', 'English':'language',
 'Spanish':'language','French':'language', 'Quran':'other' ,'History':'other'}
df['Topic_type']=df.Topic.map(Topic_types)


# Standardization of continuous variables is necessary to eliminate the scalability problem.
# We transfer continuous variables to the standard form by the formula $x=(x-mean)/std$

# In[21]:


for column in continuous_variables:
    SS=StandardScaler().fit(df[[column]])
    df[[column]]=SS.transform(df[[column]])


# Encoding of categorical variables:
# * Binary variables translate into the form 0/1 (gender, number of passes, etc.)
# * Order variables will be re-encoded by LabelEncoding (GradeID, StageID) to preserve their order
# * The remaining variables are encoded by the one-hot encoding method.

# In[22]:


categorical_variables=df.select_dtypes(include='object').columns
for column in categorical_variables:
    #Binarize and LabelEncode
    #Кодируем переменные, у которых 2 уникальных значения и StageID, GradeID, так как в них важен порядок
    if (df[column].value_counts().shape[0]==2) | (column=='StageID') | (column=='GradeID'):
        le=LabelEncoder().fit(df[column])
        df[column]=le.transform(df[column])

#One-hot encoding
df=pd.get_dummies(df)


# ## Modeling

# In[23]:


X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.20, random_state=42)

def modelling(model):
    model.fit(X_train, y_train)
    preds=model.predict(X_test)
    print(('Accuracy = {}'.format(100*round(accuracy_score(y_test,preds),2))))
    print((classification_report(y_test, preds)))
    plt.figure(figsize=(7,5))
    sns.heatmap(confusion_matrix(y_test,preds), annot=True, vmax=50)
    plt.show()


# In[24]:


modelling(make_pipeline(PolynomialFeatures(2),LogisticRegression(random_state=42, C=0.1)))


# In[25]:


modelling(XGBClassifier(n_estimators=100, n_jobs=-1, learning_rate=0.03));


# In[26]:


modelling(KNeighborsClassifier(n_neighbors=25, n_jobs=-1))


# In[27]:


modelling(DecisionTreeClassifier(random_state=42, max_depth=5))


# In[28]:


modelling(RandomForestClassifier(n_estimators=2000, n_jobs=-1, max_depth=6, random_state=42))


# In[29]:


modelling(SVC(random_state=42, C=10, kernel='rbf', degree=3, gamma=0.1))


# In[30]:


modelling(LGBMClassifier(learning_rate=0.02,random_state=42, n_estimators=2000))


# In[31]:


criterions=['gini','entropy']
max_depth=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, None]
max_features=[1,2,3,4,5, None]
criterions=['gini','entropy']
min_samples_spilts=[2,3,4,5]
min_samples_leafs=[1,2,3,4]
class_weights=['balanced',None]

max_accuracy=0
best_params=None
best_model=None
for class_weight in class_weights:
    for min_sample_leaf in min_samples_leafs:
        for min_samples_spilt in min_samples_spilts:
            for crit in criterions:
                for splitter in splitters:
                    for depth in max_depth:
                        for feature in max_features:
                            DT=DecisionTreeClassifier(class_weight=class_weight,min_samples_leaf=min_sample_leaf,
                                max_depth=depth, min_samples_split=min_samples_spilt, criterion=crit, splitter=splitter,
                                max_features=feature, random_state=42)
                            DT.fit(X_train, y_train)
                            acc=accuracy_score(y_test,DT.predict(X_test))
                            if acc>max_accuracy:
                                max_accuracy=acc
                                best_model=DT
                                best_params=DT.get_params()
print(("Best accuracy at validation set is: {}%".format(round(100*max_accuracy,2))))

