#!/usr/bin/env python
# coding: utf-8

# - 만약 데이터 사이언스, 머신러닝 또는 캐글에서 어떤 것을 해야하는 지 잘 모르는 newbie 라면, 타이타닉을 하시는 게 가장 좋은 선택입니다.
# - 타이타닉은 아시다시피, 사상 최대 해난사고로써, 1,500여명의 희생자가 생겼습니다.
# - 우리는 이 튜토리얼에서 타이타닉에 탑승한 사람들의 신상정보를 활용하여, 승선한 사람들의 생존여부를 예측하는 모델을 생성할 것입니다.
# - 본 튜토리얼에서는 여러 시각화 도구(matplotlib, seaborn, plotly), 데이터 분석 도구(pandas, numpy), 머신 러닝 도구(sklearn)을 사용할 것입니다.
# - 본 튜토리얼은 캐글에 있는 여러 커널들을 참조하여 만들었습니다. 본 튜토리얼을 공부하신 뒤에, 캐글 타이타닉 컴퍼티션에 존재하는 여러 다른 커널들을 더 공부하시면 됩니다.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn')
sns.set(font_scale=2.5) 
# 이 두줄은 본 필자가 항상 쓰는 방법입니다. 
# matplotlib 의 기본 scheme 말고 seaborn scheme 을 세팅하고, 일일이 graph 의 font size 를 지정할 필요 없이 seaborn 의 font_scale 을 사용하면 편합니다.
import missingno as msno


get_ipython().run_line_magic('matplotlib', 'inline')


# 앞으로 우리가 해야할 프로세스는 대략 아래와 같습니다.
# 
# 1. 데이터셋 확인 - 대부분의 캐글 데이터들은 잘 정제되어 있습니다. 하지만 가끔 null data가 존재합니다. 이를 확인하고, 향후 수정합니다.
# 2. 탐색적 데이터 분석(exploratory data analysis) - 여러 feature 들을 개별적으로 분석하고, feature 들 간의 상관관계를 확인합니다. 여러 시각화 툴을 사용하여 insight를 얻습니다.
# 3. feature engineering - 모델을 세우기에 앞서, 모델의 성능을 높일 수 있도록 feature 들을 engineering 합니다. one-hot encoding, class로 나누기, 구간으로 나누기, 텍스트 데이터 처리 등을 합니다.
# 4. model 만들기 - sklearn 을 사용해 모델을 만듭니다. 파이썬에서 머신러닝을 할 때는 sklearn 을 사용하면 수많은 알고리즘을 일관된 문법으로 사용할 수 있습니다. 물론 딥러닝을 위해 tensorflow, pytorch 등을 사용할 수 도 있습니다.
# 5. 모델 학습 및 예측 - trainset 을 가지고 모델을 학습시킨 후, testset 을 가지고 prediction 합니다.
# 6. 모델 평가 - 예측 성능이 원하는 수준인지 판단합니다. 풀려는 문제에 따라 모델을 평가하는 방식도 달라집니다. 학습된 모델이 어떤 것을 학습하였는 지 확인해봅니다.

# -------------

# # 1. Dataset 확인

# - 파이썬에서 테이블화 된 데이터를 다루는 데 가장 최적화되어 있으며, 많이 쓰이는 라이브러리는 pandas 입니다.
# - 우리는 pandas 를 사용하여 데이터셋의 간단한 통계적 분석 부터, 복잡한 처리들을 간단한 메소드를 사용하여 해낼 수 있습니다.
# - 파이썬으로 데이터분석을 한다고 하면 반드시 능숙해져야 할 라이브러리이니, 여러 커널들을 공부하시면서 사용법에 익숙해지도록 반복 또 반복하시길 권장합니다.
# - 캐글에서 데이터셋은 보통 train, testset 으로 나뉘어 있습니다.

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# Any results you write to the current directory are saved as output.


# In[3]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')


# In[4]:


train.head()


# - 우리가 다루는 문제에서 feature는 Pclass, Age, SibSp, Parch, Fare 이며, 예측하려는 target label 은 Survived 입니다.

# - survival :	생존여부(target label 임)(1,0 으로 표현됨) //	integer
# - Pclass :	티켓의 클래스	1 = 1st, 2 = 2nd, 3 = 3rd // integer
# - sex :	성별(male, female 로 구분됨) //	string
# - Age :	나이 //	continuous	integer
# - sibSp :	함께 탑승한 형제와 배우자의 수 //	quantitative	integer
# - parch :	함께 탑승한 부모, 아이의 수 //	quantitative	integer
# - ticket :	티켓 번호 //	alphabat + integer	string
# - fare :	탑승료 //	continuous	float
# - cabin :	객실 번호 //	alphabat + integer	string
# - embared :	탑승 항구 //	C = Cherbourg, Q = Queenstown, S = Southampton	string

# In[5]:


train.describe()
# describe()를 쓰면 각 feature 가 가진 통계치들을 반환해줍니다.


# In[6]:


test.describe()


# --------------

# # 1.1 Null data check

# In[7]:


train.isnull().sum()/len(train)*100


# In[8]:


test.isnull().sum()/len(train)*100
# Train, Test set 에서 Age(둘다 약 20%), Cabin(둘다 약 80%), Embarked(Train만 0.22%) null data 존재하는 것을 볼 수 있습니다.
# MANO 라는 라이브러리를 사용하면 null data의 존재를 더 쉽게 볼 수 있습니다.


# In[9]:


msno.matrix(df=train, figsize=(8, 8), color=(0.8, 0.5, 0.2))
# Null 값을 시각화 해줍니다.


# In[10]:


msno.matrix(df=test, figsize=(8, 8), color=(0.8, 0.5, 0.2))


# ----------

# # 1.2 Target Label 확인
# - target label 이 어떤 distribution 을 가지고 있는 지 확인해봐야 합니다.
# - 지금 같은 binary classification 문제의 경우에서, 1과 0의 분포가 어떠냐에 따라 모델의 평가 방법이 달라 질 수 있습니다.
# 

# In[11]:


train.head()


# In[12]:


f, ax = plt.subplots(1, 2, figsize=(18, 8))

train['Survived'].value_counts().plot.pie( autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Pie plot - Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=train, ax=ax[1])
ax[1].set_title('Count plot - Survived')

plt.show()
# 안타깝게도 죽은 사람이 많습니다... 38.4 % 가 살아남았습니다.
# target label 의 분포가 제법 균일(balanced)합니다. 
# 불균일한 경우, 예를 들어서 100중 1이 99, 0이 1개인 경우에는 만약 모델이 모든것을 1이라 해도 정확도가 99%가 나오게 됩니다. 
# 0을 찾는 문제라면 이 모델은 원하는 결과를 줄 수 없게 됩니다. 지금 문제에서는 그렇지 않으니 계속 진행하겠습니다.


# ------------

# # [EDA] Pclass
# 
# - 먼저 Pclass 에 대해서 살펴보겠습니다. Pclass 는 ordinal, 서수형 데이터입니다. 카테고리이면서, 순서가 있는 데이터 타입입니다.
# - 먼저 Pclass 에 따른 생존률의 차이를 살펴보겠습니다. 엑셀의 피벗 차트와 유사한 작업을 하게 되는데, pandas dataframe 에서는 groupby 를 사용하면 쉽게 할 수 있습니다. 또한 pivot 이라는 메소드도 있습니다.
# - 'Pclass', 'Survived' 를 가져온 후, pclass 로 묶습니다. 그러고 나면 각 pclass 마다 0, 1 이 count가 되는데, 이를 평균내면 각 pclass 별 생존률이 나옵니다
# - 아래와 같이 count() 를 하면, 각 class 에 몇명이 있는 지 확인할 수 있으며, sum() 을 하면, 216 명중 생존한(survived=1)사람의 총합을 주게 됩니다

# In[13]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()
# 마치 피벗함수 같습니다.


# In[14]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()


# In[15]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar()
# 보다시피, Pclass 가 좋을 수록(1st) 생존률이 높은 것을 확인할 수 있습니다.
# 좀 더 보기 쉽게 그래프를 그려보겠습니다. seaborn 의 countplot 을 이용하면, 특정 label 에 따른 개수를 확인해볼 수 있습니다.


# In[16]:


y_position = 1.02
f, ax = plt.subplots(1, 2, figsize=(18, 8))
train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'], ax=ax[0])
ax[0].set_title('Number of Passengers By Pclass', y=y_position)
ax[0].set_ylabel('Count')
sns.countplot('Pclass', hue='Survived', data=train, ax=ax[1])
ax[1].set_title('Pclass: Survived vs Dead', y=y_position)
plt.show()
# 클래스가 높을 수록, 생존 확률이 높은걸 확인할 수 있습니다. Pclass 1, 2, 3 순서대로 63%, 48%, 25% 입니다
# 우리는 생존에 Pclass 가 큰 영향을 미친다고 생각해볼 수 있으며, 나중에 모델을 세울 때 이 feature 를 사용하는 것이 좋을 것이라 판단할 수 있습니다.


# https://kaggle-kr.tistory.com/17?category=868316#1.1-Null-data-check

# ---------------

# # [EDA] Sex
# 

# In[17]:


f, ax = plt.subplots(1, 2, figsize=(18, 8))
train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex', hue='Survived', data=train, ax=ax[1])
ax[1].set_title('Sex: Survived vs Dead')
plt.show()
# 이번에는 성별로 생존률이 어떻게 달라지는 지 확인해보겠습니다.
# 마찬가지로 pandas groupby 와 seaborn countplot 을 사용해서 시각화해봅시다.


# In[18]:


train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[19]:


pd.crosstab(train['Sex'], train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
# Pclass 와 마찬가지로, Sex 도 예측 모델에 쓰일 중요한 feature 임을 알 수 있습니다


# ------------------

# # [EDA] Both Sex and Pclass
# - 이번에는 Sex, Pclass 두가지에 관하여 생존이 어떻게 달라지는 지 확인해 봅시다
# - seaborn 의 factorplot 을 이용하면, 손쉽게 3개의 차원으로 이루어진 그래프를 그릴 수 있습니다.

# In[20]:


sns.factorplot('Pclass', 'Survived', hue='Sex', data=train, size=6, aspect=1.5)
# 모든 클래스에서 female 이 살 확률이 male 보다 높은 걸 알 수 있습니다.
# 또한 남자, 여자 상관없이 클래스가 높을 수록 살 확률 높습니다.
# 위 그래프는 hue 대신 column 으로 하면 아래와 같아집니다


# In[21]:


sns.factorplot(x='Sex', y='Survived', col='Pclass',
               data=train, satureation=.5,
               size=9, aspect=1
              )


# -----------

# # [EDA] Age
# - 이번에는 Age feature 를 살펴봅시다.

# In[22]:


print(('제일 나이 많은 탑승객 : ',round(train['Age'].max())))
print(('제일 어린 탑승객 : ',round(train['Age'].min())))
print(('탑승객 평균 나이 : ',round(train['Age'].mean())))


# - 생존 여부에 따른 Age의 histogram 을 그려보겠습니다.

# In[23]:


fig, ax = plt.subplots(1, 1, figsize=(9, 5))
sns.kdeplot(train[train['Survived'] == 1]['Age'], ax=ax)
sns.kdeplot(train[train['Survived'] == 0]['Age'], ax=ax)
plt.legend(['Survived == 1', 'Survived == 0'])
plt.show()
# 보시다시피, 생존자 중 나이가 어린 경우가 많음을 볼 수 있습니다.


# In[24]:


# Age distribution withing classes
plt.figure(figsize=(8, 6))
train['Age'][train['Pclass'] == 1].plot(kind='kde')
train['Age'][train['Pclass'] == 2].plot(kind='kde')
train['Age'][train['Pclass'] == 3].plot(kind='kde')

plt.xlabel('Age')
plt.title('Age Distribution within classes')
plt.legend(['1st Class', '2nd Class', '3rd Class'])
# Class 가 높을 수록 나이 많은 사람의 비중이 커짐
# 나이대가 변하면서 생존률이 어떻게 되는 지 보려고 합니다.
# 나이범위를 점점 넓혀가며, 생존률이 어떻게 되는지 한번 봅시다.


# In[25]:


cummulate_survival_ratio = []
for i in range(1, 80):
    cummulate_survival_ratio.append(train[train['Age'] < i]['Survived'].sum() / len(train[train['Age'] < i]['Survived']))
    
plt.figure(figsize=(7, 7))
plt.plot(cummulate_survival_ratio)
plt.title('Survival rate change depending on range of Age', y=1.02)
plt.ylabel('Survival rate')
plt.xlabel('Range of Age(0~x)')
plt.show()

# 보시다시피, 나이가 어릴 수록 생존률이 확실히 높은것을 확인할 수 있습니다.
# 우리는 이 나이가 중요한 feature 로 쓰일 수 있음을 확인했습니다.


# In[26]:


train.groupby(['Embarked','Pclass'])['Fare'].median() 


# # [EDA] Pclass, Sex, Age
# - 지금까지 본, Sex, Pclass, Age, Survived 모두에 대해서 보고싶습니다. 이를 쉽게 그려주는 것이 seaborn 의 violinplot 입니다.
# - x 축은 우리가 나눠서 보고싶어하는 case(여기선 Pclass, Sex) 를 나타내고, y 축은 보고 싶어하는 distribution(Age) 입니다.
# - 한번 그려보겠습니다

# In[27]:


f,ax=plt.subplots(1,2,figsize=(18,8))

#=== Pclass - Age 간 생존 여부 ===#
sns.violinplot("Pclass","Age", hue="Survived", data=train, scale='count', split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(list(range(0,110,10)))

#=== Sex - Age간 생존 여부 ===#
sns.violinplot("Sex","Age", hue="Survived", data=train, scale='count', split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(list(range(0,110,10)))
plt.show()

# 왼쪽 그림은 Pclass 별로 Age의 distribution 이 어떻게 다른지, 거기에 생존여부에 따라 구분한 그래프입니다.
# 오른쪽 그림도 마찬가지 Sex, 생존에 따른 distribution 이 어떻게 다른지 보여주는 그래프입니다.
# 생존만 봤을 때, 모든 클래스에서 나이가 어릴 수록 생존을 많이 한것을 볼 수 있습니다.
# 오른쪽 그림에서 보면, 명확히 여자가 생존을 많이 한것을 볼 수 있습니다.
# 여성과 아이를 먼저 챙긴 것을 볼 수 있습니다.


# ---------------------------------------------------------------------

# # [EDA] Embarked
# - Embarked 는 탑승한 항구를 나타냅니다.
# - 위에서 해왔던 것과 비슷하게 탑승한 곳에 따른 생존률을 보겠습니다.

# In[28]:


f, ax = plt.subplots(1, 1, figsize=(7, 7))
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax)

# 보시다시피, 조금의 차이는 있지만 생존률은 좀 비슷한 거 같습니다. 그래도 C가 제일 높군요.
# 모델에 얼마나 큰 영향을 미칠지는 모르겠지만, 그래도 사용하겠습니다.
# 사실, 모델을 만들고 나면 우리가 사용한 feature 들이 얼마나 중요한 역할을 했는지 확인해볼 수 있습니다. 이는 추후에 모델을 만들고 난 다음에 살펴볼 것입니다.
# 다른 feature 로 split 하여 한번 살펴보겠습니다


# In[29]:


f,ax=plt.subplots(2, 2, figsize=(20,15))
sns.countplot('Embarked', data=train, ax=ax[0,0])
ax[0,0].set_title('(1) No. Of Passengers Boarded')

sns.countplot('Embarked', hue='Sex', data=train, ax=ax[0,1])
ax[0,1].set_title('(2) Male-Female Split for Embarked')

sns.countplot('Embarked', hue='Survived', data=train, ax=ax[1,0])
ax[1,0].set_title('(3) Embarked vs Survived')

sns.countplot('Embarked', hue='Pclass', data=train, ax=ax[1,1])
ax[1,1].set_title('(4) Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()


# Figure(1) - 전체적으로 봤을 때, S 에서 가장 많은 사람이 탑승했습니다.
# Figure(2) - C와 Q 는 남녀의 비율이 비슷하고, S는 남자가 더 많습니다.
# Figure(3) - 생존확률이 S 경우 많이 낮은 걸 볼 수 있습니다. (이전 그래프에서 봤었습니다)
# Figure(4) - Class 로 split 해서 보니, C가 생존확률이 높은건 클래스가 높은 사람이 많이 타서 그렇습니다. S는 3rd class 가 많아서 생존확률이 낮게 나옵니다.


# -----

# # [EDA] Family - SibSp(형제 자매) + Parch(부모, 자녀)
# - SibSp와 Parch를 합하면 Family 가 될 것입니다. Family 로 합쳐서 분석해봅시다

# In[30]:


train['FamilySize'] = train['SibSp'] + train['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다


# In[31]:


print(("Maximum size of Family: ", train['FamilySize'].max()))
print(("Minimum size of Family: ", train['FamilySize'].min()))
# FamilySize 와 생존의 관계를 한번 살펴봅시다


# In[32]:


# 직접 해보세요
train['FamilySize'].mean()


# -----------

# # [EDA] Fare
# - Fare 는 탑승요금이며, contious feature 입니다. 한번 histogram 을 그려보겠습니다.

# In[33]:


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(train['Fare'], color='b', label='Skewness : {:.2f}'.format(train['Fare'].skew()), ax=ax)
g = g.legend(loc='best')

# 보시다시피, distribution이 매우 비대칭인 것을 알 수 있습니다.(high skewness). 
# 만약 이대로 모델에 넣어준다면 자칫 모델이 잘못 학습할 수도 있습니다. 
# 몇개 없는 outlier 에 대해서 너무 민감하게 반응한다면, 실제 예측 시에 좋지 못한 결과를 부를 수 있습니다.
# outlier의 영향을 줄이기 위해 Fare 에 log 를 취하겠습니다.


# In[34]:


train.isnull().sum()
# 만약 Log를 취하는데 비어있는 값이 있다면 에러가 발생할 수 있습니다.(실제로는 해보니까 안납니다....왜안나는지는 모르겠습니다.)
# 그래서 Null 값들을 확인해보니 Test의 Fare에서 Null 값이 1개가 있습니다. 해당 값은 평균으로 대체하겠습니다.


# In[35]:


test.isnull().sum()


# In[36]:


test.loc[test.Fare.isnull(), 'Fare'] = test['Fare'].mean() 
# testset 에 있는 nan value 를 평균값으로 치환합니다.


# In[37]:


test.isnull().sum()
# 잘 채워졌습니다.


# In[38]:


print((train['Fare'].min(), test['Fare'].min()))


# In[39]:


train['Fare'] = np.log1p(train['Fare'])
test['Fare'] = np.log1p(test['Fare'])
# 로그변환을 해줍니다.


# In[40]:


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(train['Fare'], color='b', label='Skewness : {:.2f}'.format(train['Fare'].skew()), ax=ax)
g = g.legend(loc='best')
# 잘 변환됐는지 확인합니다.

# log 를 취하니, 이제 비대칭성이 많이 사라진 것을 볼 수 있습니다.
# 우리는 이런 작업을 사용해 모델이 좀 더 좋은 성능을 내도록 할 수 있습니다.
# 사실 방금한 것은 feature engineering 에 들어가는 부분인데, 여기서 작업했습니다.
# 모델을 학습시키기 위해, 그리고 그 모델의 성능을 높이기 위해 feature 들에 여러 조작을 가하거나, 새로운 feature를 추가하는 것을 feature engineering 이라고 하는데, 우리는 이제 그것을 살펴볼 것입니다.


# 

# ------------------

# # [EDA] Cabin
# - 이 feature 는 NaN 이 대략 80% 이므로, 생존에 영향을 미칠 중요한 정보를 얻어내기가 쉽지는 않습니다.
# - 그러므로 우리가 세우려는 모델에 포함시키지 않도록 하겠습니다.

# In[41]:


train.isnull().sum() / len(train)


# In[42]:


train = train.drop(['Cabin'],1)
test = test.drop(['Cabin'],1)


# In[43]:


test.head()
# 직접 해보세요
# Train의 Null 개수에 전체 Row 개수를 나눠줌
# Cabin이 사라진 것을 확인


# ------------------

# # [EDA] Ticket
# - 이 feature 는 NaN 은 없습니다. 일단 string data 이므로 우리가 어떤 작업들을 해주어야 실제 모델에 사용할 수 있는데, 이를 위해선 사실 아이디어가 필요합니다.

# In[44]:


train.isnull().sum()/len(train)
# Ticket 값은 Null 값이 없음


# In[45]:


train.head()
# 다만 글자로 되어있어서 변환에 고민이 됨


# In[46]:


train['Ticket'].value_counts()
# Ticket의 종류별로 얼마나 있는지 확인 (자주 사용되는 함수)


# - 보시다시피, ticket number 는 매우 다양합니다. 우리는 여기서 어떤 특징을 이끌어내서 생존과 연결시킬 수 있을까요?
# - 여러분이 직접 한번 아이디어를 내보세요! 이것이 본격적인 캐글 레이스의 시작점입니다 ^^
# - 이 튜토리얼에서는 튜토리얼이니 일단 ticket 은 넘기도록 하겠습니다. 튜토리얼을 끝낸 후, 여러분의 모델의 성능을 향상시키기 위해 ticket 에서 정보를 이끌어내는 것도 좋겠네요!

# --------------
# --------------

# # Feature engineering
# - 본격적인 feature engineering 을 시작해보겠습니다.
# - 가장 먼저, dataset 에 존재하는 null data를 채우려고 합니다.
# - 아무 숫자로 채울 수는 없고, null data 를 포함하는 feature 의 statistics 를 참고하거나, 다른 아이디어를 짜내어 채울 수 있습니다.
# - null data 를 어떻게 채우느냐에 따라 모델의 성능이 좌지우지될 수 있기 때문에, 신경써줘야할 부분입니다.
# - Feature engineering 은 실제 모델의 학습에 쓰려고 하는 것이므로, train 뿐만아닐라 test 도 똑같이 적용해주어야 합니다. 잊지맙시다.
# - 원문 : https://kaggle-kr.tistory.com/18?category=868316

# --------

# # Fill Null in Age using title
# - Age 에는 null data가 177개나 있습니다. 이를 채울 수 있는 여러 아이디어가 있을 것인데, 여기서 우리는 title + statistics 를 사용해 보겠습니다.
# - 영어에서는 Miss, Mrr, Mrs 같은 title이 존재합니다. 각 탑승객의 이름에는 꼭 이런 title 이 들어가게 되는데 이를 사용해보겠습니다.
# - pandas series 에는 data 를 string 으로 바꿔주는 str method, 거기에 정규표현식을 적용하게 해주는 extract method가 있습니다. 이를 사용하여 title 을 쉽게 추출할 수 있습니다. title을 Initial column에 저장하겠습니다

# In[47]:


train.isnull().sum()


# In[48]:


test.isnull().sum()


# In[49]:


train


# In[50]:


train['Name'][1].split(',')[1].split('.')[0].strip()
# split('.')는 .을 기준으로 나눈는 것이며, Strip()는 앞뒤의 공백을 지워주는 코드임
# 해당 코드는 train['Name'] Column의 1번째 값을 -> , 기준으로 나눈것의 1번째 -> .기준으로 나눈것의 0번째 -> 앞뒤 공백 삭제 한 값 = Mrs 라고 해석할 수 있습니다.


# In[51]:


train['Name'].apply(lambda x : x.split(',')[1].split('.')[0].strip())
# 위에 코드는 한개의 단어에만 적용되므로, 이것을 Column 전체에 적용하기 위해서는 Apply(lambda x) 함수를 사용해야 합니다.
# 여기서는 위에서 만든 코드를 Name Column 각 값에 각각 적용하라는 의미로 받아들이면 됩니다.
# 잘 모르겠으면 Apply(lambda) 함수를 찾아보시면 됩니다.


# In[52]:


train['Initial'] = train['Name'].apply(lambda x : x.split(',')[1].split('.')[0].strip())
# 위에서 출력한 값들을 train['Initial'] Column에 넣어줍니다.


# In[53]:


test['Initial'] = test['Name'].apply(lambda x : x.split(',')[1].split('.')[0].strip())
# Test도 역시 같이 진행!


# In[54]:


train['Initial'].value_counts()
# 잘 바뀌었는지 확인합니다.


# In[55]:


train.groupby(['Sex', 'Initial'])['Initial'].count()
# 성별별 Initial을 봅니다. 근데 뭔가 알수없는 Initial들이 많네요 이걸 다 아는 것들로 바꿔봅시다(Mr, Mrs 등등)


# In[56]:


train.tail(10)


# In[57]:


train['Initial'] = train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                                            ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'])
test['Initial'] = test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                                          ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'])
# Replace 함수를 모르신다면 검색해보시고, 여기서는 각 매칭되는 값들을 바꿔줍니다.


# In[58]:


train.groupby(['Sex', 'Initial'])['Initial'].count()
# 잘 바뀌었는지 확인합니다.
# 근데 Mr이 Female로 잘못분류되었네요........일단은 그냥 넘어갑니다ㅜ


# In[59]:


train.loc[(train['Age'].isnull())&(train['Initial']=='Mr'),'Age'] = 33
train.loc[(train['Age'].isnull())&(train['Initial']=='Mrs'),'Age'] = 36
train.loc[(train['Age'].isnull())&(train['Initial']=='Master'),'Age'] = 5
train.loc[(train['Age'].isnull())&(train['Initial']=='Miss'),'Age'] = 22
train.loc[(train['Age'].isnull())&(train['Initial']=='Other'),'Age'] = 46

test.loc[(test['Age'].isnull())&(test['Initial']=='Mr'),'Age'] = 33
test.loc[(test['Age'].isnull())&(test['Initial']=='Mrs'),'Age'] = 36
test.loc[(test['Age'].isnull())&(test['Initial']=='Master'),'Age'] = 5
test.loc[(test['Age'].isnull())&(test['Initial']=='Miss'),'Age'] = 22
test.loc[(test['Age'].isnull())&(test['Initial']=='Other'),'Age'] = 46

# 위에 코드는 조금 어려워 보일 수 있습니다. 하지만 한개씩 해석해보도록 합시다.
# 각 스탭별로 코드들을 따라치면서 확인해보셔야 합니다

#=== 1STEP ===#
# 우선 train['Age'].isnull() 를 쳐보시면 Age의 Null 값만 True가 됩니다.
# train['Initial']=='Mr' 은 Train의 Initial이 Mr인 경우만 True가 됩니다.
# 위에 두개가 동시에 충족되는 경우만 True가 되려고 (train['Age'].isnull())&(train['Initial']=='Mr') 로 작성했습니다(중간에 And(&) 확인)

#=== 2STEP ===#
# train.loc[A,'B']는 Train의 A 행과 B열에 속하는 값을 찾아줍니다.
# 앞에서 Step으로 찾은 Index들을 Row로, 우리가 찾고자 하는 Age를 열로 지정해줍니다.
# --> train.loc[(train['Age'].isnull())&(train['Initial']=='Mr'),'Age']

#=== 3STEP ===#
# 뒤에 '= 33' 로 되어있는 것은 앞에 표시한 값(loc에서 찾은 값)들을 33으로 치환하라는 의미입니다.


# ## [참고] Null값을 채우는 다른 방법
# - 여기선 간단하게 Null을 채웠지만, 좀 더 다양한 방법을 쓴 예시들이 다른 커널에 존재합니다
# - https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling 보시면서 공부해보세요!
# - 이 외에도 다른 캐글러들의 커널을 보며 여러 참신한 아이디어를 살펴보세요!

# # Fill Null in Embarked

# In[60]:


train.isnull().sum()


# In[61]:


test.isnull().sum()


# In[62]:


train['Embarked'].value_counts()
# Embarked 는 Null value 가 2개이고, S 에서 가장 많은 탑승객이 있었으므로, 간단하게 Null 을 S로 채우겠습니다.


# In[63]:


train['Embarked'] = train['Embarked'].fillna('S')
# dataframe 의 fillna method 를 이용하면 쉽게 채울 수 있습니다. 여기서 inplace=True 로 하면 df_train 에 fillna 를 실제로 적용하게 됩니다


# In[64]:


train


# In[65]:


test.isnull().sum()


# ------------

# --------

# # Change Initial, Embarked and Sex (string to numerical)

# In[66]:


train['Sex'] = train['Sex'].replace(['male','female'],[0,1])
test['Sex'] = test['Sex'].replace(['male','female'],[0,1])
# Sex의 Male ->0, Female ->1로 바꿔줍니다.


# In[67]:


train['Sex'] = train['Sex'].astype('category')
test['Sex'] = test['Sex'].astype('category')
# 이 값은 Intiger가 아니므로, Category로 Type 변경해줍니다.


# In[68]:


train = pd.get_dummies(train, columns = ['Embarked'], prefix = 'Embarked')
test = pd.get_dummies(test, columns = ['Embarked'], prefix = 'Embarked')
print((train.shape, test.shape))
# Embarked는 One Hot Encoding을 해보도록 하죠, Prefix를 Embarked로 정해서 실행시켜봅니다.
# Train, Test 모두 잘되었는지 shape를 활용해 꼭 확인해보세요


# # Target Encoding
# 

# In[69]:


train_initial_groupby = train.groupby(['Initial'])['Survived'].mean().to_dict()
train_initial_groupby
# Target Encoding은 해당 Column과 y값과의 관계로 Encoding 하는 방법입니다.
# 우리는 Initial과 Survived가 관계가 있다고 생각하고, 두 값을 Pivotting 해봅시다.
# Pivotting한 값을 mapping 시키기위해 to_dict()함수를 사용합니다.


# In[70]:


train['Initial_mean'] = train['Initial'].map(train_initial_groupby)
test['Initial_mean'] = test['Initial'].map(train_initial_groupby)
# 위에서 to_dict한 값들을 train_initial_groupby로 집어넣었는데, 이걸 Map 함수를 활용해 바꿔줍니다.


# In[71]:


train.head()


# In[72]:


train_initial_groupby = train.groupby(['Initial'])['Survived'].std().to_dict()
train_initial_groupby
# 이번에는 Target Encoding을 std(표준편차)를 가지고 해봅시다


# In[73]:


train['Initial_std'] = train['Initial'].map(train_initial_groupby)
test['Initial_std'] = test['Initial'].map(train_initial_groupby)
# 위에서 했던것과 동일하게 하죠?


# In[74]:


train = train.drop(['Initial'],1)
test = test.drop(['Initial'],1)
# 이제 필요없는 Initial은 떠나보내면 됩니다.


# In[75]:


train.shape, test.shape


# In[76]:


train = train.drop(['Name', 'Ticket'],1)
test = test.drop(['Name', 'Ticket'],1)
# Name과 Ticket은 FE하기 귀찮으니까...날려버립니다.


# In[77]:


train.head()


# In[78]:


test.head()


# # Data Spilting

# In[79]:


y = train['Survived']
X = train.drop(['Survived'],1)


# In[80]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state = 30, test_size = 0.2)
# 저희가 잘 알고있는 train_test_split함수를 사용하여 데이터를 나눠줍니다.


# In[81]:


print((X_train.shape, y_train.shape, X_valid.shape,  y_valid.shape))
# 나눠주고 난 다음에는 잘 나눠졌는지 shape를 찍어봅니다.


# ----
# # Hyperparameter Tunning

# In[82]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[83]:


lr = LogisticRegression(C=1.0, intercept_scaling=1, random_state=None)


# In[84]:


best_score = 0

for i in [0.001, 0.01, 0.1, 1, 10, 100]:
    lr = LogisticRegression(C = i, intercept_scaling=1, random_state=None)
    
    lr.fit(X_train,y_train)
    pred_valid = lr.predict(X_valid)
    score = (pred_valid == y_valid).mean()
    print(('C = ',i,'  //  score : ', score))
    
    if score > best_score:
        best_score = score
        best_parameter = {'C':i, 'intercept_scaling':1, 'random_state':None}
    


# In[85]:


lr = LogisticRegression(**best_parameter)
lr.fit(X_train, y_train)


# ---
# # KFold
# - k개의 Fold를 만듬(전체를 dataset을 k개로 나눔(행 기준))
# - 나눈 dataset 중 번갈아가면서 Valid로 지정해줌
# - 각각의 Valid_score를 계산하면서 Learning을 시켜줌(전차수의 최적값을 다음차수로 이전하여 학습시킴)

# In[107]:


X.loc[[1,2,3,4,5]]
# X.loc[행,열]


# In[109]:


index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
train_index = [1,2,3,4,5,6,7,8,9,10]
test_index = [11,12,13,14,15]

X_tr = X.loc[train_index]
y_tr = y.loc[train_index]

X_vl = X.loc[test_index]
y_vl = y.loc[test_index]

model1 = LogisticRegression()
model1.fit(X_tr, y_tr)

pred_vl = model1.predict(X_vl)
score = ((pred_vl == y_vl).mean())
print(('Validation Score : ', score))

#=======================================

index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
train_index = [4,5,6,7,8,9,10]
test_index = [11,12,13,14,15]

X_tr = X.loc[train_index]
y_tr = y.loc[train_index]

X_vl = X.loc[test_index]
y_vl = y.loc[test_index]

model2 = LogisticRegression()
model2.fit(X_tr, y_tr)

pred_vl = model2.predict(X_vl)
score = ((pred_vl == y_vl).mean())
print(('Validation Score : ', score))

#=======================================

index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
train_index = [1,2,3,4,5,6,7,8,9,10]
test_index = [11,12,13,14,15]

X_tr = X.loc[train_index]
y_tr = y.loc[train_index]

X_vl = X.loc[test_index]
y_vl = y.loc[test_index]

model3 = LogisticRegression()
model3.fit(X_tr, y_tr)

pred_vl = model3.predict(X_vl)
score = ((pred_vl == y_vl).mean())
print(('Validation Score : ', score))

#=======================================

index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
train_index = [1,2,3,4,5,6,7,8,9,10]
test_index = [11,12,13,14,15]

X_tr = X.loc[train_index]
y_tr = y.loc[train_index]

X_vl = X.loc[test_index]
y_vl = y.loc[test_index]

model4 = LogisticRegression()
model4.fit(X_tr, y_tr)

pred_vl = model4.predict(X_vl)
score = ((pred_vl == y_vl).mean())
print(('Validation Score : ', score))

#=======================================

index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
train_index = [1,2,3,4,5,6,7,8,9,10]
test_index = [11,12,13,14,15]

X_tr = X.loc[train_index]
y_tr = y.loc[train_index]

X_vl = X.loc[test_index]
y_vl = y.loc[test_index]

model = LogisticRegression()
model.fit(X_tr, y_tr)

pred_vl = model.predict(X_vl)
score = ((pred_vl == y_vl).mean())
print(('Validation Score : ', score))

#=======================================


# In[120]:





# In[126]:





# In[151]:


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression

seed = 1019
splits = 7
cv_score = []
ttl_pred_test = np.zeros(len(test))  

kf = KFold(n_splits = splits, shuffle = False, random_state = seed)

for train_index, test_index in kf.split(X):
    X_tr, X_vl = X.loc[train_index], X.loc[test_index]
    y_tr, y_vl = y.loc[train_index], y.loc[test_index]
    
    model = LogisticRegression()
    model.fit(X_tr, y_tr)
    
    pred_vl = model.predict(X_vl)
    score = round(((pred_vl == y_vl).mean()),2)
    
    cv_score.append(score)    ##########
    
    
    pred_test = model.predict(test)
    
    ttl_pred_test = ttl_pred_test + pred_test/splits        
    
    print(('Validation Score : ', score))

submission['Survived'] = ttl_pred_test

sum(cv_score)/splits


# In[136]:


submission


# In[95]:


# from sklearn.model_selection import KFold, StratifiedKFold
# from sklearn.linear_model import LogisticRegression

# seed = 1019
# splits = 5
# cv_score =[]

# kf = KFold(n_splits=splits, shuffle=True, random_state=seed)

# for train_index,test_index in kf.split(X):
#     print('='*50)
#     X_tr, X_vl = X.loc[train_index],X.loc[test_index]
#     y_tr, y_vl = y.loc[train_index],y.loc[test_index]
    
#     #model
#     lr = LogisticRegression()
#     lr.fit(X_tr,y_tr)
#     pred_vl = lr.predict(X_vl)
#     score = (pred_vl == y_vl).mean()
#     print('validation_score : ',score)
#     cv_score.append(score)    
 
#     pred_test = lr.predict_proba(test)[:,1]
#     submission['Survived'] += pred_test / splits
    
# sum(cv_score)/splits


# ---
# # StratifiedKFold
# - 예를 들어 titanic 데이터셋의 y값의 unique 개수가 2개인데, kfold를 했더니 한 폴더에는 0만, 하나는 1만 나올수도 있음
# - 그렇게 되면 학습이 엉망진창이 되니까 나눌때 y값의 unique 개수 비율대로 잘 나눠준다.
# - (ex) y 값이 '1: 20%, 2:30%, 3:50%' 이렇게 분포한다면 각각의 폴더들의 비율도 20:30:50으로 맞춰준다.

# In[153]:


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression

seed = 1019
splits = 7
cv_score = []
ttl_pred_test = np.zeros(len(test))  

kf = StratifiedKFold(n_splits=splits, shuffle=False, random_state=seed)

for train_index, test_index in kf.split(X,y):
    X_tr, X_vl = X.loc[train_index], X.loc[test_index]
    y_tr, y_vl = y.loc[train_index], y.loc[test_index]
    
    model = LogisticRegression()
    model.fit(X_tr, y_tr)
    
    pred_vl = model.predict(X_vl)
    score = round(((pred_vl == y_vl).mean()),2)
    
    cv_score.append(score)    ##########
    
    
    pred_test = model.predict(test)
    
    ttl_pred_test = ttl_pred_test + pred_test/splits        
    
    print(('Validation Score : ', score))

submission['Survived'] = ttl_pred_test

sum(cv_score)/splits


# In[96]:


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression

seed =45
splits = 5
cv_score =[]

kf = StratifiedKFold(n_splits=splits, shuffle=False, random_state=seed)

for train_index,test_index in kf.split(X,y):
    print(('='*50))
    X_tr, X_vl = X.loc[train_index],X.loc[test_index]
    y_tr, y_vl = y.loc[train_index],y.loc[test_index]
    
    #model
    lr = LogisticRegression()
    lr.fit(X_tr,y_tr)
    pred_vl = lr.predict(X_vl)
    score = (pred_vl == y_vl).mean()
    print(('validation_score : ',score))
    cv_score.append(score)    
 
    pred_test = lr.predict_proba(test)[:,1]
    submission['Survived'] += pred_test / splits
sum(cv_score)/splits


# ---
# # Group KFold
# - 사람 표정을 인식하는 AI를 만들고자 하는데(기쁨/슬픔), 한사람의 기쁨/슬픔의 표정이 들어가있음.
# - 만약 train/valid를 나눴는데, 한사람의 사진이 train에만 들어가있으면 모델의 성능 향상에 한계가 생김
# - 그래서 사람_ID가 중복되지 않도록 잘 나눠주는게 Group KFold임
# - (Ex) 여기서는 PassengerId가 사람을 나눠주는 구분자인데 겹치는건 없어서 의미가 없긴함
# 

# In[152]:


train['Pclass'].unique()


# In[101]:


from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression

seed =45
splits = 50
cv_score =[]

groups = np.array(train['Pclass'].values)
gkf = GroupKFold(n_splits=splits)

for train_index, test_index in gkf.split(X, y, groups):
    print(('='*50))
    X_tr, X_vl = X.loc[train_index],X.loc[test_index]
    y_tr, y_vl = y.loc[train_index],y.loc[test_index]
    
    #model
    lr = LogisticRegression()
    lr.fit(X_tr,y_tr)
    pred_vl = lr.predict(X_vl)
    score = (pred_vl == y_vl).mean()
    print(('validation_score : ',score))
    cv_score.append(score)    
 
    pred_test = lr.predict_proba(test)[:,1]
    submission['Survived'] += pred_test / splits
    
sum(cv_score)/splits


# In[ ]:





# In[ ]:





# In[ ]:





# ----

# # Model generation and prediction

# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[ ]:





# In[ ]:


model.fit(X_train, y_train)
pred_train = model.predict(X_train)
print(('train_score : ', (pred_train == y_train).mean()))
pred_valid = model.predict(X_valid)
print(('valid_score : ', (pred_valid == y_valid).mean()))


# ----
# # Prediction on Test set 

# In[ ]:


pred_test = model.predict(test)
# Test의 예측값을 뽑아냅니다.


# In[ ]:


submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
submission
#submission을 불러와주고


# In[ ]:


submission['Survived'] = pred_test
# pred_test를 Survived Column에 넣어줍니다.


# In[ ]:


submission.to_csv('submission_final.csv', index=False)
# Index=False를 안해주면 Output된 csv에 이상한 Index가 column으로 하나 추가되어있습니다.
# 제가 맨날 하는 실수인데 아직도 습관이 안되어있습니다....


# ----
# # Conclusion 
# - 정말 수고하셨습니다. 여러분은 titanic dataset 을 가지고 data science 를 경험해보셨습니다.
# - 이걸로 끝이 아닙니다. 앞으로 배울 것이 너무나 무궁무진합니다.
# - 좀 더 참신한 feature engineering, 머신 러닝 모델 hyperparameter tunning, ensembling 등, 무궁무진합니다..꾸준히 커널공부를 하시면 실력이 꾸준히 늘겁니다.
# - 포기하지 마시고 재밌게 하시면 됩니다
# 

# - 본 튜토리얼은 유한님의 커널 https://kaggle-kr.tistory.com/18?category=868316#3_3 을 옮겨온 자료입니다. 자세한 사항은 해당 페이지를 참조하세요!

# In[ ]:




