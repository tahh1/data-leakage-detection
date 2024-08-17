#!/usr/bin/env python
# coding: utf-8

# "이미 몇십년이나 지났지만, 제 시계는 영원히 그날, 1912년 4월 14일을 가리키고 있습니다."
# 
# 역사는 언제고 되풀이 되기 마련입니다. 16세기 조선에서 일어났던 일본의 침략이 몇 세기 후인 20세기 조선에서 또다시 되풀이되었듯이, 역사는 되풀이됩니다. 1912년 타이타닉 호 사건 이후 백년하고도 2년 후, 우리나라에서 유사한 사건이 일어났죠. 이러한 비극이 미래에 또다시 되풀이 되는 상황은 막아야 합니다. 1912년 4월 14일 타이타닉 호에 함께 바다에 수장된 영혼들이 데이터가 되어 우리에게 말하고 있습니다.
# 
# "1912년 4월 14일을 기억해주세요. 그날 우리들과 상황들을 봐주세요. 그리고 먼 훗날 이러한 일이 다시는 일어나지 않게 해주세요."
# 
# 지금 우리는 Kaggle이란 타임머신을 타고, 백년이란 시간을 거슬러 그날 타이타닉호로 갑니다. 
# 
# 
# 
# 그러면 시작하겠습니다.

# *데이터셋 확인

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
sns.set(font_scale=2.5) 
import missingno as msno
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[3]:


df_train.head()


# 위 데이터는 타이타닉 승객들 데이터 중 상위 5개만 나타낸 정보입니다. 먼저 train 정보를 살펴보도록 하죠.

# In[4]:


df_train.describe()


# In[5]:


df_train.info()


# train정보를 살펴봤으니, 이젠 test 정보를 살펴보죠.

# In[6]:


df_test.describe()


# In[7]:


df_test.head()


# In[8]:


df_test.info()


# *NaN값 체크

# In[9]:


df_train.isnull().sum()


# train에서 null값은 Age의 경우 177개나 있고, Cabin의 경우, 687개나 있네요. Embarked의 경우 2개나 있구요. 여기서 저희는 null값을 가지고 있는 Age, Cabin, Embarked를 고칠겁니다.
# 
# 그리고 위 데이터를 시각화하면 다음과 같습니다. msno라는 라이브러리를 사용했습니다.

# In[10]:


msno.matrix(df=df_train.iloc[:, :], figsize=(7, 5), color=(0.5, 0.1, 0.2))


# In[11]:


msno.bar(df=df_train.iloc[:, :], figsize=(7, 5), color=(0.2, 0.5, 0.2))


# 우선 얼마나 많이 생존했을까요?

# In[12]:


f,ax=plt.subplots(1,2,figsize=(16,6))
df_train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=False)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=df_train,ax=ax[1])
ax[1].set_title('Survived')
plt.show()


# 위 데이터를 보면 전체 타이타닉 승객들 중 오직 38.4%만이 살아남았다는 사실을 알 수 있습니다. 이 자료를 통해 저희들은 살아남은 이 38.4% 생존자들이 어떤 공통된 feature들을 가지고 있는지 살펴볼 필요가 있습니다. 그럼 이제 그 feature들을 살펴보겠습니다.

# (1). Sex

# In[13]:


df_train.groupby(['Sex','Survived'])['Survived'].count()


# 위 데이터는 성별로 생존자/사망자별로 나눈 데이터입니다. 이를 시각화해서 표시해보겠습니다.

# In[14]:


f,ax=plt.subplots(1,2,figsize=(14,4))
df_train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived',data=df_train,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')
plt.show()


# 위 데이터들을 통해 생존자 수는 여성들이 남성들보다 현저히 높다는 결과를 알 수 있습니다. 그러나 단순히 이 feature만으로는 부족합니다. 더 많은 feature가 필요합니다. 그런 의미에서 이제는 Pclass를 살펴볼 필요가 있습니다.

# (2). Pclass

# Pclass에 따른 생존률을 살펴보겠습니다. 먼저, Pclass1, 2, 3에 각각 탑승한 승객들의 수와 각 Pclass별로 생존자 수를 알 필요가 있습니다.

# In[15]:


df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()


# 위 데이터는 count()를 통해 Pclass1, 2, 3에 따라 탑승한 승객들의 총 수를 나타낸 데이터입니다. 아래는 sum()을 통해 위에 나와있는 대로 Pclass별로 탑승한 승객들 중에서 생존자들의 수를 나타낸 데이터입니다.

# In[16]:


df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()


# 그러면 이를 pandas의 crosstab으로 나타내 봅시다.

# In[17]:


pd.crosstab(df_train.Pclass,df_train.Survived,margins=True)


# 이제 그래프로 나타내봅시다.

# In[18]:


df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar()


# 위 그래프에서 알수 있듯이 pclass가 좋을수록 생존자 수가 높음을 알 수 있습니다. 역시 돈은 다다익선이라는 말을 실감할 수 있군요..
# 
# 좀 더 보기 쉽게 그래프를 그려봐요. seaborn의 countplot을 이용하면, 특정 label에 따른 개수를 확인할 수 있습니다.

# In[19]:


f,ax=plt.subplots(1,2,figsize=(16,8))
df_train['Pclass'].value_counts().plot.bar(color=['black','silver','yellow'],ax=ax[0])
ax[0].set_title('Number Of Passengers By Pclass')
ax[0].set_ylabel('Count')
sns.countplot('Pclass',hue='Survived',data=df_train,ax=ax[1])
ax[1].set_title('Pclass:Survived vs Dead')
plt.show()


# 위 그래프를 통해 우리는 "돈은 다다익선"이라는 불변의 진리를 확인할 수 있네요. 씁쓸한 세상...
# 
# 그러면 좀더 흥미로운 걸 보죠. Sex와 Pclass와 관련된 생존율을 보죠.

# 1-2. Sex and Pclass

# 이번에는 Sex와 Pclass에 따라 생존자 수가 어떻게 달라지는지 보죠. 그러기 위해선 seaborn의 factorplot을 쓸겁니다.

# In[20]:


pd.crosstab([df_train.Sex,df_train.Survived],df_train.Pclass,margins=True)


# In[21]:


sns.factorplot('Pclass','Survived',hue='Sex',data=df_train)
plt.show()


# factorplot을 사용했던 이유는 범주형 값 분리를 쉽게 해주기 때문입니다.
# Crosstab과 factorplot을 보면 Pclass가 좋을수록, 여성일수록 생존자 수가 높음을 알수 있습니다. 또한, 여성들은 Pclass에 상관없이 남성들보다 높음으로봐서 여성들 위주로 구출을 했다는 사실을 알 수 있습니다. 이를 통해 Pclass 역시 중요한 feature라는 것이 명확해졌군요. 이제 다른 feature들을 살펴봅시다.

# (3). Age

# In[22]:


print(('Oldest Passenger was of:',df_train['Age'].max(),'Years'))
print(('Youngest Passenger was of:',df_train['Age'].min(),'Years'))
print(('Average Age on the ship:',df_train['Age'].mean(),'Years'))


# 위 데이터를 통해 타이타닉 호에 탑승한 승객들 중 최고로 오래된 나이는 80세, 제일 어린 나이의 승객은 0.42살, 즉 우리 나이로 1살이라는 사실을 알수 있습니다. 타이타닉 호에 탑승한 승객들 평균 나이는 29세라는 결과도 얻을 수 있구요. 그러면 생존에 따른 Age의 히스토그램을 그려보겠습니다.

# In[23]:


fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)
sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)
plt.legend(['Survived == 1', 'Survived == 0'])
plt.show()


# 위 히스토그램을 통해 생존자들 중 젊은 사람들이 많다는 사실을 알수 있네요. 그러면 이를 Pclass와 연결시켜보죠.

# In[24]:


plt.figure(figsize=(8, 6))
df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')
df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')
df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')

plt.xlabel('Age')
plt.title('Age Distribution within classes')
plt.legend(['1st Class', '2nd Class', '3rd Class'])


# class가 높을수록 나이 많은 사람의 비중이 높네요. 그러면 나이대가 변하면서 생존율은 어떤 결과를 보일까요? 나이 범위를 점점 넓혀가면서 생존율이 어떻게 되는지 봅시다.

# In[25]:


cummulate_survival_ratio = []
for i in range(1, 80):
    cummulate_survival_ratio.append(df_train[df_train['Age'] < i]['Survived'].sum() / len(df_train[df_train['Age'] < i]['Survived']))
    
plt.figure(figsize=(7, 7))
plt.plot(cummulate_survival_ratio)
plt.title('Survival rate change depending on range of Age', y=1.02)
plt.ylabel('Survival rate')
plt.xlabel('Range of Age(0~x)')
plt.show()


# 이를 통해 나이가 어릴수록 생존율이 높다는 결과를 얻을 수 있군요. Age도 중요한 feature같습니다. 그러면 이 시점에서 Pclass와 Age를 묶어서 생존율과 연관시켜보죠. 어떤 결과가 나올까요?

# In[26]:


f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Pclass","Age", hue="Survived", data=df_train,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(list(range(0,110,10)))
sns.violinplot("Sex","Age", hue="Survived", data=df_train,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(list(range(0,110,10)))
plt.show()


# 결국, Pclass와 관계없이 10살 이하 어린이들의 생존율과 함께 증가하는 어린이들의 수는 양호하다는 결과를 얻을 수 있습니다. Pclass1에서, 20세에서 50세 사이의 그리고 여성이 생존율이 높음을 알 수 있네요. 

# 앞서 살펴본 바와 같이 Age feature는 null 값이 177개나 됩니다. 이러한 NaN 값을 대체하기 위해 dataset의 평균 연령을 할당해야 할 필요가 있습니다.
# 
# 그러나 문제는 서로 다른 많은 연령대의, 많은 사람들이 있다는 것입니다. 우리는 단지 29 세의 평균 연령으로 4 세 아동을 배정 할 수 없습니다. 그러면 어떻게 해야 할까요?
# 
# 바로 이름!! 이름 feature을 확인할 수 있습니다. 이 feature을 살펴보면 이름에 Mr이나 Mrs.과 같은 인사말이 있다는 것을 알 수 있습니다. 따라서 Mr과 Mrs의 평균값을 각 그룹에 할당 할 수 있습니다.

# (4). Name

# In[27]:


df_train['Title'] = 0
for salut in df_train:
    df_train['Title'] = df_train.Name.str.extract('([A-Za-z]+)\.')
    
df_test['Title'] = 0
for salut in df_test:
    df_test['Title'] = df_test.Name.str.extract('([A-Za-z]+)\.')  


# 여기에서 우리는 [A-Za-z] +)를 사용하고 있습니다. 이것은 A에서 Z, 또는 a에서 z 사이에 있고 '.' 뒤에 오는 문자열을 찾아줍니다. 그래서 우리는 Name에서 이니셜을 성공적으로 추출합니다. 

# In[28]:


pd.crosstab(df_train['Title'], df_train['Sex'])
#Sex와 관련된 이니셜(Initials) 체크


# Mlle나 Mme와 같은 철자가 틀린 이니셜이 있습니다. 미스를 대신해 미스와 다른 값으로 대체합니다.

# In[29]:


df_train['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
df_test['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)


# In[30]:


data_df = df_train.append(df_test)

titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']
for title in titles:
    age_to_impute = data_df.groupby('Title')['Age'].median()[titles.index(title)]
    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute


# In[31]:


# TRAIN_DF and TEST_DF에서 Age 값 대체:
df_train['Age'] = data_df['Age'][:891]
df_test['Age'] = data_df['Age'][891:]


# In[32]:


df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# 작업을 다 한 후 train과 test에 Age NaN값이 다 사라졌는지 확인합니다.

# In[33]:


df_train.Age.isnull().any()


# In[34]:


df_test.Age.isnull().any()


# False가 나온 것으로 보아 성공적으로 NaN값을 채웠다는 사실을 알수 있습니다. 이제 Age를 그래프로 나타내봅시다.

# In[35]:


f,ax=plt.subplots(1,2,figsize=(20,10))
df_train[df_train['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Survived= 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
df_train[df_train['Survived']==1].Age.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')
ax[1].set_title('Survived= 1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()


# 결국 우리는 나온 데이터를 통해 10세 미만의 유아(0~5세)가 많이 살아남았으며, 30세에서 35세 사이 나이대의 사람들이 많이 죽었음을 알수 있습니다.

# 이제 이를 Pclass와 연계시켜서 다시 그래프를 그려봅시다.

# In[36]:


sns.factorplot('Pclass','Survived',col='Title',data=df_train)
plt.show()


# 이를 통해 저희는 당시 타이타닉 호에서 여성과 아이들 위주로 구출했다는 사실을 알수 있네요. 영화 "타이타닉"은 고증이 잘 된 영화였습니다.

# (5). Embarked

# Embarked 는 탑승한 항구를 나타냅니다. 위에서 해왔던 것과 비슷하게 탑승한 곳에 따른 생존률을 보겠습니다.

# In[37]:


pd.crosstab([df_train.Embarked,df_train.Pclass],[df_train.Sex,df_train.Survived],margins=True)


# In[38]:


f, ax = plt.subplots(1, 1, figsize=(7, 7))
df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax)


# 이 셋 중에서 C가 제일 높음을 알수 있네요.

# In[39]:


f,ax=plt.subplots(2,2,figsize=(20,12))
sns.countplot('Embarked',data=df_train,ax=ax[0,0])
ax[0,0].set_title('No. Of Passengers Boarded')
sns.countplot('Embarked',hue='Sex',data=df_train,ax=ax[0,1])
ax[0,1].set_title('Male-Female Split for Embarked')
sns.countplot('Embarked',hue='Survived',data=df_train,ax=ax[1,0])
ax[1,0].set_title('Embarked vs Survived')
sns.countplot('Embarked',hue='Pclass',data=df_train,ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()


# 이를 통해 타이타닉 호에 탑승한 대부분의 승객들은 S에서 승선했었으며, Pclass3에서 왔습니다. Embarked S, C, Q 중에서 C에서 생존자 수가 높았던 이유는 S, Q에 비해 사망자수:생존자수 비율이 좋기 때문에 그런 결과가 나온것으로 판단할 수 있습니다. 하지만, 결국 그 이유는 Pclass과 관련이 있다는 사실을 알 수 있습니다. 위를 보면 S는 Pclass3 탑승자들이 많지만, C, Q에 비해 Pclass1 탑승자들이 많습니다. Q는 거의 대부분이 Pclass3 탑승자들 이구요. 
# 

# 그러면 Embarked NaN값을 채워봅시다. S, C, Q 중에서 S에서 대부분의 승객들이 탑승했기 때문에, NaN값을 S로 대체하겠습니다.

# In[40]:


df_train['Embarked'].fillna('S',inplace=True)


# In[41]:


df_train.Embarked.isnull().any()


# 이제 Embarked의 NaN값도 없어졌습니다.

# (6). SibSp

# 이 feature에서는 탑승한 승객 한 사람이 혼자인지 아니면 가족과 함께인지를 나타냅니다.

# In[42]:


pd.crosstab([df_train.SibSp],df_train.Survived)


# In[43]:


fig, ax = plt.subplots(figsize=(20, 15))

df_train.groupby(['SibSp', 'Survived']).size().unstack().plot(kind='bar', stacked=False, ax=ax)
ax.set_title('SibSp vs Survived - Count - Side by Side')
plt.show()


# In[44]:


f,ax=plt.subplots(1,2,figsize=(20,8))
sns.barplot('SibSp','Survived',data=df_train,ax=ax[0])
ax[0].set_title('SibSp vs Survived')
sns.factorplot('SibSp','Survived',data=df_train,ax=ax[1])
ax[1].set_title('SibSp vs Survived')
plt.close(2)
plt.show()


# 다음에는 Pclass별 SibSp 수를 crosstab으로 나타냅니다.

# In[45]:


pd.crosstab(df_train.SibSp,df_train.Pclass)


# 그래프를 보면 형제자매가 많아질수 생존율이 줄어든다는 특징을 얻을 수 있습니다. 이 특징이 나온 이유는 Pclass와 관련있습니다. 위 crosstab을 보면 Pclass3에서 형제자매가 3명 이상인 승객들 수가 많음을 알수 있습니다.

# (7). Parch

# Parch는 부모, 자녀입니다.

# In[46]:


pd.crosstab(df_train.Parch,df_train.Pclass)


# SibSp와 마찬가지로 Pclass3에서 대가족이 많음을 알수 있네요.

# In[47]:


f, ax = plt.subplots(figsize=(18, 10))
df_train.groupby(['Parch', 'Survived']).size().unstack().plot(kind='bar', stacked=False, ax=ax)
ax.set_title('Parch vs Survived - Count - Side by Side')
plt.show()


# In[48]:


f,ax=plt.subplots(1,2,figsize=(20,8))
sns.barplot('Parch','Survived',data=df_train,ax=ax[0])
ax[0].set_title('Parch vs Survived')
sns.factorplot('Parch','Survived',data=df_train,ax=ax[1])
ax[1].set_title('Parch vs Survived')
plt.close(2)
plt.show()


# 위에서 알수 있듯이 1~3명의 부모, 자식이 있어야 생존율이 높습니다. 그러나 4인이 넘어가면 줄어든다는 결과를 알 수 있습니다.

# 그럼 위에서 살펴본 SibSp과 Parch를 Family로 합쳐서 분석해봅시다.

# (8). FamilySize

# In[49]:


df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다


# In[50]:


print(("Maximum size of Family: ", df_train['FamilySize'].max()))
print(("Minimum size of Family: ", df_train['FamilySize'].min()))


# 그럼 FamilySize와 생존율 관계를 살펴봅시다.

# In[51]:


f,ax=plt.subplots(1, 3, figsize=(40,10))
sns.countplot('FamilySize', data=df_train, ax=ax[0])
ax[0].set_title('(1) No. Of Passengers Boarded', y=1.02)

sns.countplot('FamilySize', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('(2) Survived countplot depending on FamilySize',  y=1.02)

df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])
ax[2].set_title('(3) Survived rate depending on FamilySize',  y=1.02)

plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()


# 맨 왼쪽 그래프는 가족 크기, FamilySize를 나타내는데, 1부터 11까지 있습니다. 두번째 그래프는 FamilySize에 따른 생존자 수와 사망자 수를 나타낸 것입니다. 세번째 그래프는 가족 사이즈에 따른 생존율을 내림차순으로 나타내었습니다. 가족이 4명일 때 생존율이 높음을 알수 있습니다. FamilySize가 3, 4일때 생존율이 높음을 알수 있습니다.

# (9). Fare

# In[52]:


print(('Highest Fare was:',df_train['Fare'].max()))
print(('Lowest Fare was:',df_train['Fare'].min()))
print(('Average Fare was:',df_train['Fare'].mean()))


# 다음은 Pclass에 따른 Fare를 그래프로 나타내겠습니다.

# In[53]:


f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(df_train[df_train['Pclass']==1].Fare,ax=ax[0])
ax[0].set_title('Fares in Pclass 1')
sns.distplot(df_train[df_train['Pclass']==2].Fare,ax=ax[1])
ax[1].set_title('Fares in Pclass 2')
sns.distplot(df_train[df_train['Pclass']==3].Fare,ax=ax[2])
ax[2].set_title('Fares in Pclass 3')
plt.show()


# (10). Cabin

# 이 cabin은 NaN값이 687개나 있습니다. 압도적으로 NaN값에 많기 때문에 하자가 있어서, 우리가 만드려는 모델에 포함시키지 않겠습니다.

# In[54]:


df_train.head()


# (11). Ticket

# 이 feature는 NaN이 없습니다. 다만, string data입니다. 그래서 어떤 작업을 해줘야 실제 모델에 사용할 수 있습니다. 

# In[55]:


df_train['Ticket'].value_counts()


# 모든 feature들을 살펴본 결과, 다음과 같은 사실을 알수 있습니다.
# 
# 먼저, 여성의 생존 확률이 남성보다 높습니다.
# 두번째로, 1등석일수록 생존 확률이 높다는 사실을 알았습니다. Pclass3은 생존율이 매우 낮습니다.
# 세번째로, 5세에서 10세 미만 어린이의 생존 확률이 높습니다. 그리고 동시에 15세에서 35세 사이의 사람들이 많이 사망하였습니다.
# 네번째로, Pclass1의 승객 대다수가 S에서 왔음에도 불구하고, 생존자들은 C에서 온 승객들이 많았다는 사실은 흥미로운 결과입니다.
# 마지막으로, 3~4명의 가족들과 함께 탑승한 승객들의 생존율이 높았음을 알수 있었습니다.

# *Correlation Between The Features

# In[56]:


sns.heatmap(df_train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #df_train.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# 이 heatmap은 알파벳과 문자열 간 상관관계가 명확하지 않기 때문에, 숫자 feature만 비교됩니다.

# *Feature Engineering

# 여태까지 많은 feature들을 살펴보고 분석해봤지만, 모든 feature들이 중요하지는 않습니다. 이 과정에서는 중요하지 않은 feature들을 제거하고 다른 feature의 정보를 관찰하고 추출하여 새로운 feature를 추가하거나 얻어보겠습니다. 

# (1). Age-band

# Age는 연속성이 있는 feature입니다. 이는 기계 학습 모델에 문제가 됩니다. 그래서 우리는 Age를 몇개의 그룹으로 나누어서 범주화시킬 겁니다. 그러기 위해서는 Binning 또는 Normalization을 통해 연속 값을 범주 값 으로 변환해야 합니다. 여기서는 binning을 사용할 겁니다. 
# 
# 승객들의 나이가 0세부터 80세였으니, 0-80범위를 4개의 범위로 나눌수 있겠네요. 그리고 80/4=20이므로 각각 범위는 20씩이네요.

# In[57]:


df_train['Age_band']=0
df_train.loc[df_train['Age']<=20,'Age_band']=0
df_train.loc[(df_train['Age']>20)&(df_train['Age']<=40),'Age_band']=1
df_train.loc[(df_train['Age']>40)&(df_train['Age']<=60),'Age_band']=2
df_train.loc[df_train['Age']>60,'Age_band']=3
df_train.head(2)


# In[58]:


df_test['Age_band']=0
df_test.loc[df_test['Age']<=20,'Age_band']=0
df_test.loc[(df_test['Age']>20)&(df_test['Age']<=40),'Age_band']=1
df_test.loc[(df_test['Age']>40)&(df_test['Age']<=60),'Age_band']=2
df_test.loc[df_test['Age']>60,'Age_band']=3
df_test.head(2)


# 그리고 각 밴드에 있는 승객들의 수를 체크합니다.

# In[59]:


df_train['Age_band'].value_counts().to_frame()


# In[60]:


sns.factorplot('Age_band','Survived',data=df_train,col='Pclass')
plt.show()


# 그러면 결국, 이 데이터를 통해 Pclass와 상관없이 나이가 증가함에 따라 생존율이 낮아짐을 알 수 있습니다.

# (2). Family_Size and Alone

# 이제 'Family_Size and Alone'이라는 새로운 feature를 만들고 분석해봅시다. 이 feature는 혼자 승객이 혼자 있는지 여부와 함께, 생존율이 승객의 가족 규모와 관련이 있는지를 확인할 수 있도록 합쳐진 데이터를 제공합니다. 

# In[61]:


df_train['Family_Size']=0
df_train['Family_Size']=df_train['Parch']+df_train['SibSp']#family size
df_train['Alone']=0
df_train.loc[df_train.Family_Size==0,'Alone']=1#Alone

df_test['Family_Size']=0
df_test['Family_Size']=df_test['Parch']+df_test['SibSp']#family size
df_test['Alone']=0
df_test.loc[df_test.Family_Size==0,'Alone']=1#Alone



f,ax=plt.subplots(1,2,figsize=(18,6))
sns.factorplot('Family_Size','Survived',data=df_train,ax=ax[0])
ax[0].set_title('Family_Size vs Survived')
sns.factorplot('Alone','Survived',data=df_train,ax=ax[1])
ax[1].set_title('Alone vs Survived')
plt.close(2)
plt.close(3)
plt.show()


# Family_Size=0는 승객이 혼자라는 것을 나타냅니다. 위 그래프를 보면 승객이 혼자인 경우 생존율이 낮음을 알 수 있습니다. 또한 Family_Size가 4이상인 경우도 생존율이 줄어듬을 알 수 있습니다. 이 feature는 모델에 중요한 feature입니다. 

# (3). Fare_Range

# Fare도 연속성이 있는 feature이기 때문에 전환시켜줘야 합니다. 이때 pandas.qcut을 사용합니다.

# In[62]:


df_train['Fare_Range']=pd.qcut(df_train['Fare'],10)
df_test['Fare_Range']=pd.qcut(df_test['Fare'],10)
df_train.groupby(['Fare_Range'])['Survived'].mean().to_frame()


# 위에서 언급했듯이, fare_range가 증가함에 따라 생존 기회가 증가한다는 것을 분명히 알 수 있습니다.
# 
# 이제는 Fare_Range 값을 그대로 전달할 수 없습니다. 우리는 Age_Band 에서와 같은 싱글 톤 값으로 변환해야합니다

# In[63]:


df_train['Fare_cat']=0
df_train.loc[df_train['Fare']<=7.55,'Fare_cat']=0
df_train.loc[(df_train['Fare']>7.55)&(df_train['Fare']<=7.854),'Fare_cat']=1
df_train.loc[(df_train['Fare']>7.854)&(df_train['Fare']<=8.05),'Fare_cat']=2
df_train.loc[(df_train['Fare']>8.05)&(df_train['Fare']<=10.5),'Fare_cat']=3
df_train.loc[(df_train['Fare']>10.5)&(df_train['Fare']<=14.454),'Fare_cat']=4
df_train.loc[(df_train['Fare']>14.454)&(df_train['Fare']<=21.679),'Fare_cat']=5
df_train.loc[(df_train['Fare']>21.679)&(df_train['Fare']<=27.0),'Fare_cat']=6
df_train.loc[(df_train['Fare']>27.0)&(df_train['Fare']<=39.688),'Fare_cat']=7
df_train.loc[(df_train['Fare']>39.688)&(df_train['Fare']<=77.958),'Fare_cat']=8
df_train.loc[(df_train['Fare']>77.958)&(df_train['Fare']<=513),'Fare_cat']=9


# In[64]:


df_test['Fare_cat']=0
df_test.loc[df_test['Fare']<=7.55,'Fare_cat']=0
df_test.loc[(df_test['Fare']>7.55)&(df_test['Fare']<=7.854),'Fare_cat']=1
df_test.loc[(df_test['Fare']>7.854)&(df_test['Fare']<=8.05),'Fare_cat']=2
df_test.loc[(df_test['Fare']>8.05)&(df_test['Fare']<=10.5),'Fare_cat']=3
df_test.loc[(df_test['Fare']>10.5)&(df_test['Fare']<=14.454),'Fare_cat']=4
df_test.loc[(df_test['Fare']>14.454)&(df_test['Fare']<=21.679),'Fare_cat']=5
df_test.loc[(df_test['Fare']>21.679)&(df_test['Fare']<=27.0),'Fare_cat']=6
df_test.loc[(df_test['Fare']>27.0)&(df_test['Fare']<=39.688),'Fare_cat']=7
df_test.loc[(df_test['Fare']>39.688)&(df_test['Fare']<=77.958),'Fare_cat']=8
df_test.loc[(df_test['Fare']>77.958)&(df_test['Fare']<=513),'Fare_cat']=9


# In[65]:


sns.factorplot('Fare_cat','Survived',data=df_train,hue='Sex')
plt.show()


# 분명히 Fare_cat이 증가하면 생존율이 증가합니다. 이 feature는 sex와 함께 모델링에 있어서 중요하다는 사실을 알 수 있습니다.

# *Converting String Values into Numeric

# 문자열을 기계 학습 모델에 전달할 수 없으므로, Sex, Embarked 등의 feature를 숫자 값으로 변환시켜야 합니다.

# In[66]:


df_train['Sex'].replace(['male','female'],[0,1],inplace=True)
df_train['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
df_train['Title'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)


# In[67]:


df_test['Sex'].replace(['male','female'],[0,1],inplace=True)
df_test['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
df_test['Title'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)


# 이제 예측을 위해 feature들을 단순화시킵니다. 그런 의미에서 정확한 예측에 불필요한 column들은 drop을 이용해서 제거합니다. 

# In[68]:


df_train.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)


# In[69]:


df_test.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)


# 위의 과정을 모두 하고 나서 다시한번 df_train을 살펴봅시다.

# In[70]:


df_train.head()


# 그러면 fare, name, Ticket등이 사라져 있음을 알수 있습니다. 모델링을 위한 준비는 마친 겁니다. 이제 이를 heatmap으로 구현해보겠습니다.

# In[71]:


sns.heatmap(df_train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})
fig=plt.gcf()
fig.set_size_inches(18,15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# Model prediction 

# Pipeline, ColumnTransformer

# In[72]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# In[73]:


df_train.head()


# In[74]:


all_features = ['Pclass', 'Sex', 'FamilySize', 'Age_band', 'Fare_cat']


# In[75]:


all_transformer = Pipeline(steps = [
    ('stdscaler', StandardScaler())
])


# In[76]:


all_preprocess = ColumnTransformer(
    transformers = [
        ('allfeatures', all_transformer, all_features),
    ]
)


# 이제 sklearn을 이용해 머신러닝 모델을 만들고 예측을 해봅시다. 그 전에 데이터부터 분리시킵시다.

# In[77]:


y = df_train['Survived']


# In[78]:


X = df_train[df_train.columns[1:]]


# In[79]:


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3,random_state=0,stratify=df_train['Survived'])


# Model prediction

# 우선 hyper parameter tuning없이 classifier들을 실행시켜서 모델들을 평가합니다.

# In[80]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_score


# In[81]:


classifiers = [
    LogisticRegression(random_state=42),
    RandomForestClassifier(random_state=42),
    SVC(random_state=42),
    KNeighborsClassifier(),
    SGDClassifier(random_state=42),
    ]


# In[82]:


first_round_scores = {}
for classifier in classifiers:
    pipe = Pipeline(steps=[('preprocessor', all_preprocess),
                      ('classifier', classifier)])
    pipe.fit(X_train, y_train)   
    print(classifier)
    score = pipe.score(X_test, y_test)
    first_round_scores[classifier.__class__.__name__[:10]] = score
    print(("model score: %.3f" % score))


# 이제 위 데이터를 그래프로 시각화시킵니다.

# In[83]:


# Plot the model scores
plt.plot(list(first_round_scores.keys()), list(first_round_scores.values()), "ro", markersize=10)
fig=plt.gcf()
fig.set_size_inches(8,5)
plt.title('Model Scores of the Classifiers - with no tuning ')
plt.show()


# 이제 Hyperparameter tuning을 합니다. Hyperparameter tuning은 위 그래프로부터의 최고의 classifier를 바꿀 것입니다. 

# Hyper Parameters

# 이제 마지막 예측 모델을 선택해보죠. 
# 
# hyper parameter tuning 후의 그래프로부터 마지막 예측 모델을 선택합니다.
# 
# 전반적으로 최고의 점수를 얻었던 KNeighborsClassifier을 선택합니다.

# Prediction

# In[84]:


final_pipe = Pipeline(steps=[('preprocessor', all_preprocess)])


# In[85]:


X_final_processed = final_pipe.fit_transform(X)


# In[86]:


df_test_final_processed = final_pipe.transform(df_test)


# hyperparameter tuning을 따라서 모든 훈련 데이터에 관한 모델을 훈련시킵니다.

# In[87]:


knn_hyperparameters = {
    'n_neighbors': [6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22],
    'algorithm' : ['auto'],
    'weights': ['uniform', 'distance'],
    'leaf_size': list(range(1,50,5)),
}

gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = knn_hyperparameters,  
                cv=10, scoring = "roc_auc")

gd.fit(X_final_processed, y)
print((gd.best_score_))
print((gd.best_estimator_))


# In[88]:


gd.best_estimator_.fit(X_final_processed, y)
y_pred = gd.best_estimator_.predict(df_test_final_processed)


# 테스트됐을 때 위 예측은 더 낮은 점수를 보여줬습니다.
# 
# n_neighbors 및 n_jobs에 대해 다른 값으로 테스트되었으며 그래서 이 KNN 분류기에 도달했습니다.

# In[89]:


knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski', 
                           metric_params=None, n_jobs=None, n_neighbors=6, p=2, 
                           weights='uniform')
knn.fit(X_final_processed, y)
X_pred = knn.predict(X_final_processed)
y_pred = knn.predict(df_test_final_processed)


# In[90]:


submission = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])
submission['Survived'] = y_pred
submission.to_csv("submission.csv", index = False)


# 제가 이 코딩을 작성함에 있어 아래 링크 사이트의 영향을 받았습니다.
# 
# https://www.kaggle.com/ash316/eda-to-prediction-dietanic
# 
# https://www.kaggle.com/vidyabhandary/titanic-eda-hyperparameters
