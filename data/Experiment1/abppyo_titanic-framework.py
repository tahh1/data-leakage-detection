#!/usr/bin/env python
# coding: utf-8

# # Titanic
# https://www.kaggle.com/c/titanic

# 加入numpy, pandas, matplot等库

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


# 读入数据

# In[2]:


dataset = pd.read_csv('../input/train.csv')
testset = pd.read_csv('../input/test.csv')


# In[3]:


print((dataset.columns))
print((testset.columns))


# 0, PassengerId：乘客的数字id
# 
# 1, Survived：幸存(1)、死亡(0)
# 
# 2, Pclass：乘客船层—1st = Upper，2nd = Middle， 3rd = Lower
# 
# 3, Name：名字。
# 
# 4, Sex：性别
# 
# 5, Age：年龄
# 
# 6, SibSp：兄弟姐妹和配偶的数量。
# 
# 7, Parch：父母和孩子的数量。
# 
# 8, Ticket：船票号码。
# 
# 9, Fare：船票价钱。
# 
# 10, Cabin：船舱。
# 
# 11, Embarked：从哪个地方登上泰坦尼克号。 C = Cherbourg, Q = Queenstown, S = Southampton

# # 查看读入数据

# In[4]:


dataset.head()


# In[5]:


print((dataset.dtypes))


# In[6]:


print((dataset.describe()))


# In[7]:





# 从上面数据发现两个有意思的事情
# 
# 1. 数据有NULL元素
# 2. 数据非数值

# # 仔细观察数据

# 观察年龄
# 
# -conclusion gentleman

# In[7]:


Survived_m = dataset.Survived[dataset.Sex == 'male'].value_counts()
Survived_f = dataset.Survived[dataset.Sex == 'female'].value_counts()

df=pd.DataFrame({'male':Survived_m, 'female':Survived_f})
df.plot(kind='bar',stacked=True)
plt.title("Survived by sex")
plt.xlabel("survived")
plt.ylabel("count")
plt.show()


# 看看年龄
# -survived young ratio greater

# In[8]:


dataset['Age'].hist()
plt.ylabel("Number")
plt.xlabel("Age")
plt.title("Age distribution")
plt.show()

#plt.subplot(2,1,1)
dataset[dataset.Survived==0]['Age'].hist()
plt.ylabel("Number")
plt.xlabel("Age")
plt.title("Age distribution -non-survived")
plt.show()
#plt.subplot(2,1,2)
dataset[dataset.Survived==1]['Age'].hist()
plt.ylabel("Number")
plt.xlabel("Age")
plt.title("Age distribution -survived")
plt.show()


# 看看船票价钱

# In[9]:


dataset['Fare'].hist()
plt.ylabel("Number")
plt.xlabel("Fare")
plt.title("Fare distribution")
plt.show()

dataset[dataset.Survived==0]['Fare'].hist()
plt.ylabel("Number")
plt.xlabel("Fare")
plt.title("Fare- non-survived dist")
plt.show()

dataset[dataset.Survived==1]['Fare'].hist()
plt.ylabel("Number")
plt.xlabel("Fare")
plt.title("Fare- non-survived dist")
plt.show()


# 观察乘客舱层

# In[10]:


dataset['Pclass'].hist()
plt.show()
print((dataset['Pclass'].isnull().values.any()))

Survived_p1 = dataset.Survived[dataset['Pclass'] == 1].value_counts()
Survived_p2 = dataset.Survived[dataset['Pclass'] == 2].value_counts()
Survived_p3 = dataset.Survived[dataset['Pclass'] == 3].value_counts()
print(Survived_p1)

#sex_p1 = dataset.Sex[dataset['Pclass'] == 1].value_counts()
#print(fare_p1)

df = pd.DataFrame({'p1':Survived_p1, 'p2':Survived_p2, 'p3':Survived_p3})
print(df)
df.plot(kind='bar', stacked=True)
plt.title("survived by pclass")
plt.xlabel("pclass")
plt.ylabel("count")
plt.show()


# 观察登船地点

# In[11]:


Survived_S = dataset.Survived[dataset['Embarked'] == 'S'].value_counts()
Survived_C = dataset.Survived[dataset['Embarked'] == 'C'].value_counts()
Survived_Q = dataset.Survived[dataset['Embarked'] == 'Q'].value_counts()
print(Survived_S)

df = pd.DataFrame({'S':Survived_S, 'C':Survived_C, 'Q':Survived_Q})
df.plot(kind='bar',stacked =True)
plt.title('Survived by Embarked')
plt.xlabel('Survival')
plt.ylabel('count')
plt.show()


# # 保留下有效数据
# pclass, sex, age, fare, embarked

# # 分离label 和 训练数据

# In[12]:


label=dataset.loc[:,'Survived']
data=dataset.loc[:,['Pclass','Sex','Age','Fare','Embarked']]
testdat=testset.loc[:,['Pclass','Sex','Age','Fare','Embarked']]
print((data.shape))
print(data)


# 处理空数据

# In[13]:


def fill_NAN(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[:,'Age'] = data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy.loc[:,'Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy.loc[:,'Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy.loc[:,'Sex'] = data_copy['Sex'].fillna('female')
    data_copy.loc[:,'Embarked'] = data_copy['Embarked'].fillna('S')
    return data_copy

data_no_nan = fill_NAN(data)
testdat_no_nan = fill_NAN(testdat)


print((testdat.isnull().values.any()))
print((testdat_no_nan.isnull().values.any()))
print((data.isnull().values.any()))
print((data_no_nan.isnull().values.any()))

print(data_no_nan)



# 处理Sex 

# In[14]:


print((data_no_nan['Sex'].isnull().values.any()))

def transfer_sex(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Sex']=='female', 'Sex'] = 0
    data_copy.loc[data_copy['Sex']=='male', 'Sex'] = 1
    return data_copy

data_after_sex = transfer_sex(data_no_nan)
testdat_after_sex = transfer_sex(testdat_no_nan)
print(testdat_after_sex)


# 处理Embarked

# In[15]:


def transfer_embark(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 0
    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 1
    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 2
    return data_copy

data_after_embarked = transfer_embark(data_after_sex)
testdat_after_embarked = transfer_embark(testdat_after_sex)
print(testdat_after_embarked)


# 利用KNN训练数据

# In[16]:


data_now = data_after_embarked
testdat_now = testdat_after_embarked
from sklearn.model_selection import train_test_split

train_data,val_data,train_labels,val_labels = train_test_split(data_now, label, random_state=0, test_size=0.2)


# In[17]:


print((train_data.shape, val_data.shape, train_labels.shape, val_labels.shape))


# In[18]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
k_range = list(range(1,51))
k_scores = []
for K in k_range:
    clf = KNeighborsClassifier(n_neighbors = K)
    clf.fit(train_data, train_labels)
    print(('K=', K))
    predictions = clf.predict(val_data)
    score = accuracy_score(val_labels,predictions)
    print(score)
    k_scores.append(score)
    


# In[19]:


plt.plot(k_range, k_scores)
plt.xlabel('K for KNN')
plt.ylabel('Accuracy on validation set')
plt.show()
print((np.array(k_scores).argsort()))


# In[20]:


# 预测
clf = KNeighborsClassifier(n_neighbors = 33)
clf.fit(train_data,train_labels)
val_pred = clf.predict(val_data)
print(val_pred)


# In[21]:


# 检测模型precision， recall 等各项指标
from sklearn.metrics import confusion_matrix, classification_report
print((confusion_matrix(val_pred,val_labels)))
print((classification_report(val_pred,val_labels)))





# In[22]:





# In[22]:


# 预测
clf = KNeighborsClassifier(n_neighbors = 33)
clf.fit(data_now, label)
result = clf.predict(testdat_now)
print(result)


# In[23]:





# 打印输出

# In[23]:


df = pd.DataFrame({'PassengerId':testset['PassengerId'],"Survived":result})
df.to_csv('submission.csv',header=True,index=False)

