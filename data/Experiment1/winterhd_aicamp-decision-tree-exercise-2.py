#!/usr/bin/env python
# coding: utf-8

# ## 决策树的直接调用与Titanic数据集的探索 

# In[ ]:


# 必要的引入
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# #### 读取数据并打印基本信息

# In[ ]:


data = pd.read_csv(os.path.join("../input", "titanic.csv"), sep=",")


# In[ ]:


# 打印数据基本信息
data.info()


# In[ ]:


# 观察部分数据的形式
data.head(3)


# #### 预测目标的基本分布

# In[ ]:


# TODO 观察预测目标的分布
data['survived'].value_counts(normalize=True)


# In[ ]:


#TODO 可视化预测目标的分布
sns.countplot(data['survived'])
# plt.hist(data['survived'])


# #### 舱位与预测目标的关系

# In[ ]:


#TODO 利用sns画出每种舱对应的幸存与遇难人数
sns.countplot(x = data['pclass'], hue = data['survived'])


# #### 名字的信息

# In[ ]:


# TODO 打印部分名字信息
data['name'].head()


# #### 对名字属性进行变换  
# - 取名字的title

# In[ ]:


data['name_title'] = data['name']\
    .apply(lambda x: x.split(',')[1] if len(x.split(',')) > 1 else x.split(',')[0])\
    .apply(lambda x: x.split()[0])

# data['name_title'] = data['name'].apply(lambda x: x.split(',')[1] if len(x.split(',')) > 1 else x.split(',')[0]).apply(lambda x: x.split()[0])


# In[ ]:


# TODO 打印name title信息
# data['name_title'].head()
data['name_title'].value_counts()


# #### 名字的title与存活与否的关系

# In[ ]:


# TODO 名字title 与幸存的关系
data['survived'].groupby(data['name_title']).mean()


# #### 取名字的长度

# In[ ]:


# TODO 新增名字长度的变量
data['name_length'] = data['name'].apply(lambda x: len(x))
# data['name_length']


# #### 名字长度与存活与否的关系

# In[ ]:


# TODO 名字长度与幸存的关系
# data['survived'].groupby(data['name_length']).mean()
print((data['survived'].groupby(pd.qcut(data['name_length'], 5)).mean()))

sns.countplot(x=pd.qcut(data['name_length'], 5), hue=data['survived'])


# #### 性别的分布与最后幸存的关系

# In[ ]:


# TODO 打印性别比例
data['sex'].value_counts(normalize=True)


# In[ ]:


# TODO 性别与幸存的关系
print((data['survived'].groupby(data['sex']).mean()))

sns.countplot(x=data['sex'], hue=data['survived'])


# #### 年龄与幸存的关系  
# - 缺失数据的处理  
#   1 实值： 中位数或者平均数去补  
#   2 类别： major class去补

# In[ ]:


# TODO 年龄与幸存的关系
# check number of missing value .isna() or .isnull()
data['age'].isna().sum()


# In[ ]:


print((data['survived'].groupby(pd.qcut(data['age'], 5)).mean()))
sns.countplot(x=pd.qcut(data['age'], 5), hue=data['survived'])


# In[ ]:


age_mean = data['age'].mean()
age_median = data['age'].median()
print(('age mean is %d, age median is %d' % (age_mean, age_median)))
data['age'] = data['age'].fillna(age_mean)
data['age'].isnull().sum()


# #### 登船的地点与幸存的关系

# In[ ]:


# TODO 登船地点的分布
data['embarked'].value_counts()


# In[ ]:


# TODO 登船地点与幸存的关系
data['survived'].groupby(data['embarked']).mean()


# In[ ]:


# TODO 可视化登船地点与舱位的关系
sns.countplot(x=data['embarked'], hue=data['pclass'])


# #### 目的地与幸存的关系

# In[ ]:


data['home.dest'].head()


# In[ ]:


# data['survived'].groupby(data['home.dest'].apply(lambda x: str(x).split(',')[-1])).mean()

data['survived'].groupby(data['home.dest'].apply(lambda x: str(x).split(',')[-1])).mean()


# #### room, ticket, boat缺失数据太多，舍弃不用

# ### 新来了一个小鲜肉，基本信息如下  
# 

# ![alt text](./notebook_image/jack.jpg)
# ![alt text](./notebook_image/jack_info.png)

# #### Feature Transform

# In[ ]:


# 对使用的feature进行预处理：
#     pclass, name, age, embarked, sex, survived

# 取出姓名的length和title
def name(data):
    data['name_length'] = data['name'].apply(lambda x: len(x))
    data['name_title'] = data['name'].apply(lambda x: x.split(',')[1] if len(x.split(',')) > 1 else x.split(',')[0]).apply(lambda x: x.split()[0])
    del data['name']
    return data

def age(data):
    data['age_flag'] = data['age'].apply(lambda x: 1 if pd.isnull(x) else 0)
    grouped_age = data.groupby(['name_title', 'pclass'])['age']
    data['age'] = grouped_age.transform(lambda x: x.fillna(data['age'].mean()) if pd.isnull(x.mean()) else x.fillna(x.mean()))
    return data

# 用majority class fill null
def embark(data):
    data['embarked'] = data['embarked'].fillna('Southampton')
    return data

def label_encode(data, columns=['pclass','name_title','embarked', 'sex']):
    for col in columns:
        data[col] = data[col].apply(lambda x: str(x))
        new_cols = [col + '_' + i for i in data[col].unique()]
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)[new_cols]], axis=1)
        del data[col]
    return data


# #### 预处理输入数据  
# - 去掉不需要的特征  
# - 对某些特征进行变换

# In[ ]:


# TODO 
# 去掉row.names, home.dest, room, ticket, boat等属性
drop_columns = ['row.names', 'home.dest', 'room', 'ticket', 'boat'] 
data = data.drop(drop_columns, axis = 1)
data.head()


# In[ ]:


# TODO
# 利用name(), age(), embark(), dummies()等函数对数据进行变换
data = name(data)
data = age(data)
data = embark(data)
data = label_encode(data)
data.head()


# ####  调用决策树模型并预测结果

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn import tree

# 准备训练集合测试集， 测试集大小为0.2， 随机种子为33
trainX, testX, trainY, testY = train_test_split(data.iloc[:,1:], data.iloc[:,0], test_size=0.2, random_state=33)

# TODO 创建深度为3，叶子节点数不超过5的决策树
model = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf = 5)
model.fit(trainX, trainY)


# In[ ]:


from sklearn import metrics
def measure_performance(X, y, model, show_accuracy=True, show_classification_report=True, show_confussion_matrix=True):
    #TODO complete measure_performance函数
    y_pred = model.predict(X)
    
    if show_accuracy:
        print(("Accuracy:{0:.3f}".format(metrics.accuracy_score(y, y_pred)),"\n"))
    
    if show_classification_report:
        print("Classification report")
        print((metrics.classification_report(y, y_pred), "\n"))
    
    if show_confussion_matrix:
        print("Confusion matrix")
        print((metrics.confusion_matrix(y, y_pred), "\n"))


# In[ ]:


# TODO 调用measure_performance 观察模型在testX, testY上的表现
measure_performance(testX, testY, model)


# #### Bonus part: 利用简单特征直接调用决策树模型

# In[ ]:


# 利用 age, sex_male, sex_female做训练
sub_columns = ['age', 'sex_male', 'sex_female']
sub_trainX = trainX[sub_columns]
sub_testX = testX[sub_columns]
sub_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
sub_model.fit(sub_trainX, trainY)


# In[ ]:


measure_performance(sub_testX, testY, sub_model)


# #### 可视化决策树

# In[ ]:


import graphviz

dot_data = tree.export_graphviz(model, out_file=None, feature_names=trainX.columns) 

#TODO 生成graph文件
graph = graphviz.Source(dot_data) 
#graph.render("titanic") 
#graph.view()
graph


# #### 展示特征的重要性

# In[ ]:


# TODO 观察前20个特征的重要性
model.feature_importances_


# In[ ]:


columns = pd.DataFrame(data=trainX.columns, columns=['feature'])
importance = pd.DataFrame(data=model.feature_importances_, columns=['importance'])


# In[ ]:


pd.concat([columns, importance], axis=1).sort_values(by='importance', ascending=False)[:20]


# In[ ]:




