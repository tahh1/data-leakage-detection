#!/usr/bin/env python
# coding: utf-8

# ![header](https://i.ibb.co/n1h4QM5/header.jpg)
# 
# 

# # Report
# 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))


# In[3]:


trainset = pd.read_csv('/kaggle/input/bigdata2021-rl-recsys/trainset.csv',' ')


# In[4]:


trainset


# In[5]:


trainset.info()


# In[6]:


# there are a total of 260087 rows, we randomly choose 10k data for EDA purpose
trainset_10k = trainset.sample(n=10000)


# In[7]:


trainset_10k


# In[8]:


# we first try the easiest: we want to see the distribution of ppl who bought 0, 1, 2, ... 9 items
cnt_dict = {}  # how many ppl bought 0, 1, 2, ... 9 items
for index, row in trainset_10k.iterrows():
    item_ids = row['exposed_items'].split(',')
    is_bought = row['labels'].split(',')
    bought_amount = 0
    for i in range(0, len(is_bought)):
        if is_bought[i] == '1':
            bought_amount += 1
    if bought_amount not in cnt_dict:
        cnt_dict[bought_amount] = 1
    else:
        cnt_dict[bought_amount] += 1

cnt_dict
            


# In[9]:


data_display = []
for i in range(0, 10):
    data_display.append(cnt_dict[i])


# In[10]:


data_display


# In[11]:


import seaborn as sns
 
from matplotlib import pyplot as plt
import numpy as np
plt.plot(data_display, color = 'r')
plt.title('Distribution of People who bought X items')
plt.xlabel('# of Items')
plt.ylabel('# of People')
plt.grid(True)

plt.show()
# so most people tend to buy 8 items, 2671 ppl out of 10k


# In[12]:


x = np.arange(len(data_display))
plt.bar(x, height=data_display)

plt.xticks(x, [i for i in range(0, 10)])
plt.ylabel('# of People', fontsize=20)
plt.xlabel('# of items bought', fontsize=20)
plt.title('Distribution of People who bought X amount of items', fontsize=20)
plt.show()

# most people bought 8 items


# In[13]:


# distribution of user_protrait: choose the top 10 user protraits as classifying features
cnt_dict = {}  
for index, row in trainset_10k.iterrows():
    user_protraits = row['user_protrait'].split(',')
    for user_protrait in user_protraits:
        if user_protrait not in cnt_dict:
            cnt_dict[user_protrait] = 1
        else:
            cnt_dict[user_protrait] += 1
    


# In[14]:


user_protrait_cnt_dict = cnt_dict


# In[15]:


user_protrait_cnt_list = [(k,v) for k, v in sorted(list(user_protrait_cnt_dict.items()), key=lambda item: item[1], reverse=True)]


# In[16]:


#user_protrait_cnt_list
top_10_features = user_protrait_cnt_list[:10]
top_10_features


# In[17]:


import seaborn as sns
 
from matplotlib import pyplot as plt
import numpy as np

fig,ax = plt.subplots()
ax.set_xticks([i for i in range(0, 10)])
ax.set_xticklabels([x[0] for x in top_10_features])

plt.plot([x[1] for x in top_10_features], color = 'r')
plt.xticks(rotation=45)
plt.grid(True)
plt.title('Distribution of User Protrait')
plt.xlabel('user protraits')
plt.ylabel('People with this protrait')
plt.show()


# In[18]:


# choose top 10 user protrait for training

import seaborn as sns
 
from matplotlib import pyplot as plt
import numpy as np


x = np.arange(len(top_10_features))
plt.bar(x, height=[x[1] for x in top_10_features])


plt.xticks(x, [x[0] for x in top_10_features], rotation=45)
plt.ylabel('# of people with this protrait', fontsize=20)
plt.xlabel('User Protrait ID', fontsize=20)
plt.title('Top 10 User Protraits', fontsize=20)
plt.show()


# In[19]:


trainset_10k


# In[20]:


trainset_10k


# In[21]:


columns = ['user_id'] + [x[0] for x in top_10_features] + ['item_' + str(i) for i in range(1, 382)]


# In[22]:


# from StringIO import StringIO

processed_trainset = pd.read_csv('/kaggle/input/processed-trainset/processed_trainset.csv', ',')


# In[23]:


# processed_trainset = pd.DataFrame([], columns = columns)


# In[24]:


# processed_trainset


# In[25]:


trainset_10k.iloc[0]['user_protrait']


# In[26]:


trainset_10k.iloc[2]


# In[27]:


# columns = ['user_id'] + [x[0] for x in top_10_features] + ['item_' + str(i) for i in range(1, 382)]
# processed_trainset = pd.DataFrame([], columns = columns)
# cnt = 3
# for index, row in trainset_10k.iterrows():
#     user_protraits = row['user_protrait'].split(',')
#     user_id = str(row['user_id'])
#     to_insert = {}
#     # mark features
#     for column in processed_trainset:
#         if column == 'user_id':
#             to_insert['user_id']=user_id
#         elif not column.startswith('item_'):
#             if column in user_protraits:
#                 to_insert[column]=1
#             else:
#                 to_insert[column]=0
#     # mark items
#     exposed_items = row['exposed_items'].split(',')
#     labels = row['labels'].split(',')

#     for i in range(0, len(exposed_items)):
#         cur_item = 'item_' + exposed_items[i]
# #         print(cur_item)
# #         print(labels[i])
#         to_insert[cur_item] = labels[i]  
    
# #     for i in range(1, 382):
# #         cur_item_id = 'item_' + str(i)
# #         if cur_item_id not in to_insert:
# #             to_insert[cur_item_id] = np.nan
#     # print(to_insert)
#     processed_trainset = processed_trainset.append(to_insert, ignore_index=True)
# #     cnt -= 1
#     if cnt <= 0:
#         break
# processed_trainset.to_csv('processed_trainset.csv', index=False)


# In[28]:


processed_trainset.iloc[4]


# In[29]:


processed_trainset.iloc[1]['item_377']


# In[30]:


# processed_trainset.to_csv('processed_trainset.csv', index=False)


# In[31]:


trainset_10k


# In[32]:


processed_trainset


# In[33]:


processed_trainset.iloc[1]


# In[34]:


trainset[trainset['user_id']==122197]['user_protrait']


# In[35]:


trainset[trainset['user_id']==122197]


# In[36]:


processed_trainset.iloc[1]['item_171']


# In[37]:


from sklearn.ensemble import RandomForestClassifier

cur_item = 'item_1'

cur_Xy = processed_trainset[~processed_trainset[cur_item].isnull()]
train_test_perc = 0.8
cur_len = len(cur_Xy)

cur_Xy_train = cur_Xy.head(int(cur_len * train_test_perc))
cur_Xy_test = cur_Xy.tail(cur_len - int(cur_len * train_test_perc))


# selected_rows = df[~df['Age'].isnull()]


# In[38]:


print(cur_len)
print((len(cur_Xy_train)))
print((len(cur_Xy_test)))


# In[39]:


cur_Xy_train


# In[40]:


top_10_feature_ids = [x[0] for x in top_10_features]


# In[41]:


X_train = cur_Xy_train[[x[0] for x in top_10_features]]
y_train = cur_Xy_train[cur_item]

X_test = cur_Xy_test[[x[0] for x in top_10_features]]
y_test = cur_Xy_test[cur_item]


# In[42]:


X_train


# In[43]:


y_train


# In[44]:


model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)


# In[45]:


# prediction accuracy == 84%
(predictions==y_test).sum() / len(y_test)


# In[46]:


from sklearn.ensemble import RandomForestClassifier

all_rf = {}
accuracy_dict = {}
cnt = 10
for i in range(1, 382):
    cur_item = 'item_' + str(i)
    cur_Xy = processed_trainset[~processed_trainset[cur_item].isnull()]
    if len(cur_Xy) == 0:
        print(('Item {} never showed up.'.format(i)))
        continue
    if len(cur_Xy) < 10:
        print(('Item {} showed up less than 10 times.'.format(i)))
        continue    
    train_test_perc = 0.8
    cur_len = len(cur_Xy)

    cur_Xy_train = cur_Xy.head(int(cur_len * train_test_perc))
    cur_Xy_test = cur_Xy.tail(cur_len - int(cur_len * train_test_perc))    
    X_train = cur_Xy_train[[x[0] for x in top_10_features]]
    y_train = cur_Xy_train[cur_item]

    X_test = cur_Xy_test[[x[0] for x in top_10_features]]
    y_test = cur_Xy_test[cur_item]
    
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)    
    
    accuracy = (predictions==y_test).sum() / len(y_test)
    accuracy_dict[i] = accuracy
    print(('Item {} has accuracy {}'.format(i, accuracy)))
    
    X = cur_Xy[[x[0] for x in top_10_features]]
    y = cur_Xy[cur_item]
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X, y)
    
    all_rf[i] = model
        
#     cnt -= 1
    if cnt <=0:
        break
 


# In[47]:


len(list(accuracy_dict.keys()))


# In[48]:


accuracy_dict


# In[49]:


all_rf


# In[50]:


testset = pd.read_csv('/kaggle/input/bigdata2021-rl-recsys/track1_testset.csv',' ')
testset


# In[56]:


import time

test_result = pd.DataFrame([], columns = ['id', 'category'])

start_time = int(time.time())
cnt = 1
for index, row in testset.iterrows():
#     print(row)
    item_ids = row['exposed_items'].split(',')
    user_protraits = row['user_protrait'].split(',')
#     user_protraits = row['user_protrait'].split(',')    
    cur_X = pd.DataFrame([], columns = top_10_feature_ids)

    to_insert = {}
    for column in cur_X:
        if column in user_protraits:
            to_insert[column]=1
        else:
            to_insert[column]=0
#     print('to_insert')
#     print(to_insert)
    cur_X = cur_X.append(to_insert, ignore_index=True)
#     print('cur_X')        
#     print(cur_X)
    predictions = []
    for item_id in item_ids:
        item_id = int(item_id)
        if item_id in all_rf:
            cur_pred = all_rf[item_id].predict(cur_X)
#             print('Pred item {} is {}'.format(item_id, cur_pred))
            if cur_pred > 0.5:
                cur_pred = 1
            else:
                cur_pred = 0
        else:
            # buy is more than not-buy
            cur_pred = 1
        predictions.append(cur_pred)
#     print('predictions')
#     print(predictions)

#     test_result_to_insert['id'] = row['user_id']
    predictions = [str(p) for p in predictions]
#     test_result_to_insert['category'] = ' '.join(predictions)   
    
    test_result_to_insert = {'id': row['user_id'], 'category': ' '.join(predictions)}   
    test_result = test_result.append(test_result_to_insert, ignore_index=True)
    cnt += 1
    if cnt % 5000 == 0:
        cur_time = int(time.time())
        used_time_in_secs = cur_time - start_time
        test_result.to_csv('top_' + str(cnt) + '.csv', index=False)
        print(('{} done, used {} secs'.format(cnt, used_time_in_secs)))


print('finally all done')
test_result.to_csv('submission.csv', index=False)


# In[ ]:


test_result

