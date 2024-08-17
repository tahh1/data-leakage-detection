#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_auc_score, roc_curve

get_ipython().system('pip freeze > requirements.txt')


# # Загрузка данных

# Описания полей:
# 
# client_id - идентификатор клиента
# 
# education - уровень образования
# 
# sex - пол заемщика
# 
# age - возраст заемщика
# 
# car - флаг наличия автомобиля
# 
# car_type - флаг автомобиля иномарки
# 
# decline_app_cnt - количество отказанных прошлых заявок
# 
# good_work - флаг наличия “хорошей” работы
# 
# bki_request_cnt - количество запросов в БКИ
# 
# home_address - категоризатор домашнего адреса
# 
# work_address - категоризатор рабочего адреса
# 
# income - доход заемщика
# 
# foreign_passport - наличие загранпаспорта
# 
# sna - связь заемщика с клиентами банка
# 
# first_time - давность наличия информации о заемщике
# 
# score_bki - скоринговый балл по данным из БКИ
# 
# region_rating - рейтинг региона
# 
# app_date - дата подачи заявки
# 
# default - флаг дефолта по кредиту (target)

# In[2]:


# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# In[3]:


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


# In[4]:


df_train.info()


# In[5]:


print(('Соотношение количество значений 1 к количеству значений 0 в целевой переменной в тренировочной выборке:',
      int(df_train.default.value_counts()[1] / df_train.default.value_counts()[0] *100), '%'))


# In[6]:


df_test.info()


# In[7]:


pd.read_csv('../input/sf-dst-scoring/sample_submission.csv')


# Количество строк в тестовом датасете не соответствует количеству строк в sample_submission, что странно.

# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[8]:


# Объединение тренировочной и тестовой выборки для совместной обработки.
# Для последующего разделения создаётся флаг is_test
df_train['is_test'] = 0
df_test['is_test'] = 1

df = pd.concat([df_train, df_test]).reset_index(drop=True) # без reset_index НЕ РАБОТАЕТ!!!!!!!!
df.head()


# In[9]:


df.tail()


# In[10]:


df.shape


# # Заполнение пропусков

# In[11]:


df.isna().sum()


# Пропуски содержатся только в признаке education.

# In[12]:


df.education.value_counts()


# In[13]:


# Пропущенные значения признака education заполняются модой
df.education[df.education.isna()] = df.education.mode()[0]


# In[14]:


df.isna().sum()


# Пропуски ликвидированы

# # Обработка даты

# In[15]:


df.app_date = pd.to_datetime(df.app_date)
df.app_date.dt.year.value_counts()


# In[16]:


df.app_date.dt.month.value_counts()


# In[17]:


df[df.is_test==1].app_date.dt.month.value_counts()


# Интервал дат в выборке: январь-апрель 2014. Даты распределены в тренировочной и в тестовой выборках примерно одинаково.

# In[18]:


# новый признак: день недели подачи заявки
df['app_weekday'] = df.app_date.dt.weekday
df.app_weekday.value_counts()


# На выходные дни приходится значительно меньше заявок, чем на будни.

# In[19]:


# преобразование даты в число
df.app_date = df.app_date.dt.strftime('%m%d').astype('int64')


# In[20]:


# новый признак: количество заявок в данный день
apps_per_date = df.app_date.value_counts()


# In[21]:


df['apps_per_day'] = df.app_date.map(apps_per_date)


# # Разделение признаков на группы

# In[22]:


# бинарные признаки
bin_cols = ['sex', 'car', 'car_type', 'good_work', 'foreign_passport']

# категориальные признаки
cat_cols = ['education', 'home_address', 'work_address',
        'sna', 'first_time',
            'app_weekday'
           ]

# числовые признаки
num_cols = ['app_date', 'age', 'decline_app_cnt', 'score_bki',
        'bki_request_cnt', 'region_rating', 'income',
            'apps_per_day'
           ]


# # Обработка признаков

# ## Числовые признаки

# In[23]:


df[df.is_test == 0][num_cols].hist(figsize=(25,10),bins=100);


# In[24]:


df[df.is_test == 1][num_cols].hist(figsize=(25,10),bins=100);


# Распределение признаков в тренировочной и тестовой выборках одинаково.

# In[25]:


# попытка прологарифмировать признаки для приближения к виду нормального распределения
(np.log(df[num_cols]+1)).hist(figsize=(25,10),bins=100)


# Некоторые признаки после логарифмирования выглядят более приближенными к виду нормального распеределения.

# In[26]:


for i in ['age', 'bki_request_cnt', 'decline_app_cnt', 'income']:
    df[i] = np.log(df[i]+1)


# In[27]:


sns.heatmap(df[num_cols].corr())


# Корреляции незначительны.

# In[28]:


# # стандартизация:
sscaler = StandardScaler()
df[num_cols] = pd.DataFrame(sscaler.fit_transform(df[num_cols]),columns=num_cols)


# In[29]:


df[num_cols].describe()


# In[30]:


# оценка значимости числовых признаков:
imp_num = pd.Series(f_classif(df[df.is_test==0][num_cols], df[df.is_test==0]['default'])[0], index = num_cols)
imp_num.sort_values(inplace = True)
imp_num.plot(kind = 'barh')


# Из числовых признаков наиболее значимым является score_bki

# ## Бинарные признаки

# In[31]:


bin_cols


# In[32]:


# преобразование в числа
label_encoder = LabelEncoder()

for i in bin_cols:
    df[i] = label_encoder.fit_transform(df[i])  


# In[33]:


df[bin_cols].hist(bins=2)


# In[34]:


df[bin_cols].describe()


# In[35]:


# # стандартизация:
# df[bin_cols] = pd.DataFrame(sscaler.fit_transform(df[bin_cols]),columns=bin_cols)


# In[36]:


# оценка значимости бинарных признаков:
imp_cat = pd.Series(mutual_info_classif(df[df.is_test == 0][bin_cols], df[df.is_test == 0]['default'],
                                        discrete_features=True), index=bin_cols)
imp_cat.sort_values(inplace=True)
imp_cat.plot(kind='barh')


# Из бинарных признаков наиболее значимыми являются foreign_passport и car_type.

# ## Категориальные признаки

# In[37]:


cat_cols


# In[38]:


df[cat_cols].info()


# In[39]:


df_train['education'].value_counts() / len(df_train)


# In[40]:


df_test['education'].value_counts() / len(df_test)


# In[41]:


df_dummies = pd.get_dummies(df[cat_cols].astype('object'))
df_dummies


# In[42]:


dummy_cols = df_dummies.columns


# In[43]:


df = pd.concat([df, df_dummies], axis=1)


# In[44]:


df.columns


# In[45]:


df.drop(columns=cat_cols, inplace=True)


# In[46]:


df.info()


# In[47]:


cat_cols=dummy_cols


# # Модель

# In[48]:


x = df[df.is_test == 0].drop(columns=['client_id','default'])
y = df[df.is_test == 0]['default']


# In[49]:


x_pred = df[df.is_test == 1].drop(columns=['client_id','default'])


# In[50]:


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=13)


# In[51]:


model = LogisticRegression(random_state=13, max_iter=1000, solver='lbfgs')
model.fit(x_train, y_train)


# In[52]:


probs = model.predict_proba(x_test)
probs = probs[:, 1]


# In[53]:


fpr, tpr, threshold = roc_curve(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)

plt.figure()
plt.plot([0, 1], label='Baseline', linestyle='--')
plt.plot(fpr, tpr, label='Regression')
plt.title('Logistic Regression ROC AUC = %0.5f' % roc_auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc='lower right')
plt.show()


# In[54]:


y_pred = model.predict(x_test)


# In[55]:


pd.Series(y_pred).value_counts()


# In[56]:


pd.Series(probs).describe()


# In[57]:


confusion_matrix(y_test, y_pred)


# # Регуляризация

# In[58]:


# Добавим типы регуляризации
penalty = ['l2']


# In[59]:


# Зададим ограничения для параметра регуляризации
C = np.logspace(0, 2, 20)
C


# In[60]:


# Создадим гиперпараметры
hyperparameters = dict(C=C, penalty=penalty)


# In[61]:


# Создаем сетку поиска с использованием 5-кратной перекрестной проверки
model = LogisticRegression(random_state=13, max_iter=1000, solver='lbfgs')
clf = GridSearchCV(model, hyperparameters, cv=5, verbose=0)


# In[62]:


best_model = clf.fit(x_train, y_train)


# In[63]:


# View best hyperparameters
print(('Лучшее Penalty:', best_model.best_estimator_.get_params()['penalty']))
print(('Лучшее C:', best_model.best_estimator_.get_params()['C']))


# In[64]:


p = best_model.best_estimator_.get_params()['penalty']
c = best_model.best_estimator_.get_params()['C']


# In[65]:


model = LogisticRegression(penalty=p,C=c, random_state=13, max_iter=1000, solver='lbfgs')


# In[66]:


model.fit(x_train, y_train)


# In[67]:


probs = model.predict_proba(x_test)
probs = probs[:, 1]


# In[68]:


fpr, tpr, threshold = roc_curve(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)

plt.figure()
plt.plot([0, 1], label='Baseline', linestyle='--')
plt.plot(fpr, tpr, label='Regression')
plt.title('Logistic Regression ROC AUC = %0.5f' % roc_auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc='lower right')
plt.show()


# Регуляризация не показала видимого улучшения качества модели.

# In[69]:


y_pred = model.predict(x_test)


# In[70]:


confusion_matrix(y_test, y_pred)


# # Результат

# In[71]:


y_pred = model.predict(x_pred)
y_pred_proba = model.predict_proba(x_pred)


# In[72]:


x_pred


# In[73]:


y_pred


# In[74]:


y_pred_proba[:,1]


# In[75]:


len(y_pred)


# In[76]:


submission_proba = df_test.client_id.to_frame()
submission_proba['default'] = y_pred_proba[:,1]
submission_proba


# In[77]:


submission_proba.to_csv('submission.csv', index = False)


# In[ ]:




