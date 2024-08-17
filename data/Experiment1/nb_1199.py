#!/usr/bin/env python
# coding: utf-8

# # Выпускной проект

# Чтобы оптимизировать производственные расходы, металлургический комбинат ООО «Так закаляем сталь» решил уменьшить потребление электроэнергии на этапе обработки стали. Вам предстоит построить модель, которая предскажет температуру стали.

# **Описание этапа обработки**  
# 
# Сталь обрабатывают в металлическом ковше вместимостью около 100 тонн. Чтобы ковш выдерживал высокие температуры, изнутри его облицовывают огнеупорным кирпичом. Расплавленную сталь заливают в ковш и подогревают до нужной температуры графитовыми электродами. Они установлены в крышке ковша. 
# 
# Из сплава выводится сера (десульфурация), добавлением примесей корректируется химический состав и отбираются пробы. Сталь легируют — изменяют её состав — подавая куски сплава из бункера для сыпучих материалов или проволоку через специальный трайб-аппарат (англ. tribe, «масса»).
# 
# Перед тем как первый раз ввести легирующие добавки, измеряют температуру стали и производят её химический анализ. Потом температуру на несколько минут повышают, добавляют легирующие материалы и продувают сплав инертным газом. Затем его перемешивают и снова проводят измерения. Такой цикл повторяется до достижения целевого химического состава и оптимальной температуры плавки.
# 
# Тогда расплавленная сталь отправляется на доводку металла или поступает в машину непрерывной разливки. Оттуда готовый продукт выходит в виде заготовок-слябов (англ. *slab*, «плита»).

# **Описание данных**
# 
# Данные состоят из файлов, полученных из разных источников:
# 
# - `data_arc.csv` — данные об электродах;
# - `data_bulk.csv` — данные о подаче сыпучих материалов (объём);
# - `data_bulk_time.csv` *—* данные о подаче сыпучих материалов (время);
# - `data_gas.csv` — данные о продувке сплава газом;
# - `data_temp.csv` — результаты измерения температуры;
# - `data_wire.csv` — данные о проволочных материалах (объём);
# - `data_wire_time.csv` — данные о проволочных материалах (время).
# 
# Во всех файлах столбец `key` содержит номер партии. В файлах может быть несколько строк с одинаковым значением `key`: они соответствуют разным итерациям обработки.

# In[ ]:


import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

from scipy import stats as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from pyod.models.knn import KNN
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Изучение данных

# ### Данные об электродах

# In[ ]:


data_arc = pd.read_csv('/datasets/final_steel/data_arc.csv')
data_arc.info()
data_arc.describe()


# In[ ]:


data_arc[data_arc['Реактивная мощность'] < 0]['Реактивная мощность']


# In[ ]:


len(data_arc['key'].unique())


# In[ ]:


data_arc.tail(10)


# In[ ]:


data_arc[data_arc['Реактивная мощность'] > 0]['Реактивная мощность'].hist(bins=100, figsize=(15,8))


# In[ ]:


data_arc['Активная мощность'].hist(bins=100, figsize=(15,8))


# Есть аномалия в данных по реактивной мощности  
# Время в формат object перевести в datetime  
# Данные активной и реактивной мощности смещенны влево  

# ### Данные о подаче сыпучих материалов (объём)

# In[ ]:


data_bulk = pd.read_csv('/datasets/final_steel/data_bulk.csv')
data_bulk.info()
data_bulk.describe()


# In[ ]:


data_bulk.head(10)


# In[ ]:


len(data_bulk['key'].unique())


# Столбец key перевезти в целочисленный формат  
# Имеется большие пропуски данных, скорее всего это связано с добавлением различного количество сыпучих материалов в смеси, в том числе и не добавлением, возможно это и ошибки датчика, но данные восстановить корректно невозможно.  
# Данные перевезти из Float в int
# Есть пропуски по ключам, возможно данные потерянны.
# 

# ### Данные о подаче сыпучих материалов (время)

# In[ ]:


data_bulk_time = pd.read_csv('/datasets/final_steel/data_bulk_time.csv')
data_bulk_time.info()
data_bulk_time.describe()


# In[ ]:


data_bulk_time


# In[ ]:


len(data_bulk_time['key'].unique())


# Отсутствие данных времени похоже на отсутствие данные подачи. Необходимо сравнить наны

# ### Данные о продувке сплава газом

# In[ ]:


data_gas = pd.read_csv('/datasets/final_steel/data_gas.csv')
data_gas.info()
data_gas.describe()


# In[ ]:


data_gas


# In[ ]:


len(data_gas['key'].unique())


# In[ ]:


data_gas['Газ 1'].hist(bins=100, figsize=(16,10))


# ### Результаты измерения температуры

# In[ ]:


data_temp = pd.read_csv('/datasets/final_steel/data_temp.csv')
data_temp.info()
data_temp.describe()


# In[ ]:


data_temp.tail(15)


# In[ ]:


len(data_temp['key'].unique())


# In[ ]:


data_temp['Температура'].hist(bins=100, figsize=(16,10))


# Имеются пропуски ключевого признака: последней температуры, скорее всего приведет к удалению всех данных по ключу

# ### Данные о проволочных материалах (объём)

# In[ ]:


data_wire = pd.read_csv('/datasets/final_steel/data_wire.csv')
data_wire.info()
data_wire.describe()


# In[ ]:


data_wire


# In[ ]:


len(data_wire['key'].unique())


# ### Данные о проволочных материалах (время)

# In[ ]:


data_wire_time = pd.read_csv('/datasets/final_steel/data_wire_time.csv')
data_wire_time.info()
data_wire_time.describe()


# In[ ]:


data_wire_time


# In[ ]:


len(data_wire_time['key'].unique())


# **Выводы:**
# 
# Имеется большие пропуски данных, скорее всего это связано с добавлением различного количество сыпучих материалов в смеси, в том числе и не добавлением, возможно это и ошибки датчика, но данные восстановить корректно невозможно.    
# Во многих таблицах отсутствуют данные уникальных номеров партии 'key'.  
# Есть неправильное использование типа числа с плавающей точкой, времени в формате object вместо datetime.   
# Используется русскоязычные названия столбцов использование пробелов в названиях, что может приводить к ошибкам при обращении. Нужно переименовать.  
# Количество записей у одной партии для каждой таблицы может различаться.  
# Аномальное значение реактивной мощности в таблице data_arc.  
# Данные требуют предобработки и группировки по признаку key.   
# Данные распределены нормально со смещением влево  
# (В графике о температуре смысла мало так как это температура на различных этапах, и этапов может быть разное количество в каждой партии)  
# upd. в некоторых партиях время последнего замера температуры происходило до конца нагрева

# **План решения задачи**

# Целевой признак конечная температура.  
# Подготовим данные.  
# Создадим таблицу, где просуммируем активную и реактивную мощность для каждой партии. Это снизит вычислительную нагрузку.   
# Создадим таблицу, где есть первый и последний замер температуры. (Если бы данные собирались четко по каждому этапу, то можно было бы использовать и промежуточную температуру)  
# Для таблиц data_bulk и data_wire заполним пропуски нулями (предполагаем, что пропуски не являются ошибкой, а означают отсутствие подачи)  
# Приведем типы данных к целочисленным для ускорения работы и временным для правильной отработки алгоритмов.  
# В каждой таблице преобразуем index в значение столбца key.  
# Выполним оценку мультиколлинеарности  
# Разобьем данные на train/test  
# Проведём обучение модели  

# ## Подготовка данных

# ### Переименуем столбцы на английский язык без пробелов

# In[ ]:


data_arc.columns = ['key', 'start_time','end_time', 'active_power', 'reactive_power']
data_gas.columns = ['key', 'gas']
data_temp.columns = ['key', 'measurement_time', 'temperature']
data_bulk.columns = ['key', 'bulk_1', 'bulk_2', 'bulk_3', 'bulk_4', 'bulk_5', 'bulk_6', 'bulk_7', 'bulk_8', 'bulk_9', 'bulk_10', 'bulk_11', 'bulk_12', 'bulk_13', 'bulk_14', 'bulk_15']
data_wire.columns = ['key', 'wire_1', 'wire_2', 'wire_3', 'wire_4', 'wire_5', 'wire_6', 'wire_7', 'wire_8', 'wire_9']


# In[ ]:


data_arc = data_arc.query('reactive_power > 0') # Избавимся от ошибоке в данных


# ### Создадим таблицу, где просуммируем активную и реактивную мощность для каждой партии.

# In[ ]:


data_arc_sum = pd.pivot_table(data_arc,
                             values=['active_power','reactive_power'],
                             index='key',
                             aggfunc={'active_power': np.sum,
                                      'reactive_power': np.sum})
data_arc_sum.columns = ['sum_active_power','sum_reactive_power']
data_arc_sum


# ### Создадим таблицу, где есть первый и последний замер температуры.

# In[ ]:


keys = []
for key in list(data_temp['key'].unique()):
    try:
        if (data_temp[data_temp['key'] == key]['measurement_time'].max() < 
            data_arc[data_arc['key'] == key]['end_time'].max()): # время последнего замера температуры происходило до конца нагрева
            keys.append(key)
    except:
        keys.append(key)
data_temp = data_temp.query('key not in @keys')


# In[ ]:


data_temp = data_temp.dropna() # избавимся от пропусков замеров


# In[ ]:


for i in (data_temp['key'].unique()): # удаляем партии с одним замером температуры
    if (data_temp['key']==i).sum() < 2:
        data_temp = data_temp[data_temp.key != i]


# In[ ]:


data_temp_time = data_temp.pivot_table(index=['key'], values=('temperature', 'measurement_time'), aggfunc=['first', 'last'])


# In[ ]:


data_temp_time.columns = ['first_time', 'first_temperature', 'last_time', 'last_temperature']
data_temp_time


# ### Присваеваем индексам номер партии

# In[ ]:


data_bulk = data_bulk.set_index('key')
data_gas = data_gas.set_index('key')
data_wire = data_wire.set_index('key')


# ### Собираем все по индексу

# In[ ]:


data = pd.concat([data_temp_time, data_arc_sum, data_bulk, data_gas, data_wire], axis=1, sort=False)


# In[ ]:


data


# In[ ]:


# Время замера не информативно так, как за это время происходилос разное колличество дествий
data = data.drop('first_time',axis=1)
data = data.drop('last_time',axis=1)
data = data.dropna(subset=['last_temperature'])
data = data.fillna(0)


# In[ ]:


data


# ### Исправляем типы данных

# In[ ]:


data['first_temperature']=data['first_temperature'].astype(int)
data['last_temperature']=data['last_temperature'].astype(int)
for i in range(1,16):
    data[f'bulk_{i}'] = data[f'bulk_{i}'].astype(int)


# In[ ]:


data.info()


# ### Проведем поиск параметров с высокой корреляцией 

# In[ ]:


pd.set_option('display.max_columns', None)
numeric_col = data.columns.values.tolist()

corr = data.loc[:,numeric_col].corr()
corr


# Высокая корреляция между sum_active_power - sum_reactive_power и bulk_9 - wire_8. Удалим по одному из столбцов.

# In[ ]:


data = data.drop(['sum_reactive_power', 'wire_8'], axis=1)


# ### Разобьем данные

# In[ ]:


features = data.drop('last_temperature', axis=1)
target = data['last_temperature']

features_train, features_test, target_train, target_test = train_test_split(
                                                            features, 
                                                            target, 
                                                            test_size=0.25, 
                                                            random_state=12345)


# ### Проведем оценку важности признаков

# In[ ]:


reg = RandomForestRegressor(n_estimators=100)
reg.fit(features, target)


# In[ ]:


df_feature_importance = pd.DataFrame(reg.feature_importances_, index=features.columns.values, columns=['feature importance']).sort_values('feature importance', ascending=False)


# In[ ]:


df_feature_importance


# In[ ]:


data_clean = data[list(df_feature_importance.head(13).index)]
data = pd.concat([data_clean, target], axis=1, sort=False)


# In[ ]:


data


# ### Проведем поиск аномалий.
# Для этого воспользуемся простым алгоритмом классификации  
# Без этого MSE до идеала не доводилось

# In[ ]:


model = KNN()
data_anomaly = data
model.fit(data_anomaly)
data_anomaly['anomaly'] =  model.predict(data_anomaly) == 1
anomaly_indexes = list(data_anomaly[data_anomaly['anomaly'] == 1].index)
data_knn = data.drop(anomaly_indexes)


# In[ ]:


len(anomaly_indexes)


# In[ ]:


### Повторно разобьем важные выборки без аномалий


# In[ ]:


features = data_knn.drop('last_temperature', axis=1)
target = data_knn['last_temperature']

features_train, features_test, target_train, target_test = train_test_split(
                                                            features, 
                                                            target, 
                                                            test_size=0.25, 
                                                            random_state=12345)


# ## Обучение модели

# ### LinearRegression

# In[ ]:


get_ipython().run_cell_magic('time', '', "regressor = LinearRegression()\nprint('# Train for mean_absolute_error')\nprint()\ncv_MAE_LR = (cross_val_score(regressor, \n                             features_train, \n                             target_train, \n                             cv=5, \n                             scoring='neg_mean_absolute_error').mean() * -1)\nprint('MAE LinearRegression =', cv_MAE_LR)\n")


# ### RandomForestRegressor

# In[ ]:


get_ipython().run_cell_magic('time', '', 'regressor = RandomForestRegressor() \nhyperparams = [{\'criterion\':[\'mse\'],\n                \'n_estimators\':[x for x in range(100, 200, 10)], \n                \'random_state\':[12345]}]\nclf = GridSearchCV(regressor, hyperparams, scoring=\'neg_mean_absolute_error\', cv=5)\nclf.fit(features_train, target_train)\nprint("Перебор параметров:")\nprint()\nmeans = clf.cv_results_[\'mean_test_score\']\nstds = clf.cv_results_[\'std_test_score\']\nfor mean, std, params in zip(means, stds, clf.cv_results_[\'params\']):\n    print("%0.4f for %r"% ((mean*-1), params))\nprint()\ncv_MAE_RFR = (max(means)*-1)\nprint()\nprint("Лучшие параметры")\nprint()\nbest_params_RFR = clf.best_params_\nprint(clf.best_params_)\n')


# ### CatBoostRegressor 

# In[ ]:


get_ipython().run_cell_magic('time', '', "regressor = CatBoostRegressor(verbose=False, random_state=12345)\ncv_MAE_CBR = (cross_val_score(regressor, \n                             features_train, \n                             target_train, \n                             cv=5, \n                             scoring='neg_mean_absolute_error').mean() * -1)\nprint('MAE CatBoostRegressor =', cv_MAE_CBR)\nbest_params_CBR = CatBoostRegressor(verbose=False, \n                                    random_state=12345).fit(features_train, \n                                        target_train).get_all_params()\n")


# ### LGBMRegressor

# In[ ]:


get_ipython().run_cell_magic('time', '', 'regressor = LGBMRegressor() \nhyperparams = [{\'num_leaves\':[x for x in range(10,15)], \n                \'learning_rate\':[0.05, 0.07, 0.9],\n                \'random_state\':[12345]}]\nclf = GridSearchCV(regressor, hyperparams, scoring=\'neg_mean_absolute_error\', cv=5)\nclf.fit(features_train, target_train)\nprint("Grid scores on development set:")\nprint()\nmeans = clf.cv_results_[\'mean_test_score\']\nstds = clf.cv_results_[\'std_test_score\']\nfor mean, std, params in zip(means, stds, clf.cv_results_[\'params\']):\n    print("%0.4f for %r"% ((mean*-1), params))\nprint()\ncv_MAE_LGBMR = (max(means)*-1)\nprint()\nprint("Лучшие параметры:")\nprint()\nbest_params_LGBMR = clf.best_params_\nprint(clf.best_params_)\n')


# ### XGBRegressor

# In[ ]:


get_ipython().run_cell_magic('time', '', 'regressor = XGBRegressor() \nhyperparams = [{\'learning_rate\':[x/100 for x in range(8, 19)],\n                \'random_state\':[12345],\n                 \'silent\':[True]}]\nclf = GridSearchCV(regressor, hyperparams, scoring=\'neg_mean_absolute_error\', cv=5)\nclf.fit(features_train, target_train)\nprint("Перебор параметров:")\nprint()\nmeans = clf.cv_results_[\'mean_test_score\']\nstds = clf.cv_results_[\'std_test_score\']\nfor mean, std, params in zip(means, stds, clf.cv_results_[\'params\']):\n    print("%0.6f for %r"% ((mean*-1), params))\nprint()\ncv_MAE_XGBR = (max(means)*-1)\nprint()\nprint("Лучшие параметры:")\nprint()\nbest_params_XGBR = clf.best_params_\nprint(clf.best_params_)\n')


# ## Тестирование моделей

# ### LinearRegression

# In[ ]:


get_ipython().run_cell_magic('time', '', "model = LinearRegression()\nmodel.fit(features_train, target_train)\ntarget_predict = model.predict(features_test)\ntest_MAE_LR = mean_absolute_error(target_predict, target_test)\nprint('MAE LinearRegression =', test_MAE_LR)\n")


# ### RandomForestRegressor 

# In[ ]:


get_ipython().run_cell_magic('time', '', "model = RandomForestRegressor()\nmodel.set_params(**best_params_RFR)\nmodel.fit(features_train, target_train)\ntarget_predict = model.predict(features_test)\ntest_MAE_RFR = mean_absolute_error(target_predict, target_test)\nprint('MAE RandomForestRegressor =', test_MAE_RFR)\n")


# ### CatBoostRegressor

# In[ ]:


get_ipython().run_cell_magic('time', '', "model = CatBoostRegressor(verbose=False)\nmodel.set_params(**best_params_CBR)\nmodel.fit(features_train, target_train)\ntarget_predict = model.predict(features_test)\ntest_MAE_CBR = mean_absolute_error(target_predict, target_test)\nprint('MAE CatBoostRegressor =', test_MAE_CBR)\n")


# ### LGBMRegressor

# In[ ]:


get_ipython().run_cell_magic('time', '', "model = LGBMRegressor()\nmodel.set_params(**best_params_LGBMR)\nmodel.fit(features_train, target_train)\ntarget_predict = model.predict(features_test)\ntest_MAE_LGBMR = mean_absolute_error(target_predict, target_test)\nprint('MAE LGBMRegressor =', test_MAE_LGBMR)\n")


# ### XGBRegressor

# In[ ]:


get_ipython().run_cell_magic('time', '', "model = XGBRegressor()\nmodel.set_params(**best_params_XGBR)\nmodel.fit(features_train, target_train)\ntarget_predict = model.predict(features_test)\ntest_MAE_XGBR = mean_absolute_error(target_predict, target_test)\nprint('MAE XGBRegressor =', test_MAE_XGBR)\n")


# ## Вывод

# Лучшая модель CatBoostRegressor, MAE = 5.79  
# Другие модели так же дают хороший результат и могут использоваться для предсказаний.  
# Для улучшения метрики использовали оценку важности гиперпараметров и поиск аномалий.
# 
