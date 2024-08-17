#!/usr/bin/env python
# coding: utf-8

# # Описание проекта

# Допустим, вы работаете в добывающей компании «ГлавРосГосНефть». Нужно решить, где бурить новую скважину.
# 
# Вам предоставлены пробы нефти в трёх регионах: в каждом 10 000 месторождений, где измерили качество нефти и объём её запасов. Постройте модель машинного обучения, которая поможет определить регион, где добыча принесёт наибольшую прибыль. Проанализируйте возможную прибыль и риски техникой *Bootstrap.*
# 
# Шаги для выбора локации:
# 
# - В избранном регионе ищут месторождения, для каждого определяют значения признаков;
# - Строят модель и оценивают объём запасов;
# - Выбирают месторождения с самым высокими оценками значений. Количество месторождений зависит от бюджета компании и стоимости разработки одной скважины;
# - Прибыль равна суммарной прибыли отобранных месторождений.

# <a id="1"></a>
# # 1. Загрузка и подготовка данных

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


geo_data_0 = pd.read_csv('/datasets/geo_data_0.csv')
geo_data_1 = pd.read_csv('/datasets/geo_data_1.csv')
geo_data_2 = pd.read_csv('/datasets/geo_data_2.csv')


# In[ ]:


# функция для выведения данных для анализа датафреймов
def research(data):

    display(data.head())
    print((data.info())) 
    display(data.describe())
    print(('дубликаты', geo_data_1.duplicated().sum()))


# <a id="11"></a>
# #### Регион 0

# In[ ]:


research(geo_data_0)


# <a id="12"></a>
# #### Регион 1

# In[ ]:


research(geo_data_1)


# <a id="13"></a>
# #### Регион 2

# In[ ]:


research(geo_data_2)


# ### Вывод:  
# * Дубликатов нет, 
# * id не несет нужной для расчетов информации, можно удалить
# * Есть 0 значения product, необходимо проанализировать
#     

# Избавимся от столбцов id, так как они имеют для модели пользы

# In[ ]:


geo_data_0 = geo_data_0.drop(['id'], axis=1)
geo_data_1 = geo_data_1.drop(['id'], axis=1)
geo_data_2 = geo_data_2.drop(['id'], axis=1)


# <a id="14"></a>
# #### Нулевые значения product

# In[ ]:


print((geo_data_0[geo_data_0['product'] == 0]['product'].count()))
print((geo_data_1[geo_data_1['product'] == 0]['product'].count()))
print((geo_data_2[geo_data_2['product'] == 0]['product'].count()))


# In[ ]:


print((len(geo_data_0['product'].unique())))
print((len(geo_data_1['product'].unique())))
print((len(geo_data_2['product'].unique())))


# In[ ]:


geo_data_1['product'].value_counts()


# In[ ]:


plt.figure(figsize=[12,9])
plt.hist(geo_data_1['product'], bins = 12)
plt.title("Гистограмма распределения по объему скважин")
plt.show()


# ### Вывод
# Большое количество нулевых значений, равное распределение и малая вариабельность данных: возможно данные подверглись вручную категоризации, и вследствие этого получилось большое количество нулевых значений.  Нулевые значения нельзя заменить на среднее. В связи с большим количеством этих данных удалять тоже не стоит. 

# <a id="2"></a>
# # 2. Обучение и проверка модели

# <a id="21"></a>
# Для обучения модели подходит только линейная регрессия  
# Напишем функцию: На вход она получает регион, делит его на признаки и целевой признак, разбивает данные на обучающую и валидационные выборки, делаем стандартизацию данных рассчитывает метрики RMSE.  
# Так же рассчитаем среднее значение запасов месторождений в регионе

# In[ ]:


def region_prediction(data):
   
    features = data.drop(["product"], axis=1)
    target = data["product"]
    features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.25, random_state=12345)
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_valid = scaler.transform(features_valid)
    
    lr = LinearRegression(normalize=False)
    lr.fit(features_train, target_train)
    predictions = lr.predict(features_valid)
    predictions = pd.Series(predictions)
    rmse = np.sqrt(mean_squared_error(target_valid, predictions))  
    stock_mean = data['product'].mean()
    stock_mean_pred = predictions.mean()
    return predictions, rmse, stock_mean, stock_mean_pred, target_valid.reset_index(drop=True)


# In[ ]:


pred_0, rmse_0, stock_mean_0, stock_mean_pred_0, target_valid_0 = region_prediction(geo_data_0)
print(('RMSE модели в регионе 0 = {:.3f}'.format(rmse_0)))
print(('Средний запас предсказанного сырья 0 = {:.3f} т. баррелей'.format(stock_mean_pred_0)))


# In[ ]:


pred_1, rmse_1, stock_mean_1, stock_mean_pred_1, target_valid_1 = region_prediction(geo_data_1)
print(('RMSE модели в регионе 1 = {:.3f}'.format(rmse_1)))
print(('Средний запас предсказанного сырья 1 = {:.3f} т. баррелей'.format(stock_mean_pred_1)))


# In[ ]:


pred_2, rmse_2, stock_mean_2, stock_mean_pred_2, target_valid_2 = region_prediction(geo_data_2)
print(('RMSE модели в регионе 2 = {:.3f}'.format(rmse_2)))
print(('Средний запас предсказанного сырья 2 = {:.3f} т. баррелей'.format(stock_mean_pred_2)))


# ### Вывод:
# Лучшие показатели RMSE модели в Регионе 1 (RMSE = 0.893), этот регион более предсказуем. Возможно это следствии явно отредактированных данных.  
# 
# Средний запас предсказанного сырья 1 = 68.729 т. баррелей, что меньше чем в двух других регионах(92.593  и 94.965)  
# В этих регионах высокий показатель RMSE, что говорит о том что модель работает хуже, это дает более непредсказуемые результаты, RMSE для постоянного предсказания среднего таргета по Регионам = 44.67 и 44,66, соответственно)
# 

# <a id="3"></a>
# # 3. Подготовка к расчёту прибыли

# <a id="31"></a>
# Сохраним ключевые значения для расчетов в отдельных переменных

# In[ ]:


BUDGET = 10000000000 
BEST_WELLS = 200
BARREL_PROFIT = 450*1000
TOTAL_WELLS = 500 
RISK_LOSS = 0.025 


# In[ ]:


DRILLING_COST = BUDGET/BEST_WELLS
print(('Бюджет бурения одного месторождения, руб:', DRILLING_COST))


# In[ ]:


print(('Объём сырья для безубыточной разработки новой скважины  = {:.3f} т. баррелей'.format(DRILLING_COST / BARREL_PROFIT)))


# In[ ]:


print(('Средний запас сырья региона 0 = {:.3f} т. баррелей'.format(stock_mean_0)))
print(('Средний запас сырья региона 1 = {:.3f} т. баррелей'.format(stock_mean_1)))
print(('Средний запас сырья региона 2 = {:.3f} т. баррелей'.format(stock_mean_2)))


# ### Вывод:  
# * Средний запас сырья в скважинах меньше, чем необходимый объем сырья для безубыточности.  
# * Необходимо для безубыточной разработки новой скважины  = 111.111 т. баррелей
# * Необходимо разрабатывать только перспективные скважины

# <a id="4"></a>
# # 4. Расчёт прибыли и рисков 

# <a id="41"></a>
# #### Функция для расчёта прибыли по выбранным скважинам и предсказаниям модели

# In[ ]:


def profit(prediction, target):
    data = pd.concat([prediction, target],axis=1)
    data.columns = ['prediction','target']
    data = data.sort_values(by = 'prediction', ascending = False)[:BEST_WELLS]
    return (data['target'].sum() * BARREL_PROFIT - BUDGET)



# In[ ]:


revenue_0 = profit(pred_0, target_valid_0)

print(('Прибыль для полученного объёма сырья региона 0 = {} млн. руб'.format(revenue_0  / 10e6)))


# In[ ]:


revenue_0 = profit(pred_1, target_valid_1)

print(('Прибыль для полученного объёма сырья региона 1 = {} млн. руб'.format(revenue_0 / 10e6)))


# In[ ]:


revenue_0 = profit(pred_2, target_valid_2)

print(('Прибыль для полученного объёма сырья региона 2 = {} млн. руб'.format(revenue_0 / 10e6)))


# <a id="42"></a>
# #### Функция для расчетов прибыли и рисков для каждого региона с применением техники Bootstrap 

# In[ ]:


def estimate(prediction, target):
    state = np.random.RandomState(12345)
    values = []
    for i in range(1000):
        target_subsample = target.sample(n=500, replace=True, random_state=state)
        pred_subsample = prediction[target_subsample.index]
        values.append(profit(pred_subsample, target_subsample))
    values = pd.Series(values)
    mean = np.mean(values) / 10e6
    lower = values.quantile(0.025) / 10e6
    upper = values.quantile(0.975) / 10e6
    confidence_interval = (lower, upper)
    risk_of_loss = (values < 0).sum() / values.count()
    
    print(('Средняя прибыль = {:.2f} млн.руб.'.format(mean)))
    print(('95% доверительный интервал от {:.2f} до {:.2f} млн.руб.'.format(lower, upper)))
    print(('Процент риска {:.1%}'.format(risk_of_loss)))


# #### Регион 0

# In[ ]:


region_0 = estimate(pred_0, target_valid_0)


# #### Регион 1

# In[ ]:


region_1 = estimate(pred_1, target_valid_1)


# #### Регион 2

# In[ ]:


region_2 = estimate(pred_2, target_valid_2)


# <a id="5"></a>
# ### Общий вывод
# Изучили предоставленные данные трёх регионов, обучили модель линейной регрессии и сделали расчёт прибыли и рисков для каждого региона.  
# 
# Наиболее перспективный регион 1. Добыча нефти в этом регионе связана с наименьшим риском и предполагается получит наибольшую среднюю прибыль.   
# Остальные регионы имеют вероятность убытков больше 2.5%, что не соответствует условию поставленной задачи.
