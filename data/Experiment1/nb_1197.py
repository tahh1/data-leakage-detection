#!/usr/bin/env python
# coding: utf-8

# ## Описание проекта

# Вам нужно защитить данные клиентов страховой компании «Хоть потоп». Разработайте такой метод преобразования данных, чтобы по ним было сложно восстановить персональную информацию. Обоснуйте корректность его работы.
# 
# Нужно защитить данные, чтобы при преобразовании качество моделей машинного обучения не ухудшилось. Подбирать наилучшую модель не требуется.

# ## Загрузка данных

# In[ ]:


# импорт библиотек

import pandas as pd
import sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[ ]:


# <чтение файла с данными с сохранением в data_full>

data = pd.read_csv('/datasets/insurance.csv')


# In[ ]:


# изучаю датафрейм

print((data.info()))
data.head()


# In[ ]:


data.describe()


# In[ ]:


data["Пол"].value_counts()


# **Вывод:**  
# Данные состроят из 5000 строк  
# Признаки: Пол, Возраст, Зарплата, Члены, семьи  
# Целевой признак: количество страховых выплат  
# Данные в предобработке не нуждаются  
# Данные разделены по полу почти в равном колличестве
# 

# ## Умножение матриц

# В этом задании вы можете записывать формулы в *Jupyter Notebook.*
# 
# Чтобы записать формулу внутри текста, окружите её символами доллара \\$; если снаружи —  двойными символами \\$\\$. Эти формулы записываются на языке вёрстки *LaTeX.* 
# 
# Для примера мы записали формулы линейной регрессии. Можете их скопировать и отредактировать, чтобы решить задачу.
# 
# Работать в *LaTeX* необязательно.

# Обозначения:
# 
# - $X$ — матрица признаков (нулевой столбец состоит из единиц)
# 
# - $y$ — вектор целевого признака
# 
# - $P$ — матрица, на которую умножаются признаки
# 
# - $w$ — вектор весов линейной регрессии (нулевой элемент равен сдвигу)

# Предсказания:
# 
# $$
# a = Xw
# $$
# 
# Задача обучения:
# 
# $$
# w = \arg\min_w MSE(Xw, y)
# $$
# 
# Формула обучения:
# 
# $$
# w = (X^T X)^{-1} X^T y
# $$

# **Признаки умножают на обратимую матрицу. Изменится ли качество линейной регрессии?**

# **Ответ:** Не изменится.

# **Обоснование:**  
# 
# Признаки умножают на обратимую матрицу $P$ : 
# 
# $$
# w_1 = ((XP)^T XP)^{-1} (XP)^T y
# $$
# 

# Транспонированное произведение матриц равно произведению транспонированных матриц, взятых в обратном порядке
# 
# $$
# w_1 = (P^T X^T XP)^{-1} P^T X^T  y
# $$

# Используя формулу
# $$
# (A * B)^{-1} = B^{-1} * A^{-1}  
# $$  
# 
# где $A$ и $B$ квадратные  
# 
# $P$, $P^T$ и $X^T X$ квадратные  
# 
# 
# $$
# w_1= P^{-1} (X^T X)^{-1} (P^T)^{-1} P^T X^T  y
# $$
# 

# Умножение матрицы на обратную матрицу равно единичной матрице  
# $$
# w_1= P^{-1} (X^T X)^{-1} I X^T  y
# $$
# 
# Умножение любой матрицы на единичную равно этой самой матрице.
# $$
# w_1= P^{-1} (X^T X)^{-1} X^T  y
# $$

# $$
# a_1 = XPw_1
# $$  
# 
# $$
# a_1 = X P P^{-1} (X^T X)^{-1} X^T  y
# $$  
# 
# $$
# a_1 = X (X^T X)^{-1} X^T  y
# $$  
# 
# $$
# a_1 = Xw
# $$

# **Вывод:**  
# Преобразовав формулы мы выяснили что при умножении признаков на обратимую матрицу качество не меняется

# ## Алгоритм преобразования

# **Алгоритм**  
# Необходимо умножить матрицу признаков на матрицу шифрования так как при умножении количество столбцов матрицы A должно быть равно количеству строк матрицы, полученная матрица будет иметь количество строк матрицы A и количество столбцов матрицы B. Поэтому размер матрицы щифрования на которую умножают равен 4х4
# 
# Матрицу шифрования сгенерируем с помощью np.random.normal()

# **Обоснование**
# 

# In[ ]:


#разобьем data на признаки и целевой признак
features = data.drop('Страховые выплаты', axis=1)
target = data['Страховые выплаты']


# In[ ]:


# Создадим обратимую квадратную матрицу со случайными числами размерностью столбцов features
state = np.random.RandomState(12345)
matrix_cr=np.random.normal(size=(features.shape[1],features.shape[1]))


# In[ ]:


# зашифруем даннные путем умножения матриц
features_cr=features @ matrix_cr
features_cr


# **Вывод**  
# После умножения размер матрицы признаков и кодированных признаков совпадает

# ## Проверка алгоритма

# ### Качество линейной регресии на нешифрованных данных

# In[ ]:


model = LinearRegression()

model.fit(features, target)

predictions = model.predict(features)

print(('R2:', r2_score(target,predictions)))


# ### Качество линейной регресии c шифрованием данных

# In[ ]:


model = LinearRegression()

model.fit(features_cr, target)

predictions = model.predict(features_cr)

print(('R2:', r2_score(target,predictions)))


# **Вывод:**  
# Качество моделей с шифрованием при помощи умножения на обратимую матрицу со случайными числами не изменяется  
# Небольшое расхождение предположу связанно с тем, что операции с float накапливают небольшую погрешность.
