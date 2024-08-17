#!/usr/bin/env python
# coding: utf-8

# # АНСАМБЛИ МОДЕЛЕЙ

# In[64]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

#устраним ошибки со шрифтами
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']


# # Часть 1. Бэггинг

# ## Описание задачи
# 
# Используем данные страхового подразделения BNP Paribas из соревнования
# 
# https://www.kaggle.com/c/bnp-paribas-cardif-claims-management
# 
# Решается задача классификации страховых случаев:
#     1. Случаи, требующие дополнительных документов для подтвердения (0)
#     2. Случаи, которые можно подтверждать автоматически на основе имеющейся информации (1)

# ## Загрузка данных

# In[65]:


data = pd.read_csv('datasets/ensembles/train.csv')

data.head()


# Уменьшим размер данных для ускорения обучения, возмем случайную подвыборку 20% данных со стратификацией

# In[66]:


from sklearn.model_selection import StratifiedShuffleSplit

random_splitter = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=777)

for train_index, test_index in random_splitter.split(data, data.target):
    data = data.iloc[test_index]


# Разбиение на обучение и hold-out тест 70/30. Данных досттаточно много, поэтому можно принебречь честной кросс-валидацией и оценивать модель
# на тесте

# In[67]:


splitter = StratifiedShuffleSplit(n_splits=1, test_size=.3, random_state=777)

for train_index, test_index in splitter.split(data, data.target):
    d_train = data.iloc[train_index]
    d_test = data.iloc[test_index]
    
    y_train = data['target'].iloc[train_index]
    y_test = data['target'].iloc[test_index]


# ## Первичный анализ

# Размер датасета

# In[68]:


data.shape


# Распределение значений таргета (event rate)

# In[69]:


data.target.value_counts()/len(data)


# ## Предобработка данных

# Находим категориальные признаки
# 
# Чтобы в разы не увеличивать число признаков при построении dummi, будем использовать категориальные
# признаки с < 30 уникальных значений

# In[70]:


data.dtypes.head(50)


# In[71]:


cat_feat = list(data.dtypes[data.dtypes == object].index)

# закодируем пропущенные знаения строкой, факт пропущенного значения тоже может нести в себе информацию
data[cat_feat] = data[cat_feat].fillna('nan')

# отфильтруем непрерывные признаки
num_feat = [f for f in data if f not in (cat_feat + ['ID', 'target'])]

cat_nunique = d_train[cat_feat].nunique()
print(cat_nunique)
cat_feat = list(cat_nunique[cat_nunique < 30].index)


# In[72]:


from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression


# ## Композиция моделей одного семейства

# ### Будем использовать решеющие деревья
# 
# 1. Неустойчивы к входным данным
# 2. Склонны к переобучению
# 3. Быстро обучаются
# 
# => отличный выбор для построения композиций
# 
# **Создаем признаки для "деревянных" моделей**
# 
# 1. Заменяем пропуски на специальны значения -999, чтобы деревья могли их отличить
# 2. Создаем дамми-переменный для категорий

# In[73]:


dummi_train = pd.get_dummies(d_train[cat_feat], columns=cat_feat)
dummi_test = pd.get_dummies(d_test[cat_feat], columns=cat_feat)

dummi_cols = list(set(dummi_train) & set(dummi_test))

dummi_train = dummi_train[dummi_cols]
dummi_test = dummi_test[dummi_cols]

X_train = pd.concat([d_train[num_feat].fillna(-999), dummi_train], axis=1)

X_test = pd.concat([d_test[num_feat].fillna(-999), dummi_test], axis=1)


# Обучаем решающее дерево
# 
# Немного ограничим глубину и минимальное количество объектов в листе для уменьшения переобучения

# In[74]:


dummy_train = pd.get_dummies(d_train[cat_feat], columns=cat_feat)
dummy_test = pd.get_dummies(d_test[cat_feat], columns=cat_feat)

dummy_cols = list(set(dummy_train) & set(dummy_test))

dummy_train = dummy_train[dummy_cols]
dummy_test = dummy_test[dummy_cols]


X_train = pd.concat([d_train[num_feat].fillna(-999),
                     dummy_train], axis=1)

X_test = pd.concat([d_test[num_feat].fillna(-999),
                     dummy_test], axis=1)


# In[75]:


from sklearn.tree import DecisionTreeClassifier

clf_tree = DecisionTreeClassifier(max_depth=15, min_samples_leaf=20)
clf_tree.fit(X_train, y_train)


# #### Считаем ROS AUC

# In[76]:


def calc_auc(y, y_pred, plot_label='', prin=True):
    fpr, tpr, _ = roc_curve(y, y_pred)
    auc_val = auc(fpr, tpr)
    if prin:
        print(f'ROC AUC: {auc_val:.4f}')
    if plot_label:
        plt.plot(fpr, tpr, label=plot_label)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
    return auc_val


# Предсказываем  вероятность класса 1 и считаем ROC AUC

# In[77]:


y_pred_test = clf_tree.predict_proba(X_test)[:, 1]
y_pred_train = clf_tree.predict_proba(X_train)[:, 1]

print('Train')
calc_auc(y_train, y_pred_train, 'train')
print('Test')
calc_auc(y_test, y_pred_test, 'test')
plt.legend();


# ### Бэггинг

# In[78]:


import numpy as np


# In[79]:


np.arange(y_train.shape[0])


# In[80]:


BAGGING_ITERS = 20

y_pred_test = np.zeros_like(y_pred_test)
y_pred_train = np.zeros_like(y_pred_train)

for i in tqdm.trange(BAGGING_ITERS):
    new_index = np.random.choice(np.arange(y_train.shape[0]), size=y_train.shape[0], replace=True)
    clf_tree.fit(X_train.iloc[new_index], y_train.iloc[new_index])
    
    y_pred_test += clf_tree.predict_proba(X_test)[:, 1]
    y_pred_train += clf_tree.predict_proba(X_train)[:, 1]

y_pred_test /= BAGGING_ITERS
y_pred_train /= BAGGING_ITERS


# In[81]:


print('Train')
calc_auc(y_train, y_pred_train, 'train')
print('Test')
calc_auc(y_test, y_pred_test, 'test')
plt.legend();


# ### Бэггинг

# Используем готовый алгоритм из sklearn

# In[82]:


from sklearn.ensemble import BaggingClassifier

bag_clf = BaggingClassifier(n_estimators=20, base_estimator=clf_tree, n_jobs=-1)

get_ipython().run_line_magic('time', 'bag_clf.fit(X_train, y_train)')

y_pred_test = bag_clf.predict_proba(X_test)[:, 1]
y_pred_train = bag_clf.predict_proba(X_train)[:, 1]


# In[83]:


print('Train')
calc_auc(y_train, y_pred_train, 'train')
print('Test')
calc_auc(y_test, y_pred_test, 'test')
plt.legend();


# # Часть 2. Случайный лес
# 
# Бэггинг + случайные подпространства = случайный лес
# 
# **Важные гиперпараметры алгоритма**
# 
# а. Параметры деревьев
#     1. criterion - критерий построения дерева
#     2. max_dept - максимальная глубина дерева(обычно 10-20, больше глубина -> больше риск переобучения)
#     3. min_samples_leaf - минимальное число объектов в листе (обычно 20+, больше объектов -> меньше
#                                                              риск переобучения)
# b. Параметры леса
#     1. n_estimators - количество деревьев (чем больше, тем лучше)
#     2. max_features - число признаков случайного подпространства
#     3. bootstrap - использовать ли бэггинг
#     4. n_jobs - количество потоков для одновременного построения деревьев (большая прибавка к скорости на
#                                                                            многоядерных процессарах)

# In[84]:


'минимальное число объектов в листе'.upper()


# In[85]:


from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=20, max_features=.8, n_jobs=-1)

get_ipython().run_line_magic('time', 'clf_rf.fit(X_train, y_train)')


# In[86]:


y_pred_rf_test = clf_rf.predict_proba(X_test)[:, 1]
y_pred_rf_train = clf_rf.predict_proba(X_train)[:, 1]

print('Train')
calc_auc(y_train, y_pred_rf_train, 'train')
print('Test')
calc_auc(y_test, y_pred_rf_test, 'test')
plt.legend();


# #### Важность признаков

# В sklearn - усредненное по всем деревьям в ансамбле колчество сплитов по признаку, взвешенное на прирост
# информации (information gain) и долю объектов в вершине, в которой производится этот сплит
# 
# это не единственный вариант, см. здесь:
# 
# https://medium.com/@ceshine/feature-importance-measures-for-tree-models-part-i-47f187c1a2c3
# 
# Важность признаков случайного леса лежат в атрибуте **feature\_importances\_**

# In[87]:


imp = pd.Series(clf_rf.feature_importances_, X_train.columns)
imp.sort_values(ascending=False)


# In[88]:


imp.sort_values(ascending=False).iloc[:20].plot(kind='barh');


# # Часть 3. Композиции моделей разных типов

# ### Линейная комбинация моделей разного типа
# 
# Смешаем дерево и логистическую регрессию
# 
# **Создадим признаки для логистической регрессии**
# 
# 1. Заменяем пропуски на медианы
# 2. Создаем индикаторы пропущенных значений
# 3. Создаём дамми-переменные для категорий

# In[97]:


d_train[num_feat + cat_feat].isnull().astype(np.int8).add_suffix('_NaN')


# In[91]:


from sklearn.preprocessing import StandardScaler

train_median = d_train[num_feat].median()

X_train_lin = pd.concat([d_train[num_feat].fillna(train_median),
                       d_train[num_feat + cat_feat].isnull().astype(np.int8).add_suffix('_NaN'),
                       dummy_train], axis=1)

X_test_lin = pd.concat([d_test[num_feat].fillna(train_median),
                       d_test[num_feat + cat_feat].isnull().astype(np.int8).add_suffix('_NaN'),
                       dummy_test], axis=1)

scaler = StandardScaler()
scaler.fit(X_train_lin[num_feat])

X_train_lin[num_feat] = scaler.transform(X_train_lin[num_feat])
X_test_lin[num_feat] = scaler.transform(X_test_lin[num_feat])


# Обучим логистическую регрессию

# In[93]:


clf_lr = LogisticRegression(solver='liblinear', penalty='l1', C=.1)

clf_lr.fit(X_train_lin, y_train)


# In[99]:


y_pred_lin_test = clf_lr.predict_proba(X_test_lin)[:, 1]
y_pred_lin_train = clf_lr.predict_proba(X_train_lin)[:, 1]

print('Train')
calc_auc(y_train, y_pred_lin_train, 'train')
print('Test')
calc_auc(y_test, y_pred_lin_test, 'test')
plt.legend();


# Будем строить линейную комбинацию вида
# 
# $y=\alpha y + (1 - \alpha)y_2$
# 
# Параметр $\alpha$ переберем по сетке от 0 до 1, оценивая качество на тестовой выборке

# In[103]:


np.linspace(0, 1, 100)


# In[102]:


aucs = []
alpha_space = np.linspace(0, 1, 100)
for alpha in alpha_space:
    y_pred_weight = alpha * y_pred_lin_test + (1 - alpha) * y_pred_rf_test
    aucs.append(calc_auc(y_test, y_pred_weight, prin=False))

aucs = np.array(aucs)

max_ind = np.where(aucs == aucs.max())[0]
alpha = alpha_space[max_ind]

plt.plot(alpha_space, aucs)
plt.plot(alpha_space[max_ind], aucs[max_ind], 'o', c='r')
plt.xlabel('alpha')
plt.ylabel('auc')

# Итоговое взвешенное предсказание
y_pred_weight = alpha * y_pred_lin_test * (1 - alpha) * y_pred_rf_test


# Сравним 3 метода (приблизим график ROC кривой, чтобы увидеть разницу)

# In[104]:


print('Weighted:')
calc_auc(y_test, y_pred_weight, 'weighted')
print('Log regression:')
calc_auc(y_test, y_pred_lin_test, 'LR')
print('Random forest:')
calc_auc(y_test, y_pred_rf_test, 'RF')
plt.legend();
plt.xlim(.2, .5)
plt.ylim(.5, .8)


# ### Стэкинг

# #### Среднее значение таргета

# Создадим новые признаки, на основе категориальных переменных. Каждому уникальному знаению $V$ переменной $X_i$
# сопоставим среднее значение тергета среди всех объектов, у которых переменная $X_i$ принимает значение $V$
# 
# Новый признак со средним значением таргета в категории можно считать за предсказание вероятности красса 1
# простого классификатора "усреднения"
# 
# Опишем класс этого классификатора

# In[119]:


class MeanClassifier():
    
    def __init__(self, col):
        self._col = col
        
    
    
    def fit(self, X, y):
        self._y_mean = y.mean()
        self._means = y.groupby(X[self._col].astype(str)).mean()
        
    
    def predict_proba(self, X):
        new_feature = X[self._col].astype(str).map(self._means.to_dict()).fillna(self._y_mean)
        return np.stack([1-new_feature, new_feature], axis=1)
        


# Делаем предсказание по фолдам кросс-валидации. **Главное не допустить утечки информации!** <br>
# Опишем цункцию для стекинга.

# In[117]:


def get_meta_features(clf, X_train, y_train, X_test, stack_cv):

    meta_train = np.zeros_like(y_train, dtype=float)
    meta_test = np.zeros_like(y_test, dtype=float)
    
    for i, (train_ind, test_ind) in enumerate(stack_cv.split(X_train, y_train)):
        
        clf.fit(X_train.iloc[train_ind], y_train.iloc[train_ind])
        meta_train[test_ind] = clf.predict_proba(X_train.iloc[test_ind])[:, 1]
        meta_test += clf.predict_proba(X_test)[:, 1]
    
    return meta_train, meta_test / stack_cv.n_splits


# #### Стэкинг нескольких моделей

# 0. Средние значения
# 1. Random Forest
# 2. Log reg
# 3. SVM

# Посмотрим, какое качество дает линейный SVM
# 
# для совместимости с общим кодом стекинга немного модифицируем класс SVM

# In[108]:


from sklearn.svm import LinearSVC

def norm(x):
    return (x - x.min()) / (x.max() - x.min())

class SVMWrapper(LinearSVC):
    
    def predict_proba(self, X):
        df = norm(self.decision_function(X))
        return np.stack([1-df, df], axis=1)


clf_svm = SVMWrapper(C=.001)
clf_svm.fit(X_train_lin, y_train)


y_pred_svm_test = clf_svm.predict_proba(X_test_lin)[:, 1]
y_pred_svm_train = clf_svm.predict_proba(X_train_lin)[:, 1]

print('Train')
calc_auc(y_train, y_pred_svm_train, 'train')
print('Test')
calc_auc(y_test, y_pred_svm_test, 'test')
plt.legend();


# Теперь получим мета признаки для 3х моделей:
# * SVM
# * Logreg
# * Random Forest
# и средних значений по каждой категориальной переменной

# In[121]:


from sklearn.model_selection import StratifiedKFold

stack_cv = StratifiedKFold(n_splits=10, random_state=555, shuffle=True)

meta_train = []
meta_test = []
col_names = []

print('mean futures...')
for c in cat_nunique.index.tolist():
    clf = MeanClassifier(c)
    
    meta_tr, meta_te = get_meta_features(clf, d_train, y_train, d_test, stack_cv)
    
    meta_train.append(meta_tr)
    meta_test.append(meta_te)
    col_names.append(f'mean_pred_{c}')
    
print('SVM futures...')
meta_tr, meta_te = get_meta_features(clf_svm, X_train_lin, y_train, X_test_lin, stack_cv)

meta_train.append(meta_tr)
meta_test.append(meta_te)
col_names.append('svm_pred')

print('LR futures...')
meta_tr, meta_te = get_meta_features(clf_lr, X_train_lin, y_train, X_test_lin, stack_cv)

meta_train.append(meta_tr)
meta_test.append(meta_te)
col_names.append('lr_pred')

print('RF features...')
meta_tr, meta_te = get_meta_features(clf_rf, X_train, y_train, X_test, stack_cv)

meta_train.append(meta_tr)
meta_test.append(meta_te)
col_names.append('rf_pred')


# In[122]:


X_meta_train = pd.DataFrame(np.stack(meta_train, axis=1), columns=col_names)
X_meta_test = pd.DataFrame(np.stack(meta_test, axis=1), columns=col_names)


# #### Стэкинг мета-признаков с помощью LR

# Используем регуляризованную лог регрессию в качестве алгоритма второго уровня

# In[125]:


clf_lr_meta = LogisticRegression(penalty='l2', C=1, max_iter=200)

clf_lr_meta.fit(X_meta_train, y_train)


# In[126]:


y_pred_meta_test = clf_lr_meta.predict_proba(X_meta_test)[:, 1]

calc_auc(y_test, y_pred_meta_test, 'test')
plt.legend();


# #### Посмотрим на коэффициенты объединяющей линейной модели

# Получим интерпретацию общей модели

# In[127]:


pd.Series(clf_lr_meta.coef_.flatten(), index=X_meta_train.columns).plot(kind='barh')


# # Домашняя работа
# 
# #### Простая
# 1. Теперь решаем задачу регрессии - предскажем цены на недвижимость. Использовать датасет https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data (train.csv)
# 2. Данных немного, поэтому необходимо использовать 10-fold кросс-валидацию для оценки качества моделей
# 3. Построить случайный лес, вывести важность признаков
# 4. Обучить стекинг как минимум 3х моделей, использовать хотя бы 1 линейную модель и 1 нелинейную
# 5. Для валидации модели 2-го уровня использовать отдельный hold-out датасет, как на занятии
# 6. Показать, что использование ансамблей моделей действительно улучшает качество (стекинг vs другие модели сравнивать на hold-out)
# 7. В качестве решения:
#     Jupyter notebook с кодом, комментариями и графиками
# 
# #### Средняя
# 0. Все то же, что и в части 1, плюс:
# 1. Попробовать другие оценки важности переменных, например Boruta
# http://danielhomola.com/2015/05/08/borutapy-an-all-relevant-feature-selection-method/#comments
# 3. Изучить extremely randomized trees (ExtraTreesRegressor в sklearn), сравнить с Random Forest
# 4. Проводить настройку гиперпараметров для моделей первого уровня в стекинге (перебирать руками и смотреть на CV или по сетке: GridSearchCV, RandomizedSearchCV)
# 5. Попробовать другие алгоритмы второго уровня
# 6. Сделать сабмиты на kaggle (минимум 3: отдельные модели vs стекинг), сравнить качество на локальной валидации и на leaderboard
# 7. В качестве решения:
#     * Jupyter notebook с кодом, комментариями и графиками
#     * сабмит на kaggle (ник на leaderboard)

# In[ ]:




