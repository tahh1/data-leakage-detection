#!/usr/bin/env python
# coding: utf-8

# # Проект для «Викишоп»

# Интернет-магазин «Викишоп» запускает новый сервис. Теперь пользователи могут редактировать и дополнять описания товаров, как в вики-сообществах. То есть клиенты предлагают свои правки и комментируют изменения других. Магазину нужен инструмент, который будет искать токсичные комментарии и отправлять их на модерацию. 
# 
# Обучите модель классифицировать комментарии на позитивные и негативные. В вашем распоряжении набор данных с разметкой о токсичности правок.
# 
# Постройте модель со значением метрики качества *F1* не меньше 0.75. 
# 
# **Инструкция по выполнению проекта**
# 
# 1. Загрузите и подготовьте данные.
# 2. Обучите разные модели. 
# 3. Сделайте выводы.
# 
# Для выполнения проекта применять *BERT* необязательно, но вы можете попробовать.
# 
# **Описание данных**
# 
# Данные находятся в файле `toxic_comments.csv`. Столбец *text* в нём содержит текст комментария, а *toxic* — целевой признак.

# ## Подготовка

# In[ ]:


import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer 
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')


# In[ ]:


toxic_comments = pd.read_csv('/datasets/toxic_comments.csv')
toxic_comments.head(5)


# In[ ]:


toxic_comments.info()


# In[ ]:


display(toxic_comments['toxic'].value_counts())


# **Вывод**  
# Классы несбалансированы. Необходимо учитывать это при обучении моделей.

# Напишем функцию для лемматизации и очистки текста

# In[ ]:


"""# функция лемматизации и удаления лишних символов
m = WordNetLemmatizer()

def lemmatize_text(text):
    text = text.lower()
    lemm_text = "".join(m.lemmatize(text))
    cleared_text = re.sub(r'[^a-zA-Z]', ' ', lemm_text) 
    return " ".join(cleared_text.split())
"""


# In[ ]:


# функция лемматизации и удаления лишних символов

def clear_text(text):
    re_list = re.sub(r"[^a-zA-Z']", ' ', text)
    re_list = re_list.split()
    re_list = " ".join(re_list)
    return re_list

m = WordNetLemmatizer()

def lemmatize_text(text):
    word_list = nltk.word_tokenize(text)
    
    return ' '.join([m.lemmatize(w) for w in word_list])


# In[ ]:


toxic_comments['text'] = toxic_comments['text'].apply(clear_text)


# In[ ]:


toxic_comments['lemm_text'] = toxic_comments['text'].apply(lemmatize_text)


# In[ ]:


toxic_comments


# In[ ]:


toxic_comments = toxic_comments.drop(['text'], axis=1)


# Создадим корпус из лематизированных и очищеных тексов

# In[ ]:


corpus = toxic_comments['lemm_text'].values


# Разобьем выборки

# In[ ]:


features = corpus
target = toxic_comments['toxic'].values

train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=12345)


# In[ ]:


stopwordss = set(stopwords.words('english'))

count_tf_idf = TfidfVectorizer(stop_words = stopwordss)
train_features = count_tf_idf.fit_transform(train_features)


# In[ ]:


test_features = count_tf_idf.transform(test_features)


# ## Обучение

# ### LogisticRegression

# In[ ]:


lr_model = LogisticRegression()
hyperparams = [{'C':[10],   # так же подбирал [0.1, 1, 3]
                'class_weight':['balanced']}]
clf = GridSearchCV(lr_model, hyperparams, scoring='f1',cv=3)
clf.fit(train_features, train_target)
print("Лучшие параметры модели:")
print()
LR_best_params = clf.best_params_
print(LR_best_params)
print()
print(('F1:', clf.best_score_))


# ### CatBoostClassifier 

# In[ ]:


train_featurescat =  train_features.toarray() 
valid_featurescat =  valid_features.toarray() 
test_featurescat = test_features.toarray() 
cat_model = CatBoostClassifier(eval_metric="F1", 
                                   iterations=100, 
                                   max_depth=6, 
                                   learning_rate=0.9, 
                                   random_state=43)
cat_model.fit(train_featurescat, train_target, verbose=20)
print("Лучшие параметры модели:")
print()
CAT_best_params = clf.best_params_
print(CAT_best_params)
print()
print(('F1:', clf.best_score_))


# F1: 0.7601224906430758   

# ### LightGBM model

# Закомитил LightGBM model потому что в тренажере обрабатываются больше часа

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# подбирал параметры кросс валидацией\n\nLightGBM_model = LGBMClassifier()\nhyperparams = [{\'max_depth\' : [-1], # -1, 1\n                \'learning_rate\':[0.1], # 0.03, 0.1, 0.3\n                \'n_estimators\' : [1000],  # 200, 500, 1000\n                \'random_state\':[12345]}]\nclf = GridSearchCV(LightGBM_model, hyperparams, scoring=\'f1\',cv=3)\nclf.fit(train_features, train_target)\nprint("Лучшие параметры модели:")\nprint()\nLGBM_best_params = clf.best_params_\nprint(LGBM_best_params)\nprint()\nprint(\'F1:\', clf.best_score_)\n')


# 
# F1: 0.7679683884224614  
# Wall time: 9min 24s

# ## Выводы

# ### LogisticRegression

# In[ ]:


get_ipython().run_cell_magic('time', '', "lr_model = LogisticRegression()\nlr_model.set_params(**LR_best_params)\nlr_model.fit(train_features, train_target)\nprediction = lr_model.predict(test_features)\nf1 = f1_score(test_target, prediction)\nprint('F1 регрессии:', f1)\nprint()\nprint('Матрица ошибок')\nprint(confusion_matrix(test_target, prediction))\nprint()\n")


# ### CatBoostClassifier

# In[ ]:


cat_model = LogisticRegression()
cat_model.set_params(**CAT_best_params)
cat_model.fit(train_features, train_target)
prediction = cat_model.predict(test_features)
f1 = f1_score(test_target, prediction)
print(('F1', f1))
print()
print('Матрица ошибок')
print((confusion_matrix(test_target, prediction)))
print()


# F1 CatBoost: 0.7551954913702007
# 
# Матрица ошибок  
# [[14191   170]  
#  [  525  1072]]

# ### LightGBM model

# In[ ]:


LightGBM_model = LogisticRegression()
LightGBM_model.set_params(**LGBM_best_params)
LightGBM_model.fit(train_features, train_target)
prediction = LightGBM_model.predict(test_features)
f1 = f1_score(test_target, prediction)
print(('F1:', f1))
print()
print('Матрица ошибок')
print((confusion_matrix(test_target, prediction)))
print()


# 
# F1 регрессии: 0.7600950118764845  
# 
# Матрица ошибок  
# [[13730   617]  
#  [  248  1363]]

# **Вывод**
# Все модели имеют удовлетворительное значение F1. Лучше всех справилась LightGBM model. Но у LogisticRegression меньше ложно отрицательных предсказаний. Значит меньше токсичных коментариев пройдут мимо модерации. То что у LogisticRegression болльше ложно положительных предсказаний не страшно. Так как после модерации они вернуться обратно.
# 
# Такрим образом, для бизнеса, лучше подходит модель LogisticRegression
