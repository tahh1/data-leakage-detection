#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system('pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887')

get_ipython().system('apt update && apt install -y libsm6 libxext6')


# In[ ]:


from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[ ]:





# In[ ]:





# In[ ]:


df_raw = pd.read_csv('../input/train.csv', low_memory=False)
test_raw = pd.read_csv('../input/test.csv', low_memory=False)


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)


# In[ ]:


display_all(df_raw.tail().T)


# In[ ]:


df_raw.head()


# In[ ]:


df_raw.drop(['Name','Ticket','Cabin','Embarked'], axis=1, inplace=True)
test_raw.drop(['Name','Ticket','Cabin','Embarked'], axis=1, inplace=True)


# In[ ]:


train_cats(df_raw)
apply_cats(test_raw, df_raw)


# In[ ]:


df_raw.Sex.cat.categories


# In[ ]:


df_raw.Sex = df_raw.Sex.cat.codes


# In[ ]:


display_all(df_raw.isnull().sum().sort_index()/len(df_raw))


# In[ ]:


os.makedirs('tmp', exist_ok=True)
df_raw.to_feather('tmp/titanic-raw')


# In[ ]:


df_raw = pd.read_feather('tmp/titanic-raw')


# In[ ]:


df, y, nas = proc_df(df_raw, 'Survived')
test, _, nas = proc_df(test_raw, na_dict=nas)


# In[ ]:


m = RandomForestClassifier(n_jobs=-1, oob_score=True, n_estimators=60)
m.fit(df,y)
m.score(df,y)

test_prediction = m.predict(test)


# In[ ]:


submission = pd.DataFrame({'PassengerId': test['PassengerId'],'Survived': test_prediction})
submission.to_csv('submission.csv', index=False)

