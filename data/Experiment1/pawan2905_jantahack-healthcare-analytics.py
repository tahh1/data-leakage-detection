#!/usr/bin/env python
# coding: utf-8

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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import textwrap
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Scikit-Learn ML Libraries :

from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.metrics import *


# In[4]:


from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold,KFold


# In[5]:


from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))


# In[6]:


train_df = pd.read_csv("../input/janatahack-healthcare-analytics-ii/Train/train.csv")
test_df = pd.read_csv('../input/janatahack-healthcare-analytics-ii/test.csv')
sample_df = pd.read_csv("../input/janatahack-healthcare-analytics-ii/sample_submission.csv")


# In[7]:


train_df.head()


# In[8]:


test_df.head()


# In[9]:


print(("Shape of Training Data ... "+str(train_df.shape)))
print(("Shape of Testing Data ... "+str(test_df.shape)))


# In[10]:


# Genersting the pie chart
def pie_chart(df,col,path):
  label = df[col].value_counts().index.tolist()
  fig = plt.figure(figsize=(10,6))
  ax = (df[col].value_counts()*100.0 /len(df))\
  .plot.pie(startangle=90,autopct='%.1f%%', labels =label, fontsize=12)                                                                           
  ax.set_title('% '+str(col))
  # plt.savefig(path+str(col1)+'.png')
  plt.show()
    
# Relation between the categorical variable    
def rel_cat(df,x_axis,y_axis,path,stacked=None):
    temp =pd.crosstab(df[x_axis],df[y_axis])
    temp.plot(kind='bar',stacked=stacked,grid=False)
    plt.xlabel(str(x_axis),weight='bold',fontsize=12)
    plt.ylabel(str(y_axis),weight='bold',fontsize=12)
    plt.title(str(x_axis)+'_'+'and'+'_'+str(y_axis),weight='bold',fontsize=14)
    plt.xticks(rotation=0,fontsize=12)
    plt.yticks(fontsize=12)
    labels = df[x_axis].value_counts().index.tolist()
    labels.sort()
    labels=[textwrap.fill(text,10) for text in labels]
    pos = np.arange(len(labels)) 
    plt.xticks(pos, labels)
#     plt.legend()
#     plt.savefig(path+str(x_axis)+'_'+'and'+'_'+str(y_axis)+'.jpg')    
    plt.show()


# In[11]:


train_df.isnull().sum()


# In[ ]:





# In[12]:


pie_chart(train_df,'Hospital_type_code',8)


# In[13]:


sns.set(font_scale = 1.5)
plt.figure(figsize=(15,8))
ax = sns.countplot(x='Stay',data=train_df)
plt.xlabel("Category of Stay",weight='bold',fontsize=18)
plt.ylabel("Count Category of Stay",weight='bold',fontsize=18)
plt.title("Count of Stay by each category",weight='bold',fontsize=24)

#adding the text labels
rects = ax.patches
labels = train_df['Stay'].value_counts().tolist()
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=18)

c_labels = train_df['Stay'].value_counts().index.tolist()
c_name=[textwrap.fill(text,10) for text in c_labels]
pos = np.arange(len(c_name)) 
plt.xticks(pos, c_name)


plt.show


# In[14]:


# Check the cardinality of each columns
cardinality_df = pd.DataFrame()
l = []
for i in range(len(train_df.columns)):
#     print(i)
    distinct_count = train_df[train_df.columns[i]].value_counts().count()
    l.append(distinct_count)
col_name = train_df.columns.tolist()
cardinality_df["Columns_Name"]= col_name
cardinality_df['Cardinality_Count'] = l
cardinality_df


# In[15]:


pie_chart(train_df,'Severity of Illness',8)


# In[16]:


sns.set(font_scale = 1)
plt.figure(figsize=(10,6))
t1 = train_df.groupby(['Type of Admission','Severity of Illness'])[['Stay']].count().add_prefix('count_of_').reset_index()
t2 = t1.pivot('Type of Admission','Severity of Illness','count_of_Stay')
t2.plot(kind='bar',stacked=True)
plt.legend(loc=2)
plt.show()


# In[17]:


sns.set(font_scale = 1)
plt.figure(figsize=(10,6))
rel_cat(train_df,'Type of Admission','Severity of Illness',8,stacked=None)


# In[18]:


train_df.columns


# In[19]:


cols = ['Hospital_type_code','City_Code_Hospital',
       'Hospital_region_code'
       ,'Department'
       ,'Ward_Type',
       'Ward_Facility_Code',
       'Bed Grade','Type of Admission','Severity of Illness','Age']


# In[20]:


# Concatenate train and test data into single DataFrame - df :

train_df['is_train'] = 1
test_df['is_train'] = 0
df = pd.concat([train_df,test_df])


# In[21]:


from sklearn.preprocessing import LabelEncoder

for i in cols:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i].astype('str'))


# In[22]:


df.shape


# In[23]:


df['Stay'].value_counts()


# In[24]:


# Mapping Values to Label ENCODED Values :

df['Stay'] = df['Stay'].map({'0-10':0,'11-20':1,'21-30':2,
                             '31-40':3,'41-50':4,'51-60':5,'61-70':6,
                             '71-80':7,'81-90':8,'91-100':9,'More than 100 Days':10})


# In[25]:


# Get Back train data from df with a condition on column is_train == 1 :

train = df[df['is_train'] == 1]


# In[26]:


train.head()


# In[27]:


# split train into 5 folds and apply random forest and check accuracy of each fold

predictor_train = train.drop(['Stay','is_train','case_id'],axis=1)
target_train    = train['Stay']


# In[28]:


predictor_test = test_df.drop(['is_train','case_id'],axis=1)


# In[29]:


def data_encoding( encoding_strategy , encoding_data , encoding_columns ):
    
    if encoding_strategy == "LabelEncoding":
        Encoder = LabelEncoder()
        for column in encoding_columns :
            encoding_data[ column ] = Encoder.fit_transform(tuple(encoding_data[ column ]))
        
    elif encoding_strategy == "OneHotEncoding":
#         display(encoding_data[encoding_columns])
        encoding_data = pd.get_dummies( encoding_data  )
        
    elif encoding_strategy == "TargetEncoding":
        ## Code Coming soon
        print("TargetEncoding")

    else :
        encoding_data = pd.get_dummies( encoding_data[encoding_columns]  )
        
    dtypes_list =['float64','float32','int64','int32']
    # BEST CODE : 0.6872386379302422
#     encoding_data.astype( dtypes_list[0] ).dtypes # UNCOMMENTED EARLIER
    # NEW CODE : 0.6872386379302422 - NO CHANGE !!!
    # encoding_data.astype( dtypes_list[0] ).dtypes - COMMENTED NOW
    
    return encoding_data


# In[30]:


encoding_columns  = cols
encoding_strategy = [ "OneHotEncoding", "LabelEncoding", "TargetEncoding", "ELSE"]

predictor_train_encode = data_encoding( encoding_strategy[1] , predictor_train , encoding_columns )
predictor_test_encode  = data_encoding( encoding_strategy[1] , predictor_test ,  encoding_columns )


# In[31]:


print(("predictor_train_encode SHAPE   : ",predictor_train_encode.shape))
display("predictor_train_encode COLUMNS : ",predictor_train_encode.head())

print(("predictor_test_encode SHAPE   : ",predictor_test_encode.shape))
display("predictor_test_encode COLUMNS : ",predictor_test_encode.head())


# In[32]:


# Mention Categorical Values of the Light GBM Model to Handle :
categorical_features = cols

lgb_model = LGBMClassifier()

# Apply Stratified K-Fold Cross Validation where K=5 or n_splits=5 :
kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=10)
acc = []

# Pass predictor_train,target_train for Cross Validation :
for fold,(t_id,v_id) in enumerate(kf.split(predictor_train,target_train)):
    
    # Split train and validation data :
    tx = predictor_train.iloc[t_id]; ty = target_train.iloc[t_id]
    vx = predictor_train.iloc[v_id]; vy = target_train.iloc[v_id]
    
    # Train/Fit the Data to LighGBM Model :
    lgb_model.fit(tx,ty, categorical_feature = categorical_features )
    
    # Predict the Validation Data to Train LighGBM Model :
    val_y = lgb_model.predict(vx)
    
    # Get Accuracy Score on Validation Data for Each Fold :
    acc_score = accuracy_score(vy,val_y)
    acc.append(acc_score)
    print(f"fold {fold} accuracy {acc_score}")

# Get Mean of Accuracy Score on Validation Data for All 5 Folds :
print(f"Mean accuracy score {np.mean(acc)}")


# In[33]:


# Tuned the Hyperparameters of LighGBM Classifier :
lgb_model = LGBMClassifier(
                                   boosting_type='gbdt', 
                                   max_depth=15, 
                                   learning_rate=0.15, 
                                   objective='multiclass', # Multi Class Classification
                                   random_state=100,  
                                   n_estimators=1000 ,
                                   reg_alpha=0, 
                                   reg_lambda=1, 
                                   n_jobs=-1
                                 )


# In[34]:


# Mention Categorical Values of the Light GBM Model to Handle :
categorical_features = cols

# Apply Stratified K-Fold Cross Validation where K=5 or n_splits=5 :
kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=10)
acc = []

# Pass predictor_train,target_train for Cross Validation :
for fold,(t_id,v_id) in enumerate(kf.split(predictor_train,target_train)):
    
    # Split train and validation data :
    tx = predictor_train.iloc[t_id]; ty = target_train.iloc[t_id]
    vx = predictor_train.iloc[v_id]; vy = target_train.iloc[v_id]
    
    # Train/Fit the Data to LighGBM Model :
    lgb_model.fit(tx,ty, categorical_feature = categorical_features )
    
    # Predict the Validation Data to Train LighGBM Model :
    val_y = lgb_model.predict(vx)
    
    # Get Accuracy Score on Validation Data for Each Fold :
    acc_score = accuracy_score(vy,val_y)
    acc.append(acc_score)
    print(f"fold {fold} accuracy {acc_score}")

# Get Mean of Accuracy Score on Validation Data for All 5 Folds :
print(f"Mean accuracy score {np.mean(acc)}")


# In[35]:


def model_train_predict_submit( Classifiers_model_name, model_name ,X_train, y_train, X_test, target):
    
    categorical_features = cols
    Classifiers_model_name.fit( X_train, y_train , categorical_feature = categorical_features )
    final_predictions = Classifiers_model_name.predict( X_test )
    print(final_predictions)  
   
    Result_Promoted = pd.DataFrame({'case_id': sample_df['case_id'], target : final_predictions})
    Result_Promoted['Stay'] = Result_Promoted['Stay'].astype(int)
    Result_Promoted[ target ]=Result_Promoted[ target ].map({0:'0-10',
                                                              1:'11-20',2:'21-30',
                                                              3:'31-40',4:'41-50',
                                                              5:'51-60',6:'61-70',7:'71-80',8:'81-90',
                                                              9:'91-100',10:'More than 100 Days'})
    print((Result_Promoted[ target ].unique()))
    Result_Promoted.to_csv(model_name +"_Labelling=Yes_Scaling=Yes"+".csv",index=False)
    return Result_Promoted

model_name       = "LGBM_Tuned_BEST"
model_classifier = lgb_model
sub = model_train_predict_submit( model_classifier, model_name, predictor_train_encode,target_train, predictor_test_encode, target = 'Stay')

