#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
# !pip install lightgbm
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


# SMOTE and Near Miss
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler


from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.explainers import MetricTextExplainer, MetricJSONExplainer
# from aif360.algorithms.preprocessing import Reweighing

plt.style.use("fivethirtyeight")

from IPython.display import display, HTML

import warnings
print('Ok')


# In[2]:


data = pd.read_csv("""..\data\interim\data_100000-30percent.csv""")
data2 = data.drop(['customer'], axis=1)
# data2 = data2.drop(['tenure'], axis=1)
# data2 = data2.drop(['comeback_product'], axis=1)

data2['gender'] = data2['gender'].astype(int)
# data2['churn'] = data2['churn'].astype(object)
data2['is_senior'] = data2['is_senior'].astype(int)
data2['contract_period'] = data2['contract_period'].astype(object)

numerical = data2.select_dtypes(['number']).columns
print(f'Numerical: {numerical}\n')

categorical = data2.columns.difference(numerical)

data2[categorical] = data2[categorical].astype('object')
print(f'Categorical: {categorical}')

data2 = pd.get_dummies(data2)

X_original = data2.drop('churn', axis=1)
y_original = data2['churn']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state = 42)
print((len(list(data2.keys()))))
list(data2.keys())


# In[3]:


models = []
models.append(('Random Forest', RandomForestClassifier()))
models.append(("LightGBM", LGBMClassifier()))
models.append(('Gradient Boosting',GradientBoostingClassifier()))
models.append(('Logistic Reg.', LogisticRegression(max_iter=1000)))

models.append(('XGBClassifier', XGBClassifier()))
models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))

methods = ['original', 'adasyn', 'smote', 'rus', 'ros']


# In[4]:


# AIF360 Dataset Preparation

all_labels = ['age', 'gender', 'hsbb_area', 'speed', 'price_start', 'complain_count',
       'churn', 'median_outstanding', 'technical_problem_count', 'is_senior',
       'avg_download', 'avg_upload', 'avg_voice_usage', 'race_B', 'race_C',
       'race_I', 'race_M', 'race_O', 't_location_Zone-AJP',
       't_location_Zone-AKM', 't_location_Zone-BAL', 't_location_Zone-BKK',
       't_location_Zone-BLS', 't_location_Zone-CSM', 't_location_Zone-GCK',
       't_location_Zone-GHP', 't_location_Zone-GPP', 't_location_Zone-GRT',
       't_location_Zone-IRM', 't_location_Zone-KDS', 't_location_Zone-KRP',
       't_location_Zone-NSN', 't_location_Zone-PBP', 't_location_Zone-PDK',
       't_location_Zone-RLK', 't_location_Zone-SRJ', 't_location_Zone-TLK',
       't_location_Zone-TLS', 't_location_Zone-UBS', 't_location_Zone-URJ',
       't_location_Zone-UWT', 
        # 'contract_period_1', 
        'contract_period_12',
       'contract_period_18', 'contract_period_24', 'contract_period_36']

features = ['age', 'gender', 'hsbb_area', 'speed', 'price_start', 'complain_count',
       'churn', 'median_outstanding', 'technical_problem_count', 'is_senior',
       'avg_download', 'avg_upload', 'avg_voice_usage', 'race_B', 'race_C',
       'race_I', 'race_M', 'race_O', 't_location_Zone-AJP',
       't_location_Zone-AKM', 't_location_Zone-BAL', 't_location_Zone-BKK',
       't_location_Zone-BLS', 't_location_Zone-CSM', 't_location_Zone-GCK',
       't_location_Zone-GHP', 't_location_Zone-GPP', 't_location_Zone-GRT',
       't_location_Zone-IRM', 't_location_Zone-KDS', 't_location_Zone-KRP',
       't_location_Zone-NSN', 't_location_Zone-PBP', 't_location_Zone-PDK',
       't_location_Zone-RLK', 't_location_Zone-SRJ', 't_location_Zone-TLK',
       't_location_Zone-TLS', 't_location_Zone-UBS', 't_location_Zone-URJ',
       't_location_Zone-UWT', 
        # 'contract_period_1', 
        'contract_period_12',
       'contract_period_18', 'contract_period_24', 'contract_period_36']


class TMDataset(StandardDataset):
    def __init__(self, 
             label_name='churn',
             favorable_classes=[1.0],
                 
             protected_attribute_names=[
                'gender', 
#                 'is_senior',
#                 'race_O',
             ],
    
             privileged_classes=[
                [1.0,], 
#                 [0.0,],
#                 [0.0,],
             ],
                 
             instance_weights_name=None,
             categorical_features=[],
             features_to_keep=features, 
             features_to_drop=[],
             custom_preprocessing=None,
             metadata=None,
             csv_file_name='',
             data_frame=None,
    ):
        
        if data_frame is not None:
            aif_df = data_frame
        else:
            aif_df = pd.read_csv(csv_file_name)
        
        #df.reset_index(drop=True, inplace=True)
        # Preprocessing
        
        super().__init__(
            df=aif_df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop,
            custom_preprocessing=custom_preprocessing, 
            metadata=metadata,
     
        )


# In[5]:


def make_sampling(method_name, X, y):
    if method_name == 'original':
        return (X, y)
    elif method_name == 'adasyn':
        adasyn = ADASYN(sampling_strategy='minority', random_state=420, n_neighbors=5)
        return adasyn.fit_sample(X, y)
    elif method_name == 'smote':
        os = SMOTE(random_state=41)
        return os.fit_sample(X, y)
    elif method_name == 'ros':
        random_over_sampler = RandomOverSampler(random_state=42)
        return random_over_sampler.fit_resample(X, y)
    elif method_name == 'rus':
        random_under_sampler = RandomUnderSampler(random_state=42)
        return random_under_sampler.fit_resample(X, y)
    else:
        print('UNKNOWN METHOD !!') 


# In[6]:


for name, model in models:
    display(HTML(f'<h2> {name} </h2>'))
    for method in methods:
        display(HTML(f'<h3>{method} </h3>'))
        X, y = make_sampling(method, X_original, y_original)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state = 42)
        score = cross_val_score(model, X, y, cv = 10, scoring='accuracy')
        
        print('Charn values Original')
        print((y.value_counts()))
        print(('Total data: {}'.format(y.count())))
        
        print('\nCharn values Train')
        print((y_train.value_counts()))
        print(('Total data: {}'.format(y_train.count())))
        
        print('\nCharn values Test')
        print((y_test.value_counts()))
        print(('Total data: {}'.format(y_test.count())))
        
        print("\n")
        print(("*_" * 20))
        print(f"Mean scores : {score.mean()}")

        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        print((metrics.classification_report(y_pred_test, y_test)))
        
        pred_df = X_test.copy()
        pred_df['churn'] = y_pred_test
        
        print('\n Gender Count in predited value with X_test set')
        print(('# ' * 10))
        print((pred_df.gender.value_counts()))
        p_df = pred_df[['churn', 'gender']]
        print('\n')
        p_df.insert(2, 'counter', 1)
        print((p_df.groupby(['churn','gender',]).sum()))
        print(('# ' * 10))
        
        print((metrics.confusion_matrix(y_test, y_pred_test)))
        metrics.plot_confusion_matrix(model, X_test, y_test)
        plt.show()

        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc_score = metrics.roc_auc_score(y_test, y_proba)
        print('\n\n')
        print(f'ROC AUC Score {roc_auc_score}')
        

        # Additional matrix
        df_pred = X_test
        df_pred.reset_index(drop=True, inplace=True)
        
        pred = pd.Series(y_pred_test)
        df_pred = df_pred.assign(churn=pred)
        
        # df_pred = df_pred.assign(churn=y_test)
        
        aif_df = TMDataset(data_frame=df_pred)
        aif_df_labeled = aif_df.copy()
        aif_df_labeled.labels = y_pred_test
        
        result_tbl_cols = [
            'Attribute',
            'Mean difference', 
            'Positive Outcome',
            'Negative', 
            'Differences', 
            'Disparate impact', 
            'Consistency',
            'Statistical parity dif',
        ]

        result_rows = []
        for p_attribute in aif_df.protected_attribute_names:
            result_row = []
            privileged_groups = [{p_attribute: 1}]
            print((' * ' * 10))
            print(privileged_groups)
            unprivileged_groups = [{p_attribute: 0}]
            
            metric_orig_train = BinaryLabelDatasetMetric(
                aif_df,  unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
            
            clsf_metric = ClassificationMetric(
                aif_df, aif_df_labeled, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
            
            print('\n AIF 360 ')
            print(('\n Statistical parity difference', clsf_metric.statistical_parity_difference()))
           
            print(('\n True possitve rate {} \t True negative rate {} '.format(
                clsf_metric.true_positive_rate(), clsf_metric.true_negative_rate())))
            
            print(('\n Desparate impact ', clsf_metric.disparate_impact()))
            print(('\n Equal opportunity difference ', clsf_metric.equal_opportunity_difference()))
            print(('\n Average odds difference ', clsf_metric.average_odds_difference()))
            print(('\n Theil Index ', clsf_metric.theil_index()))
            print('\n Binary Confusion Matric ')
            print((clsf_metric.binary_confusion_matrix()))
            print('\n ..........................')

            text_expl = MetricTextExplainer(metric_orig_train)
            
            result_row.append(p_attribute)
            result_row.append(metric_orig_train.mean_difference())
            result_row.append(metric_orig_train.num_positives())
            result_row.append(metric_orig_train.num_negatives())

            # this is to shutup the warning msg from sklearn
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                result_row.append(metric_orig_train.consistency())
                
            result_row.append(metric_orig_train.disparate_impact())
            result_row.append(metric_orig_train.consistency())
            result_row.append(metric_orig_train.statistical_parity_difference())

            result_rows.append(result_row)

        result_df = pd.DataFrame(result_rows, columns=result_tbl_cols)
        display(result_df)
        # End Aif matrix


# In[7]:


# pred_df.group_by(['churn'])
print((pred_df.gender.value_counts()))
p_df = pred_df[['churn', 'gender']]
p_df.insert(2, 'counter', 1)
p_df.groupby(['churn','gender',]).sum()


# In[ ]:




