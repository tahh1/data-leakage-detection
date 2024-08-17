#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns

from sklearn.metrics import classification_report
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score 
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import re
import string
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(10,7)})
pd.set_option('display.max_columns', 500)

import warnings
warnings.filterwarnings("ignore")
# def plot_feature_importance(importance,names,model_type):

#     #Create arrays from feature importance and feature names
#     feature_importance = np.array(importance)
#     feature_names = np.array(names)

#     #Create a DataFrame using a Dictionary
#     data={'feature_names':feature_names,'feature_importance':feature_importance}
#     fi_df = pd.DataFrame(data)

#     #Sort the DataFrame in order decreasing feature importance
#     fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

#     #Define size of bar plot
#     plt.figure(figsize=(10,8))
#     #Plot Searborn bar chart
#     sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
#     #Add chart labels
#     plt.title(model_type + 'FEATURE IMPORTANCE')
#     plt.xlabel('FEATURE IMPORTANCE')
#     plt.ylabel('FEATURE NAMES')

# plot_feature_importance(rf.feature_importances_,X_train.columns,'RANDOM FOREST')


# ### Read data 

# In[2]:


#URL Reading: too slow 

# url = 'https://media.githubusercontent.com/media/AlexanderLawson17/WCFCandidateChallenge/master/loan.csv'
# df = pd.read_csv(url)
# print(df.head(5))

loan  = pd.read_csv('loan.txt')
loan.head()


# In[3]:


loan.shape


# In[4]:


(loan.isnull().sum()*100/loan.shape[0]).sort_values(ascending = False)[:20]


# Lot of columns have high amount of missing values. Id, member_id and url have 100% missing. I am going to assign a unique id to each row since id and member id are empty 

# In[5]:


loan.id =list(loan.index)
loan.head()


# In[6]:


ax=(loan.loan_status.value_counts()*100/loan.shape[0]).plot(kind='bar')
#ax.title('asd')
plt.xlabel('Loan Status', fontsize = 16)
plt.ylabel('Proportion', fontsize = 16)
ax.tick_params(axis='both', which='major', labelsize=16)
#loan = loan.drop(['recoveries'], axis = 1)
#loan = loan.drop(['collection_recovery_fee'], axis = 1)


# Consider Charged Off,Does not meet the credit policy. Status:Charged Off and Default as bad loans. Create a binary variable as bad loan and drop the loan status column 

# In[8]:


bad_list=  ['Charged Off','Does not meet the credit policy. Status:Charged Off', 'Default' ]
good_list = ['Fully Paid', 'Late (31-120 days)' , 'In Grace Period', 'Late (16-30 days)',
             'Does not meet the credit policy. Status:Fully Paid']

loan['bad_loan'] = loan['loan_status'].replace(bad_list, [1]*len(bad_list))
loan['bad_loan'] = loan['bad_loan'].replace(good_list, [0]*len(good_list))
loan['bad_loan'] = loan['bad_loan'].replace(['Current'], [np.nan]*1)

loan = loan.drop(['loan_status'], axis = 1)

loan.bad_loan.value_counts()


# In[9]:


# loan.bad_loan.isnull().sum() + loan.bad_loan.value_counts()[0] + loan.bad_loan.value_counts()[1] 


# In[10]:


loan.bad_loan.isnull().sum()*100/loan.shape[0]


# 40.6% of the loans are current and we do not know the outcome yet. So, let's use those loans in the test/scoring set. Rest 59.4% will be used as training and validation datasets 

# In[11]:


scoring = loan[loan['bad_loan'].isnull()]
loan = loan[~loan['bad_loan'].isnull()]


# In[12]:


# scoring.shape


# In[13]:


# loan.shape


# In[14]:


# loan.shape[0] + scoring.shape[0]


# In[15]:


cols_more_50_percent =loan.columns[loan.isnull().mean() > 0.50]
loan.drop(cols_more_50_percent, axis=1, inplace=True)
loan.shape


# ### Exploratory Analysis 

# In[16]:


fig, ax = plt.subplots(1, 3, figsize=(16,5))

sns.distplot(loan['loan_amnt'], ax=ax[0]) ; ax[0].set_title("Loan amount")
sns.distplot(loan['funded_amnt'], ax=ax[1]); ax[1].set_title("Funded amount")
sns.distplot(loan['funded_amnt_inv'], ax=ax[2]) ; ax[2].set_title("Invester amounts")


# ### Term 

# In[17]:


loan.term.value_counts()/loan.shape[0]


# Around 75% of the loans are 36 months. Others are 60 months. 

# In[18]:


loan.groupby(['term'])['bad_loan'].agg(['mean']).plot(kind='bar')


# ### Harship flag

# In[19]:


loan.hardship_flag.value_counts()


# ### initial_list_status

# In[20]:


loan.initial_list_status.value_counts()


# In[ ]:





# ### debt_settlement_flag

# In[21]:


loan.debt_settlement_flag.value_counts()


# In[22]:


loan.groupby(['debt_settlement_flag'])['bad_loan'].agg(['mean']).plot(kind='bar')


# In[23]:


loan['term_36'] = np.where(loan['term'] == '36 months', 1 , 0 )
loan = loan.drop('term', 1)

loan['pymnt_plan'] = np.where(loan['pymnt_plan'] == 'y', 1 , 0 )
#loan = loan.drop('term', 1)
loan['initial_list_status_w'] = np.where(loan['initial_list_status'] == 'w', 1 , 0 )
loan = loan.drop('initial_list_status', 1)

loan['hardship_flag'] = np.where(loan['hardship_flag'] == 'Y', 1 , 0 )
#loan = loan.drop('initial_list_status', 1)

#loan['debt_settlement_flag'] = np.where(loan['debt_settlement_flag'] == 'Y', 1 , 0 )


# ### interest rate 

# In[24]:


sns.boxplot(x= 'bad_loan', y = 'int_rate', data= loan)


# In[25]:


loan.groupby(['bad_loan'])['int_rate'].agg(['mean']).plot(kind='bar')


# ### Grade 

# In[26]:


loan.groupby(['grade'])['bad_loan'].agg(['mean']).plot(kind='bar')


# ### subgrade 

# In[27]:


loan.groupby(['sub_grade'])['bad_loan'].agg(['mean']).plot(kind='bar')


# Since both of these contain the similar information, I am ignore grade for now. Since sub_grade is more granualr, I will use it. It does increase dimensionality (when one-hot encoded) 

# In[28]:


loan = loan.drop(['grade'], axis = 1)


# ### Employment title 

# In[29]:


loan.emp_title.value_counts()


# Too many unique values. Let's drop it 

# In[30]:


loan = loan.drop(['emp_title'], axis = 1)


# ### Employment Length 

# In[31]:


loan.emp_length.value_counts()


# ### Home Ownership 

# In[32]:


loan.groupby(['home_ownership'])['bad_loan'].agg(['mean']).plot(kind='bar')


# ### Annual Income 

# In[33]:


loan.groupby(['bad_loan'])['annual_inc'].agg(['mean']).plot(kind='bar')


# ### Payment Plan

# In[34]:


loan.pymnt_plan.value_counts()


# In[35]:


loan.groupby(['pymnt_plan'])['bad_loan'].agg(['mean']).plot(kind='bar')


# Payment plan = yes has no bad loans. But is 678 enough to run into a conclusion? 

# In[36]:


(loan.isnull().sum()*100/loan.shape[0]).sort_values(ascending = False)


# ### loan_amnt, funded_amnt and funded_amnt_inv

# In[37]:


loan[['loan_amnt', 'funded_amnt', 'funded_amnt_inv']].corr()


# In[38]:


loan = loan.drop(['funded_amnt','funded_amnt_inv'], axis = 1)


# ### Zip code

# In[39]:


print((loan.zip_code.nunique())) ; loan = loan.drop(['zip_code'], axis = 1)


# ### Policy code

# In[40]:


print((loan.policy_code.nunique()))  ;loan = loan.drop(['policy_code'], axis = 1) 


# In[41]:


# #zero variance columns.. 
# (loan.nunique()*100/loan.shape[0]).sort_values()


# ### disbursement_method

# In[42]:


loan.groupby(['disbursement_method'])['bad_loan'].agg(['mean']).plot(kind='bar')


# ### hardship_flag

# In[43]:


loan.groupby(['hardship_flag'])['bad_loan'].agg(['mean']).plot(kind='bar')


# ### application_type

# In[44]:


loan.groupby(['application_type'])['bad_loan'].agg(['mean']).plot(kind='bar')


# In[45]:


#loan.groupby(['initial_list_status'])['bad_loan'].agg(['mean']).plot(kind='bar')


# ### Addr_state 

# In[46]:


loan.groupby(['addr_state'])['bad_loan'].agg(['mean']).plot(kind='bar')


# can reduce dimentionality by grouping into regions

# In[47]:


loan = loan.drop(['addr_state'], axis = 1)


# In[48]:


# loan.groupby(['title'])['bad_loan'].agg(['mean']).plot(kind='bar')


# In[49]:


loan = loan.drop(['title'], axis = 1)


# ### Purpose 

# In[50]:


loan.groupby(['purpose'])['bad_loan'].agg(['mean']).plot(kind='bar')


# ### Transformation 
# 
# 1) difference of days between earliest credit line and issued date 

# In[51]:


loan['test'] = np.where(loan['last_pymnt_d'] == loan['last_credit_pull_d'], 1, 0  )
loan['issue_d'] = pd.to_datetime(loan.issue_d, dayfirst=True)
loan['earliest_cr_line'] = pd.to_datetime(loan.earliest_cr_line, dayfirst=True)
loan['issue_earliest_diff'] = (loan['issue_d'] -  loan['earliest_cr_line']).dt.days
loan = loan.drop(['issue_d', 'earliest_cr_line' , 'last_pymnt_d', 'last_credit_pull_d'], axis = 1)


# In[52]:


loan.select_dtypes(include=['object']).isnull().sum()


# In[53]:


loan.groupby(['emp_length'])['bad_loan'].agg(['mean']).plot(kind='bar')


# emp_length is a good feature (assumption). However, due to time constraints, I am not trying any advacned imputation. We could try imputing it based on employement, annual income etc.. For now, since I do not see much variation in the bad loan rate, I am ignoring this column 

# In[54]:


loan.select_dtypes(include=['object']).isnull().sum()


# #### Train - test split

# In[55]:


from sklearn.model_selection import train_test_split
drop_feat = ['out_prncp' , 'out_prncp_inv' , 'total_pymnt', 'total_pymnt_inv', 'total_rec_int', 'total_rec_prncp', 
            'total_rec_late_fee', 'last_pymnt_amnt' , 'num_tl_120dpd_2m',  'num_tl_30dpd', 'recoveries', 
             'collection_recovery_fee', 'emp_length' ,'id', 'debt_settlement_flag']

loan= loan.drop(drop_feat, axis = 1)

features  = list(loan.columns)
len(features)
X = loan[features]
y = loan['bad_loan']
X = X.drop(['bad_loan', 'test'],axis=  1)


# In[56]:


loan.select_dtypes(include=['object']).columns


# In[57]:


ohe_cols = [ 'sub_grade', 'home_ownership', 'verification_status',
       'pymnt_plan', 'purpose', 'application_type', 'disbursement_method'] 


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42,stratify=y)


# In[59]:


y_test.value_counts(normalize=True)


# In[60]:


y_train.value_counts(normalize=True)


# ### Missing value imputation: median for now 

# In[61]:


X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())


# ### One hot encoding 

# In[62]:


enc = OneHotEncoder(handle_unknown = 'ignore')
enc.fit(X_train[ohe_cols])
train_ohe_df=pd.DataFrame(enc.transform(X_train[ohe_cols]).toarray())
train_ohe_df.columns = enc.get_feature_names()
X_train.reset_index(inplace=True)
X_train= X_train.drop(ohe_cols, 1)
X_train = pd.concat([X_train,train_ohe_df], axis = 1)
X_train.shape
X_train.drop('index', axis = 1, inplace=True)


# In[63]:


test_ohe_df=pd.DataFrame(enc.transform(X_test[ohe_cols]).toarray())
test_ohe_df.columns = enc.get_feature_names()
X_test= X_test.drop(ohe_cols, 1)
X_test.reset_index(inplace=True)
X_test = pd.concat([X_test,test_ohe_df], axis = 1)
X_test.drop('index', axis = 1, inplace=True)
#X_test = pd.concat([X_test,test_ohe_df], axis = 1)
X_test.head()


# ### highly correlated features

# In[64]:


corr_matrix = X_train.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
to_drop


# In[65]:


X_train = X_train.drop(to_drop, 1)
X_test = X_test.drop(to_drop, 1)


# In[66]:


X_train.shape


# In[67]:


X_test.shape


# ### Random forest: taking too much time on local machine

# In[68]:


# from sklearn.model_selection import StratifiedKFold
# n_optimal_param_grid = {'bootstrap': [True],  'max_depth': [10], 
#     'min_samples_leaf': [1],  'min_samples_split': [2],
#     'n_estimators': [20]
# }
# rf = RandomForestClassifier(random_state=42)

# nn_grid_search = GridSearchCV(estimator = rf, param_grid = n_optimal_param_grid, 
#                           cv =StratifiedKFold(), n_jobs = 1, verbose = 2)
# nn_grid_search.fit(X_train, y_train)

# y_pred = nn_grid_search.predict(X_test)


# In[69]:


# nn_grid_search.best_params_


# In[70]:


# rf= RandomForestClassifier(random_state=42,bootstrap=nn_grid_search.best_params_['bootstrap'],
#                           max_depth= nn_grid_search.best_params_['max_depth'],
#                            min_samples_leaf= nn_grid_search.best_params_['min_samples_leaf'],
#                           min_samples_split= nn_grid_search.best_params_['min_samples_split'],
#                           n_estimators= nn_grid_search.best_params_['n_estimators'])

# rf.fit(X_train, y_train)
# y_pred = rf.predict(X_test)


# In[71]:


# accuracy_score(y_pred, y_test)


# In[72]:


# confusion_matrix(y_pred, list(y_test))


# In[73]:


# feature_importance = np.array(rf.feature_importances_)
# feature_names = np.array(X_train.columns)

#     #Create a DataFrame using a Dictionary
# data={'feature_names':feature_names,'feature_importance':feature_importance}
# fi_df = pd.DataFrame(data)

#     #Sort the DataFrame in order decreasing feature importance
# fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)


# In[74]:


# top_features = fi_df['feature_names'][:20].values


# ### Model 1 

# ### Cross Validation 

# In[75]:


# grid={"C":np.logspace(-3,3,7), "penalty":["l1"]}
# logisticRegr = LogisticRegression()

# logreg_cv=GridSearchCV(logisticRegr,grid,cv=5,scoring = 'recall')
# logreg_cv.fit(X_train,y_train)
# # all parameters not specified are set to their defaults
# #)
# #lgisticRegr.fit(X_train, y_train)
# predictions = logreg_cv.predict(X_test)
# #predictions_actual = logisticRegr.predict(test)

# print ("Accuracy: {}".format(accuracy_score(predictions, y_test)) )
# print ("F1 Score: {}".format(f1_score(predictions, y_test, average='weighted')))
# print ("AUC: {}".format(roc_auc_score(predictions, y_test)) )


# In[ ]:





# In[76]:


#np.logspace(-3,3,7)


# In[77]:


#logreg_cv.best_params_ 


# In[78]:


#best model 
log_reg = LogisticRegression( C=0.001 ,penalty = 'l1' ,random_state  = 42 )
log_reg.fit(X_train,y_train)


# In[79]:


predictions = log_reg.predict(X_test)


# In[80]:


print(("Accuracy: {}".format(accuracy_score(predictions, y_test)) ))
print(("F1 Score: {}".format(f1_score(y_test, predictions, average='weighted'))))
print(("AUC: {}".format(roc_auc_score(y_test, predictions)) ))
print((classification_report(y_test, predictions)))


# In[81]:


data = confusion_matrix(y_test, predictions)
df_cm = pd.DataFrame(data, columns=np.unique(y_test), index = np.unique(y_test))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})


# In[82]:


print(('False Negative Rate: {}'.format(data[1][0]/len(y_test))))
print(('False Positive Rate: {}'.format(data[0][1]/len(y_test))))


# In[ ]:





# In[83]:


logReg_coeff = pd.DataFrame({'feature_name': X_train.columns, 'model_coefficient': log_reg.coef_.transpose().flatten()})
logReg_coeff = logReg_coeff.sort_values('model_coefficient',ascending=False)
logReg_coeff


# ### Model 2 

# ### Feature Selection: Remove zero coefficient features

# In[84]:


zero_coef =logReg_coeff[logReg_coeff.model_coefficient == 0 ]['feature_name'].values
select_features = [i for i in X_train.columns if i not in zero_coef] 


# In[85]:


log_reg = LogisticRegression( C=0.001 ,penalty = 'l1',random_state  = 42 )
log_reg.fit(X_train[select_features],y_train)


# In[86]:


predictions = log_reg.predict(X_test[select_features])


# In[87]:


print(("Accuracy: {}".format(accuracy_score(predictions, y_test)) ))
print(("F1 Score: {}".format(f1_score(y_test,  predictions,average='weighted'))))
print(("AUC: {}".format(roc_auc_score( y_test,predictions)) ))

print((classification_report(y_test, predictions)))


# In[88]:


logReg_coeff = pd.DataFrame({'feature_name': X_train[select_features].columns, 'model_coefficient': log_reg.coef_.transpose().flatten()})
logReg_coeff = logReg_coeff.sort_values('model_coefficient',ascending=False)
logReg_coeff


# In[89]:


# predictions = log_reg.predict(X_test[select_features])


# ### Classfication report

# In[90]:


# print(classification_report(y_test, predictions))


# In[91]:


# roc_auc_score(y_test, predictions)


# ### ROC Curve

# In[92]:


logit_roc_auc = roc_auc_score(y_test, log_reg.predict(X_test[select_features]))
fpr, tpr, thresholds = roc_curve(y_test, log_reg.predict_proba(X_test[select_features])[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# ### Threshold Selection 

# In[93]:


from numpy import sqrt
from numpy import argmax
# calculate the g-mean for each threshold
gmeans = sqrt(tpr * (1-fpr))
# locate the index of the largest g-mean
ix = argmax(gmeans)
print(('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix])))


# In[94]:


predictions_prob = log_reg.predict_proba(X_test[select_features])
#yhat =predictions_prob[:, 1]


# ### ROC Curve

# In[95]:


# pr curve for logistic regression model
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot
# calculate roc curves
#fpr, tpr, thresholds = roc_curve(testy, yhat)
# calculate the g-mean for each threshold
gmeans = sqrt(tpr * (1-fpr))
# locate the index of the largest g-mean
ix = argmax(gmeans)
print(('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix])))
# plot the roc curve for the model
pyplot.plot([0,1], [0,1], linestyle='--', label='Random Chance')
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
# show the plot
pyplot.show()


# In[96]:


pred_prob = []
for i,v in enumerate(predictions_prob):
    pred_prob.append(predictions_prob[i][1])

new_items = [1 if x > thresholds[ix] else 0 for x in pred_prob]


# In[97]:


roc_auc_score(y_test, new_items)


# ### Classification report
# 

# In[98]:


from sklearn.metrics import classification_report
print((classification_report(y_test, new_items)))


# ### Confusion matrix
# 

# In[99]:


data = confusion_matrix(y_test, new_items)
df_cm = pd.DataFrame(data, columns=np.unique(y_test), index = np.unique(y_test))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})


# In[100]:


data


# In[ ]:





# In[101]:


print(('False Negative Rate: {}'.format(data[1][0]/len(y_test))))

print(('False Positive Rate: {}'.format(data[0][1]/len(y_test))))


# In[ ]:





# In[102]:


86000/len(y_test)


# Imbalance, metrics, roc curve, confusion matrix 

# In[ ]:





# In[103]:


import statsmodels.api as sm 
import pandas as pd 


# In[104]:


log_reg = sm.Logit(list(y_train), X_train[select_features]).fit() 


# In[105]:


print((log_reg.summary()))


# In[106]:


# corr_matrix = X_train.corr().abs()

# # Select upper triangle of correlation matrix
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# # Find features with correlation greater than 0.95
# to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# to_drop


# In[ ]:




