#!/usr/bin/env python
# coding: utf-8

# ## **Please upvote if you like!!!**

# # BANK LENDING

# ## Task

# Objective :This is a binary classification where you need predict custusmer will default

# In[169]:


#loading_all libraries
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.metrics import f1_score,roc_curve,recall_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
from sklearn import metrics
import matplotlib as mpl
from scipy import stats
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import csv


# In[170]:


#to ignore warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading the train and test data-set using pandas.read_table
# 
# 

# In[171]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))


# In[172]:


df_train=pd.read_table('/kaggle/input/xyzcorp-lendingdata/XYZCorp_LendingData.txt',parse_dates=['issue_d'],low_memory=False)


# In[173]:


#to display the entire dataframe 
pd.set_option("display.max.columns", None)


# In[174]:


df_train.shape


# # Data Description:
# Train.csv : 855969 x 73 [including headers] 

# In[175]:


#to view top 5 rows of dataframe
df_train.head(n=5)


# In[176]:


#to known about the dataframe of featurs
df_train.dtypes


# In[177]:


#function to find missing Value
def missing_data(df_train):
    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return(missing_data.head(20))


# In[178]:


#missing_data function on dataset
missing_data(df_train)


# In[179]:


#removing top 26 feature with most missing value
df_train_new=df_train.drop(['dti_joint','verification_status_joint','annual_inc_joint','il_util','mths_since_rcnt_il',
'total_bal_il','inq_last_12m','open_acc_6m','open_il_6m','open_il_24m','open_il_12m',
'open_rv_12m','open_rv_24m','max_bal_bc','all_util','inq_fi','total_cu_tl','desc','mths_since_last_record',
'mths_since_last_major_derog','mths_since_last_delinq','next_pymnt_d','tot_cur_bal',
'tot_coll_amt','total_rev_hi_lim','emp_title'],axis=1)


# In[180]:


#imputing the missing value with mean value in revol_util featture
df_train_new['revol_util'].fillna(df_train_new['revol_util'].mean(),inplace=True)


# In[181]:


missing_data(df_train_new)


# In[182]:


#feature enginearing on last_credit_pull_d feature
df_train_new['last_credit_pull_d'] = pd.to_datetime(df_train_new['last_credit_pull_d'])
df_train_new['Month'] = df_train_new['last_credit_pull_d'].apply(lambda x: x.month)
df_train_new['Year'] = df_train_new['last_credit_pull_d'].apply(lambda x: x.year)
df_train_new = df_train_new.drop(['last_credit_pull_d'], axis = 1)


# In[183]:


#imputing missing value with mode value 
df_train_new['Month'].fillna(df_train_new.mode()['Month'][0],inplace=True)
df_train_new['Year'].fillna(df_train_new.mode()['Year'][0],inplace=True)


# In[184]:


df_train_new.shape


# In[185]:


print((df_train_new['title'].value_counts()))


# In[186]:


#to impute missing na value with debt_consolidation because its is most repeted value
df_train_new['title'].fillna('Debt consolidation ',inplace=True)


# In[187]:


df_train_new.dtypes


# In[188]:


#removing/droping the unwanted features 
df_train_new = df_train_new.drop(['id'],axis=1)

df_train_new = df_train_new.drop(['member_id'],axis=1)

df_train_new = df_train_new.drop(['earliest_cr_line'],axis=1)

df_train_new = df_train_new.drop(['zip_code'],axis=1)

df_train_new = df_train_new.drop(['last_pymnt_d'],axis=1)

df_train_new = df_train_new.drop(['policy_code'],axis=1)


# In[189]:


df_train_new.head(n=5)


# In[190]:


#replace the categorial to numeric 
df_train_new=df_train_new.replace(to_replace='10+ years',value=10)
df_train_new=df_train_new.replace(to_replace='1 year',value=1)
df_train_new=df_train_new.replace(to_replace='2 years',value=2)
df_train_new=df_train_new.replace(to_replace='3 years',value=3)
df_train_new=df_train_new.replace(to_replace='4 years',value=4)
df_train_new=df_train_new.replace(to_replace='5 years',value=5)
df_train_new=df_train_new.replace(to_replace='6 years',value=6)
df_train_new=df_train_new.replace(to_replace='7 years',value=7)
df_train_new=df_train_new.replace(to_replace='8 years',value=8)
df_train_new=df_train_new.replace(to_replace='9 years',value=9)
df_train_new=df_train_new.replace(to_replace='< 1 year',value=0.5)


# In[191]:


df_train_new['title'].value_counts()


# In[192]:


counts = df_train_new['title'].value_counts()

df_train_new = df_train_new[~df_train_new['title'].isin(counts[counts < 100].index)]


# In[193]:


df_train_new.head(n=5)


# In[194]:


#to remove all na values throughout the dataset
df_train_new = df_train_new.dropna(axis = 0, how ='any') 


# In[195]:


df_train_new['emp_length'].value_counts()


# In[196]:


#dataset for data visulization in tabelau
df_train_bin=df_train_new
df_train_new.to_csv('Bank Lending.csv')


# ##  Correlation Matrix
# When two sets of data are strongly linked together we say they have a High Correlation.
# 
# The word Correlation is made of Co- (meaning "together"), and Relation
# 
# Correlation is Positive when the values increase together, and
# Correlation is Negative when one value decreases as the other increases
# A correlation is assumed to be linear (following a line).
# 
# correlation examples
# Correlation can have a value:
# 
# 1 is a perfect positive correlation
# 0 is no correlation (the values don't seem linked at all)
# -1 is a perfect negative correlation
# The value shows how good the correlation is (not how steep the line is), and if it is positive or negative.

# In[197]:


#correlation matrix
corrmat = df_train_new.corr()
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corrmat, vmax=.8, square=True);


# # Data Visualization
# 
# Data visualization is the graphic representation of data. It involves producing images that communicate relationships among the represented data to viewers of the images. This communication is achieved through the use of a systematic mapping between graphic marks and data values in the creation of the visualization

# In[198]:


df_train['default_ind'].value_counts().plot.bar()


# In[199]:


sns.countplot('initial_list_status',data=df_train_new,hue='default_ind')


# In[200]:


sns.boxplot('grade','int_rate',data=df_train_new)


# In[201]:


plt.figure(figsize=(10,5))
sns.distplot(df_train_new['int_rate'])
plt.show()


# In[202]:


sns.violinplot('default_ind','int_rate',data=df_train_new,bw='scott')


# In[203]:


#plotting histogram of all features
df_train_new.hist(figsize=(15,20))


# In[204]:


sns.stripplot('default_ind','annual_inc',data=df_train_new,jitter=True)


# In[205]:


plt.figure(figsize=(15,10))
sns.catplot(x='verification_status',y='loan_amnt',data=df_train_new,hue='default_ind',height=5,aspect=3,kind='box')
plt.title('boxplot')


# In[206]:


plt.figure(figsize=(15,10))
sns.relplot(x='funded_amnt', y='funded_amnt_inv', data=df_train_new,
            kind='line', hue='term', col='default_ind')


# In[207]:


fig, ax = plt.subplots(1, 3, figsize=(16,5))

sns.distplot(df_train['loan_amnt'], ax=ax[0])
sns.distplot(df_train['funded_amnt'], ax=ax[1])
sns.distplot(df_train['funded_amnt_inv'], ax=ax[2])

ax[1].set_title("Amount Funded by the Lender")
ax[0].set_title("Loan Applied by the Borrower")
ax[2].set_title("Total committed by Investors")


# In[208]:


df_train.purpose.value_counts(ascending=False).plot.bar(figsize=(10,5))
plt.xlabel('purpose'); plt.ylabel('Density'); plt.title('Purpose of loan');


# In[209]:


plt.figure(figsize=(10,5))
df_train['issue_year'] = df_train['issue_d'].dt.year
sns.barplot(x='issue_year',y='loan_amnt',data=df_train)


# In[210]:


# Loan Status 
fig, ax = plt.subplots(1, 2, figsize=(16,5))
df_train['default_ind'].value_counts().plot.pie(explode=[0,0.25],labels=['good loans','bad loans'],
                                             autopct='%1.2f%%',startangle=70,ax=ax[0])
sns.kdeplot(df_train.loc[df_train['default_ind']==0,'issue_year'],label='default_ind = 0')
sns.kdeplot(df_train.loc[df_train['default_ind']==1,'issue_year'],label='default_ind = 1')
plt.xlabel('Year'); plt.ylabel('Density'); plt.title('Yearwise Distribution of defaulter')


# In[211]:


defaulter = df_train_new.loc[df_train_new['default_ind']==1]
plt.figure(figsize=(10,10))
plt.subplot(211)
sns.boxplot(data=defaulter,x = 'home_ownership',y='loan_amnt',hue='default_ind')
plt.subplot(212)
sns.boxplot(data=defaulter,x='Year',y='loan_amnt',hue='home_ownership')


# In[212]:


sns.countplot('verification_status',data=df_train_new,hue='default_ind')


# In[213]:


sns.stripplot('default_ind','total_rec_prncp',data=df_train_new,jitter=True)


# In[214]:


plt.figure(figsize=(25,20))
sns.factorplot(data=df_train_new,x='verification_status',y='loan_amnt',hue='default_ind')


# In[215]:


# Plotting
sns.catplot(x='verification_status', y='loan_amnt', data=df_train_new, kind='boxen', aspect=2)
plt.title('Boxen Plot', weight='bold', fontsize=16)
plt.show()


# In[216]:


df_train_new.columns


# In[217]:


df_train_new.describe()


# In[345]:


dftrain_bin=df_train_new


# ## Label Encoding 

# In[218]:


from sklearn import preprocessing

le1 = preprocessing.LabelEncoder()
le1.fit(df_train_new['term'])
list(le1.classes_)
df_train_new['term'] = le1.transform(df_train_new['term'])
df_train_new.head()


# In[219]:


le1 = preprocessing.LabelEncoder()
le1.fit(df_train_new['grade'])
list(le1.classes_)
df_train_new['grade'] = le1.transform(df_train_new['grade'])
df_train_new.head()


# In[220]:


le1 = preprocessing.LabelEncoder()
le1.fit(df_train_new['sub_grade'])
list(le1.classes_)
df_train_new['sub_grade'] = le1.transform(df_train_new['sub_grade'])
df_train_new.head()


# In[221]:


le1 = preprocessing.LabelEncoder()
le1.fit(df_train_new['home_ownership'])
list(le1.classes_)
df_train_new['home_ownership'] = le1.transform(df_train_new['home_ownership'])
df_train_new.head()


# In[222]:


le1 = preprocessing.LabelEncoder()
le1.fit(df_train_new['verification_status'])
list(le1.classes_)
df_train_new['verification_status'] = le1.transform(df_train_new['verification_status'])
df_train_new.head()


# In[223]:


le1 = preprocessing.LabelEncoder()
le1.fit(df_train_new['purpose'])
list(le1.classes_)
df_train_new['purpose'] = le1.transform(df_train_new['purpose'])
df_train_new.head()


# In[224]:


le1 = preprocessing.LabelEncoder()
le1.fit(df_train_new['addr_state'])
list(le1.classes_)
df_train_new['addr_state'] = le1.transform(df_train_new['addr_state'])
df_train_new.head()


# In[225]:


le1 = preprocessing.LabelEncoder()
le1.fit(df_train_new['application_type'])
list(le1.classes_)
df_train_new['application_type'] = le1.transform(df_train_new['application_type'])
df_train_new.head()


# In[226]:


le1 = preprocessing.LabelEncoder()
le1.fit(df_train_new['pymnt_plan'])
list(le1.classes_)
df_train_new['pymnt_plan'] = le1.transform(df_train_new['pymnt_plan'])
df_train_new.head()


# In[227]:


le1 = preprocessing.LabelEncoder()
le1.fit(df_train_new['initial_list_status'])
list(le1.classes_)
df_train_new['initial_list_status'] = le1.transform(df_train_new['initial_list_status'])
df_train_new.head()


# In[228]:


df_train_new.dtypes


# # Train And Test Split 

# In[229]:


train = df_train_new[df_train_new['issue_d'] < '2015-6-01']
test = df_train_new[df_train_new['issue_d'] >= '2015-6-01']


# In[230]:


x_train=train.drop(['default_ind','title','issue_d'],axis=1)
y_train=train['default_ind']
x_test=test.drop(['default_ind','title','issue_d'],axis=1)
y_test=test['default_ind']


# # LogisticRegression 
# 
# Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of binary regression). Mathematically, a binary logistic model has a dependent variable with two possible values, such as pass/fail which is represented by an indicator variable, where the two values are labeled "0" and "1". In the logistic model, the log-odds (the logarithm of the odds) for the value labeled "1" is a linear combination of one or more independent variables ("predictors"); the independent variables can each be a binary variable (two classes, coded by an indicator variable) or a continuous variable (any real value). The corresponding probability of the value labeled "1" can vary between 0 (certainly the value "0") and 1 (certainly the value "1"),

# In[231]:


log =LogisticRegression()
log.fit(x_train,y_train)


# In[232]:


#model on train using all the independent values in df
log_prediction = log.predict(x_train)
log_score= accuracy_score(y_train,log_prediction)
print(('Accuracy score on train set using Logistic Regression :',log_score))


# # confusion matrix
# A confusion matrix is a table that is often used to describe the performance of a classification model (or “classifier”) on a set of test data for which the true values are known. It allows the visualization of the performance of an algorithm.

# In[233]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, log_prediction)


# # AUC 
# Compute Area Under the Curve (AUC) using the trapezoidal rule

# In[234]:


from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_train,log_prediction)
print(("AUC on train using Logistic Regression :",metrics.auc(fpr, tpr)))


# # average precision recall score
# 

# In[235]:


from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_train, log_prediction)

print(('Average precision-recall score: {0:0.2f}'.format(
      average_precision)))


# # recall score
# 
# The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
# 
# The best value is 1 and the worst value is 0.

# In[236]:


from sklearn.metrics import recall_score
print(('recall_score on train set :',recall_score(y_train, log_prediction)))


#  # F1 Score
# Compute the F1 score, also known as balanced F-score or F-measure
# 
# The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:
# 
# F1 = 2 * (precision * recall) / (precision + recall)

# In[237]:


from sklearn.metrics import f1_score
print(('F1_sccore on train set :',f1_score(y_train, log_prediction)))


# ## Classification report

# In[238]:


print((classification_report(y_train,log_prediction)))


# In[239]:


#model on test using all the independent values in df
log_prediction = log.predict(x_test)
log_score= accuracy_score(y_test,log_prediction)
print(('accuracy score on test using Logisitic Regression :',log_score))


# In[240]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, log_prediction)


# In[241]:


from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test,log_prediction)
print(("AUC on test using Logistic Regression :",metrics.auc(fpr, tpr)))


# In[242]:


from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, log_prediction)

print(('Average precision-recall score: {0:0.2f}'.format(
      average_precision)))


# In[243]:


from sklearn.metrics import recall_score
print(('recall_score on train set :',recall_score(y_test, log_prediction)))


# In[244]:


from sklearn.metrics import f1_score
print(('F1_sccore on train set :',f1_score(y_test, log_prediction)))


# In[245]:


print((classification_report(y_test, log_prediction)))


# ## Kfold cross validation

# In[246]:


lr = LogisticRegression()
scores = cross_val_score(lr, x_train, y_train, cv=5, scoring = "accuracy")
print(("Scores:", scores))
print(("Mean:", scores.mean()))
print(("Standard Deviation:", scores.std()))


# ## ROC Curve

# In[247]:


lr_prob=log.predict_proba(x_train)
lr_prob=lr_prob[:,1]
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_train, lr_prob)
print(('auc_score for Logistic Regression(train): ', roc_auc_score(y_train, lr_prob)))
# Plot ROC curves
plt.subplots(1, figsize=(5,5))
plt.title('Receiver Operating Characteristic(train) - logistic regression')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
lr_prob_test=log.predict_proba(x_test)
lr_prob_test=lr_prob_test[:,1]
false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, lr_prob_test)
print(('auc_score for Logistic Regression(test): ', roc_auc_score(y_test, lr_prob_test)))
# Plot ROC curves
plt.subplots(1, figsize=(5,5))
plt.title('Receiver Operating Characteristic(test) - logistic regression')
plt.plot(false_positive_rate2, true_positive_rate2)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # XGBoost Algorithm
# 
# XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. The same code runs on major distributed environment (Hadoop, SGE, MPI) and can solve problems beyond billions of examples.

# In[248]:


xgboost = xgb.XGBClassifier(max_depth=3,n_estimators=300,learning_rate=0.05)


# In[249]:


xgboost.fit(x_train,y_train)


# In[250]:


#XGBoost model on the train set
XGB_prediction = xgboost.predict(x_train)
XGB_score= accuracy_score(y_train,XGB_prediction)
print(('accuracy score on train using XGBoost ',XGB_score))


# In[251]:


confusion_matrix(y_train, XGB_prediction)


# In[252]:


fpr, tpr, thresholds = metrics.roc_curve(y_train,XGB_prediction)
print(("AUC on train using XGBoost :",metrics.auc(fpr, tpr)))


# In[253]:


average_precision = average_precision_score(y_train, XGB_prediction)

print(('Average precision-recall score: {0:0.2f}'.format(
      average_precision)))


# In[254]:


print(('recall_score on train set :',recall_score(y_train, XGB_prediction)))


# In[255]:


print(('F1_sccore on train set :',f1_score(y_train, XGB_prediction)))


# In[256]:


print('classification Report on  train using XGBoost :')
print((classification_report(y_train,XGB_prediction)))


# In[257]:


#XGBoost model on the test
XGB_prediction = xgboost.predict(x_test)
XGB_score= accuracy_score(y_test,XGB_prediction)
print(('accuracy score on test using XGBoost :',XGB_score))


# In[258]:


confusion_matrix(y_test, XGB_prediction)


# In[259]:


fpr, tpr, thresholds = metrics.roc_curve(y_test,XGB_prediction)
print(("AUC on test using XGBoost :",metrics.auc(fpr, tpr)))


# In[260]:


average_precision = average_precision_score(y_test, XGB_prediction)

print(('Average precision-recall score: {0:0.2f}'.format(
      average_precision)))


# In[261]:


print(('recall_score on test set :',recall_score(y_test, XGB_prediction)))


# In[262]:


print(('F1_sccore on test set :',f1_score(y_test, XGB_prediction)))


# In[263]:


print('classification Report on  test using XGBoost :')
print((classification_report(y_test,XGB_prediction)))


# ## ROC Curve

# In[264]:


xg_prob=xgboost.predict_proba(x_train)
xg_prob=xg_prob[:,1]
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_train, xg_prob)
print(('auc_score for Xgboost: (train): ', roc_auc_score(y_train, xg_prob)))
# Plot ROC curves
plt.subplots(1, figsize=(5,5))
plt.title('Receiver Operating Characteristic(train) - XGBoost ')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
xg_prob_test=xgboost.predict_proba(x_test)
xg_prob_test=xg_prob_test[:,1]
false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, xg_prob_test)
print(('auc_score for Xgboost(test): ', roc_auc_score(y_test, xg_prob_test)))
# Plot ROC curves
plt.subplots(1, figsize=(5,5))
plt.title('Receiver Operating Characteristic(test) - XGBoost ')
plt.plot(false_positive_rate2, true_positive_rate2)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## Feature Importance Graph

# In[265]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
#do code to support model
#"data" is the X dataframe and model is the SKlearn object

feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(x_train.columns, xgboost.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
#plt.figure(figsize=(15,7))
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=45,figsize=(15,7))


# ## Kfold Cross Validation

# In[266]:


xg = xgb.XGBClassifier()
scores = cross_val_score(xg, x_test, y_test, cv=10, scoring = "accuracy")
print(("Scores:", scores))
print(("Mean:", scores.mean()))
print(("Standard Deviation:", scores.std()))


# # RandomForestClassifier
# Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.Random decision forests correct for decision trees' habit of overfitting to their training set.

# In[267]:


rfc2=RandomForestClassifier()
rfc2.fit(x_train,y_train)


# In[268]:


#model on train using all the independent values in df
rfc_prediction = rfc2.predict(x_train)
rfc_score= accuracy_score(y_train,rfc_prediction)
print(('accuracy Score on train using RandomForest :',rfc_score))


# In[269]:


confusion_matrix(y_train, rfc_prediction)


# In[270]:


fpr, tpr, thresholds = metrics.roc_curve(y_train,rfc_prediction)
print(("AUC on train using RandomForest :",metrics.auc(fpr, tpr)))


# In[271]:


average_precision = average_precision_score(y_train, rfc_prediction)

print(('Average precision-recall score: {0:0.2f}'.format(
      average_precision)))


# In[272]:


print(('recall_score on train set :',recall_score(y_train, rfc_prediction)))


# In[273]:


print(('F1_sccore on train set :',f1_score(y_train, rfc_prediction)))


# In[274]:


print('classification Report on  train using RandomForest :')
print((classification_report(y_train,rfc_prediction)))


# In[275]:


#model on test using all the indpendent values in df
rfc_prediction = rfc2.predict(x_test)
rfc_score= accuracy_score(y_test,rfc_prediction)
print(('accuracy score on test using RandomForest ',rfc_score))


# In[276]:


confusion_matrix(y_test, rfc_prediction)


# In[277]:


fpr, tpr, thresholds = metrics.roc_curve(y_test,rfc_prediction)
print(("AUC on test using RandomForest :",metrics.auc(fpr, tpr)))


# In[278]:


average_precision = average_precision_score(y_test, rfc_prediction)

print(('Average precision-recall score: {0:0.2f}'.format(
      average_precision)))


# In[279]:


print(('recall_score on test set :',recall_score(y_test, rfc_prediction)))


# In[280]:


print(('F1_sccore on train set :',f1_score(y_test, rfc_prediction)))


# In[281]:


print('classification Report on  test using RandomForest :')
print((classification_report(y_test,rfc_prediction)))


# ## ROC Curve

# In[282]:


rf_prob=rfc2.predict_proba(x_train)
rf_prob=rf_prob[:,1]
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_train, rf_prob)
print(('auc_score for Random Forest : (train): ', roc_auc_score(y_train, rf_prob)))
# Plot ROC curves
plt.subplots(1, figsize=(5,5))
plt.title('Receiver Operating Characteristic(train) - Random Forest :')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
rf_prob_test=rfc2.predict_proba(x_test)
rf_prob_test=rf_prob_test[:,1]
false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, rf_prob_test)
print(('auc_score for Random forest (test): ', roc_auc_score(y_test, rf_prob_test)))
# Plot ROC curves
plt.subplots(1, figsize=(5,5))
plt.title('Receiver Operating Characteristic(test) - Random Forest : ')
plt.plot(false_positive_rate2, true_positive_rate2)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## Feature Importance graph

# In[283]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
#do code to support model
#"data" is the X dataframe and model is the SKlearn object

feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(x_train.columns, rfc2.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
#plt.figure(figsize=(15,7))
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=45,figsize=(15,7))


# ## Kfold Cross Validation

# In[284]:


lr = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(lr, x_train, y_train, cv=3, scoring = "accuracy")
print(("Scores:", scores))
print(("Mean:", scores.mean()))
print(("Standard Deviation:", scores.std()))


# # Decision Tree CLassifier
# 
# A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements.
# 
# Decision trees are commonly used in operations research, specifically in decision analysis, to help identify a strategy most likely to reach a goal, but are also a popular tool in machine learning.

# In[285]:


from sklearn.tree import DecisionTreeClassifier
dec=DecisionTreeClassifier()


# In[286]:


dec.fit(x_train,y_train)


# In[287]:


#model on train using all the independent values in df
dec_prediction = dec.predict(x_train)
dec_score= accuracy_score(y_train,dec_prediction)
print(('Accuracy score on train using Decision Tree :',dec_score))


# In[288]:


print((confusion_matrix(y_train, dec_prediction)))
fpr, tpr, thresholds = metrics.roc_curve(y_train,dec_prediction)
print(("AUC on train using DecisionTree :",metrics.auc(fpr, tpr)))
average_precision = average_precision_score(y_train, dec_prediction)
print(('Average precision-recall score: {0:0.2f}'.format(average_precision)))
print(('recall_score on train set :',recall_score(y_train, dec_prediction)))
print(('F1_sccore on train set :',f1_score(y_train, dec_prediction)))
print(('classification report on train using Decision tree ',classification_report(y_train,dec_prediction)))


# In[289]:


#model on test using all the independent values in df
dec_prediction = dec.predict(x_test)
dec_score= accuracy_score(y_test,dec_prediction)
print(('Accuracy Score on tree using Decision Tree  :',dec_score))


# In[290]:


print((confusion_matrix(y_test, dec_prediction)))
fpr, tpr, thresholds = metrics.roc_curve(y_test,dec_prediction)
print(("AUC on test using DecisionTree :",metrics.auc(fpr, tpr)))
average_precision = average_precision_score(y_test, dec_prediction)
print(('Average precision-recall score: {0:0.2f}'.format(average_precision)))
print(('recall_score on test set :',recall_score(y_test, dec_prediction)))
print(('F1_sccore on test set :',f1_score(y_test, dec_prediction)))
print(('classification report on test using Decision tree ',classification_report(y_test,dec_prediction)))


# ## ROC Curve

# In[291]:


rf_prob=dec.predict_proba(x_train)
rf_prob=rf_prob[:,1]
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_train, rf_prob)
print(('auc_score for decision tree : (train): ', roc_auc_score(y_train, rf_prob)))
# Plot ROC curves
plt.subplots(1, figsize=(5,5))
plt.title('Receiver Operating Characteristic(train) - decision tre :')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
rf_prob_test=dec.predict_proba(x_test)
rf_prob_test=rf_prob_test[:,1]
false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, rf_prob_test)
print(('auc_score for decision tree (test): ', roc_auc_score(y_test, rf_prob_test)))
# Plot ROC curves
plt.subplots(1, figsize=(5,5))
plt.title('Receiver Operating Characteristic(test) - decision tree : ')
plt.plot(false_positive_rate2, true_positive_rate2)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## Kfold Cross Validation

# In[292]:


lr = DecisionTreeClassifier()
scores = cross_val_score(lr, x_train, y_train, cv=5, scoring = "accuracy")
print(("Scores:", scores))
print(("Mean:", scores.mean()))
print(("Standard Deviation:", scores.std()))


# # ExtraTreeClassifier
# 
# Each Decision Tree in the Extra Trees Forest is constructed from the original training sample. Then, at each test node, Each tree is provided with a random sample of k features from the feature-set from which each decision tree must select the best feature to split the data based on some mathematical criteria (typically the Gini Index). This random sample of features leads to the creation of multiple de-correlated decision trees.

# In[293]:


from sklearn.tree import ExtraTreeClassifier
etc=ExtraTreeClassifier()
etc.fit(x_train,y_train)


# In[294]:


#model on train using all the independent values in df
etc_prediction = etc.predict(x_train)
etc_score= accuracy_score(y_train,etc_prediction)
print(('Accuracy score on train using extratree :',etc_score))


# In[295]:


print((confusion_matrix(y_train, etc_prediction)))
fpr, tpr, thresholds = metrics.roc_curve(y_train,etc_prediction)
print(("AUC on train using ExtraTree :",metrics.auc(fpr, tpr)))
average_precision = average_precision_score(y_train, etc_prediction)
print(('Average precision-recall score: {0:0.2f}'.format(average_precision)))
print(('recall_score on train set :',recall_score(y_train, etc_prediction)))
print(('F1_sccore on train set :',f1_score(y_train, etc_prediction)))
print(('classification report on train using Extra tree ',classification_report(y_train,etc_prediction)))


# In[296]:


#model on test using all the independent values in df
etc_prediction = etc.predict(x_test)
etc_score= accuracy_score(y_test,etc_prediction)
print(('Accuracy score on test using extratree :',etc_score))


# In[297]:


print((confusion_matrix(y_test, etc_prediction)))
fpr, tpr, thresholds = metrics.roc_curve(y_test,etc_prediction)
print(("AUC on train using ExtraTree :",metrics.auc(fpr, tpr)))
average_precision = average_precision_score(y_test, etc_prediction)
print(('Average precision-recall score: {0:0.2f}'.format(average_precision)))
print(('recall_score on test set :',recall_score(y_test, dec_prediction)))
print(('F1_sccore on test set :',f1_score(y_test, etc_prediction)))
print(('classification report on test using Extra tree ',classification_report(y_test,etc_prediction)))


# ## ROC Curve 

# In[298]:


rf_prob=etc.predict_proba(x_train)
rf_prob=rf_prob[:,1]
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_train, rf_prob)
print(('auc_score for Extra tree : (train): ', roc_auc_score(y_train, rf_prob)))
# Plot ROC curves
plt.subplots(1, figsize=(5,5))
plt.title('Receiver Operating Characteristic(train) - Extra tree :')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
rf_prob_test=etc.predict_proba(x_test)
rf_prob_test=rf_prob_test[:,1]
false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, rf_prob_test)
print(('auc_score for Extra Tree (test): ', roc_auc_score(y_test, rf_prob_test)))
# Plot ROC curves
plt.subplots(1, figsize=(5,5))
plt.title('Receiver Operating Characteristic(test) - Extra tree : ')
plt.plot(false_positive_rate2, true_positive_rate2)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## Kfold Cross Validation

# In[299]:


lr = ExtraTreeClassifier()
scores = cross_val_score(lr, x_train, y_train, cv=5, scoring = "accuracy")
print(("Scores:", scores))
print(("Mean:", scores.mean()))
print(("Standard Deviation:", scores.std()))


# # AdaBoostClassifier
# 
# An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.

# In[300]:


from sklearn.ensemble import AdaBoostClassifier
ada =AdaBoostClassifier(n_estimators=100)


# In[301]:


ada.fit(x_train,y_train)


# In[302]:


#model on train using all the independent values in df
ada_prediction = ada.predict(x_train)
ada_score= accuracy_score(y_train,ada_prediction)
print(('Accuracy score on train using AdaBoost :',ada_score))


# In[303]:


print((confusion_matrix(y_train, ada_prediction)))
fpr, tpr, thresholds = metrics.roc_curve(y_train,ada_prediction)
print(("AUC on train using AdaBoost :",metrics.auc(fpr, tpr)))
average_precision = average_precision_score(y_train, ada_prediction)
print(('Average precision-recall score: {0:0.2f}'.format(average_precision)))
print(('recall_score on train set :',recall_score(y_train, ada_prediction)))
print(('F1_sccore on train set :',f1_score(y_train, ada_prediction)))
print(('classification report on train using Extra tree ',classification_report(y_train,ada_prediction)))


# In[304]:


#model on test using all the independent values in df
ada_prediction = ada.predict(x_test)
ada_score= accuracy_score(y_test,ada_prediction)
print(('accuracy score on test using AdaBoost :',ada_score))


# In[305]:


print((confusion_matrix(y_test, ada_prediction)))
fpr, tpr, thresholds = metrics.roc_curve(y_test,ada_prediction)
print(("AUC on test using AdaBoost :",metrics.auc(fpr, tpr)))
average_precision = average_precision_score(y_test, ada_prediction)
print(('Average precision-recall score: {0:0.2f}'.format(average_precision)))
print(('recall_score on test set :',recall_score(y_test, ada_prediction)))
print(('F1_sccore on test set :',f1_score(y_test, ada_prediction)))
print(('classification report on test using Extra tree ',classification_report(y_test,ada_prediction)))


# ## ROC Curve

# In[306]:


rf_prob=ada.predict_proba(x_train)
rf_prob=rf_prob[:,1]
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_train, rf_prob)
print(('auc_score for ADAboost : (train): ', roc_auc_score(y_train, rf_prob)))
# Plot ROC curves
plt.subplots(1, figsize=(5,5))
plt.title('Receiver Operating Characteristic(train) -ADAboost :')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
rf_prob_test=ada.predict_proba(x_test)
rf_prob_test=rf_prob_test[:,1]
false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, rf_prob_test)
print(('auc_score for ADAboost (test): ', roc_auc_score(y_test, rf_prob_test)))
# Plot ROC curves
plt.subplots(1, figsize=(5,5))
plt.title('Receiver Operating Characteristic(test) - ADAboost : ')
plt.plot(false_positive_rate2, true_positive_rate2)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## Kfold Cross Validation

# In[307]:


lr = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(lr, x_train, y_train, cv=3, scoring = "accuracy")
print(("Scores:", scores))
print(("Mean:", scores.mean()))
print(("Standard Deviation:", scores.std()))


# <h3 style='padding: 10px'>Comparison Table (LABEL ENCODING)</h2><table border-style:solid; class='table table-striped'> <thead> <tr> <th>Algorithm Used</th> <th>Accuracy Score On Train</th> <th>Accuracy Score On Test</th></tr> </thead> <tbody> <tr> <th scope='row'>XGBoost Classifier </th> <td>0.997</td> <td>0.781</td></tr> 
#     <tr> <th scope='row'>Random Forest Classifier</th> <td>0.995</td> <td>0.391</td></tr> <tr> 
#     <th scope='row'>Logisitic Regresion</th> <td>0.996</td> <td>0.998
#     </td></tr> <tr><th scope='row'>Decision Tree Classifier</th> <td>1.0</td> <td>0.30</td></tr>
#     <tr><th scope='row'>Extra tree classifier</th><td>1.0</td><td>0.612</td></tr>
#     <tr><th scope='row'>ADA boost classifier</th><td>0.996</td><td>0.862</td></tr>
#     </tbody> </table>

# # Binary Encoding

# In[308]:


df_train_bin=df_train_new


# In[309]:


from sklearn.preprocessing import LabelBinarizer
le1 = preprocessing.LabelBinarizer()
le1.fit(df_train_new['term'])
list(le1.classes_)
df_train_new['term'] = le1.transform(df_train_new['term'])
df_train_new.head()


# In[310]:


le1 = preprocessing.LabelBinarizer()
le1.fit(df_train_new['grade'])
list(le1.classes_)
df_train_new['grade'] = le1.transform(df_train_new['grade'])
df_train_new.head()


# In[311]:


le1 = preprocessing.LabelBinarizer()
le1.fit(df_train_new['sub_grade'])
list(le1.classes_)
df_train_new['sub_grade'] = le1.transform(df_train_new['sub_grade'])
df_train_new.head()


# In[312]:


le1 = preprocessing.LabelBinarizer()
le1.fit(df_train_new['home_ownership'])
list(le1.classes_)
df_train_new['home_ownership'] = le1.transform(df_train_new['home_ownership'])
df_train_new.head()


# In[313]:


le1 = preprocessing.LabelBinarizer()
le1.fit(df_train_new['verification_status'])
list(le1.classes_)
df_train_new['verification_status'] = le1.transform(df_train_new['verification_status'])
df_train_new.head()


# In[314]:


le1 = preprocessing.LabelBinarizer()
le1.fit(df_train_new['purpose'])
list(le1.classes_)
df_train_new['purpose'] = le1.transform(df_train_new['purpose'])
df_train_new.head()


# In[315]:


le1 = preprocessing.LabelBinarizer()
le1.fit(df_train_new['addr_state'])
list(le1.classes_)
df_train_new['addr_state'] = le1.transform(df_train_new['addr_state'])
df_train_new.head()


# In[316]:


le1 = preprocessing.LabelBinarizer()
le1.fit(df_train_new['application_type'])
list(le1.classes_)
df_train_new['application_type'] = le1.transform(df_train_new['application_type'])
df_train_new.head()


# In[317]:


le1 = preprocessing.LabelBinarizer()
le1.fit(df_train_new['pymnt_plan'])
list(le1.classes_)
df_train_new['pymnt_plan'] = le1.transform(df_train_new['pymnt_plan'])
df_train_new.head()


# In[318]:


le1 = preprocessing.LabelBinarizer()
le1.fit(df_train_new['initial_list_status'])
list(le1.classes_)
df_train_new['initial_list_status'] = le1.transform(df_train_new['initial_list_status'])
df_train_new.head()


# In[319]:


df_train_new.dtypes


# ## Train Test Split 

# In[320]:


train = df_train_new[df_train_new['issue_d'] < '2015-6-01']
test = df_train_new[df_train_new['issue_d'] >= '2015-6-01']


# In[321]:


del df_train_new['issue_d']


# In[322]:


x_train=train.drop(['default_ind','title','issue_d'],axis=1)
y_train=train['default_ind']
x_test=test.drop(['default_ind','title','issue_d'],axis=1)
y_test=test['default_ind']


# ## Logisitic Regression on Binary encoded Dataset
# 

# In[323]:


log =LogisticRegression()
log.fit(x_train,y_train)


# In[324]:


#model on train using all the independent values in df
log_prediction = log.predict(x_train)
log_score= accuracy_score(y_train,log_prediction)
print(('Accuracy score on train set using Logistic Regression :',log_score))


# In[325]:


print((confusion_matrix(y_train, log_prediction)))
fpr, tpr, thresholds = metrics.roc_curve(y_train,log_prediction)
print(("AUC on train using Logistic regression :",metrics.auc(fpr, tpr)))

average_precision = average_precision_score(y_train, log_prediction)

print(('Average precision-recall score: {0:0.2f}'.format(
      average_precision)))
print(('recall_score on train set :',recall_score(y_train, log_prediction)))
print(('F1_sccore on train set :',f1_score(y_train, log_prediction)))
print(('classification report on train using Logistic regression  ',
      classification_report(y_train,log_prediction)))


# In[326]:


#model on train using all the independent values in df
log_prediction = log.predict(x_test)
log_score= accuracy_score(y_test,log_prediction)
print(('accuracy score on test using Logisitic Regression :',log_score))


# In[327]:


print((confusion_matrix(y_test, log_prediction)))
fpr, tpr, thresholds = metrics.roc_curve(y_test,log_prediction)
print(("AUC on test using Logistic regression :",metrics.auc(fpr, tpr)))

average_precision = average_precision_score(y_test, log_prediction)

print(('Average precision-recall score: {0:0.2f}'.format(
      average_precision)))
print(('recall_score on test set :',recall_score(y_test, log_prediction)))
print(('F1_sccore on test set :',f1_score(y_test, log_prediction)))
print(('classification report on test using Logistic regression  ',classification_report(y_test,log_prediction)))


# ## ROC Curve

# In[328]:


lr_prob=log.predict_proba(x_train)
lr_prob=lr_prob[:,1]
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_train, lr_prob)
print(('auc_score for Logistic Regression(train): ', roc_auc_score(y_train, lr_prob)))
# Plot ROC curves
plt.subplots(1, figsize=(5,5))
plt.title('Receiver Operating Characteristic(train) - logistic regression')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
lr_prob_test=log.predict_proba(x_test)
lr_prob_test=lr_prob_test[:,1]
false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, lr_prob_test)
print(('auc_score for Logistic Regression(test): ', roc_auc_score(y_test, lr_prob_test)))
# Plot ROC curves
plt.subplots(1, figsize=(5,5))
plt.title('Receiver Operating Characteristic(test) - logistic regression')
plt.plot(false_positive_rate2, true_positive_rate2)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## KFold Cross Validation 

# In[329]:


lr = LogisticRegression()
scores = cross_val_score(lr, x_train, y_train, cv=5, scoring = "accuracy")
print(("Scores:", scores))
print(("Mean:", scores.mean()))
print(("Standard Deviation:", scores.std()))


# # XGBoost on Binary Encoded data

# In[330]:


xgboost = xgb.XGBClassifier(max_depth=3,n_estimators=300,learning_rate=0.05)


# In[331]:


xgboost.fit(x_train,y_train)


# In[332]:


#XGBoost model on the train set
XGB_prediction = xgboost.predict(x_train)
XGB_score= accuracy_score(y_train,XGB_prediction)
print(('accuracy score on train using XGBoost ',XGB_score))


# In[333]:


print((confusion_matrix(y_train, XGB_prediction)))
fpr, tpr, thresholds = metrics.roc_curve(y_train,XGB_prediction)
print(("AUC on train using XGBClassifiers:",metrics.auc(fpr, tpr)))

average_precision = average_precision_score(y_train, XGB_prediction)

print(('Average precision-recall score: {0:0.2f}'.format(
      average_precision)))
print(('recall_score on train set :',recall_score(y_train, XGB_prediction)))
print(('F1_sccore on train set :',f1_score(y_train, XGB_prediction)))
print('classification report on train using XGBoost  ')
print((classification_report(y_train,XGB_prediction)))


# In[334]:


#XGBoost model on the test
XGB_prediction = xgboost.predict(x_test)
XGB_score= accuracy_score(y_test,XGB_prediction)
print(('accuracy score on test using XGBoost :',XGB_score))


# In[335]:


print((confusion_matrix(y_test, XGB_prediction)))
fpr, tpr, thresholds = metrics.roc_curve(y_test,XGB_prediction)
print(("AUC on test using XGBClassifiers:",metrics.auc(fpr, tpr)))

average_precision = average_precision_score(y_test, XGB_prediction)

print(('Average precision-recall score: {0:0.2f}'.format(
      average_precision)))
print(('recall_score on test set :',recall_score(y_test, XGB_prediction)))
print(('F1_sccore on test set :',f1_score(y_test, XGB_prediction)))
print('classification report on test using XGBoost  ')
print((classification_report(y_test,XGB_prediction)))


# ## ROC Curve

# In[336]:


xg_prob=xgboost.predict_proba(x_train)
xg_prob=xg_prob[:,1]
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_train, xg_prob)
print(('auc_score for Xgboost: (train): ', roc_auc_score(y_train, xg_prob)))
# Plot ROC curves
plt.subplots(1, figsize=(5,5))
plt.title('Receiver Operating Characteristic(train) - XGBoost ')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
xg_prob_test=xgboost.predict_proba(x_test)
xg_prob_test=xg_prob_test[:,1]
false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, xg_prob_test)
print(('auc_score for Xgboost(test): ', roc_auc_score(y_test, xg_prob_test)))
# Plot ROC curves
plt.subplots(1, figsize=(5,5))
plt.title('Receiver Operating Characteristic(test) - XGBoost ')
plt.plot(false_positive_rate2, true_positive_rate2)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## Kfold crossValiddation

# In[337]:


xg = xgb.XGBClassifier()
scores = cross_val_score(xg, x_test, y_test, cv=5, scoring = "accuracy")
print(("Scores:", scores))
print(("Mean:", scores.mean()))
print(("Standard Deviation:", scores.std()))


# # RandomForestclassifier on BinaryEncoded dataset

# In[338]:


rfc2=RandomForestClassifier(n_estimators=100)
rfc2.fit(x_train,y_train)


# In[339]:


#model on train using all the independent values in df
rfc_prediction = rfc2.predict(x_train)
rfc_score= accuracy_score(y_train,rfc_prediction)
print(('accuracy Score on train using RandomForest :',rfc_score))


# In[340]:


print((confusion_matrix(y_train, rfc_prediction)))
fpr, tpr, thresholds = metrics.roc_curve(y_train,rfc_prediction)
print(("AUC on train using RandomForest :",metrics.auc(fpr, tpr)))

average_precision = average_precision_score(y_train, rfc_prediction)

print(('Average precision-recall score: {0:0.2f}'.format(
      average_precision)))
print(('recall_score on train set :',recall_score(y_train, rfc_prediction)))
print(('F1_sccore on train set :',f1_score(y_train, rfc_prediction)))
print('classification Report on  train using RandomForest :')
print((classification_report(y_train,rfc_prediction)))


# In[341]:


#model on test using all the indpendent values in df
rfc_prediction = rfc2.predict(x_test)
rfc_score= accuracy_score(y_test,rfc_prediction)
print(('accuracy score on test using RandomForest ',rfc_score))


# In[342]:


print((confusion_matrix(y_test, rfc_prediction)))
fpr, tpr, thresholds = metrics.roc_curve(y_test,rfc_prediction)
print(("AUC on test using RandomForest :",metrics.auc(fpr, tpr)))

average_precision = average_precision_score(y_test, rfc_prediction)

print(('Average precision-recall score: {0:0.2f}'.format(
      average_precision)))
print(('recall_score on test set :',recall_score(y_test, rfc_prediction)))
print(('F1_sccore on test set :',f1_score(y_test, rfc_prediction)))
print('classification Report on  test using RandomForest :')
print((classification_report(y_test,rfc_prediction)))


# ## ROC Curve 

# In[343]:


rf_prob=rfc2.predict_proba(x_train)
rf_prob=rf_prob[:,1]
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_train, rf_prob)
print(('auc_score for Random Forest : (train): ', roc_auc_score(y_train, rf_prob)))
# Plot ROC curves
plt.subplots(1, figsize=(5,5))
plt.title('Receiver Operating Characteristic(train) - Random Forest :')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
rf_prob_test=rfc2.predict_proba(x_test)
rf_prob_test=rf_prob_test[:,1]
false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, rf_prob_test)
print(('auc_score for Random forest (test): ', roc_auc_score(y_test, rf_prob_test)))
# Plot ROC curves
plt.subplots(1, figsize=(5,5))
plt.title('Receiver Operating Characteristic(test) - Random Forest : ')
plt.plot(false_positive_rate2, true_positive_rate2)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## KcrossFold Validation

# In[344]:


lr = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(lr, x_train, y_train, cv=5, scoring = "accuracy")
print(("Scores:", scores))
print(("Mean:", scores.mean()))
print(("Standard Deviation:", scores.std()))


# <h2 style='padding: 10px'>Comparison Table (BINARY ENCODING)</h2><table border-style:solid; class='table table-striped'> <thead> <tr> <th>Algorithm Used</th> <th>Accuracy Score On Train</th> <th>Accuracy Score On Test</th></tr> </thead> <tbody> <tr> <th scope='row'>XGBoost Classifier </th> <td>0.997</td> <td>0.594</td></tr> 
#     <tr> <th scope='row'>Random Forest Classifier</th> <td>0.999</td> <td>0.358</td></tr> <tr> 
#     <th scope='row'>Logisitic Regresion</th> <td>0.996</td> <td>0.998
#     </tbody> </table>

# # Conclusion

# From all above analyis we can conclude that after binary encoding of dataset and applying logisitic regression model gives best results with accuracy score 0.996 on train and 0.998 on train. 
# 
# 
# Hence logistic model can be used for further predicting.
