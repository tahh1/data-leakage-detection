#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pprint import pprint
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, roc_auc_score 
import scikitplot as skplt
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Data Preparation

# In[2]:


df = pd.read_csv('creditcard.csv')
df.head()


# In[3]:


# Check for empty cells and if data types are correct for the respective columns
df.info()


# ### Exploratory Data Analysis

# In[4]:


# Plot with seaborn
sns.set_context('talk')
sns.set_palette('dark')
sns.set_style('ticks')

classes = pd.value_counts(df['Class'], sort = True).sort_index()
classes.plot(kind = 'bar')
plt.figsize = (20,10)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency');


# In[5]:


# Distribution of target variable
print((df['Class'].value_counts()))


# In[6]:


print(('Non-fraudulent transactions represents', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset.'))
print(('Fraudulent transactions represents', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset.'))


# In[7]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20,9))
bins = 50

ax1.hist(df.Time[df.Class==0], bins=bins)
ax1.set(xlabel='Time (in seconds)',
        ylabel='Number of Transactions',
        title='Non-fraudulent Transactions') 

ax2.hist(df.Time[df.Class==1], bins=bins)
ax2.set(xlabel='Time (in seconds)',
        ylabel='Number of Transactions',
        title='Fraudulent Transactions')

plt.tight_layout()


# ### Benford's Law

# In[8]:


# Expected frequency for every digit from 1 to 9
exp=pd.Series()
for i in range(1,10):
    exp.at[i] =  math.log(1 + 1/i,9) * 100

df_BL = pd.DataFrame(columns=['Frequency (%)'])
df_BL['Frequency (%)'] = exp

df_BL.plot.bar(figsize=(13,7), title='Frequency of Occurrence of Leading Digits According to Benford’s Law');


# In[9]:


# Expected frequency for every digit from 1 to 9
exp = pd.Series()
for i in range(1,10):
    exp.at[i] =  math.log(1 + 1/i,9) * 100
    
# Actual frequency based on dataset   
amt = df['Amount'][df['Amount'] >= 1].apply(lambda x: x // 10**(len(str(math.floor(x))) - 1))
num_counts = amt.value_counts()
num_total = amt.count()
num_percent = num_counts.apply(lambda x: 100 * x / num_total)

df = pd.DataFrame(columns=['Expected', 'Actual'])
df['Expected'] = exp
df['Actual'] = num_percent
df.plot.bar(figsize = (18,8), title = 'Frequency of Occurrence of Leading Digits');


# ### Data Modelling

# In[8]:


# Scale columns 'Amount' and 'Time'
std_scaler = StandardScaler()

df['amount_scaled'] = std_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['time_scaled'] = std_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)


# In[9]:


df.info()


# In[10]:


X = df.iloc[:,np.r_[:,0:28,29,30]] # independent columns - features
y = df.iloc[:,28]                  # target column - Class


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 1)


# ### Model 1: Baseline Logistic Regression Model

# In[12]:


# Create baseline logistic regression classifier
LR = LogisticRegression()

# Fit training data and training labels
LR.fit(X_train, y_train)


# In[13]:


print(f'Baseline Logistic Regression Model Accuracy for train data: {LR.score(X_train, y_train)}')
print(f'Baseline Logistic Regression Model Accuracy for test data: {LR.score(X_test, y_test)}')


# ### Synthetic Minority Oversampling Technique ("SMOTE")

# In[14]:


sm = SMOTE(sampling_strategy='minority', random_state=1, k_neighbors=5)

X_train_res, y_train_res = sm.fit_sample(df.drop('Class', axis=1), df['Class'])


# In[15]:


new_df = pd.concat([pd.DataFrame(X_train_res), pd.DataFrame(y_train_res)], axis=1)
new_df['Class'].value_counts()


# In[16]:


new_df.info()


# ### Correlation Matrix with Heat Map

# In[17]:


# Obtain correlations of each features in dataset
sns.set(font_scale=1.4)
corrmat = new_df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(30,30))

# Plot heat map
correlation = sns.heatmap(new_df[top_corr_features].corr(),annot=True,fmt=".2f",cmap='Blues')


# ### Box Plots

# In[18]:


sns.set_style('ticks')
f, axes = plt.subplots(ncols=4, figsize=(30,8))

sns.boxplot(x='Class', y='V2', data=new_df, palette='Set1', ax=axes[0])
axes[0].set_title('Positive Correlation - V2 vs Class')

sns.boxplot(x='Class', y='V4', data=new_df, palette='Set1', ax=axes[1])
axes[1].set_title('Positive Correlation - V4 vs Class')

sns.boxplot(x='Class', y='V11', data=new_df, palette='Set1', ax=axes[2])
axes[2].set_title('Positive Correlation - V11 vs Class')

sns.boxplot(x='Class', y='V19', data=new_df, palette='Set1', ax=axes[3])
axes[3].set_title('Positive Correlation - V19 vs Class')

plt.show()


# In[19]:


f, axes = plt.subplots(ncols=4, figsize=(30,8))

sns.boxplot(x='Class', y='V10', data=new_df, palette='husl', ax=axes[0])
axes[0].set_title('Negative Correlation - V10 vs Class')

sns.boxplot(x='Class', y='V12', data=new_df, palette='husl', ax=axes[1])
axes[1].set_title('Negative Correlation - V12 vs Class')

sns.boxplot(x='Class', y='V14', data=new_df, palette='husl', ax=axes[2])
axes[2].set_title('Negative Correlation - V14 vs Class')

sns.boxplot(x='Class', y='V16', data=new_df, palette='husl', ax=axes[3])
axes[3].set_title('Negative Correlation - V16 vs Class')

plt.show()


# ### Distribution Plot

# In[20]:


print('Top 4 Features - Positive Correlation with Class\n')
f, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(22, 6))

v2_fraud_dist = new_df['V2'].loc[new_df['Class'] == 1].values
sns.distplot(v2_fraud_dist, ax=ax1, fit=norm, color='#23b458')
ax1.set_title('V2 Distribution \n (Fraudulent Transactions)', fontsize=14)

v4_fraud_dist = new_df['V4'].loc[new_df['Class'] == 1].values
sns.distplot(v4_fraud_dist, ax=ax2, fit=norm, color='#ee2424')
ax2.set_title('V4 Distribution \n (Fraudulent Transactions)', fontsize=14)

v11_fraud_dist = new_df['V11'].loc[new_df['Class'] == 1].values
sns.distplot(v11_fraud_dist, ax=ax3, fit=norm, color='#f0701a')
ax3.set_title('V11 Distribution \n (Fraudulent Transactions)', fontsize=14)

v19_fraud_dist = new_df['V19'].loc[new_df['Class'] == 1].values
sns.distplot(v19_fraud_dist, ax=ax4, fit=norm, color='#2783d6')
ax4.set_title('V19 Distribution \n (Fraudulent Transactions)', fontsize=14)

plt.show()


# In[21]:


print('Top 4 Features - Negative Correlation with Class\n')
f, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(22, 6))

v10_fraud_dist = new_df['V10'].loc[new_df['Class'] == 1].values
sns.distplot(v10_fraud_dist, ax=ax1, fit=norm, color='#23b458')
ax1.set_title('V10 Distribution \n (Fraudulent Transactions)', fontsize=14)

v12_fraud_dist = new_df['V12'].loc[new_df['Class'] == 1].values
sns.distplot(v12_fraud_dist, ax=ax2, fit=norm, color='#ee2424')
ax2.set_title('V12 Distribution \n (Fraudulent Transactions)', fontsize=14)

v14_fraud_dist = new_df['V14'].loc[new_df['Class'] == 1].values
sns.distplot(v14_fraud_dist, ax=ax3, fit=norm, color='#f0701a')
ax3.set_title('V14 Distribution \n (Fraudulent Transactions)', fontsize=14)

v16_fraud_dist = new_df['V16'].loc[new_df['Class'] == 1].values
sns.distplot(v16_fraud_dist, ax=ax4, fit=norm, color='#2783d6')
ax4.set_title('V16 Distribution \n (Fraudulent Transactions)', fontsize=14)

plt.show()


# ### Removal of Extreme Outliers

# In[22]:


print('Removal of Extreme Outliers from Top 4 Features which are Positively Correlated with Class\n')

# V2 - Removing extreme outliers from fraudulent transactions
print('V2 Analysis')
v2_fraud = new_df['V2'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v2_fraud, 25), np.percentile(v2_fraud, 75)
print(('25th Quartile: {} | 75th Quartile: {}'.format(q25, q75)))
v2_iqr = q75 - q25
print(('IQR: {}'.format(v2_iqr)))

v2_cut_off = v2_iqr * 1.5
v2_lower, v2_upper = q25 - v2_cut_off, q75 + v2_cut_off
print(('Cut-Off: {}'.format(v2_cut_off)))
print(('Lower Limit: {}'.format(v2_lower)))
print(('Upper Limit: {}'.format(v2_upper)))

outliers = [x for x in v2_fraud if x < v2_lower or x > v2_upper]
print(('Number of Outliers: {}'.format(len(outliers))))
new_df = new_df.drop(new_df[(new_df['V2'] > v2_upper) | (new_df['V2'] < v2_lower)].index)
print(('Number of Instances after Removing Outliers: {}'.format(len(new_df))))
print('\n')

# V4 - Removing extreme outliers from fraudulent transactions
print('V4 Analysis')
v4_fraud = new_df['V4'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v4_fraud, 25), np.percentile(v4_fraud, 75)
print(('25th Quartile: {} | 75th Quartile: {}'.format(q25, q75)))
v4_iqr = q75 - q25
print(('IQR: {}'.format(v4_iqr)))

v4_cut_off = v4_iqr * 1.5
v4_lower, v4_upper = q25 - v4_cut_off, q75 + v4_cut_off
print(('Cut-Off: {}'.format(v4_cut_off)))
print(('Lower Limit: {}'.format(v4_lower)))
print(('Upper Limit: {}'.format(v4_upper)))

outliers = [x for x in v4_fraud if x < v4_lower or x > v4_upper]
print(('Number of Outliers: {}'.format(len(outliers))))
new_df = new_df.drop(new_df[(new_df['V4'] > v4_upper) | (new_df['V4'] < v4_lower)].index)
print(('Number of Instances after Removing Outliers: {}'.format(len(new_df))))
print('\n')

# V11 - Removing extreme outliers from fraudulent transactions
print('V11 Analysis')
v11_fraud = new_df['V11'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v11_fraud, 25), np.percentile(v11_fraud, 75)
print(('25th Quartile: {} | 75th Quartile: {}'.format(q25, q75)))
v11_iqr = q75 - q25
print(('IQR: {}'.format(v11_iqr)))

v11_cut_off = v11_iqr * 1.5
v11_lower, v11_upper = q25 - v11_cut_off, q75 + v11_cut_off
print(('Cut-Off: {}'.format(v11_cut_off)))
print(('Lower Limit: {}'.format(v11_lower)))
print(('Upper Limit: {}'.format(v11_upper)))

outliers = [x for x in v11_fraud if x < v11_lower or x > v11_upper]
print(('Number of Outliers: {}'.format(len(outliers))))
new_df = new_df.drop(new_df[(new_df['V11'] > v11_upper) | (new_df['V11'] < v11_lower)].index)
print(('Number of Instances after Removing Outliers: {}'.format(len(new_df))))
print('\n')

# V19 - Removing extreme outliers from fraudulent transactions
print('V19 Analysis')
v19_fraud = new_df['V19'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v19_fraud, 25), np.percentile(v19_fraud, 75)
print(('25th Quartile: {} | 75th Quartile: {}'.format(q25, q75)))
v19_iqr = q75 - q25
print(('IQR: {}'.format(v19_iqr)))

v19_cut_off = v19_iqr * 1.5
v19_lower, v19_upper = q25 - v19_cut_off, q75 + v19_cut_off
print(('Cut-Off: {}'.format(v19_cut_off)))
print(('Lower Limit: {}'.format(v19_lower)))
print(('Upper Limit: {}'.format(v19_upper)))

outliers = [x for x in v19_fraud if x < v19_lower or x > v19_upper]
print(('Number of Outliers: {}'.format(len(outliers))))
new_df = new_df.drop(new_df[(new_df['V19'] > v19_upper) | (new_df['V19'] < v19_lower)].index)
print(('Number of Instances after Removing Outliers: {}'.format(len(new_df))))


# In[23]:


print('Removal of Extreme Outliers from Top 4 Features which are Negatively Correlated with Class\n')

# V10 - Removing extreme outliers from fraudulent transactions
print('V10 Analysis')
v10_fraud = new_df['V10'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
print(('25th Quartile: {} | 75th Quartile: {}'.format(q25, q75)))
v10_iqr = q75 - q25
print(('IQR: {}'.format(v10_iqr)))

v10_cut_off = v10_iqr * 1.5
v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
print(('Cut-Off: {}'.format(v10_cut_off)))
print(('Lower Limit: {}'.format(v10_lower)))
print(('Upper Limit: {}'.format(v10_upper)))

outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]
print(('Number of Outliers: {}'.format(len(outliers))))
new_df = new_df.drop(new_df[(new_df['V10'] > v10_upper) | (new_df['V10'] < v10_lower)].index)
print(('Number of Instances after Removing Outliers: {}'.format(len(new_df))))
print('\n')

# V12 - Removing extreme outliers from fraudulent transactions
print('V12 Analysis')
v12_fraud = new_df['V12'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
print(('25th Quartile: {} | 75th Quartile: {}'.format(q25, q75)))
v12_iqr = q75 - q25
print(('IQR: {}'.format(v12_iqr)))

v12_cut_off = v12_iqr * 1.5
v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
print(('Cut-Off: {}'.format(v12_cut_off)))
print(('Lower Limit: {}'.format(v12_lower)))
print(('Upper Limit: {}'.format(v12_upper)))

outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]
print(('Number of Outliers: {}'.format(len(outliers))))
new_df = new_df.drop(new_df[(new_df['V12'] > v12_upper) | (new_df['V12'] < v12_lower)].index)
print(('Number of Instances after Removing Outliers: {}'.format(len(new_df))))
print('\n')

# V14 - Removing extreme outliers from fraudulent transactions
print('V14 Analysis')
v14_fraud = new_df['V14'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
print(('25th Quartile: {} | 75th Quartile: {}'.format(q25, q75)))
v14_iqr = q75 - q25
print(('IQR: {}'.format(v14_iqr)))

v14_cut_off = v14_iqr * 1.5
v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
print(('Cut-Off: {}'.format(v14_cut_off)))
print(('Lower Limit: {}'.format(v14_lower)))
print(('Upper Limit: {}'.format(v14_upper)))

outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]
print(('Number of Outliers: {}'.format(len(outliers))))
new_df = new_df.drop(new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)
print(('Number of Instances after Removing Outliers: {}'.format(len(new_df))))
print('\n')

# V16 - Removing extreme outliers from fraudulent transactions
print('V16 Analysis')
v16_fraud = new_df['V16'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v16_fraud, 25), np.percentile(v16_fraud, 75)
print(('25th Quartile: {} | 75th Quartile: {}'.format(q25, q75)))
v16_iqr = q75 - q25
print(('IQR: {}'.format(v16_iqr)))

v16_cut_off = v16_iqr * 1.5
v16_lower, v16_upper = q25 - v16_cut_off, q75 + v16_cut_off
print(('Cut-Off: {}'.format(v16_cut_off)))
print(('Lower Limit: {}'.format(v16_lower)))
print(('Upper Limit: {}'.format(v16_upper)))

outliers = [x for x in v16_fraud if x < v16_lower or x > v16_upper]
print(('Number of Outliers: {}'.format(len(outliers))))
new_df = new_df.drop(new_df[(new_df['V16'] > v16_upper) | (new_df['V16'] < v16_lower)].index)
print(('Number of Instances after Removing Outliers: {}'.format(len(new_df))))


# In[24]:


new_df.info()


# In[25]:


X_clean = new_df.iloc[:,0:30] # independent columns - features
y_clean = new_df.iloc[:,30]   # target column - Class


# In[26]:


X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean, y_clean,
                                                                            test_size = 0.2,
                                                                            random_state = 1)


# ### Model 2: Optimized Logistic Regression Model

# In[27]:


# Set the model parameters for grid search
log_reg_params = {'penalty': ['l1', 'l2'], 
                  'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# Set up grid search meta-estimator
grid_search = GridSearchCV(LogisticRegression(), log_reg_params, 
                           n_jobs=-1, scoring='roc_auc', cv=3)

# Train the grid search meta-estimator to obtain optimal model
opt_LR = grid_search.fit(X_train_clean, y_train_clean)

# Print optimal hyperparameters
pprint(opt_LR.best_estimator_.get_params())


# In[28]:


print(f'Optimized Logistic Regression Model Accuracy for train data: {opt_LR.score(X_train_clean, y_train_clean)}')
print(f'Optimized Logistic Regression Model Accuracy for test data: {opt_LR.score(X_test_clean, y_test_clean)}')


# ### Model 3: Naive Bayes Model

# In[29]:


# Create Naive Bayes classifier
NB = BernoulliNB()

# Fit training data and training labels
NB.fit(X_train_clean, y_train_clean)


# In[30]:


print(f'Naive Bayes Model Accuracy for train data: {NB.score(X_train_clean, y_train_clean)}')
print(f'Naive Bayes Model Accuracy for test data: {NB.score(X_test_clean, y_test_clean)}')


# ### Performance Evaluation of Models

# In[31]:


# Predict target variables (ie. labels) for each classifer
lr_classifier_name = ['Baseline Logistic Regression']
lr_predicted_labels = LR.predict(X_test)

opt_lr_classifier_name = ['Optimized Logistic Regression']
opt_lr_predicted_labels = opt_LR.predict(X_test_clean)

nb_classifier_name = ['Naive Bayes']
nb_predicted_labels = NB.predict(X_test_clean)


# ### 1. Classification Report

# In[32]:


print(("Classification Report for", lr_classifier_name, " :\n ",
      metrics.classification_report(y_test, lr_predicted_labels, 
                                    target_names=['Non-Fraudulent','Fraud'])))

print(("Classification Report for ", opt_lr_classifier_name, " :\n ",
      metrics.classification_report(y_test_clean, opt_lr_predicted_labels,
                                   target_names=['Non-Fraudulent','Fraud'])))

print(("Classification Report for ", nb_classifier_name, " :\n ",
      metrics.classification_report(y_test_clean, nb_predicted_labels,
                                   target_names=['Non-Fraudulent','Fraud'])))


# ### 2. Confusion Matrix

# In[33]:


print(("Confusion Matrix for", lr_classifier_name))
skplt.metrics.plot_confusion_matrix(y_test, lr_predicted_labels, normalize=True)
plt.show()

print(("Confusion Matrix for", opt_lr_classifier_name))
skplt.metrics.plot_confusion_matrix(y_test_clean, opt_lr_predicted_labels, normalize=True)
plt.show()

print(("Confusion Matrix for", nb_classifier_name))
skplt.metrics.plot_confusion_matrix(y_test_clean, nb_predicted_labels, normalize=True)
plt.show()


# ### 3. Precision-Recall Curve

# In[34]:


fig, axList = plt.subplots(ncols=3)
fig.set_size_inches(21,6)

# Plot the Precision-Recall curve for Baseline Logistic Regression  
ax = axList[0]
lr_predicted_proba = LR.predict_proba(X_test)
precision, recall, _ = precision_recall_curve(y_test, lr_predicted_proba[:,1])
ax.plot(recall, precision,color='blue')
ax.set(xlabel='Recall', ylabel='Precision', xlim=[0, 1], ylim=[0, 1],
       title='Precision-Recall Curve - Baseline Logistic Regression ')
ax.step(recall, precision, color='blue', alpha=0.2, where='post')
ax.fill_between(recall, precision, step='post', alpha=0.2, color='blue')
ax.grid(True)

# Plot the Precision-Recall curve for Optimized Logistic Regression
ax = axList[1]
opt_lr_predicted_proba = opt_LR.predict_proba(X_test_clean)
precision, recall, _ = precision_recall_curve(y_test_clean, opt_lr_predicted_proba[:,1])
ax.plot(recall, precision,color='black')
ax.set(xlabel='Recall', ylabel='Precision', xlim=[0, 1], ylim=[0, 1],
       title='Precision-Recall Curve - Optimized Logistic Regression')
ax.step(recall, precision, color='black', alpha=0.2, where='post')
ax.fill_between(recall, precision, step='post', alpha=0.2, color='black')
ax.grid(True)

# Plot the Precision-Recall curve for Naive Bayes
ax = axList[2]
nb_predicted_proba = NB.predict_proba(X_test_clean)
precision, recall, _ = precision_recall_curve(y_test_clean, nb_predicted_proba[:,1])
ax.plot(recall, precision,color='green')
ax.set(xlabel='Recall', ylabel='Precision', xlim=[0, 1], ylim=[0, 1],
       title='Precision-Recall Curve - Naive Bayes')
ax.step(recall, precision, color='green', alpha=0.2, where='post')
ax.fill_between(recall, precision, step='post', alpha=0.2, color='green')
ax.grid(True)
plt.tight_layout()


# ### 4. ROC Curve and AUC

# In[35]:


fig, axList = plt.subplots(ncols=3)
fig.set_size_inches(21,6)

# Plot the ROC-AUC curve for Baseline Logistic Regression
ax = axList[0]
lr = LR.fit(X_train, y_train.values.ravel()) 
lr_predicted_label_r = LR.predict_proba(X_test)

def plot_auc(y, probs):
    fpr, tpr, threshold = roc_curve(y, probs[:,1])
    auc = roc_auc_score(y_test, lr_predicted_labels)
    ax.plot(fpr, tpr, color = 'blue', label = 'AUC_Baseline Logistic Regression = %0.2f' % auc)
    ax.plot([0, 1], [0, 1],'r--')
    ax.legend(loc = 'lower right')
    ax.step(fpr, tpr, color='blue', alpha=0.2, where='post')
    ax.fill_between(fpr, tpr, step='post', alpha=0.2, color='blue')
    ax.set(xlabel='False Positive Rate',
           ylabel='True Positive Rate',
           xlim=[0, 1], ylim=[0, 1],
           title='ROC curve')       
    
plot_auc(y_test, lr_predicted_label_r)
ax.grid(True)

# Plot the ROC-AUC curve for Optimized Logistic Regression
ax = axList[1]
opt_lr = opt_LR.fit(X_train_clean, y_train_clean.values.ravel()) 
opt_lr_predicted_label_r = opt_LR.predict_proba(X_test_clean)

def plot_auc(y, probs):
    fpr, tpr, threshold = roc_curve(y, probs[:,1])
    auc = roc_auc_score(y_test_clean, opt_lr_predicted_labels)
    ax.plot(fpr, tpr, color = 'black', label = 'AUC_Optimized Logistic Regression = %0.2f' % auc)
    ax.plot([0, 1], [0, 1],'r--')
    ax.legend(loc = 'lower right')
    ax.step(fpr, tpr, color='black', alpha=0.2, where='post')
    ax.fill_between(fpr, tpr, step='post', alpha=0.2, color='black')
    ax.set(xlabel='False Positive Rate',
           ylabel='True Positive Rate',
           xlim=[0, 1], ylim=[0, 1],
           title='ROC curve') 
    
plot_auc(y_test_clean, opt_lr_predicted_label_r);
ax.grid(True)

# Plot the ROC-AUC curve for Naive Bayes
ax = axList[2]
nb = NB.fit(X_train_clean, y_train_clean.values.ravel()) 
nb_predicted_label_r = NB.predict_proba(X_test_clean)

def plot_auc(y, probs):
    fpr, tpr, threshold = roc_curve(y, probs[:,1])
    auc = roc_auc_score(y_test_clean, nb_predicted_labels)
    ax.plot(fpr, tpr, color = 'green', label = 'AUC_Naive Bayes = %0.2f' % auc)
    ax.plot([0, 1], [0, 1],'r--')
    ax.legend(loc = 'lower right')
    ax.step(fpr, tpr, color='green', alpha=0.2, where='post')
    ax.fill_between(fpr, tpr, step='post', alpha=0.2, color='green')
    ax.set(xlabel='False Positive Rate',
           ylabel='True Positive Rate',
           xlim=[0, 1], ylim=[0, 1],
           title='ROC curve') 
    
plot_auc(y_test_clean, nb_predicted_label_r);
ax.grid(True)
plt.tight_layout()


# ### 5. Calibration Curve

# In[36]:


# Plot calibration curves for a set of classifier probability estimates
lr_probas = LR.fit(X_train, y_train).predict_proba(X_test)

probas_list = [lr_probas]
clf_names = ['Baseline Logistic Regression']

skplt.metrics.plot_calibration_curve(y_test, probas_list, clf_names, cmap='winter', figsize=(10,6))
plt.legend(loc = 'best')
plt.show()


# In[37]:


# Plot calibration curves for a set of classifier probability estimates
opt_lr_probas = opt_LR.fit(X_train_clean, y_train_clean).predict_proba(X_test_clean)
nb_probas = NB.fit(X_train_clean, y_train_clean).predict_proba(X_test_clean)

probas_list = [opt_lr_probas, nb_probas]
clf_names = ['Optimized Logistic Regression','Naive Bayes']

skplt.metrics.plot_calibration_curve(y_test_clean, probas_list, clf_names, figsize=(10,6))
plt.legend(loc = 'best')
plt.show()


# In[ ]:




