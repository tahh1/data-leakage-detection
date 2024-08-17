#!/usr/bin/env python
# coding: utf-8

# # CHURN PREDICTION PROJECT
# 
# 
# ------
# 
# 
# A bank is investigating a very high rate of customer leaving the bank. Here is a 10.000 records dataset to investigate and predict which of the customers are more likely to leave the bank soon.
# 
# ### Dataset
# https://www.kaggle.com/mathchi/churn-for-bank-customers
# 
# About dataset
# 
# **RowNumber** : Corresponds to the record (row) number and has no effect on the output.
# 
# **CustomerId** :Contains random values and has no effect on customer leaving the bank.
# 
# **Surname** : The surname of a customer has no impact on their decision to leave the bank.
# 
# **CreditScore** : Can have an effect on customer churn, since a customer with a higher credit score is less likely to leave the bank.
# 
# **Geography** : A customer’s location can affect their decision to leave the bank.
# 
# **Gender** : It’s interesting to explore whether gender plays a role in a customer leaving the bank.
# 
# **Age** : This is certainly relevant, since older customers are less likely to leave their bank than younger ones.
# 
# **Tenure** : Refers to the number of years that the customer has been a client of the bank. Normally, older clients are more loyal and less likely to leave a bank.
# 
# **Balance** : Also a very good indicator of customer churn, as people with a higher balance in their accounts are less likely to leave the bank compared to close with lower balances.
# 
# **NumOfProducts** : Refers to the number of products that a customer has purchased through the bank.
# 
# **HasCrCard** : Denotes whether or not a customer has a credit card. This column is also relevant, since people with a credit card are less likely to leave the bank.
# 
# **IsActiveMember** : Active customers are less likely to leave the bank.
# 
# **EstimatedSalary** : As with balance, people with lower salaries are more likely to leave the bank compared to those with higher salaries.
# 
# **Exited** : Whether or not the customer left the bank.

# Importing the libraries

# In[452]:


# Importing libraries
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import seaborn as sns #visualization
import matplotlib.pyplot as plt #visualization
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import (plot_confusion_matrix, confusion_matrix, 
                             accuracy_score, mean_squared_error, r2_score, 
                             roc_auc_score, roc_curve, classification_report, 
                             precision_recall_curve, auc, f1_score, 
                             average_precision_score, precision_score, recall_score)
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import scale, StandardScaler, RobustScaler, MinMaxScaler


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

# to display all columns and rows:
pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);  # to display all columns and rows
pd.set_option('display.float_format', lambda x: '%.2f' % x) # The number of numbers that will be shown after the comma.


# Reading the dataset

# In[453]:


churn = pd.read_csv("../input/churn-for-bank-customers/churn.csv", index_col = 0)
churn.head() # first five row of the dataset


# ## Data overview

# In[454]:


# checking dataset

print(("Rows     : " ,churn.shape[0]))
print(("Columns  : " ,churn.shape[1]))
print(("\nFeatures : \n" ,churn.columns.tolist()))
print(("\nMissing values :  ", churn.isnull().sum().values.sum()))
print(("\nUnique values :  \n",churn.nunique()))


# In[455]:


churn.describe().T


# In[456]:


churn["Exited"].value_counts()


# In[457]:


#Separating churn and non churn customers
exited     = churn[churn["Exited"] == 1]
not_exited = churn[churn["Exited"] == 0]


# ### Dropping Irrelevant Feature
# CustomerId and Surname are irrelivant, so we drop those features.

# In[458]:


df = churn.drop(['CustomerId', 'Surname'], axis = 1)
df.head()


# ## Data Visualization¶
# 

# In[459]:


fig, axarr = plt.subplots(2, 3, figsize=(18, 6))
sns.countplot(x = 'Geography', hue = 'Exited',data = df, ax = axarr[0][0])
sns.countplot(x = 'Gender', hue = 'Exited',data = df, ax = axarr[0][1])
sns.countplot(x = 'HasCrCard', hue = 'Exited',data = df, ax = axarr[0][2])
sns.countplot(x = 'IsActiveMember', hue = 'Exited',data = df, ax = axarr[1][0])
sns.countplot(x = 'NumOfProducts', hue = 'Exited',data = df, ax = axarr[1][1])
sns.countplot(x = 'Tenure', hue = 'Exited',data = df, ax = axarr[1][2])


# Customer with 3 or 4 products are higher chances to Churn
# 
# 

# In[460]:


_, ax = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.3)
sns.swarmplot(x = "NumOfProducts", y = "Age", hue="Exited", data = df, ax= ax[0])
sns.swarmplot(x = "HasCrCard", y = "Age", data = df, hue="Exited", ax = ax[1])
sns.swarmplot(x = "IsActiveMember", y = "Age", hue="Exited", data = df, ax = ax[2])


# In[461]:


facet = sns.FacetGrid(df, hue = "Exited", aspect = 3)
facet.map(sns.kdeplot, "Age", shade = True)
facet.set(xlim = (0, df["Age"].max()))
facet.add_legend()

plt.show();


# In[462]:


_, ax =  plt.subplots(1, 2, figsize = (15, 7))
cmap = sns.cubehelix_palette(light = 1, as_cmap = True)
sns.scatterplot(x = "Age", y = "Balance", hue = "Exited", cmap = cmap, sizes = (10, 200), data = df, ax = ax[0])
sns.scatterplot(x = "CreditScore", y = "Balance", hue = "Exited", cmap = cmap, sizes = (10, 200), data = df, ax = ax[1]);


# - 40 to 65 years old customers are higher chances to churn
# - Customer with CreditScore less then 450 are higher chances to churn

# In[463]:


plt.figure(figsize = (10, 10))
sns.swarmplot(x = "HasCrCard", y = "Age", data = df, hue = "Exited")


# ### Checking Correlation
# 

# In[464]:


corr = df.corr()
corr.style.background_gradient(cmap = 'coolwarm')


# In[465]:


# NumOfProducts variable is converted to string values.
NumOfProd = []
for i in df['NumOfProducts']:
    if i == 1:
        NumOfProd.append('A')
    elif i == 2:
        NumOfProd.append('B')
    elif i == 3:
        NumOfProd.append('C')
    else:
        NumOfProd.append('D')
        
df['NumOfProducts'] = NumOfProd
df.head()


# In[466]:


dummies = pd.get_dummies(df[['Geography', 'Gender', 'NumOfProducts']], drop_first = True) 
X_ = df.drop(['Geography', 'Gender', 'NumOfProducts'], axis = 1)
df_1 = pd.concat([X_, dummies], axis = 1)
df_1.head()


# In[467]:


df_1.Balance = df_1.Balance + 1 # To get rid of the problem of dividing by 0
df_1['SalBal'] = df_1.EstimatedSalary / df_1.Balance #The ratio of variables EstimatedSalary and Balance is assigned as a new variable
df_1.head()


# ### Standardization

# In[468]:


df_1.head()


# In[469]:


# Standardization on four features
X_s = pd.DataFrame(df_1[['CreditScore', 'Balance', 'EstimatedSalary', 'SalBal']], 
                   columns = ['CreditScore', 'Balance', 'EstimatedSalary', 'SalBal'])

MinMax = MinMaxScaler(feature_range = (0, 1)).fit(X_s)
X_s = MinMax.transform(X_s)
X_st = pd.DataFrame(X_s, columns = ['CreditScore', 'Balance', 'EstimatedSalary', 'SalBal'])
X_st.index = X_st.index + 1
X_st.head()


# In[470]:


# We define the dataset with standardized variables as df_2.
df_2 = df_1.drop(['CreditScore', 'Balance', 'EstimatedSalary', 'SalBal'], axis = 1)
df_2 = pd.concat([df_2, X_st], axis = 1, ignore_index = False)
df_2.head()


# In[471]:


# credit scores are divided into 6 classes.
CreditScoreClass = []
for cs in churn.CreditScore:
    if 400 <= cs < 500:
        CreditScoreClass.append(1)
    elif 500 <= cs < 700:
        CreditScoreClass.append(2)
    elif  700 <= cs < 800:
        CreditScoreClass.append(3)
    elif  800 <=  cs < 850:
        CreditScoreClass.append(4)
    elif  850 <= cs: 
        CreditScoreClass.append(5)
    elif 400 > cs :
        CreditScoreClass.append(0)

df_2['CreditScoreClass'] = CreditScoreClass
df_2.drop('CreditScore', axis = 1, inplace = True)
df_2.head()


# ## Machine Learning:
# 
# 
# We will train out data on different machine learning models and use different techniques on each model and then compare our finding at the end to determine which model is working best for out data.
# 
# 
# ----- Model Performance and Comparison -----
# 
# To measure the performance of a model, we need several elements
# Confusion matrix : also known as the error matrix, allows visualization of the performance of an algorithm
# 
# - True Positive (TP) : Exited correctly identified as exited
# - True Negative (TN) : Nonexited correctly identified as nonexited
# - False Positive (FP) : Nonexited incorrectly identified as exited
# - False Negative (FN) : Exited incorrectly identified as nonexited
# 
# 
# Metrics
# 
# - Accuracy : (TP + TN) / (TP + TN + FP +FN)
# - Precision : TP / (TP + FP)
# - Recall : TP / (TP + FN)
# - F1 score : 2 x ((Precision x Recall) / (Precision + Recall))
# 

# In[472]:


y = df_2['Exited']
X = df_2.drop('Exited', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12345)
models = [
    LogisticRegression(),
    KNeighborsClassifier(),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    LGBMClassifier(),
    XGBClassifier()]

result = []
results = pd.DataFrame(columns = ["Models","Accuracy"])

for model in models:
    names = model.__class__.__name__
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    scores = cross_val_score(model, X_test, y_test, cv = 10, scoring = 'accuracy')
    result = pd.DataFrame([[names, acc * 100, 
                            np.mean(scores) * 100]], 
                          columns = ["Models", "Accuracy", "Avg_Accuracy"])
    results = results.append(result)
results


# ### Defining variables to store the outputs.

# In[473]:


avg_accuracies={}
accuracies={}
roc_auc={}
pr_auc={}


# ### Defining function to calculate the Cross-Validation score.
# 

# In[474]:


def cv_score(name, model, folds):
    scores = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    avg_result = []
    for sc in scores:
        scores = cross_val_score(model, X_test, y_test, cv = folds, scoring = sc)
        avg_result.append(np.average(scores))
    df_avg_score = pd.DataFrame(avg_result)
    df_avg_score = df_avg_score.rename(index={0: 'Accuracy',
                                             1:'Precision',
                                             2:'Recall',
                                             3:'F1 score',
                                             4:'Roc auc'}, columns = {0: 'Average'})
    avg_accuracies[name] = np.round(df_avg_score.loc['Accuracy'] * 100, 2)
    values = [np.round(df_avg_score.loc['Accuracy'] * 100, 2),
            np.round(df_avg_score.loc['Precision'] * 100, 2),
            np.round(df_avg_score.loc['Recall'] * 100, 2),
            np.round(df_avg_score.loc['F1 score'] * 100, 2),
            np.round(df_avg_score.loc['Roc auc'] * 100, 2)]
    plt.figure(figsize = (15, 8))
    sns.set_palette('mako')
    ax = sns.barplot(x = ['Accuracy', 'Precision', 'Recall', 'F1 score', 'Roc auc'], y = values)
    plt.yticks(np.arange(0, 100, 10))
    plt.ylabel('Percentage %', labelpad = 10)
    plt.xlabel('Scoring Parameters', labelpad = 10)
    plt.title('Cross Validation ' + str(folds) + '-Folds Average Scores', pad = 20)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()), xytext = (p.get_x() + 0.3, p.get_height() + 1.02))
    plt.show()


# ### Defining function to create Confusion Matrix.

# In[475]:


def conf_matrix(ytest, pred):
    plt.figure(figsize = (15, 8))
    global cm1
    cm1 = confusion_matrix(ytest, pred)
    ax = sns.heatmap(cm1, annot = True, cmap = 'Blues')
    plt.title('Confusion Matrix', pad = 30)


# ### Defining function to calculate the Metrics Scores.

# In[476]:


def metrics_score(cm):
    total = sum(sum(cm))
    accuracy = (cm[0, 0] + cm[1, 1]) / total
    precision = cm[1, 1] / (cm[0, 1] + cm[1, 1])
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    specificity = cm[0,0] / (cm[0, 1] + cm[0, 0])
    values = [np.round(accuracy * 100, 2),
            np.round(precision * 100, 2),
            np.round(sensitivity * 100, 2),
            np.round(f1 * 100, 2),
            np.round(specificity * 100, 2)]
    plt.figure(figsize = (15, 8))
    sns.set_palette('magma')
    ax = sns.barplot(x = ['Accuracy', 'Precision', 'Recall', 'F1 score', 'Specificity'], y = values)
    plt.yticks(np.arange(0, 100, 10))
    plt.ylabel('Percentage %', labelpad = 10)
    plt.xlabel('Scoring Parameter', labelpad = 10)
    plt.title('Metrics Scores', pad = 20)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()), xytext = (p.get_x() + 0.3, p.get_height() + 1.02))
    plt.show()


# ### Defining function to plot ROC Curve.

# In[477]:


def plot_roc_curve(fpr, tpr):
    plt.figure(figsize = (8, 6))
    plt.plot(fpr, tpr, color = 'Orange', label = 'ROC')
    plt.plot([0, 1], [0, 1], color = 'black', linestyle = '--')
    plt.ylabel('True Positive Rate', labelpad = 10)
    plt.xlabel('False Positive Rate', labelpad = 10)
    plt.title('Receiver Operating Characteristic (ROC) Curve', pad = 20)
    plt.legend()
    plt.show()


# ### Defining function to plot Precision-Recall Curve.

# In[478]:


def plot_precision_recall_curve(recall, precision):
    plt.figure(figsize = (8,6))
    plt.plot(recall, precision, color = 'orange', label = 'PRC')
    plt.ylabel('Precision', labelpad = 10)
    plt.xlabel('Recall', labelpad = 10)
    plt.title('Precision Recall Curve', pad = 20)
    plt.legend()
    plt.show()


# ## 1. Logistic Regression Classifier:

# In[479]:


y = df_2['Exited']
X = df_2.drop('Exited', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12345)
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
prediction1 = log_model.predict(X_test)
accuracy1 = log_model.score(X_test, y_test) 
print(('Model Accuracy:',accuracy1 * 100))


# Storing model accuracy to plot for comparison with other Machine Learning models.

# In[480]:


accuracies['Linear Regression'] = np.round(accuracy1 * 100, 2)


# 1. Plotting Confusion Matrix to describe the performance of Linear Regression Classifier on a set of test data.

# In[481]:


conf_matrix(y_test, prediction1)


# Plotting different metrics scores for the Linear Regression Classifier for evaluation.

# In[482]:


metrics_score(cm1)


# * Plotting the average of different metrics scores for further evaluation.

# In[483]:


cv_score('Linear Regression', log_model, 5)


# Plotting Receiver Operating Characteristic (ROC) Curve, to illustrate the diagnostic ability of Linear Regression Classifier as its discrimination threshold is varied and showing the Area under the ROC Curve (AUC) value which will tell us how much our model is capable of distinguishing between churn und nonchurn customers.

# In[484]:


probs = log_model.predict_proba(X_test)
probs = probs[:, 1]
auc1 = roc_auc_score(y_test, probs)
roc_auc['Linear Regression'] = np.round(auc1, 2)
print(('Area under the ROC Curve (AUC): %.2f' % auc1))
fpr1, tpr1, _ = roc_curve(y_test, probs)
plot_roc_curve(fpr1, tpr1)


# Plotting Precision-Recall Curve for different thresholds of precision and recall much like the ROC Curve and showing the Area under the Precision-Recall Curve (AUCPR), it gives the number summary of the information in the Precision-Recall Curve.

# In[485]:


precision1, recall1, _ = precision_recall_curve(y_test, probs)
auc_score1 = auc(recall1, precision1)
pr_auc['Linear Regression'] = np.round(auc_score1, 2)
print(('Area under the PR Curve (AUCPR): %.2f' % auc_score1))
plot_precision_recall_curve(recall1, precision1)


# ## 2. KNNeighbors Classifier:

# KNN is a non-parametric, lazy learning algorithm. Its purpose is to use a database in which the data points are separated into several classes to predict the classification of a new sample point.

# In[486]:


y = df_1['Exited']
X = df_2.drop('Exited', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12345)
KNN_model = KNeighborsClassifier()
KNN_model.fit(X_train, y_train)
prediction2 = KNN_model.predict(X_test)
accuracy2 = KNN_model.score(X_test, y_test) 
print(('Model Accuracy:',accuracy2 * 100))


# Storing model accuracy to plot for comparison with other Machine Learning models.

# In[487]:


accuracies['KNeighbors Classifier'] = np.round(accuracy2 * 100, 2)


# 1. Plotting Confusion Matrix to describe the performance of KNN Classifier on a set of test data.

# In[488]:


conf_matrix(y_test, prediction2)


# Plotting different metrics scores for the KNN Classifier for evaluation.

# In[489]:


metrics_score(cm1)


# * Plotting the average of different metrics scores for further evaluation.

# In[490]:


cv_score('KNeighbors Classifier', KNN_model, 5)


# Plotting Receiver Operating Characteristic (ROC) Curve, to illustrate the diagnostic ability of KNN Classifier as its discrimination threshold is varied and showing the Area under the ROC Curve (AUC) value which will tell us how much our model is capable of distinguishing between churn und nonchurn customers.

# In[491]:


probs = KNN_model.predict_proba(X_test)
probs = probs[:, 1]
auc2 = roc_auc_score(y_test, probs)
roc_auc['KNeighbors Classifier'] = np.round(auc2, 2)
print(('Area under the ROC Curve (AUC): %.2f' % auc2))
fpr2, tpr2, _ = roc_curve(y_test, probs)
plot_roc_curve(fpr2, tpr2)


# Plotting Precision-Recall Curve for different thresholds of precision and recall much like the ROC Curve and showing the Area under the Precision-Recall Curve (AUCPR), it gives the number summary of the information in the Precision-Recall Curve.

# In[492]:


precision2, recall2, _ = precision_recall_curve(y_test, probs)
auc_score2 = auc(recall2, precision2)
pr_auc['KNeighbors Classifier'] = np.round(auc_score2, 2)
print(('Area under the PR Curve (AUCPR): %.2f' % auc_score2))
plot_precision_recall_curve(recall2, precision2)


# ## 3. Support Vector Machine Classifier:
# 

# In[493]:


y = df_1['Exited']
X = df_2.drop('Exited', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12345)
SVC_model = SVC(probability = True)
SVC_model.fit(X_train, y_train)
prediction3 = SVC_model.predict(X_test)
accuracy3 = SVC_model.score(X_test, y_test) 
print(('Model Accuracy:',accuracy3 * 100))


# Storing model accuracy to plot for comparison with other Machine Learning models.

# In[494]:


accuracies['Support Vector Machine Classifier'] = np.round(accuracy3 * 100, 2)


# 1. Plotting Confusion Matrix to describe the performance of SVM Classifier on a set of test data.

# In[495]:


conf_matrix(y_test, prediction3)


# Plotting different metrics scores for the SVM Classifier for evaluation.

# In[496]:


metrics_score(cm1);


# * Plotting the average of different metrics scores for further evaluation.

# In[497]:


cv_score('Support Vector Machine Classifier', SVC_model, 5)


# Plotting Receiver Operating Characteristic (ROC) Curve, to illustrate the diagnostic ability of SVM Classifier as its discrimination threshold is varied and showing the Area under the ROC Curve (AUC) value which will tell us how much our model is capable of distinguishing between churn und nonchurn customers.

# In[498]:


probs = SVC_model.predict_proba(X_test)
probs = probs[:, 1]
auc3 = roc_auc_score(y_test, probs)
roc_auc['Support Vector Machine Classifier'] = np.round(auc3, 2)
print(('Area under the ROC Curve (AUC): %.2f' % auc3))
fpr3, tpr3, _ = roc_curve(y_test, probs)
plot_roc_curve(fpr3, tpr3)


# Plotting Precision-Recall Curve for different thresholds of precision and recall much like the ROC Curve and showing the Area under the Precision-Recall Curve (AUCPR), it gives the number summary of the information in the Precision-Recall Curve.

# In[499]:


precision3, recall3, _ = precision_recall_curve(y_test, probs)
auc_score3 = auc(recall3, precision3)
pr_auc['Support Vector Machine Classifier'] = np.round(auc_score3, 2)
print(('Area under the PR Curve (AUCPR): %.2f' % auc_score3))
plot_precision_recall_curve(recall3, precision3)


# ## 4. Classification and Regression Tree:
# 
# 
# Decision Trees are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

# In[500]:


y = df_1['Exited']
X = df_2.drop('Exited', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12345)
CART_model = DecisionTreeClassifier(max_depth = 10, min_samples_split = 50)
CART_model.fit(X_train, y_train)
prediction4 = CART_model.predict(X_test)
accuracy4 = CART_model.score(X_test, y_test) 
print(('Model Accuracy:',accuracy4 * 100))


# Storing model accuracy to plot for comparison with other Machine Learning models.

# In[501]:


accuracies['Classification and Regression Tree'] = np.round(accuracy4 * 100, 2)


# 1. Plotting Confusion Matrix to describe the performance of CART Classifier on a set of test data.

# In[502]:


conf_matrix(y_test, prediction4)


# Plotting different metrics scores for the CART Classifier for evaluation.

# In[503]:


metrics_score(cm1)


# * Plotting the average of different metrics scores for further evaluation.

# In[504]:


cv_score('Classification and Regression Tree', CART_model, 5)


# Plotting Receiver Operating Characteristic (ROC) Curve, to illustrate the diagnostic ability of CART Classifier as its discrimination threshold is varied and showing the Area under the ROC Curve (AUC) value which will tell us how much our model is capable of distinguishing between churn und nonchurn customers.

# In[505]:


probs = CART_model.predict_proba(X_test)
probs = probs[:, 1]
auc4 = roc_auc_score(y_test, probs)
roc_auc['Desicion Tree Classifier']=np.round(auc4, 2)
print(('Area under the ROC Curve (AUC): %.2f' % auc4))
fpr4, tpr4, _ = roc_curve(y_test, probs)
plot_roc_curve(fpr4, tpr4)


# Plotting Precision-Recall Curve for different thresholds of precision and recall much like the ROC Curve and showing the Area under the Precision-Recall Curve (AUCPR), it gives the number summary of the information in the Precision-Recall Curve.

# In[506]:


precision4, recall4, _ = precision_recall_curve(y_test, probs)
auc_score4 = auc(recall4, precision4)
pr_auc['Desicion Tree Classifier'] = np.round(auc_score4, 2)
print(('Area under the PR Curve (AUCPR): %.2f' % auc_score4))
plot_precision_recall_curve(recall4, precision4)


# ## 5. Random Forests:
# 
# 
# A Random Forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

# In[507]:


y = df_1['Exited']
X = df_2.drop('Exited', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12345)
rf_model = RandomForestClassifier(max_features = 3, min_samples_split = 10, n_estimators = 200)
rf_model.fit(X_train, y_train)
prediction5 = rf_model.predict(X_test)
accuracy5 = rf_model.score(X_test, y_test) 
print(('Model Accuracy:',accuracy5 * 100))


# In[508]:


#rf_params = {"n_estimators": [100, 200, 500, 1000], "max_features": [3, 5, 7, 8], "min_samples_split": [2, 5, 10, 20]}
#rf_cv_model = GridSearchCV(rf, rf_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)
#rf_cv_model.best_params_


# Storing model accuracy to plot for comparison with other Machine Learning models.

# In[509]:


accuracies['Random Forests'] = np.round(accuracy5 * 100, 2)


# 1. Plotting Confusion Matrix to describe the performance of Random Forest Classifier on a set of test data.

# In[510]:


conf_matrix(y_test, prediction5)


# Plotting different metrics scores for the Random Forest Classifier for evaluation.

# In[511]:


metrics_score(cm1)


# * Plotting the average of different metrics scores for further evaluation.

# In[512]:


cv_score('Random Forests', rf_model, 5)


# Plotting Receiver Operating Characteristic (ROC) Curve, to illustrate the diagnostic ability of Random Forest Classifier as its discrimination threshold is varied and showing the Area under the ROC Curve (AUC) value which will tell us how much our model is capable of distinguishing between churn und nonchurn customers.

# In[513]:


probs = rf_model.predict_proba(X_test)
probs = probs[:, 1]
auc5 = roc_auc_score(y_test, probs)
roc_auc['Random Forests Classifier']=np.round(auc5, 2)
print(('Area under the ROC Curve (AUC): %.2f' % auc5))
fpr5, tpr5, _ = roc_curve(y_test, probs)
plot_roc_curve(fpr5, tpr5)


# Plotting Precision-Recall Curve for different thresholds of precision and recall much like the ROC Curve and showing the Area under the Precision-Recall Curve (AUCPR), it gives the number summary of the information in the Precision-Recall Curve.

# In[514]:


precision5, recall5, _ = precision_recall_curve(y_test, probs)
auc_score5 = auc(recall5, precision5)
pr_auc['Random Forests'] = np.round(auc_score5,3)
print(('Area under the PR Curve (AUCPR): %.2f' % auc_score5))
plot_precision_recall_curve(recall5, precision5)


# ## Feature Importance:

# In[515]:


feature_imp = pd.Series(rf_model.feature_importances_,
                        index = X_train.columns).sort_values(ascending = False)

sns.barplot(x = feature_imp, y = feature_imp.index)
plt.xlabel('Feature Important Scores')
plt.ylabel('Features')
plt.title("Feature Important Range")
plt.show()


# ## 6. Gradient Boosting Machines

# In[516]:


y = df_1['Exited']
X = df_2.drop('Exited', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12345)
gbm_model = GradientBoostingClassifier(learning_rate = 0.01, max_depth = 5, n_estimators = 300)
gbm_model.fit(X_train, y_train)
prediction6 = gbm_model.predict(X_test)
accuracy6 = gbm_model.score(X_test, y_test) 
print(('Model Accuracy:',accuracy6 * 100))


# In[517]:


#gbm_params = {"learning_rate": [0.1, 0.01, 0.001, 0.05],"n_estimators": [100, 300, 500, 1000], "max_depth":[2, 3, 5, 8]}
#gbm_cv_model= GridSearchCV(gbm_model, gbm_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)
#gbm_cv_model.best_params_


# Storing model accuracy to plot for comparison with other Machine Learning models.

# In[518]:


accuracies['Gradient Boosting Machines'] = np.round(accuracy6 * 100, 2)


# 1. Plotting Confusion Matrix to describe the performance of GBM Classifier on a set of test data.

# In[519]:


conf_matrix(y_test, prediction6)


# Plotting different metrics scores for the GBM Classifier for evaluation.

# In[520]:


metrics_score(cm1)


# * Plotting the average of different metrics scores for further evaluation.

# In[521]:


cv_score('Gradient Boosting Machines', gbm_model, 5)


# Plotting Receiver Operating Characteristic (ROC) Curve, to illustrate the diagnostic ability of GBM Classifier as its discrimination threshold is varied and showing the Area under the ROC Curve (AUC) value which will tell us how much our model is capable of distinguishing between churn und nonchurn customers.

# In[522]:


probs = gbm_model.predict_proba(X_test)
probs = probs[:, 1]
auc6 = roc_auc_score(y_test, probs)
roc_auc['Gradient Boosting Machine Classifier'] = np.round(auc6, 2)
print(('Area under the ROC Curve (AUC): %.2f' % auc6))
fpr6, tpr6, _ = roc_curve(y_test, probs)
plot_roc_curve(fpr6, tpr6)


# Plotting Precision-Recall Curve for different thresholds of precision and recall much like the ROC Curve and showing the Area under the Precision-Recall Curve (AUCPR), it gives the number summary of the information in the Precision-Recall Curve.

# In[523]:


precision6, recall6, _ = precision_recall_curve(y_test, probs)
auc_score6 = auc(recall6, precision6)
pr_auc['Gradient Boosting Machine Classifier'] = np.round(auc_score6, 2)
print(('Area under the PR Curve (AUCPR): %.2f' % auc_score6))
plot_precision_recall_curve(recall6, precision6)


# ## Feature Importance:

# In[524]:


feature_imp = pd.Series(gbm_model.feature_importances_,
                        index = X_train.columns).sort_values(ascending = False)

sns.barplot(x = feature_imp, y = feature_imp.index)
plt.xlabel('Feature Important Scores')
plt.ylabel('Features')
plt.title("Feature Important Range")
plt.show()


# ## 7. XGBoost:

# In[525]:


y = df_1['Exited']
X = df_2.drop('Exited', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12345)
xgb_model = XGBClassifier(learning_rate = 0.01, max_depth = 5, n_estimators = 1000, subsample = 0.8)
xgb_model.fit(X_train, y_train)
prediction7 = xgb_model.predict(X_test)
accuracy7 = xgb_model.score(X_test, y_test) 
print(('Model Accuracy:',accuracy7 * 100))


# In[526]:


#xgb_params = {"n_estimators": [100, 500, 1000], "subsample":[0.5, 0.8 ,1], "max_depth":[3, 5, 7], "learning_rate":[0.1, 0.001, 0.01, 0.05]}
#xgb_cv_model = GridSearchCV(xgb_model, xgb_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)
#xgb_cv_model.best_params_


# Storing model accuracy to plot for comparison with other Machine Learning models.

# In[527]:


accuracies['XGBoost Classifier'] = np.round(accuracy7 * 100, 2)


# 1. Plotting Confusion Matrix to describe the performance of XGBM Classifier on a set of test data.

# In[528]:


conf_matrix(y_test, prediction7)


# Plotting different metrics scores for the XGBM Classifier for evaluation.

# In[529]:


metrics_score(cm1)


# * Plotting the average of different metrics scores for further evaluation.

# In[530]:


cv_score('XGBoost Classifier', xgb_model, 5)


# Plotting Receiver Operating Characteristic (ROC) Curve, to illustrate the diagnostic ability of XGBM Classifier as its discrimination threshold is varied and showing the Area under the ROC Curve (AUC) value which will tell us how much our model is capable of distinguishing between churn und nonchurn customers.

# In[531]:


probs = xgb_model.predict_proba(X_test)
probs = probs[:, 1]
auc7 = roc_auc_score(y_test, probs)
roc_auc['XGB Machine Classifier']=np.round(auc7, 2)
print(('Area under the ROC Curve (AUC): %.2f' % auc7))
fpr7, tpr7, _ = roc_curve(y_test, probs)
plot_roc_curve(fpr7, tpr7)


# Plotting Precision-Recall Curve for different thresholds of precision and recall much like the ROC Curve and showing the Area under the Precision-Recall Curve (AUCPR), it gives the number summary of the information in the Precision-Recall Curve.

# In[532]:


precision7, recall7, _ = precision_recall_curve(y_test, probs)
auc_score7 = auc(recall7, precision7)
pr_auc['XGB Machine Classifier'] = np.round(auc_score7, 2)
print(('Area under the PR Curve (AUCPR): %.2f' % auc_score7))
plot_precision_recall_curve(recall7, precision7)


# ## Feature Importance:

# In[533]:


feature_imp = pd.Series(gbm_model.feature_importances_,
                        index = X_train.columns).sort_values(ascending = False)

sns.barplot(x = feature_imp, y = feature_imp.index)
plt.xlabel('Feature Important Scores')
plt.ylabel('Features')
plt.title("Feature Important Range")
plt.show()


# ## 8. Light GBM

# In[534]:


y = df_1['Exited']
X = df_2.drop('Exited', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12345)
lgbm_model = LGBMClassifier(learning_rate = 0.1, max_depth = 2, n_estimators = 500)
lgbm_model.fit(X_train, y_train)
prediction8 = lgbm_model.predict(X_test)
accuracy8 = lgbm_model.score(X_test, y_test) 
print(('Model Accuracy:',accuracy8 * 100))


# In[535]:


#lgbm_params = {"learning_rate": [0.001, 0.01, 0.1], "n_estimators": [200, 500, 100], "max_depth":[1,2,5,8]}
#lgbm_cv_model = GridSearchCV(lgbm_model,lgbm_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)
#lgbm_cv_model.best_params_


# Storing model accuracy to plot for comparison with other Machine Learning models.

# In[536]:


accuracies['LightGBM Classifier'] = np.round(accuracy8 * 100, 2)


# 1. Plotting Confusion Matrix to describe the performance of LGBM Classifier on a set of test data.

# In[537]:


conf_matrix(y_test, prediction8)


# Plotting different metrics scores for the LGBM Classifier for evaluation.

# In[538]:


metrics_score(cm1)


# * Plotting the average of different metrics scores for further evaluation.

# In[539]:


cv_score('LightGBM Classifier', lgbm_model, 5)


# Plotting Receiver Operating Characteristic (ROC) Curve, to illustrate the diagnostic ability of LGBM Classifier as its discrimination threshold is varied and showing the Area under the ROC Curve (AUC) value which will tell us how much our model is capable of distinguishing between churn und nonchurn customers.

# In[540]:


probs = lgbm_model.predict_proba(X_test)
probs = probs[:, 1]
auc8 = roc_auc_score(y_test, probs)
roc_auc['LightGBM Classifier'] = np.round(auc8, 2)
print(('Area under the ROC Curve (AUC): %.2f' % auc8))
fpr8, tpr8, _ = roc_curve(y_test, probs)
plot_roc_curve(fpr8, tpr8)


# Plotting Precision-Recall Curve for different thresholds of precision and recall much like the ROC Curve and showing the Area under the Precision-Recall Curve (AUCPR), it gives the number summary of the information in the Precision-Recall Curve.

# In[541]:


precision8, recall8, _ = precision_recall_curve(y_test, probs)
auc_score8 = auc(recall8, precision8)
pr_auc['LightGBM Classifier'] = np.round(auc_score8, 2)
print(('Area under the PR Curve (AUCPR): %.2f' % auc_score8))
plot_precision_recall_curve(recall8, precision8)


# ## Feature Importance:

# In[542]:


feature_imp = pd.Series(gbm_model.feature_importances_,
                        index = X_train.columns).sort_values(ascending = False)

sns.barplot(x = feature_imp, y = feature_imp.index)
plt.xlabel('Feature Important Scores')
plt.ylabel('Features')
plt.title("Feature Important Range")
plt.show()


# ## Performance Comparison
# 
# Plotting the accuracy metric score of the machine learning models for comparison.

# In[543]:


models_tuned = [
    log_model,
    KNN_model,
    SVC_model,
    CART_model,
    rf_model,
    gbm_model,
    lgbm_model,
    xgb_model]

result = []
results = pd.DataFrame(columns = ["Models","Accuracy"])

for model in models_tuned:
    names = model.__class__.__name__
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    scores = cross_val_score(model, X_test, y_test, cv = 10, scoring = 'accuracy')
    result = pd.DataFrame([[names, acc * 100, 
                            np.mean(scores) * 100]], 
                          columns = ["Models", "Accuracy", "Avg_Accuracy"])
    results = results.append(result)
results


# In[544]:


plt.figure(figsize = (15, 8))
sns.set_palette('cividis')
ax = sns.barplot(x = list(accuracies.keys()), y = list(accuracies.values()))
plt.yticks(np.arange(0, 100, 10))
plt.ylabel('Percentage %', labelpad = 10)
plt.xlabel('Algorithms', labelpad = 10)
plt.title('Accuracy Scores Comparison', pad = 20)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()), xytext = (p.get_x() + 0.3, p.get_height() + 1.02))
plt.show()


# 
# Plotting the average accuracy metric score of the machine learning models for comparison.

# In[545]:


plt.figure(figsize = (15, 8))
sns.set_palette('viridis')
ax=sns.barplot(x = list(avg_accuracies.keys()), y = list(avg_accuracies.values()))
plt.yticks(np.arange(0, 100, 10))
plt.ylabel('Percentage %', labelpad = 10)
plt.xlabel('Algorithms', labelpad = 10)
plt.title('Average Accuracy Scores Comparison', pad = 20)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()),xytext=(p.get_x() + 0.3, p.get_height() + 1.02))
plt.show()


# 
# Plotting the ROC Curve of the machine learning models for comparison.

# In[546]:


plt.figure(figsize = (8, 6))
sns.set_palette('Set1')
plt.plot(fpr1, tpr1, label = 'Linear Regression')
plt.plot(fpr2, tpr2, label = 'KNeiihbors Classifier')
plt.plot(fpr3, tpr3, label = 'SVM')
plt.plot(fpr4, tpr4, label = 'Decision Tree')
plt.plot(fpr5, tpr5, label = 'Random Forests')
plt.plot(fpr6, tpr6, label = 'Gradient Boosting MachineC')
plt.plot(fpr7, tpr7, label = 'XGBoost')
plt.plot(fpr8, tpr8, label = 'LightGBM')
plt.plot([0, 1], [0, 1], linestyle = '--')
plt.ylabel('True Positive Rate', labelpad = 10)
plt.xlabel('False Positive Rate', labelpad = 10)
plt.title('Receiver Operating Characteristic (ROC) Curves', pad = 20)
plt.legend()
plt.show()


# Plotting the AUC values of ROC Curve of the machine learning models for comparison.

# In[547]:


plt.figure(figsize = (15, 8))
sns.set_palette('magma')
ax = sns.barplot(x = list(roc_auc.keys()), y = list(roc_auc.values()))
#plt.yticks(np.arange(0,100,10))
plt.ylabel('Score', labelpad = 10)
plt.xlabel('Algorithms', labelpad = 10)
plt.title('Area under the ROC Curves (AUC)', pad = 20)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()), xytext = (p.get_x() + 0.3, p.get_height() + 0.01))
plt.show()


# 
# Plotting the PR Curve of the machine learning models for comparison.

# In[548]:


plt.figure(figsize = (8, 6))
sns.set_palette('Set1')
plt.plot(recall1, precision1, label = 'Linear Regression PRC')
plt.plot(recall2, precision2, label = 'KNN PRC')
plt.plot(recall3, precision3, label = 'SVM PRC')
plt.plot(recall4, precision4, label = 'CART PRC')
plt.plot(recall5, precision5, label = 'Random Forests PRC')
plt.plot(recall6, precision6, label = 'GBM PRC')
plt.plot(recall7, precision7, label = 'XGB PRC')
plt.plot(recall8, precision8, label = 'LGBM PRC')
plt.ylabel('Precision', labelpad = 10)
plt.xlabel('Recall', labelpad = 10)
plt.title('Precision Recall Curves', pad = 20)
plt.legend()
plt.show()


# Plotting the AUC values of PR Curve of the machine learning models for comparison.

# In[549]:


plt.figure(figsize = (15, 8))
sns.set_palette('mako')
ax = sns.barplot(x = list(pr_auc.keys()), y = list(pr_auc.values()))
plt.ylabel('Score', labelpad = 10)
plt.xlabel('Algorithms', labelpad = 10)
plt.title('Area under the PR Curves (AUCPR)', pad = 20)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()), xytext = (p.get_x() + 0.3, p.get_height() + 0.01))
plt.show()


# # - - - -  REPORTING  - - - -
# 
# 
# 
# 

# ### Our aim in this project was to develop a churn prediction model using machine learning algorithms.
# 
# ### There were 10000 rows in the data set and there were no missing values.
# 
# ### The dataset consisted of 13 variables.
# 
# ### The following conclusions came from the analysis on the features:
# 
# * Most customers who using products 3 and 4 stopped working with the bank. In fact, all customers using product number 4 were gone.
# * Customers between the ages of 40 and 65 were more likely to quit the bank.
# * Those who had a credit score below 450 had high abandonment rates.
# * Predictions were made with a total of 8 classification models. The highest head was taken with Random Forests.
# * Accuracy and cross validation scores were calculated for each model and results were displayed.
# 
