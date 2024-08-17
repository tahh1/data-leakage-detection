#!/usr/bin/env python
# coding: utf-8

# ## Heart Disease Prediction
# 
# 

# <h2>Objective</h2>
# 
# <p>Build a classification model that predicts heart disease in a subject. (Note the target column to predict is 'TenYearCHD' where CHD = Coronary heart disease) </p>

# <h2>Attributes:</h2>
#     <ol>
#     <li>male: male(0) or female(1);(Nominal)</li>
#     <li>age: age of the patient;(Continuous - Although the recorded ages have been truncated to whole numbers, the concept of age is continuous)</li>
#     <li>education</li>
#     <li>currentSmoker: whether or not the patient is a current smoker (Nominal)</li>
#     <li>cigsPerDay: the number of cigarettes that the person smoked on average in one day.(can be considered continuous as one can have any number of cigarretts, even half a cigarette.)</li>
#     <li>BPMeds: whether or not the patient was on blood pressure medication (Nominal)</li>
#     <li>prevalentStroke: whether or not the patient had previously had a stroke (Nominal)</li>
#     <li>prevalentHyp: whether or not the patient was hypertensive (Nominal)</li>
#     <li>diabetes: whether or not the patient had diabetes (Nominal)</li>
#     <li>totChol: total cholesterol level (Continuous)</li>
#     <li>sysBP: systolic blood pressure (Continuous)</li>
#     <li>diaBP: diastolic blood pressure (Continuous)</li>
#     <li>BMI: Body Mass Index (Continuous)</li>
#     <li>heartRate: heart rate (Continuous - In medical research, variables such as heart rate though in fact discrete, yet are considered continuous because of large number of possible values.)</li>
#     <li>glucose: glucose level (Continuous)</li>
#     <li>10 year risk of coronary heart disease CHD (binary: “1”, means “Yes”, “0” means “No”) - Target Variable</li>
#     </ol>

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib.mlab as mlab 
get_ipython().run_line_magic('matplotlib', 'inline')


import scipy.optimize as opt
import warnings
warnings.simplefilter("ignore")
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB




# <h2>1. Read the file and display columns</h2>

# In[2]:


df=pd.read_csv(r"../input/heart-disease-prediction-using-logistic-regression/framingham.csv")
df


# In[3]:


df.head(10)


# In[4]:


df.tail()


# In[5]:


df.columns


# In[6]:


df.shape


# In[7]:


df.columns.nunique()


# In[8]:


df['male'].value_counts()


# In[9]:


df['education'].value_counts()


# In[10]:


df['currentSmoker'].value_counts()


# In[11]:


df['BPMeds'].value_counts()


# In[12]:


df['prevalentStroke'].value_counts()


# In[13]:


df['diabetes'].value_counts()


# In[14]:


df['TenYearCHD'].value_counts()


# In[15]:


df.info()


# ## 2. Handle missing values, Outliers and Duplicate Data

# In[16]:


df.isnull().sum()


# In[17]:


# percentage of missing data per category
total = df.isnull().sum().sort_values(ascending=False)
percent_total = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)*100
missing = pd.concat([total, percent_total], axis=1, keys=["Total", "Percentage"])
missing_data = missing[missing['Total']>0]
missing_data


# In[18]:


plt.figure(figsize=(9,6))
sns.barplot(x=missing_data.index, y=missing_data['Percentage'], data = missing_data)
plt.title('Percentage of missing data by feature')
plt.xlabel('Features', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.show()


# At 9.15%, the blood glucose entry has the highest percentage of missing data. The otherfeatures have very few missing entries.
# 

# In[19]:


# we can drop education as it doesnt effect heart disease
df = df.drop(['education'], axis=1)


# In[20]:


print((df.isnull().sum().sum()))
df=df.dropna()
print((df.isnull().sum().sum()))
df.shape


# In[21]:


df.isna().sum()


# In[22]:


#Outliers
cols =['age','BMI','heartRate','sysBP','totChol','diaBP']
plt.title("OUTLIERS VISUALIZATION")
for i in cols:
    df[i]
    sns.distplot(df[i],color='grey')
    plt.show()


# ## 3.	Calculate  statistics and EDA of data.

# In[23]:


df.describe().T


# In[24]:


plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),linewidths=0.1,annot=True)
# linewidths is white space between boxes and annot gives value
plt.show()


# <h3>Observations:</h3>
# <ol>
#     <li>sysBP and diaBP,currentSmoker and cigsPerDay  are highly correlated with values around 0.8</li>
#     <li>sysBP and diaBP and prevalentHyp, diabetes and glucose are correlated to some extent with values arouund 0.62 </li>
# </ol>

# In[25]:


sns.boxplot(y='age',x='TenYearCHD',data=df)


# In[26]:


sns.violinplot(y='age',x='TenYearCHD',data=df)


# <h3>Observations:</h3>
# <ol>
#     <li>Patients who got CHD are in the age group:50- 65</li>
#     <li>Patients around the age group:35- 45 does not suffer from CHD mostly</li>
# </ol>

# In[27]:


sns.violinplot(y='cigsPerDay',x='TenYearCHD',data=df)


# <h3>Observations:</h3>
# <ol>
#     <li>It's weird that patients who didn't smoke suffered from CHD</li>
#     <li>More the cigarretes they smoke higher chance of getting CHD </li>
# </ol>

# In[28]:


sns.violinplot(y='sysBP',x='TenYearCHD',data=df)


# <h3>Observations:</h3>
# <ol>
#     <li>Patients who have higher systole BP have higher chances of getting CHD</li>
#     <li>Patients whose systole BP is around 120 are mostly safe</li>
# </ol>

# In[29]:


sns.boxplot(y='diaBP',x='TenYearCHD',data=df)


# <h3>Observations:</h3>
# <ol>
#     <li>Patients who have higher diastole BP have higher chances of getting CHD</li>
#     <li>Patients whose diastole BP is around 75-80 are mostly safe</li>
# </ol>

# In[30]:


sns.violinplot(y='BMI',x='TenYearCHD',data=df)


# <h3>Observations:</h3>
# <ol>
#     <li>It seems BMI doesn't affect chance of getting CHD</li>
# </ol>

# In[31]:


sns.boxplot(y='heartRate',x='TenYearCHD',data=df)


# <h3>Observations:</h3>
# <ol>
#     <li>If your heart rate is in range of 70-80 is safe, but if their heart rate goes above or below can cause CHD</li>
# </ol>

# In[32]:


sns.countplot(x=df['male'], hue=df['TenYearCHD'])


# <h3>Observations:</h3>
# <ol>
#     <li>Males are at higher risk of getting CHD</li>
# </ol>

# In[33]:


sns.countplot(x='currentSmoker',data=df,hue='TenYearCHD')


# In[34]:


sns.countplot(x='prevalentHyp',data=df,hue='TenYearCHD')


# <h3>Observations:</h3>
# <ol>
#     <li>Higher percentage of people having hypertension suffer from CHD</li>
# </ol>

# In[35]:


sns.countplot(x='BPMeds',data=df,hue='TenYearCHD')


# <h3>Observations:</h3>
# <ol>
#     <li>It seems as if 50-60% of patients taking BP meds get CHD</li>
# </ol>

# In[36]:


sns.countplot(x='diabetes',data=df,hue='TenYearCHD')


# <h3>Observations:</h3>
# <ol>
#     <li>It seems as if 60-80% of diabetic patients  get CHD</li>
# </ol>

# In[37]:


sns.countplot(x='prevalentStroke',data=df,hue='TenYearCHD')


# <h3>Observations:</h3>
# <ol>
#     <li>Same here as well, it seems as if 90% of stroke patients  get CHD</li>
# </ol>

# In[38]:


plt.figure(figsize=(10,10))
sns.boxplot(x='TenYearCHD', y='age', data=df, hue='currentSmoker')


# In[39]:


plt.figure(figsize=(10,10))
sns.violinplot(x='TenYearCHD', y='age', data=df, hue='currentSmoker', split=True)


# <h3>Observations:</h3>
# <ol>
#     <li>We see that most of smokers having no risk of CHD are in age around 40 years</li>
#     <li>most of non-smokers having risk are in age around 65-70 years</li>
#     <li>most smokers having risk are in age around 50 years</li>
# </ol>

# In[40]:


sns.boxplot(y='sysBP',x='prevalentHyp',data=df)


# In[41]:


plt.figure(figsize=(20,10))
sns.boxplot(y='diaBP',hue='prevalentHyp',data=df,x='TenYearCHD')
#split=True combines two plots


# <h3>Observations:</h3>
# <ol>
#     <li>Higher sysBP and diaBP, higher the risk of Hypertension, which means higher risk of CHD</li>
# </ol>

# In[42]:


plt.figure(figsize=(20,10))
sns.violinplot(y='glucose',hue='diabetes',data=df,x='TenYearCHD',split=True)


# <h3>Observations:</h3>
# <ol>
#     <li>
# In diabetic patients those having higher level of glucose ranging from 200-400, have higher risk of getting CHD".
# 
# </li>
# </ol>

# In[43]:


# plot histogram to see the distribution of the data
fig = plt.figure(figsize = (15,20))
ax = fig.gca()
df.hist(ax = ax)
plt.show()


# ### Observations:

# The data on the prevalent stroke, diabetes, and blood pressure meds are poorly balanced.
# The no. of cases of CHD is  in patients  suffering from prevalant Stroke/ diabetes/taking BP meds is very low compared to those not suffering from it.There is a huge gap between the two extremes of suffering and not suffering"
# 
# 

# # Feature Selection 

# Feature selection is a technique where we choose those features in our data that contribute most to the target variable. In other words we choose the best predictors for the target variable.
# 
# The classes in the sklearn.feature_selection module can be used for feature selection/dimensionality reduction on sample sets, either to improve estimators’ accuracy scores or to boost their performance on very high-dimensional datasets.
# 
# 

# In[44]:


# Identify the features with the most importance for the outcome variable Heart Disease

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


X = df.iloc[:,0:14]  
y = df.iloc[:,-1]    

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  
print((featureScores.nlargest(11,'Score')))  


# In[45]:


featureScores = featureScores.sort_values(by='Score', ascending=False)
featureScores


# In[46]:


plt.figure(figsize=(20,5))
sns.barplot(x='Specs', y='Score', data=featureScores, palette = "plasma")
plt.box(False)
plt.title('Feature importance', fontsize=16)
plt.xlabel('\n Features', fontsize=14)
plt.ylabel('Importance \n', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# ### Observations:
#     

# We can say the sysBP , Glucose ,Age are the first three important features in the data .

# In[47]:


# selecting the 10 most impactful features for the target variable
features_list = featureScores["Specs"].tolist()[:10]
features_list


# These features which have strongest relationship with the output variable are:
# 1. Systolic Blood Pressure
# 2. Glucose
# 3. Age
# 4. Cholesterin
# 5. Cigarettes per Day
# 6. Diastolic Blood Pressure
# 7. Hypertensive
# 8. Diabetes
# 9. Blood Pressure Medication
# 10. Gender

# ### New dataframe with selected features

# In[48]:


df = df[['sysBP', 'glucose','age','totChol','cigsPerDay','diaBP','prevalentHyp','diabetes','BPMeds','male','TenYearCHD']]
df


# In[49]:


# Checking for outliers again
df.describe()
sns.pairplot(df)


# In[50]:


sns.boxplot(df.totChol)
outliers = df[(df['totChol'] > 500)] 
outliers


# In[51]:


#Dropping 2 outliers in cholesterin
df = df.drop(df[df.totChol > 599].index)
sns.boxplot(df.totChol)


# ### Observations:

# We have observed outliers in totChol and by specifying the range we have  dropped the 2 outliers in totChol.

# In[52]:


df_clean = df


# # Feature Scaling

# Feature scaling is a method used to normalize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.

# In[53]:


scaler = MinMaxScaler(feature_range=(0,1)) 
scaled_df= pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)


# In[54]:


scaled_df.describe()
df.describe()


# # Taining and Testing the Data

# In[55]:


y = scaled_df['TenYearCHD']
X = scaled_df.drop(['TenYearCHD'], axis = 1)

# divide train test: 60 % - 40 %
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=29)


# In[56]:


print((len(X_train)))
print((len(X_test)))


# ### Resampling imbalanced Dataset 

# In[57]:


target_count = scaled_df.TenYearCHD.value_counts()
print(('Class 0:', target_count[0]))
print(('Class 1:', target_count[1]))
print(('Proportion:', round(target_count[0] / target_count[1], 2), ': 1'))

sns.countplot(scaled_df.TenYearCHD, palette="OrRd")
plt.box(False)
plt.xlabel('Heart Disease No/Yes',fontsize=11)
plt.ylabel('Patient Count',fontsize=11)
plt.title('Count Outcome Heart Disease\n')
plt.savefig('Balance Heart Disease.png')
plt.show()


# ### Undersampling methods

# In[58]:


# Shuffle df
shuffled_df = scaled_df.sample(frac=1,random_state=4)

# Put all the fraud class in a separate dataset.
CHD_df = shuffled_df.loc[shuffled_df['TenYearCHD'] == 1]

#Randomly select 492 observations from the non-fraud (majority class)
non_CHD_df = shuffled_df.loc[shuffled_df['TenYearCHD'] == 0].sample(n=611,random_state=42)

# Concatenate both dataframes again
normalized_df = pd.concat([CHD_df, non_CHD_df])

# check new class counts
normalized_df.TenYearCHD.value_counts()

# plot new count
sns.countplot(normalized_df.TenYearCHD, palette="OrRd")
plt.box(False)
plt.xlabel('Heart Disease No/Yes',fontsize=11)
plt.ylabel('Patient Count',fontsize=11)
plt.title('Count Outcome Heart Disease after Resampling\n')
#plt.savefig('Balance Heart Disease.png')
plt.show()


# # Models

# ### The algorithms that we  will be used are:  
# 1. Logistic Regression
# 2. k-Nearest Neighbours
# 3. Decision Trees
# 4. Support Vector Machine
# 5. Random Forest Classification
# 6. Naive Bayes

# ## 1. Logistic Regression

# Logistic Regression is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.).
# In other words, the logistic regression model predicts P(Y=1) as a function of X.

# In[59]:


#initialize model
logreg = LogisticRegression()

# fit model
logreg.fit(X_train, y_train)

# prediction = knn.predict(x_test)
normalized_df_logreg_pred = logreg.predict(X_test)

# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total
acc = accuracy_score(y_test, normalized_df_logreg_pred)
print(f"The accuracy score for LogisticRegression is: {round(acc,3)*100}%")

# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
f1 = f1_score(y_test, normalized_df_logreg_pred)
print(f"The f1 score for LogisticRegression is: {round(f1,3)*100}%")

# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes
precision = precision_score(y_test, normalized_df_logreg_pred)
print(f"The precision score for LogisticRegression is: {round(precision,3)*100}%")

# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes
recall = recall_score(y_test, normalized_df_logreg_pred)
print(f"The recall score for LogisticRegression is: {round(recall,3)*100}%")


# ### Observations:
# ###### The accuracy score for LogisticRegression is: 84.89%
# ###### The f1 score for LogisticRegression is: 6.60%

# ## 2. KNN (k-nearest neighbors)

# The k-nearest neighbors (KNN) algorithm is a simple, easy-to-implement supervised machine learning algorithm that can be used to solve both classification and regression problems.

# In[60]:


knn = KNeighborsClassifier(n_neighbors = 2)

#fit model
knn.fit(X_train, y_train)

# prediction = knn.predict(x_test)
normalized_df_knn_pred = knn.predict(X_test)


# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total
acc = accuracy_score(y_test, normalized_df_knn_pred)
print(f"The accuracy score for KNN is: {round(acc,3)*100}%")

f1 = f1_score(y_test, normalized_df_knn_pred)
print(f"The f1 score for KNN is: {round(f1,3)*100}%")

# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes
precision = precision_score(y_test, normalized_df_knn_pred)
print(f"The precision score for KNN is: {round(precision,3)*100}%")

# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes
recall = recall_score(y_test, normalized_df_knn_pred)
print(f"The recall score for KNN is: {round(recall,3)*100}%")


# ### Observations :
# 
# #####  KNearestNeighors performs best at n = 10  with a accuracy of 84.1%

# ##### F1 score : 12.5%

# ## 3. Decision Trees

# A decision tree is a flowchart-like structure in which each internal node represents a test on a feature (e.g. whether a coin flip comes up heads or tails) , each leaf node represents a class label (decision taken after computing all features) and branches represent conjunctions of features that lead to those class labels. 

# In[61]:


#initialize model
dtc_up = DecisionTreeClassifier()

# fit model
dtc_up.fit(X_train, y_train)

normalized_df_dtc_up_pred = dtc_up.predict(X_test)

# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total
acc = accuracy_score(y_test, normalized_df_dtc_up_pred)
print(f"The accuracy score for DTC is: {round(acc,3)*100}%")

# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
f1 = f1_score(y_test, normalized_df_dtc_up_pred)
print(f"The f1 score for DTC is: {round(f1,3)*100}%")

# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes
precision = precision_score(y_test, normalized_df_dtc_up_pred)
print(f"The precision score for DTC is: {round(precision,3)*100}%")

# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes
recall = recall_score(y_test, normalized_df_dtc_up_pred)
print(f"The recall score for DTC is: {round(recall,3)*100}%")


# ### Observations:
# ###### The accuracy score for DTC is: 74.1%
# ###### The f1 score for DTC is: 22.7%

# ## 4. Support vector Machine

# A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane. In other words, given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples. In two dimentional space this hyperplane is a line dividing a plane in two parts where in each class lay in either side.

# In[62]:


#initialize model
svm = SVC()

#fit model
svm.fit(X_train, y_train)

normalized_df_svm_pred = svm.predict(X_test)

print('Observations:')
# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total
acc = accuracy_score(y_test, normalized_df_svm_pred)
print(f"The accuracy score for SVM is: {round(acc,3)*100}%")

# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
f1 = f1_score(y_test, normalized_df_svm_pred)
print(f"The f1 score for SVM is: {round(f1,3)*100}%")

# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes
precision = precision_score(y_test, normalized_df_svm_pred)
print(f"The precision score for SVM is: {round(precision,3)*100}%")

# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes
recall = recall_score(y_test, normalized_df_svm_pred)
print(f"The recall score for SVM is: {round(recall,3)*100}%")


# ### Observations:
# ###### The accuracy score for SVM is: 84.7%
# ###### The f1 score for SVM is: 1.70%

# ## 5. Random Forest Classification

# The Random Forest Classifier is a set of decision trees from randomly selected subset of training set. It aggregates the votes from different decision trees to decide the final class of the test object.

# In[63]:


rfc =  RandomForestClassifier()

#fit model
rfc.fit(X_train, y_train)

normalized_df_rfc_pred = rfc.predict(X_test)

print('Observations:')
# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total
acc = accuracy_score(y_test, normalized_df_rfc_pred)
print(f"The accuracy score for RFC is: {round(acc,3)*100}%")

# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
f1 = f1_score(y_test, normalized_df_rfc_pred)
print(f"The f1 score for RFC is: {round(f1,3)*100}%")

# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes
precision = precision_score(y_test, normalized_df_rfc_pred)
print(f"The precision score for RFC is: {round(precision,3)*100}%")

# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes
recall = recall_score(y_test, normalized_df_rfc_pred)
print(f"The recall score for RFC is: {round(recall,3)*100}%")


# ### Observations:
# ###### Accuracy Score : 83.89%
# ###### F1 score : 13.0%

# ## 6. Naive Bayes Algorithm

# Naive Bayes classifiers are a collection of classification algorithms based on Bayes’ Theorem. It is not a single algorithm but a family of algorithms where all of them share a common principle, i.e. every pair of features being classified is independent of each other.

# In[64]:


nb =  GaussianNB()

#fit model
nb.fit(X_train, y_train)

normalized_df_nb_pred = nb.predict(X_test)

print('Observations:')
# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total
acc = accuracy_score(y_test, normalized_df_nb_pred)
print(f"The accuracy score for Naive Bayes is: {round(acc,3)*100}%")

# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
f1 = f1_score(y_test, normalized_df_nb_pred)
print(f"The f1 score for Naive Bayes is: {round(f1,3)*100}%")

# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes
precision = precision_score(y_test, normalized_df_nb_pred)
print(f"The precision score for Naive Bayes is: {round(precision,3)*100}%")

# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes
recall = recall_score(y_test, normalized_df_nb_pred)
print(f"The recall score for Naive Bayes is: {round(recall,3)*100}%")




# ### Observations:
# ###### Accuracy : 81.8%
# ###### f1 score :  26.0%

# # F1 SCORES

# In[65]:


data = {'Model':['Logistic Regression','KNN','Decision Tree','SVM','Random Forest','Naive Bayes'],
        'F1 Score':[6.60,12.5,22.7,1.70,13.0,26.0],'Accuracies':[84.89,84.1,74.1,84.7,83.89,81.8],'Recall':[3.40,7.30,24.6,0.89,7.80,20.70],'Precision':[72.70,41.50,21.00,100.00,40.00,35.00]}

# Create DataFrame
df = pd.DataFrame(data)
 
# Print the output.
print(df)


# # Comparing the Models 

# In[66]:


Accuracies=[84.89,84.1,73.2,84.7,83.89,81.8]
Accuracies


# In[67]:


plt.figure(figsize=(9,6))
sns.barplot(x='Model', y='Accuracies', data = df)
plt.title('Comparison of accuracy of models')
plt.xlabel('model algorithms', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.show()


# By the above visualization we see that all the six models are being compared to eachother with respect to their accuracies .
# Logistic Regression has the highest accuracy in all the models as per the observation in the above barplot.
# 
# 1. Logistic Regression : 84.89%
# 2. KNN                 : 84.10%
# 3. Decision Tree       : 73.20%
# 4. SVM                 : 84.70%
# 5. Random Forest        :83.89%
# 6. Naive Bayes         : 81.80%

# ### Observations:

# Logistic regression has the highest accuracy.

# In[68]:


acc_test = logreg.score(X_test, y_test)
print(("The accuracy score of the test data is: ",acc_test*100,"%"))
acc_train = logreg.score(X_train, y_train)
print(("The accuracy score of the training data is: ",round(acc_train*100,2),"%"))


# ### Observations:

# The scores for test and training data for the logistic regression model are similar. Therefore we do not expect the model to overfit

# # Checking cross validation

# In[69]:


cnf_matrix_logreg = confusion_matrix(y_test, normalized_df_logreg_pred)

sns.heatmap(pd.DataFrame(cnf_matrix_logreg), annot=True,cmap="Reds" , fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix Logistic Regression\n', y=1.1)


# ### AU ROC CURVE LOGISTIC REGRESSION
# 

# In[70]:


fpr, tpr, _ = roc_curve(y_test, normalized_df_logreg_pred)
auc = roc_auc_score(y_test, normalized_df_logreg_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.box(False)
plt.title ('ROC CURVE LOGREG')
plt.show()

print(f"The score for the AUC ROC Curve is: {round(auc,3)*100}%")


# # CONCLUSION:

# #### The following results were found: 
# 
# #### DATASET :  
# With the dataset that was provided, age was ranged 30-60 (majority), number of cigsperday() ranges 10-40(majority) ,sysBP ranges 100-200,glucose ranges 65-100(majority),totChol ranges 150-300. 
# 
# The above mentioned are the important features that were given by the order of highest importance with the help of Feature Selection. 
# 
# #### FINAL RESULT :
# The accuracy was observed the highest at Logistic Regression with -           
#  1. Accuracy of 84.89% 
#  2. f1 score of 6.60%
#  3. Precision of 72.7%
#  4. Recall of 3.40%.
#  
# Therefore  Logistic Regression model is the recommended model
# 
# As observed by the visualizations,
# 1. Age is directly proportional to the target variable (TenYearCHD)
# 2. No: of cigs per day  is a major factor for predicting the heart disease .
# 3. Diabetic patients those having higher level of glucose ranging from 200-400, have        higher risk of getting CHD.
# 4. 90% of stroke patients get CHD
# 5. Patients who have higher systole BP have higher chances of getting CHD
# 6. Patients whose diastole BP is around 75-80 are mostly safe
# 
