#!/usr/bin/env python
# coding: utf-8

# # World Values Survey
# ##### In this exercise we use World Values Survey data available at http://www.worldvaluessurvey.org/wvs.jsp (The data is free to be downloaded from the webpage). It is a survey, conducted every few years in a number of countries. Here we use wave 6 data, mostly from 2013-2014. Note that not all countries are participating in each wave.
# ##### The questions revolve around different opinion topics, including trust, work, religion, family, gender equality, and nationalism. The details of the questions and code used in the data is available in the attached files of "Official Questionnaire" and "Codebook". 
# 
# ##### In this exercise we focus on what the respondents think about abortion: 
# 
# **"Please tell if abortion can always be justified, never be justified, or something in between".** 
# The responses range between 1 - never justifiable (conservative attitude), and 10 - always justifiable (liberal attitude). Besides of the numeric range 1..10, a number of cases have negative codes (this applies to many variables). 
# 
# These are various types of missing information (-5: missing, -4: not asked, -3: not applicable, -2: no answer, -1: don't know). We treat all these as just missing below.
# 
# ## Loading and preparing the data

# In[1]:


#Loading necessary libraries
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Loading the data
os.chdir('/kaggle/input/world-values-survey-wave-6')
wvs = pd.read_csv("wvs.csv.bz2",sep="\t")
wvs


# Each column here is a survey question and its values (observations) are individual's responses. Column V2 is the country code, the details of what country code corresponds to which country is present in the Official Questionnaire. The dataset has 328 variables and 90350 observations.
# 
# #### Identify anomalies/missing data
# 
# The column we are most interested in here is the **Abortion column, which is the variable V204**. This column has values from 1-10 indicating respondants' response on the question "whether you think it can always be justified, never be justified, or something in between?" where 1 is Never Justifiable and 10 is Always Justifiable. The responses also have some responses that are negative. They mean the following:
# 
# -5-.- DE,SE:Inapplicable; HT: Missing-Dropped out survey; RU:Inappropriate response; Missing
# 
# -4-.- Not asked in survey
# 
# -3-.- Not applicable
# 
# -2-.- No answer
# 
# -1-.- Don´t know
# 
# For this analysis, we will only consider the responses that answer this question. Hence, we will not the considering the responses that have negative values.
# Looking at the summary of the survey question on Abortion..

# In[3]:


abortion = wvs[(wvs.V204 > 0)]
abortion.V204.describe()


# After removing the negative responses which mean there no response recording by an individual, we see that the values fall in the range of 1-10, as expected. There are 85742 non-missing (positive) values for V204 (abortion) in this dataset. The average response of the global pool of respondents say Abortion is not justifiable (3.22 - mean). The 75th percentile of the response shows a neutral response (5).
# 
# Since country (V2) seems to be an important column in this dataset, we will drop the observation that do not have a valid country code (has negative values in V2 column). 
# 
# As part of cleaning the data, let's also drop the observations that have missing values (NA) for any columns. We will keep the negative values for the rest of the variables, otherwise we will lose a lot of data.

# In[4]:


#Removing negative values from V2 as well from abortion, and drop missing values (NA) from entire dataset
wvs_ab = abortion[(abortion.V2 > 0)].dropna()
wvs_ab.shape


# After dropping NA values from the rest of the data and negative values from V2 and V204, we are left with 79267 observations. In order to better simplify the analysis, let's create a new binary variable abortion as abortion = 1 (V204 > 3) and 0 (otherwise)

# In[5]:


wvs_ab['abortion'] = [1 if x > 3 else 0 for x in wvs_ab['V204']]
wvs_ab.abortion.describe()


# The modified column has 0/1 response and the mean response is biased towards conservative attitude (0.36)
# 
# To investigate which variables/opinions are most related to Abortion, let's look at the Pearson correlation values of abortion with all the other variables.

# In[6]:


wvs_ab_corr_all = wvs_ab.corr(method="pearson")[['abortion']].sort_values(by='abortion',ascending=False)
wvs_ab_corr_all


# Strong correlation coefficient is generally described as values from 0.7 to 1.0 or -0.7 to -1.0. Looking at the co-efficients listed above, the only variables satisfying this criteria is the abortion column itself and V204 which is the original column from which abortion column was generated. Hence we will consider values greater than 0.4 or lesser than -0.4 as strong correlation co-efficient.

# In[7]:


wvs_ab_corr = wvs_ab_corr_all[(wvs_ab_corr_all.abortion > 0.4) | (wvs_ab_corr_all.abortion < -0.4)]
#Not seeing the first and second row related to abortion
wvs_ab_corr[2::]


# Out of the values seen above, the following responses seem to be correlated with abortion:
# 
# V205: Divorce
# 
# V203: Homosexuality
# 
# V206: Sex before marriage
# 
# V207: Suicide
# 
# #### One-Hot Encoding
# 
# For futher analysis, we will create dummies of the country column which is categorical for regression. 

# In[8]:


wvs_ab = wvs_ab.rename(columns={"V2":"country"})
wvs_ab_d = pd.get_dummies(wvs_ab,columns=['country']) #This step removes the country variable
wvs_ab_d.shape


# After converting country column into dummies, we now have a total of 386 variablles. Let's check if the number of dummy columns created equals to the unique country codes we had.

# In[9]:


#Number of columns with names starting with country - dummies we created == Unique countries in original dataset
wvs_ab_d[wvs_ab_d.columns[pd.Series(wvs_ab_d.columns).str.startswith('country')]].shape[1] == wvs_ab.country.unique().size


# There are a total of 58 country dummy columns which is same as the total number of countries in country column. To avoid perfect multicollinearity we will delete one column, hence let us delete the last country code column - country_887

# In[10]:


wvs_dummy = wvs_ab_d.drop(['country_887','V204'],axis=1)
#Also dropping the V204 column - responses for abortion - from the dataframe
wvs_dummy.shape


# Thus after cleaning and preparing the data, we finally have come down to a total of 384 variables and 79267 observations.
# 
# ## Cross Validation Implementation
# Instead of using an existing implementation, we will create our own implementation of k-fold Cross Validation. ALong with cross-validation, we will also simultaneaously calculate some performance metrics for the models: F1-score, Accuracy, RMSE, AUC of ROC

# In[11]:


def kcv(k,unfit_m,X,y):
    indices = X.index.values
    i_shuffle = shuffle(indices)
    f1=[]
    accuracy=[]
    rmse=[]
    for i in np.arange(k):
        v = i_shuffle[i::k]
        X_valid = X.loc[v,:]
        X_train = X[~X.index.isin(X_valid.index)]
        y_valid = y.loc[v]
        y_train = y[~y.index.isin(y_valid.index)]
        m = unfit_m.fit(X_train,y_train)
        y_predict = m.predict(X_valid)
        f1.append(f1_score(y_valid,y_predict,average='weighted'))
        accuracy.append(accuracy_score(y_valid,y_predict))
        rmse.append(np.sqrt(np.mean([np.square(m - n) for m,n in zip(y_valid,y_predict)])))
    return (np.mean(f1),np.mean(accuracy),np.mean(rmse))


# ## Find the best model for this data
# ### k-Nearest Neighbours
# Now before starting with the model, let's extract a random set of data. Here we will be selecting a sample (without replacement) of size 10000, to avoid too heavy processing that takes too long.

# In[12]:


#Picking a sample of 7000 observations to avoid forever run
wvs_sample = wvs_dummy.sample(n=10000,random_state=1)
X_sample = wvs_sample.loc[:, wvs_sample.columns != 'abortion']
y_sample = wvs_sample['abortion']
#X and y for the entire dataset
X = wvs_dummy.loc[:, wvs_dummy.columns != 'abortion']
y = wvs_dummy['abortion']


# To keep a track of the performance metrics of all the observed models, we will be creating a dataframe

# In[13]:


#Create a structure to store accuracy and F-scores
mycolumns = ['model','accuracy','f-score','RMSE','runtime']
models = pd.DataFrame(columns=mycolumns)
models.set_index('model')


# Trying kNN model on the selected sample of data for different values of k
# #### Cluster size k = 5

# In[14]:


k = 5
start_time = time.clock()
knn_5 = KNeighborsClassifier(n_neighbors=k)
#5 fold cross validation for sample of original data
f1_knn_5,accuracy_knn_5,rmse_knn_5 = kcv(5,knn_5,X_sample,y_sample)
print(("F1-score :",f1_knn_5))
print(("Accuracy :",accuracy_knn_5))
models.loc[len(models)] = ['knn, k=5',accuracy_knn_5,f1_knn_5,rmse_knn_5,time.clock() - start_time]


# #### Cluster size k = 3

# In[15]:


k = 3
start_time = time.clock()
knn = KNeighborsClassifier(n_neighbors=k)
#5 fold cross validation for original data
f1_knn_3,accuracy_knn_3,rmse_knn_3 = kcv(5,knn,X_sample,y_sample)
print(("F1-score :",f1_knn_3))
print(("Accuracy :",accuracy_knn_3))
models.loc[len(models)] = ['knn, k=3',accuracy_knn_3,f1_knn_3,rmse_knn_3,time.clock() - start_time]


# #### Cluster size k = 7

# In[16]:


k = 7
start_time = time.clock()
knn = KNeighborsClassifier(n_neighbors=k)
#5 fold cross validation for original data
f1_knn_7,accuracy_knn_7,rmse_knn_7 = kcv(5,knn,X_sample,y_sample)
print(("F1-score :",f1_knn_7))
print(("Accuracy :",accuracy_knn_7))
models.loc[len(models)] = ['knn, k=7',accuracy_knn_7,f1_knn_7,rmse_knn_7,time.clock() - start_time]


# #### Cluster size k = 9

# In[17]:


k = 9
start_time = time.clock()
knn = KNeighborsClassifier(n_neighbors=k)
#5 fold cross validation for original data
f1_knn_9,accuracy_knn_9,rmse_knn_9 = kcv(5,knn,X_sample,y_sample)
print(("F1-score :",f1_knn_9))
print(("Accuracy :",accuracy_knn_9))
models.loc[len(models)] = ['knn, k=9',accuracy_knn_9,f1_knn_9,rmse_knn_9,time.clock() - start_time]


# #### Cluster size k = 13

# In[18]:


k = 13
start_time = time.clock()
knn = KNeighborsClassifier(n_neighbors=k)
#5 fold cross validation for original data
f1_knn_13,accuracy_knn_13,rmse_knn_13 = kcv(5,knn,X_sample,y_sample)
print(("F1-score :",f1_knn_13))
print(("Accuracy :",accuracy_knn_13))
models.loc[len(models)] = ['knn, k=13',accuracy_knn_13,f1_knn_13,rmse_knn_13,time.clock() - start_time]


# Looking at the accuracy of different size clusters in kNN to see which models fits best...

# In[19]:


models.sort_values(by=['accuracy','f-score'],ascending=False)


# The performance of the models, accuracy and F-score wise, is almost equal to roughly 76-78%. We will further look at more models to compare their performance as well.
# ### Logistic Regression

# In[20]:


start_time = time.clock()
logreg = LogisticRegression(random_state=0)
#5 fold cross validation
f1_log,accuracy_log,rmse_log = kcv(5,logreg,X_sample,y_sample)
print(("F1-score :",f1_log))
print(("Accuracy :",accuracy_log))
models.loc[len(models)] = ['logistic regression',accuracy_log,f1_log,rmse_log,time.clock() - start_time]


# ### Support Vector Machines
# Repeating the process for SVM model, trying different kernel options and gamma/degree values. 
# #### Linear kernel

# In[21]:


start_time = time.clock()
svm_linear = SVC(kernel='linear', gamma='auto')
#5 fold cross validation
f1_svm_lin,accuracy_svm_lin,rmse_svm_lin = kcv(5,svm_linear,X_sample,y_sample)
print(("F1-score :",f1_svm_lin))
print(("Accuracy :",accuracy_svm_lin))
models.loc[len(models)] = ['svm, linear',accuracy_svm_lin,f1_svm_lin,rmse_svm_lin,time.clock() - start_time]


# #### Radial kernel, gamma = 5

# In[22]:


start_time = time.clock()
#Rbf kernel with gamma=5
svm_radial = SVC(kernel='rbf', gamma=5)
#5 fold cross validation
f1_svm_rad_5,accuracy_svm_rad_5,rmse_svm_rad = kcv(5,svm_radial,X_sample,y_sample)
print(("F1-score :",f1_svm_rad_5))
print(("Accuracy :",accuracy_svm_rad_5))
models.loc[len(models)] = ['svm, radial, y=5',accuracy_svm_rad_5,f1_svm_rad_5,rmse_svm_rad,time.clock() - start_time]


# #### Radial kernel, gamma = 10

# In[23]:


start_time = time.clock()
#Rbf kernel with gamma=10
svm_radial = SVC(kernel='rbf', gamma=10)
#5 fold cross validation
f1_svm_rad_10,accuracy_svm_rad_10,rmse_svm_rad_10 = kcv(5,svm_radial,X_sample,y_sample)
print(("F1-score :",f1_svm_rad_10))
print(("Accuracy :",accuracy_svm_rad_10))
models.loc[len(models)] = ['svm, radial, y=10',accuracy_svm_rad_10,f1_svm_rad_10,rmse_svm_rad_10,time.clock() - start_time]


# #### 2nd degree Polynomial kernel

# In[24]:


start_time = time.clock()
#Polynomial kernel with degree=2
svm_poly_2 = SVC(kernel='poly', gamma='auto',degree=2)
#5 fold cross validation
f1_svm_poly_2,accuracy_svm_poly_2,rmse_svm_poly_2 = kcv(5,svm_poly_2,X_sample,y_sample)
print(("F1-score :",f1_svm_poly_2))
print(("Accuracy :",accuracy_svm_poly_2))
models.loc[len(models)] = ['svm, polynomial, d=2',accuracy_svm_poly_2,f1_svm_poly_2,rmse_svm_poly_2,time.clock() - start_time]


# #### 3rd degree Polynomial kernel

# In[25]:


start_time = time.clock()
#Polynomial kernel with degree=3
svm_poly_3 = SVC(kernel='poly', gamma='auto',degree=3)
#5 fold cross validation
f1_svm_poly_3,accuracy_svm_poly_3,rmse_svm_poly_3 = kcv(5,svm_poly_3,X_sample,y_sample)
print(("F1-score :",f1_svm_poly_3))
print(("Accuracy :",accuracy_svm_poly_3))
models.loc[len(models)] = ['svm, polynomial, d=3',accuracy_svm_poly_3,f1_svm_poly_3,rmse_svm_poly_3,time.clock() - start_time]


# #### 8th degree Polynomial kernel

# In[26]:


start_time = time.clock()
#Polynomial kernel with degree=8
svm_poly_8 = SVC(kernel='poly', gamma='auto',degree=8)
#5 fold cross validation
f1_svm_poly_8,accuracy_svm_poly_8,rmse_svm_poly_8 = kcv(5,svm_poly_8,X_sample,y_sample)
print(("F1-score :",f1_svm_poly_8))
print(("Accuracy :",accuracy_svm_poly_8))
models.loc[len(models)] = ['svm, polynomial, d=8',accuracy_svm_poly_8,f1_svm_poly_8,rmse_svm_poly_8,time.clock() - start_time]


# #### Sigmoid kernel, gamma = 5

# In[27]:


start_time = time.clock()
#Sigmoid kernel with gamma=5
svm_sig_5 = SVC(kernel='sigmoid', gamma=5)
#5 fold cross validation
f1_svm_sig_5,accuracy_svm_sig_5,rmse_svm_sig_5 = kcv(5,svm_sig_5,X_sample,y_sample)
print(("F1-score :",f1_svm_sig_5))
print(("Accuracy :",accuracy_svm_sig_5))
models.loc[len(models)] = ['svm, sigmoid, y=5',accuracy_svm_sig_5,f1_svm_sig_5,rmse_svm_sig_5,time.clock() - start_time]


# Observing the values of accuracy and f-score above of several SVM kernels, we can see that there is huge difference between them i.e. the accuracy value is alright, however f-score is quite low. F1-score takes into account both precision and recall, because false negatives and false positives can be crucial to the study of models while true negatives may be less important. Recall is the ability of a classification model to identify all relevant instances and precision is the ability to identify only the relevant instances. The F1-score is the harmonic mean of precision and recall. Low precision and recall mean the perhaps due to other types of errors, despite the higer accuracy, we have very few of our positive predictions as true (low precision) and most of the positive values were not predicted at all (low recall). One of the reasons for this phenomenon could be that there exists imbalance in positive/negative data sizes, or the model is able to predict only the true negatives correctly and severely lacks in correctly predicting true positives.
# ### Random Forest

# In[28]:


start_time = time.clock()
rdf = RandomForestClassifier(n_estimators=100)
#5 fold cross validation
f1_rf,accuracy_rf,rmse_rf = kcv(5,rdf,X_sample,y_sample)
print(("F1-score :",f1_rf))
print(("Accuracy :",accuracy_rf))
models.loc[len(models)] = ['random forest',accuracy_rf,f1_rf,rmse_rf,time.clock() - start_time]


# ### Finally, compare all the models 
# Let's look at the performance metrics in terms of accuracy and f-score produced by these models on the selected sample of data

# In[29]:


models.sort_values(by=['accuracy','f-score'],ascending=False)


# Looking at the performance of several models on this particular dataset, it is quite clear the fastest and most accurate model (with least RMSE) was Random Forest. We will repeat this exercise with all of the data to see if any change in accuracy can be observed as opposed to the accuracy achieved on a subset of the data

# In[30]:


start_time = time.clock()
rdf = RandomForestClassifier(n_estimators=100)
#5 fold cross validation
f1_rf,accuracy_rf,rmse_rf = kcv(5,rdf,X,y)
print(("F1-score :",f1_rf))
print(("Accuracy :",accuracy_rf))
models.loc[len(models)] = ['random forest - all',accuracy_rf,f1_rf,rmse_rf,time.clock() - start_time]


# Running the model on the entire model gave even higher accuracy. 
# 
# ## Feature Importance
# 
# Now that we have selected Random Forest as our primary and best fit model, let's look at which features are more important

# In[31]:


rdf.fit(X,y)
feature_imp = pd.Series(rdf.feature_importances_,index=X.columns).sort_values(ascending=False)


# In[32]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# The figure is too crowded making it impossible to see which features are important, however, looking at the score below for Feature importance shows that the score nonetheless is in ranges from 0.0-0.05 which is very low. Hence, let's look at the top dozen features...

# In[33]:


feature_imp.sort_values(ascending=False).head(12)


# Getting the corresponding questions from the attached file Official Questionnaire.
# 
# V205: Opinion on Divorce
# 
# V203: Opinion on Homosexuality
# 
# V206: Opinion on Sex before Marriage
# 
# V203A: Opinion on Prostitution
# 
# V207: Opinion on Suicide
# 
# V207A: Opinion on Euthanasia
# 
# V152: How important is God in your life?
# 
# V210: Opinion on Violence against other poeple
# 
# V202: Opinion on accepting bribes
# 
# V9: How important is Religion?
# 
# V145: Apart from weddings and funerals, about how often do you attend religious services these days?
# 
# V200: Stealing property
# 
# Now let's run the model again with only these features to see if we get a better performance

# In[34]:


X_imp = X[['V205','V203','V206','V203A','V207','V207A','V152','V210','V202','V9','V145','V200']]
start_time = time.clock()
rdf = RandomForestClassifier(n_estimators=100)
#5 fold cross validation
f1_rf,accuracy_rf,rmse_rf = kcv(5,rdf,X_imp,y)
print(("F1-score :",f1_rf))
print(("Accuracy :",accuracy_rf))
models.loc[len(models)] = ['random forest - important',accuracy_rf,f1_rf,rmse_rf,time.clock() - start_time]


# Well, reducing the features for this data does not help in improving the accuracy of the predictions. Thus, for this data, the best results are obtained when considering the entire dataset
# ## Social Sciences
# Switching from Machine Learning to Social sciences, we are aware that public opinion differs from country to country. Does that mean that the country dummies are playing a huge role in the prediction. We will repeat the training and predictions on the entire dataset now without the country variables and see if they perform better than with the country variables (accuracy : 85%)

# In[35]:


#Drop the columns starting with country_ from sample of X
X_nocountry = X[X.columns.drop(list(X.filter(regex='country_')))]


# In[36]:


start_time = time.clock()
rdf = RandomForestClassifier(n_estimators=100)
#5 fold cross validation
f1_rf,accuracy_rf,rmse_rf = kcv(5,rdf,X_nocountry,y)
print(("F1-score :",f1_rf))
print(("Accuracy :",accuracy_rf))
models.loc[len(models)] = ['random forest - no country',accuracy_rf,f1_rf,rmse_rf,time.clock() - start_time]


# ![](http://)As seen, the accuracy is exactly same even after removing the country variable from the dataset. This means that country does not affect the prediction and hence even though as per social sciences, country might play an important role in a person's opinion on abortion, data science wise, it does not seem to affect the decision as whole.
