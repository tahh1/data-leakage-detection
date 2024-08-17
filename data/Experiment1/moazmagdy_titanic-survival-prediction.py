#!/usr/bin/env python
# coding: utf-8

# In this notebook, we tackle the problem of Titanic Survival Prediction.
# 
# This will include:
#     1. Data exploration: 
#         - Check whether the data is balanced.
#         - Check features with missing values.
#         - Try to find a pattern in data.
#         - What features are most correlated with the target?
#         - What are the continuous and categorical features?
#         - Check the distribution of continuous features.
#         - Check the frequency of category occurrence for categorical features.
#     2. Data Cleaning.
#         - Impute missing values.
#     3. Feature Engineering.
#         - Create hand crafted features from existing features.
#         - Convert categorical features into numeric form.
#         - Rescale features.
#     4. Data splitting.
#     5. Seperating features from target.
#     6. Classification models.
#         - Features selection.
#         - Model hyperparameters tuning.
#         - Variance check to avoid overfitting.
#         - Model Assessment.
#     8. Models comparison.

# In[1]:


# Importing liberaries for data preprocessing, visualization, modeling and scoring.
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.ensemble as ens
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import sklearn.feature_selection
import sklearn.metrics
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, Normalizer, LabelEncoder, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('white')


# # Loading Data

# In[2]:


labelled = pd.read_csv('../input/train.csv') # Labelled Data for training, validation, and model assessment. 


# In[3]:


unlabelled = pd.read_csv('../input/test.csv') # Unlabelled Data for final submission.


# In[4]:


# Keep PassengerId for final submission in seperate variable.
passengerID = unlabelled[['PassengerId']]


# Concatenate both labelled and unlabelled data so that all data cleaning and feature engineering will applied to both of them.

# In[5]:


data = pd.concat([labelled, unlabelled], axis= 0, sort= False)


# # 1. Data Exploring

# In[6]:


data.head()


# ## 1.1 Visualizing null values.

# In[7]:


sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap= 'viridis')


# - Fare column has only one null value.<br/>
# - Age column has many null values.<br/>
# - Cabin column has a majority of null values.<br/>
# - Survived column has null values for the test data.

# In[8]:


data.info()


# ## 1.2 Is data balanced?

# In[9]:


sns.countplot(data = data, x= 'Survived')


# ## 1.3 Which is the most survived gender?

# In[10]:


sns.countplot(data = data, x= 'Survived', hue= 'Sex')
plt.legend(loc =(1.1,0.9)),


# ## 1.4 Does first class have more survival rate?

# In[11]:


sns.countplot(data = data, x='Survived', hue='Pclass')


# ## 1.5 The distribution of passengers' age.

# In[12]:


sns.distplot(data['Age'].dropna(), kde = False, bins = 35)


# ## 1.6 The distribution of number of siblings.

# In[13]:


sns.countplot(x = 'SibSp', data = data)


# ## 1.7 Number of passenger's in each class.

# In[14]:


sns.countplot(data= data.dropna(), x='Pclass')


# ## 1.8 Proportion of each gender in different classes.

# In[15]:


sns.countplot(data= data, x='Pclass', hue= 'Sex')


# ## 1.9 Ticket fare for each class.

# In[16]:


sns.boxplot(data= data.dropna(), x='Pclass', y= 'Fare')


# In[17]:


data.describe()


# # 2. Data cleaning

# ## 2.1 Imputing missing values in Age with the median age for the corresponding class

# In[18]:


class_mean_age = data.pivot_table(values='Age', index='Pclass', aggfunc='median')


# In[19]:


null_age = data['Age'].isnull()


# In[20]:


data.loc[null_age,'Age'] = data.loc[null_age,'Pclass'].apply(lambda x: class_mean_age.loc[x] )


# In[21]:


data.Age.isnull().sum()


# ## 2.2 Imputing the missing value in Fare with the median fare for the corresponding class.

# In[22]:


class_mean_fare = data.pivot_table(values= 'Fare', index= 'Pclass', aggfunc='median')


# In[23]:


null_fare = data['Fare'].isnull()


# In[24]:


data.loc[null_fare, 'Fare'] = data.loc[null_fare, 'Pclass'].apply(lambda x: class_mean_fare.loc[x] )


# In[25]:


data.Fare.isnull().sum()


# ## 2.3 Imputing the missing values in Embarked with the most common port for corresponding class.

# In[26]:


data.Embarked.value_counts()


# In[27]:


data['Embarked'] = data.Embarked.fillna('S')


# In[28]:


data.Embarked.isnull().sum()


# # 3. Feature Engineering

# ## 3.1 Create New features

# ### 3.1.1 Create a new feature with the title of each passenger.

# In[29]:


data['Title'] = data.Name.apply(lambda x : x[x.find(',')+2:x.find('.')])


# In[30]:


data.Title.value_counts()


# We can notice that only 4 titles have significant frequency and the others are repeated only 8 time or less.<br/> So, we will combine all titles with small frequency under one title (say, Other).

# In[31]:


rare_titles = (data['Title'].value_counts() < 10)


# In[32]:


data['Title'] = data['Title'].apply(lambda x : 'Other' if rare_titles.loc[x] == True else x)


# ### 3.1.2 Create a new feature for the family size

# This feature combines the number of siblings and parents/children (SibSp and Parch) +1 (The passenger himself).

# In[33]:


data['FamilySize'] = data['SibSp'] + data['Parch'] + 1


# ### 3.1.3 Create a new feature to indicate whether the passenger was alone.

# In[34]:


data['IsAlone'] = 0


# In[35]:


data['IsAlone'].loc[ data['FamilySize'] == 1] = 1


# ### 3.1.4 Create a new feature by discretizing Age into buckets/bins

# Age is discretized into 4 bins coresponding to 4 stages of human life:<br/>
# 1. Childhood.
# 2. Adolescence.
# 3. Adulthood.
# 4. Old Age. <br/>
# Check this link for more details: https://bit.ly/2LkPFPf

# In[36]:


data['AgeBins'] = 0


# In[37]:


data['AgeBins'].loc[(data['Age'] >= 11) & (data['Age'] < 20)] = 1
data['AgeBins'].loc[(data['Age'] >= 20) & (data['Age'] < 60)] = 2
data['AgeBins'].loc[data['Age'] >= 60] = 3


# ### 3.1.5 Create new feature by discretizing Fare into 4 buckets/bins based on quantiles.

# In[38]:


data['FareBins'] = pd.qcut(data['Fare'], 4)


# ### 3.1.6 Drop unused columns from data.

# 1. Some features are expected to not have effect of the classification such as PassengerId, Name and Ticket. <br/> 
# 2. Also some futures have too much missing values such as the Cabin which render it useless.
# 3. We'll also drop the original features we used to create the new features because there will be high correlation between these features which may confuse the model about feature importance.

# In[39]:


data.columns


# In[40]:


data.drop(columns=['PassengerId','Name','Ticket', 'Cabin', 'Age', 'Fare', 'SibSp', 'Parch'], inplace= True)


# ## 3.2 Convert qualitative features into numeric form.

# ### 3.2.1 Convert categorical features (Embarked, Sex, Title) to numerical features and drop one dummy variable for each.

# In[41]:


data = pd.get_dummies(
    data, columns=['Embarked', 'Sex', 'Title'], drop_first=True)


# ### 3.2.2 Convert qualitative ordinal features (FareBins) into numeric form.

# In[42]:


label = LabelEncoder()
data['FareBins'] = label.fit_transform(data['FareBins'])


# In[43]:


data.head(7)


# ## 3.3 Splitting Data back to labelled/unlabelled sets.

# This is an important step before scaling features. Since the scaler should be fit on the training set only and then applied to both training and test sets.

# In[44]:


labelled = data[data.Survived.isnull() == False].reset_index(drop=True)
unlabelled = data[data.Survived.isnull()].drop(columns = ['Survived']).reset_index(drop=True)


# In[45]:


labelled['Survived'] = labelled.Survived.astype('int64')


# ## 3.4 Rescaling features using different scalers

# We will try the following scalers on a copy of the original data frame and we'll select the best one:
# 1. MinMaxScaler
# 2. MaxAbsScaler
# 3. StandardScaler
# 4. RobustScaler
# 5. Normalizer
# 6. QuantileTransformer
# 7. PowerTransformer

# In[46]:


scalers = [MinMaxScaler(), MaxAbsScaler(), StandardScaler(), RobustScaler(),
            Normalizer(), QuantileTransformer(), PowerTransformer()]


# In[47]:


scaler_score = {}
labelled_copy = labelled.copy(deep= True) # Creat a copy of the original Labelled DF.
for scaler in scalers:
    scaler.fit(labelled_copy[['FamilySize']])
    labelled_copy['FamilySize'] = scaler.transform(labelled_copy[['FamilySize']])
    lr = LogisticRegressionCV(cv = 10, scoring= 'accuracy')
    lr.fit(labelled_copy.drop(columns=['Survived']), labelled_copy.Survived)
    score = lr.score(labelled_copy.drop(columns=['Survived']), labelled_copy.Survived)
    scaler_score.update({scaler:score})


# In[48]:


scaler_score


# We can notice that the top scalers: MinMaxScaler, MaxAbsScaler, StandardScaler, and RobustScaler results in the same accuracy score. So, we will use the StandardScaler.

# In[49]:


scaler = StandardScaler()
scaler.fit(labelled[['FamilySize']])
labelled['FamilySize'] = scaler.transform(labelled[['FamilySize']])
unlabelled['FamilySize'] = scaler.transform(unlabelled[['FamilySize']])


# # 4. Train/Validation/Test.

# We will split the labelled data into 3 sets:
# 1. Training set: used for model training. (Size = %70)
# 2. Validation set: used for hyperparameter tunning. (Size = %15)
# 3. Test set: used for model assessment and comparison of different models. (Size = %15)

# We will perform data split on two steps using train_test_split function:
#    1. we split data into training set and other set.
#    2. we split the other set into validation set and test set.

# In[50]:


x_train, x_other, y_train, y_other = train_test_split(
                labelled.drop(columns=['Survived']), labelled.Survived, train_size=0.7)


# In[51]:


x_valid, x_test, y_valid, y_test = train_test_split(
                                    x_other, y_other, train_size=0.5)


# # 5. Features/Target

# We will seperate the features and target columns from the label data so that it can be used in the feature selection step.

# In[52]:


features = labelled.drop(columns=['Survived'])
target = labelled.Survived


# # 6. Classification Models

# ##  6.1 Logistic Regression Model - Baseline model

# ### 6.1.1 Feature selection

# Here, we will try different method to select the features with the highest explainatory power. We will try the following methods, then we select the best method:
# 1. VarianceThreshold
# 2. SelectKBest
# 3. RFECV
# 4. SelectFromModel

# We will use cross-validation with number of folds = 7 because the size of training data is divisible by 7.

# In[53]:


logistic_reg = LogisticRegressionCV(cv= 7)


# #### 6.1.1.1 VarianceThreshold method

# In[54]:


threshold = np.arange(1, 10, 0.5) *1e-1


# In[55]:


scores = []
for i in threshold:
    selector = sklearn.feature_selection.VarianceThreshold(threshold= i)
    selected_features = selector.fit_transform(features)
    logistic_reg.fit(selected_features, target)
    y_pred = logistic_reg.predict(features.loc[:, selector.get_support()])
    scores.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot variance threshold VS. cross-validated scores for training sets.
plt.figure()
plt.xlabel("variance threshold")
plt.ylabel("Cross validated accuracy score")
plt.plot(np.arange(1, 10, 0.5) *1e-1, np.array(scores))


# In[56]:


print(('The highest accuracy score is:', np.max(np.array(scores))))


# The highest accuracy is obtained after execluding features whose variance is less than 0.1

# #### 6.1.1.2 SelectKbest method

# In[57]:


number_of_features = list(range(1,13))


# In[58]:


scores_k = []
for i in number_of_features:
    selector = sklearn.feature_selection.SelectKBest(k=i)
    selected_features = selector.fit_transform(features, target)
    logistic_reg.fit(selected_features, target)
    y_pred = logistic_reg.predict(features.loc[:, selector.get_support()])
    scores_k.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot number of selected features VS. cross-validated scores for training sets.
plt.figure()
plt.xlabel("Number of Selected Features")
plt.ylabel("Cross validated accuracy score")    
plt.plot(list(range(1,13)), scores_k)


# In[59]:


print(("Maximum accuracy score is :", max(scores_k)))


# In[60]:


print(("Optimal number of features :", np.argmax(np.array(scores_k)) + 1))


# The highest accuracy score is obtained after selecting the best 11 features.

# #### 6.1.1.3 RFECV method

# In[61]:


selector = sklearn.feature_selection.RFECV(logistic_reg, step= 1, cv= 5)
selector.fit(features, target)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(list(range(1, len(selector.grid_scores_) + 1)), selector.grid_scores_)


# In[62]:


print(("Optimal number of features : %d" % selector.n_features_))


# In[63]:


print(("Maximum accuracy score is :", np.max(selector.grid_scores_)))


# #### 6.1.1.4 SelectFromModel method

# In[64]:


threshold = np.arange(1, 5, 0.1) *1e-1


# In[65]:


scores_sfm = []
for i in threshold:
    selector = sklearn.feature_selection.SelectFromModel(logistic_reg, threshold= i)
    selector.fit(features, target)
    selected_features = features.loc[:, selector.get_support()]
    logistic_reg.fit(selected_features, target)
    y_pred = logistic_reg.predict(selected_features)
    scores_sfm.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Threshold Value")
plt.ylabel("Cross validation score")    
plt.plot(np.arange(1, 5, 0.1) *1e-1, scores_sfm)


# In[66]:


print(("Maximum accuracy score is :", np.max(np.array(scores_sfm))))


# In[67]:


print(("Optimal threshold :", threshold[np.argmax(np.array(scores_sfm))]))


# We conclude the best feature selection method is SelectFromModel with threshold = 0.26.

# In[68]:


# Fit the model with features selected by SelectFromModel method and the training set
selector = sklearn.feature_selection.SelectFromModel(logistic_reg, threshold= 0.25)
selector.fit(features, target)
lr_selected_features = selector.get_support()


# ### 6.1.2 Logistic Regression Hyper-parameters tunning

# Fit our model and the use the cross-validation value of 7 since the training set size if divisible by 7.

# In[69]:


logistic_reg = LogisticRegressionCV(
    Cs=1, cv= 7, scoring='accuracy', max_iter=1000, refit=True)


# We will use Randomized search to find the best solver and penalty. we decided to make this process on two steps because of the limitations of some solvers that work with only one type of penalty.

# In[70]:


lr_parameters_1 = {'solver': ['liblinear', 'saga'], 'penalty': ['l1']}
lr_parameters_2 = {'solver': ['newton-cg', 'lbfgs', 'sag'], 'penalty': ['l2']}


# In[71]:


rs_lr = RandomizedSearchCV(logistic_reg, param_distributions= lr_parameters_2, n_iter= 100)


# In[72]:


rs_lr.fit(x_train.loc[:, lr_selected_features], y_train)


# In[73]:


print(('Best Parameters are:\n', rs_lr.best_params_,
      '\nTraining accuracy score is:\n', rs_lr.best_score_))


# In[74]:


print(('Validation accuracy score is:\n', rs_lr.score(
    x_valid.loc[:, lr_selected_features], y_valid)))


# ### 6.1.3 Variance check

# First, we want to tune our model such that we minimize the variance, which is sensitivity of the prediction score to the change in training set. We will use the validation curve to help us choose the best value for regularization factor.

# In[75]:


param_name = 'Cs'
param_range = [1, 10, 100, 1000]
train_score, valid_score = [], []
for cs in param_range:
    lr = LogisticRegressionCV(Cs=cs, cv=7, scoring='accuracy', solver= 'newton-cg',
                              penalty= 'l2', refit=True, max_iter=1000)
    lr.fit(x_train.loc[:, lr_selected_features], y_train)
    train_score.append(
        lr.score(x_train.loc[:, lr_selected_features], y_train))
    valid_score.append(
        lr.score(x_valid.loc[:, lr_selected_features], y_valid))


# In[76]:


# Plot Regularization factor VS. cross-validated scores for training and Validation sets.
plt.figure()
plt.xlabel("Regularization factor")
plt.ylabel("Cross validated accuracy score")
plt.plot([1, 10, 100, 1000], train_score, color = 'blue')
plt.plot([1, 10, 100, 1000], valid_score, color = 'red')


# In[77]:


train_test_diff = np.array(train_score) - np.array(valid_score)

# Plot number of folds VS. difference of cross-validated scores between train and Dev sets.
plt.figure()
plt.xlabel("Regularization Factor")
plt.ylabel("Diff. Cross validated accuracy score")
plt.plot([1, 10, 100, 1000], train_test_diff)


# At regularization factor Cs = 10, the accuracy score is high and the difference between the train and validation sets is minimum.

# ### 6.1.4 Model Assessment

# Fit the model with best parameters that minimize both bias and variance.

# In[78]:


lr = LogisticRegressionCV(Cs= 10, cv= 7, solver= 'newton-cg', penalty= 'l2')
lr.fit(x_train.loc[:, lr_selected_features], y_train)


# In[79]:


# Finding the ROC curve for different threshold values.
# probability estimates of the positive class.
y_scores_lr = lr.predict_proba(x_test.loc[:, lr_selected_features])[:, 1]
lr_fpr, lr_tpr, lr_thresholds = sklearn.metrics.roc_curve(y_test, y_scores_lr)


# In[80]:


# Finding the AUC for the logistic classification model.
lr_auc = sklearn.metrics.auc(x=lr_fpr, y=lr_tpr)


# In[81]:


# Model accuracy score on test data.
lr_acc = lr.score(x_test.loc[:, lr_selected_features], y_test)


# In[82]:


print(('For logistic Regression: \n Area Under Curve: {}, \n Test Accuracy score: {}'.format(
    lr_auc, lr_acc)))


# ## 6.2 Gaussian Naive Bayes Model

# ### 6.2.1 Feature selection

# Here, we will try different method to select the features with the highest explainatory power. We will try the following methods, then we select the best method:
# 1. VarianceThreshold
# 2. SelectKBest
# 3. RFECV
# 4. SelectFromModel

# In[83]:


nb = GaussianNB()


# #### 6.2.1.1 VarianceThreshold method

# In[84]:


threshold = np.arange(1, 10, 0.5) *1e-1


# In[85]:


scores = []
for i in threshold:
    selector = sklearn.feature_selection.VarianceThreshold(threshold= i)
    selected_features = selector.fit_transform(features)
    nb.fit(selected_features, target)
    y_pred = nb.predict(features.loc[:, selector.get_support()])
    scores.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot variance threshold VS. cross-validated scores for training sets.
plt.figure()
plt.xlabel("variance threshold")
plt.ylabel("Cross validated accuracy score")
plt.plot(np.arange(1, 10, 0.5) *1e-1, np.array(scores))


# In[86]:


print(('The highest accuracy score is:', np.max(np.array(scores))))


# The highest accuracy is obtained after execluding features whose variance is less than 0.1

# #### 6.2.1.2 SelectKbest method

# In[87]:


number_of_features = list(range(1,13))


# In[88]:


scores_k = []
for i in number_of_features:
    selector = sklearn.feature_selection.SelectKBest(k=i)
    selected_features = selector.fit_transform(features, target)
    nb.fit(selected_features, target)
    y_pred = nb.predict(features.loc[:, selector.get_support()])
    scores_k.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot number of selected features VS. cross-validated scores for training sets.
plt.figure()
plt.xlabel("Number of Selected Features")
plt.ylabel("Cross validated accuracy score")    
plt.plot(list(range(1,13)), scores_k)


# In[89]:


print(("Maximum accuracy score is :", max(scores_k)))


# In[90]:


print(("Optimal number of features :", np.argmax(np.array(scores_k)) + 1))


# The highest accuracy score is obtained after selecting the best 11 features.

# #### 6.2.1.3 RFECV method

# This method can't be used with Gaussian Naive Bayes algorithm because this classifier does not expose 'coef_' or 'feature_importances_' attributes.

# #### 6.2.1.4 SelectFromModel method

# This method can't be used with Gaussian Naive Bayes algorithm because this classifier does not expose 'coef_' or 'feature_importances_' attributes.

# We conclude the best feature selection method is Variance threshold method with threshold = 0.1.

# In[91]:


# Fit the model with features selected by Variance threshold method and the training set
selector = sklearn.feature_selection.VarianceThreshold(threshold= 0.1)
selector.fit(features, target)
nb_selected_features = selector.get_support()


# ### 6.2.2 Gaussian NB Hyper-parameters tunning

# We will use Randomized search to find the best priors.

# In[92]:


nb_params = {'priors': [[0.7, 0.3], [0.6, 0.4],
                        [0.5, 0.5], [0.4, 0.6], [0.3, 0.7]]}


# In[93]:


rs_nb = RandomizedSearchCV(nb, param_distributions= nb_params,cv= 7 ,n_iter= 200)


# In[94]:


rs_nb.fit(x_train.loc[:, nb_selected_features], y_train)


# In[95]:


print(('Best Parameters are:\n', rs_nb.best_params_,
      '\nTraining accuracy score is:\n', rs_nb.best_score_))


# In[96]:


print(('Validation accuracy score is:\n', rs_nb.score(
    x_valid.loc[:, nb_selected_features], y_valid)))


# ### 6.2.3 Model Assessment

# Fit the model with best parameters that minimize both bias and variance.

# In[97]:


nb = GaussianNB(priors= [0.4, 0.6])
nb.fit(x_train.loc[:, nb_selected_features], y_train)


# In[98]:


# Finding the ROC curve for different threshold values.
# probability estimates of the positive class.
y_scores_nb = nb.predict_proba(x_test.loc[:, nb_selected_features])[:, 1]
nb_fpr, nb_tpr, nb_thresholds = sklearn.metrics.roc_curve(y_test, y_scores_nb)


# In[99]:


# Finding the AUC for the naive bayes classification model.
nb_auc = sklearn.metrics.auc(x=nb_fpr, y=nb_tpr)


# In[100]:


# Model Accuracy score on test data
nb_acc = nb.score(x_test.loc[:, nb_selected_features], y_test)


# In[101]:


print(('For Gaussian Naive Bayes: \n Area Under Curve: {}, \n Test Accuracy score: {}'.format(
    nb_auc, nb_acc)))


# ## 6.3 KNN Classification Model

# ### 6.3.1 Feature selection

# Here, we will try different method to select the features with the highest explainatory power. We will try the following methods, then we select the best method:
# 1. VarianceThreshold
# 2. SelectKBest
# 3. RFECV
# 4. SelectFromModel

# In[102]:


knn = KNeighborsClassifier(n_neighbors= 5)


# #### 6.3.1.1 VarianceThreshold method

# In[103]:


threshold = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]


# In[104]:


scores = []
for i in threshold:
    selector = sklearn.feature_selection.VarianceThreshold(threshold= i)
    selected_features = selector.fit_transform(features)
    knn.fit(selected_features, target)
    y_pred = knn.predict(selected_features)
    scores.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot variance threshold VS. cross-validated scores for training sets.
plt.figure()
plt.xlabel("variance threshold")
plt.ylabel("Cross validated accuracy score")
plt.plot([0.001, 0.005, 0.01, 0.05, 0.1, 0.2], np.array(scores))


# In[105]:


np.max(np.array(scores))


# #### 6.3.1.2 SelectKbest method

# In[106]:


number_of_features = list(range(1,13))


# In[107]:


scores_k = []
for i in number_of_features:
    selector = sklearn.feature_selection.SelectKBest(k=i)
    selected_features = selector.fit_transform(features, target)
    knn.fit(selected_features, target)
    y_pred = knn.predict(selected_features)
    scores_k.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot number of selected features VS. cross-validated scores for training sets.
plt.figure()
plt.xlabel("Number of Selected Features")
plt.ylabel("Cross validated accuracy score")    
plt.plot(list(range(1,13)), scores_k)


# In[108]:


print(("Maximum accuracy score is :", max(scores_k)))


# In[109]:


print(("Optimal number of features :", np.argmax(np.array(scores_k)) + 1))


# #### 6.3.1.3 RFECV method

# This method can't be used with KNN algorithm because this classifier does not expose 'coef_' or 'feature_importances_' attributes.

# #### 6.3.1.4 SelectFromModel method

# This method can't be used with KNN algorithm because this classifier does not expose 'coef_' or 'feature_importances_' attributes.

# We conclude that, the highest accuracy is obtained using VarianceThreshold method after execluding features whose variance is less than 0.1

# Fit the model with the selected features.

# In[110]:


selector = sklearn.feature_selection.VarianceThreshold(threshold= 0.1)
selector.fit(features, target)
knn_selected_features = selector.get_support()


# ### 6.3.2 KNN hyperparamters tunning

# We'll use randomized search to tune the hyperparamters of KNN. And we'll follow a coarse-to-fine strategy.

# In[111]:


knn_params = {'n_neighbors': [5, 7, 9] , 'weights': [
    'uniform', 'distance'], 'leaf_size': [5, 10, 20], 'p': [1, 2, 3]}


# In[112]:


rs_knn = RandomizedSearchCV(knn, param_distributions= knn_params,
                      scoring='accuracy', cv= 7, n_iter= 200, refit=True)


# In[113]:


rs_knn.fit(x_train.loc[:, knn_selected_features], y_train)


# In[114]:


print(('Best Parameters are:\n', rs_knn.best_params_,
      '\nTraining accuracy score is:\n', rs_knn.best_score_))


# In[115]:


print(('Validation accuracy score is:\n', rs_knn.score(
    x_valid.loc[:, knn_selected_features], y_valid)))


# ### 6.3.3 Variance check

# First, we want to tune our model such that we minimize the variance, which is sensitivity of the prediction score to the change in training set. We will use the validation curve to help us choose the best number of neighbours (K).

# In[116]:


param_name = 'n_neighbors'
param_range = np.arange(3,21)
train_score, valid_score = [], []
for k in param_range:
    knn = KNeighborsClassifier(n_neighbors= k, weights= 'uniform', p= 2,leaf_size= 5)
    knn.fit(x_train.loc[:, knn_selected_features], y_train)
    train_score.append(
        knn.score(x_train.loc[:, knn_selected_features], y_train))
    valid_score.append(
        knn.score(x_valid.loc[:, knn_selected_features], y_valid))


# In[117]:


# Plot number of neighbours VS. cross-validated scores for training and Validation sets.
plt.figure()
plt.xlabel("Number of Neighbours")
plt.ylabel("Cross validated accuracy score")
plt.plot(np.arange(3,21), train_score, color = 'blue')
plt.plot(np.arange(3,21), valid_score, color = 'red')


# In[118]:


train_test_diff = np.array(train_score) - np.array(valid_score)

# Plot Number of Neighbours VS. difference of cross-validated scores between train and validation sets.
plt.figure()
plt.xlabel("Number of Neighbours")
plt.ylabel("Diff. Cross validated accuracy score")
plt.plot(np.arange(3,21), train_test_diff)


# It seems that the minimum variance is obtained at number of neighbours K = 4.

# ### 6.3.4 Model Assessment

# Fit the model with best parameters that minimize bias and variance.

# In[119]:


knn = KNeighborsClassifier(n_neighbors= 4, weights= 'uniform', p= 2, leaf_size= 5)


# In[120]:


knn.fit(x_train.loc[:, knn_selected_features], y_train)


# In[121]:


# Finding the ROC curve for different threshold values.
# probability estimates of the positive class.
y_scores_knn = knn.predict_proba(x_test.loc[:, knn_selected_features])[:, 1]
knn_fpr, knn_tpr, knn_thresholds = sklearn.metrics.roc_curve(y_test, y_scores_knn)


# In[122]:


# Finding the AUC for the naive bayes classification model.
knn_auc = sklearn.metrics.auc(x=knn_fpr, y=knn_tpr)


# In[123]:


# Model Accuracy score on test data
knn_acc = knn.score(x_test.loc[:, knn_selected_features], y_test)


# In[124]:


print(('Area Under Curve: {}, Accuracy: {}'.format(knn_auc, knn_acc)))


# We found that, using only train set for training the KNN model will result in higher accuracy score when submiting predictions compared to using the whole labelled data for training the model.

# ## 6.4 Support Vector Machine Classification model

# ### 6.4.1 Feature selection

# Here, we will try different method to select the features with the highest explainatory power. We will try the following methods, then we select the best method:
# 1. VarianceThreshold
# 2. SelectKBest
# 3. RFECV
# 4. SelectFromModel

# We will use cross-validation with number of folds = 7 because the size of training data is divisible by 7.

# In[125]:


svm = SVC(probability=True)


# #### 6.4.1.1 VarianceThreshold method

# In[126]:


threshold = np.arange(1, 10, 0.5) *1e-1


# In[127]:


scores = []
for i in threshold:
    selector = sklearn.feature_selection.VarianceThreshold(threshold= i)
    selected_features = selector.fit_transform(features)
    svm.fit(selected_features, target)
    y_pred = svm.predict(features.loc[:, selector.get_support()])
    scores.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot variance threshold VS. cross-validated scores for training sets.
plt.figure()
plt.xlabel("variance threshold")
plt.ylabel("Cross validated accuracy score")
plt.plot(np.arange(1, 10, 0.5) *1e-1, np.array(scores))


# In[128]:


print(('The highest accuracy score is:', np.max(np.array(scores))))


# The highest accuracy is obtained after execluding features whose variance is less than 0.1

# #### 6.4.1.2 SelectKbest method

# In[129]:


number_of_features = list(range(1,13))


# In[130]:


scores_k = []
for i in number_of_features:
    selector = sklearn.feature_selection.SelectKBest(k=i)
    selected_features = selector.fit_transform(features, target)
    svm.fit(selected_features, target)
    y_pred = svm.predict(features.loc[:, selector.get_support()])
    scores_k.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot number of selected features VS. cross-validated scores for training sets.
plt.figure()
plt.xlabel("Number of Selected Features")
plt.ylabel("Cross validated accuracy score")    
plt.plot(list(range(1,13)), scores_k)


# In[131]:


print(("Maximum accuracy score is :", max(scores_k)))


# In[132]:


print(("Optimal number of features :", np.argmax(np.array(scores_k)) + 1))


# The highest accuracy score is obtained when using all features for model fitting.

# #### 6.4.1.3 RFECV method

# This method can't be used with SVC algorithm because of a bug in the code of RFECV function, it shows an error message claiming that SVC classifier does not expose 'coef_' or 'feature_importances_' attributes. But it does expose 'coef_' attribute.

# #### 6.4.1.4 SelectFromModel method

# This method can't be used with SVC algorithm because of a bug in the code of SelectFromModel function, it shows an error message claiming that SVC classifier does not expose 'coef_' or 'feature_importances_' attributes. But it does expose 'coef_' attribute.

# We conclude the best feature selection method is Variance threshold with threshold = 0.1.

# In[133]:


# Fit the model with features selected by SelectFromModel method and the training set
selector = sklearn.feature_selection.VarianceThreshold(threshold= 0.1)
selector.fit(features, target)
svm_selected_features = selector.get_support()


# ### 6.4.2 Logistic Regression Hyper-parameters tunning

# We will use Randomized search to find the best solver and penalty. we decided to make this process on two steps because of the limitations of some solvers that work with only one type of penalty.

# In[134]:


svm = SVC(probability=True)


# In[135]:


svm_parameters = {'kernel': ['linear', 'rbf', 'sigmoid'], 'gamma': [
    'auto', 'scale'], 'shrinking': [True, False]}


# In[136]:


rs_svm = RandomizedSearchCV(svm, cv= 7, param_distributions= svm_parameters, n_iter= 200)


# In[137]:


rs_svm.fit(x_train.loc[:, svm_selected_features], y_train)


# In[138]:


print(('Best Parameters are:\n', rs_svm.best_params_,
      '\nTraining accuracy score is:\n', rs_svm.best_score_))


# In[139]:


print(('Validation accuracy score is:\n', rs_svm.score(
    x_valid.loc[:, svm_selected_features], y_valid)))


# After trying different paramters, we found that the best paramters are the default.

# ### 6.4.3 Variance check

# First, we want to tune our model such that we minimize the variance, which is sensitivity of the prediction score to the change in training set. We will use the validation curve to help us choose the best value for regularization factor (C).

# In[140]:


param_name = 'C'
param_range = np.arange(1,31)
train_score, valid_score = [], []
for c in param_range:
    svm = SVC(C= c,probability= True)
    svm.fit(x_train.loc[:, svm_selected_features], y_train)
    train_score.append(
        svm.score(x_train.loc[:, svm_selected_features], y_train))
    valid_score.append(
        svm.score(x_valid.loc[:, svm_selected_features], y_valid))


# In[141]:


# Plot Regularization factor VS. cross-validated scores for training and Validation sets.
plt.figure()
plt.xlabel("Regularization factor")
plt.ylabel("Cross validated accuracy score")
plt.plot(np.arange(1,31), train_score, color = 'blue')
plt.plot(np.arange(1,31), valid_score, color = 'red')


# In[142]:


train_test_diff = np.array(train_score) - np.array(valid_score)

# Plot number of folds VS. difference of cross-validated scores between train and Dev sets.
plt.figure()
plt.xlabel("Regularization Factor")
plt.ylabel("Diff. Cross validated accuracy score")
plt.plot(np.arange(1,31), train_test_diff)


# At regularization factor C = 3, the accuracy score of validation set is maximum and the difference between the train and validation sets is minimum.

# ### 6.4.4 Model Assessment

# Fit the model with best parameters that minimize both bias and variance.

# In[143]:


svm = SVC(C=3, probability= True)
svm.fit(features.loc[:, svm_selected_features], target)


# In[144]:


# Finding the ROC curve for different threshold values.
# probability estimates of the positive class.
y_scores_svm = svm.predict_proba(x_test.loc[:, svm_selected_features])[:, 1]
svm_fpr, svm_tpr, svm_thresholds = sklearn.metrics.roc_curve(y_test, y_scores_svm)


# In[145]:


# Finding the AUC for the logistic classification model.
svm_auc = sklearn.metrics.auc(x=svm_fpr, y=svm_tpr)


# In[146]:


# Model accuracy score on test data.
svm_acc = svm.score(x_test.loc[:, svm_selected_features], y_test)


# In[147]:


print(('For logistic Regression: \n Area Under Curve: {}, \n Test Accuracy score: {}'.format(
    svm_auc, svm_acc)))


# ## 6.5 Decision Tree Classification Model

# ### 6.5.1 Feature selection

# Here, we will try different method to select the features with the highest explainatory power. We will try the following methods, then we select the best method:
# 1. VarianceThreshold
# 2. SelectKBest
# 3. RFECV
# 4. SelectFromModel

# In[148]:


dt = DecisionTreeClassifier()


# #### 6.5.1.1 VarianceThreshold method

# In[149]:


threshold = np.arange(1, 10, 0.5) *1e-1


# In[150]:


scores = []
for i in threshold:
    selector = sklearn.feature_selection.VarianceThreshold(threshold= i)
    selected_features = selector.fit_transform(features)
    dt.fit(selected_features, target)
    y_pred = dt.predict(features.loc[:, selector.get_support()])
    scores.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot variance threshold VS. cross-validated scores for training sets.
plt.figure()
plt.xlabel("variance threshold")
plt.ylabel("Cross validated accuracy score")
plt.plot(threshold, np.array(scores))


# In[151]:


print(('The highest accuracy score is:', np.max(np.array(scores))))


# The highest accuracy is obtained after execluding features whose variance is less than 0.1

# #### 6.5.1.2 SelectKbest method

# In[152]:


number_of_features = list(range(1,13))


# In[153]:


scores_k = []
for i in number_of_features:
    selector = sklearn.feature_selection.SelectKBest(k=i)
    selected_features = selector.fit_transform(features, target)
    dt.fit(selected_features, target)
    y_pred = dt.predict(features.loc[:, selector.get_support()])
    scores_k.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot number of selected features VS. cross-validated scores for training sets.
plt.figure()
plt.xlabel("Number of Selected Features")
plt.ylabel("Cross validated accuracy score")    
plt.plot(number_of_features, scores_k)


# In[154]:


print(("Maximum accuracy score is :", max(scores_k)))


# In[155]:


print(("Optimal number of features :", np.argmax(np.array(scores_k)) + 1))


# The highest accuracy score is obtained after selecting the whole 12 features.

# #### 6.5.1.3 RFECV method

# In[156]:


selector = sklearn.feature_selection.RFECV(dt, step= 1, cv= 5)
selector.fit(features, target)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(list(range(1, len(selector.grid_scores_) + 1)), selector.grid_scores_)


# In[157]:


print(("Optimal number of features : %d" % selector.n_features_))


# In[158]:


print(("Maximum accuracy score is :", np.max(selector.grid_scores_)))


# #### 6.5.1.4 SelectFromModel method

# In[159]:


threshold = [0.001, 0.0025, 0.005, 0.01, 0.025 ,0.05, 0.1]


# In[160]:


scores_sfm = []
for i in threshold:
    selector = sklearn.feature_selection.SelectFromModel(dt, threshold= i)
    selector.fit(features, target)
    selected_features = features.loc[:, selector.get_support()]
    dt.fit(selected_features, target)
    y_pred = dt.predict(selected_features)
    scores_sfm.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Threshold Value")
plt.ylabel("Cross validation score")    
plt.plot(threshold, scores_sfm)


# In[161]:


print(("Maximum accuracy score is :", np.max(np.array(scores_sfm))))


# In[162]:


print(("Optimal threshold :", threshold[np.argmax(np.array(scores_sfm))]))


# We conclude the highest accuracy score can be obtained while using the whole 12 features for prediction. So we will use all features to the model.

# ### 6.5.2 Decision tree hyperparamters tunning

# We will use randomized search method and we will follow coarse-to-fine strategy.

# In[163]:


dt_params = {'criterion': ['gini'], 'min_samples_split': [
     21, 22, 23], 'max_features': ['auto', 'log2', None]}


# In[164]:


rs_dt = RandomizedSearchCV(dt, param_distributions= dt_params,
                     scoring='accuracy', cv= StratifiedKFold(7), refit=True, n_iter= 500)


# In[165]:


rs_dt.fit(x_train, y_train)


# In[166]:


print(('Best Parameters are:\n', rs_dt.best_params_,
      '\nTraining accuracy score is:\n', rs_dt.best_score_))


# In[167]:


print(('Validation accuracy score is:\n', rs_dt.score(x_valid, y_valid)))


# ### 6.5.3 Variance check

# We will use the max_depth of tree to minimize the variance. We'll use the validation curve to select the best value of max_depth.

# In[168]:


param_name = 'max_depth'
param_range = np.arange(1, 21)
train_score, valid_score = [], []
for depth in param_range:
    dt = DecisionTreeClassifier(
        criterion='gini', max_features=None, min_samples_split=22, max_depth= depth)
    dt.fit(x_train, y_train)
    train_score.append(dt.score(x_train, y_train))
    valid_score.append(dt.score(x_valid, y_valid))


# In[169]:


# Plot Regularization factor VS. cross-validated scores for training and Validation sets.
plt.figure()
plt.xlabel("Regularization factor")
plt.ylabel("Cross validated accuracy score")
plt.plot(param_range, train_score, color = 'blue')
plt.plot(param_range, valid_score, color = 'red')


# In[170]:


train_test_diff = np.array(train_score) - np.array(valid_score)

# Plot number of folds VS. difference of cross-validated scores between train and Dev sets.
plt.figure()
plt.xlabel("Regularization Factor")
plt.ylabel("Diff. Cross validated accuracy score")
plt.plot(param_range, train_test_diff)


# From the above graphs, we find that the maximum validation accuracy score is at max_depth of 3. So, we will choose max_depth of 3 because the validation accuracy score is maximum and the variance is relatively small.

# ### 6.5.4 Model Assessment

# Fit the model with best parameters that minimize both bias and variance.

# In[171]:


dt = DecisionTreeClassifier(criterion='gini', max_features=None, min_samples_split=22, max_depth= 3)
dt.fit(features,target)


# In[172]:


# Finding the ROC curve for different threshold values.
# probability estimates of the positive class.
y_scores_dt = dt.predict_proba(x_test)[:, 1]
dt_fpr, dt_tpr, dt_thresholds = sklearn.metrics.roc_curve(y_test, y_scores_dt)
# Finding the AUC for the Decision Tree classification model.
dt_auc = sklearn.metrics.auc(x=dt_fpr, y=dt_tpr)


# In[173]:


dt_acc = dt.score(x_test, y_test)


# In[174]:


print(('Area Under Curve: {}, Accuracy: {}'.format(dt_auc, dt_acc)))


# ## 6.6 Random Forest Classification Model

# ### 6.6.1 Feature selection

# Here, we will try different method to select the features with the highest explainatory power. We will try the following methods, then we select the best method:
# 1. VarianceThreshold
# 2. SelectKBest
# 3. RFECV
# 4. SelectFromModel

# In[175]:


rf = ens.RandomForestClassifier()


# #### 6.6.1.1 VarianceThreshold method

# In[176]:


threshold = np.arange(1, 10, 0.5) *1e-1


# In[177]:


scores = []
for i in threshold:
    selector = sklearn.feature_selection.VarianceThreshold(threshold= i)
    selected_features = selector.fit_transform(features)
    rf.fit(selected_features, target)
    y_pred = rf.predict(features.loc[:, selector.get_support()])
    scores.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot variance threshold VS. cross-validated scores for training sets.
plt.figure()
plt.xlabel("variance threshold")
plt.ylabel("Cross validated accuracy score")
plt.plot(threshold, np.array(scores))


# In[178]:


print(('The highest accuracy score is obtained after execluding features whose variance is less than: ', 
              np.round(threshold[np.argmax(np.array(scores))],3)))


# In[179]:


print(('The highest accuracy score is:', np.max(np.array(scores))))


# #### 6.6.1.2 SelectKbest method

# In[180]:


number_of_features = list(range(1,13))


# In[181]:


scores_k = []
for i in number_of_features:
    selector = sklearn.feature_selection.SelectKBest(k=i)
    selected_features = selector.fit_transform(features, target)
    rf.fit(selected_features, target)
    y_pred = rf.predict(features.loc[:, selector.get_support()])
    scores_k.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot number of selected features VS. cross-validated scores for training sets.
plt.figure()
plt.xlabel("Number of Selected Features")
plt.ylabel("Cross validated accuracy score")    
plt.plot(number_of_features, scores_k)


# In[182]:


print(("Maximum accuracy score is :", max(scores_k)))


# In[183]:


print(("Optimal number of features :", np.argmax(np.array(scores_k)) + 1))


# #### 6.6.1.3 RFECV method

# In[184]:


selector = sklearn.feature_selection.RFECV(rf, step= 1, cv= 5)
selector.fit(features, target)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(list(range(1, len(selector.grid_scores_) + 1)), selector.grid_scores_)


# In[185]:


print(("Optimal number of features : %d" % selector.n_features_))


# In[186]:


print(("Maximum accuracy score is :", np.max(selector.grid_scores_)))


# #### 6.6.1.4 SelectFromModel method

# In[187]:


threshold = [0.001, 0.0025, 0.005, 0.01, 0.025 ,0.05, 0.1, 0.15]


# In[188]:


scores_sfm = []
for i in threshold:
    selector = sklearn.feature_selection.SelectFromModel(rf, threshold= i)
    selector.fit(features, target)
    selected_features = features.loc[:, selector.get_support()]
    rf.fit(selected_features, target)
    y_pred = rf.predict(selected_features)
    scores_sfm.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Threshold Value")
plt.ylabel("Cross validation score")    
plt.plot(threshold, scores_sfm)


# In[189]:


print(("Maximum accuracy score is :", np.max(np.array(scores_sfm))))


# In[190]:


print(("Optimal threshold :", threshold[np.argmax(np.array(scores_sfm))]))


# We conclude the highest accuracy score can be also obtained while using the whole 12 features for prediction. So we will use all features to the model.

# ### 6.6.2 Random Forest hyperparamters tunning

# We will use randomized search method and we will follow coarse-to-fine strategy.

# In[191]:


rf_params = {'n_estimators': [200, 300, 400], 'criterion': ['gini'], 'min_samples_split': [
    22, 20, 25], 'max_features': ['auto', 'log2', None], 'class_weight': [{0: 0.6, 1: 0.4}, {0: 0.6, 1: 0.4}, {0: 0.5, 1: 0.5}]}


# In[192]:


rs_rf = RandomizedSearchCV(rf, param_distributions= rf_params,
                     scoring='accuracy', cv= StratifiedKFold(7), refit=True, n_iter= 200)


# In[193]:


rs_rf.fit(x_train, y_train)


# In[194]:


print(('Best Parameters are:\n', rs_rf.best_params_,
      '\nTraining accuracy score is:\n', rs_rf.best_score_))


# In[195]:


print(('Validation accuracy score is:\n', rs_rf.score(x_valid, y_valid)))


# ### 6.6.3 Variance check

# We will use the max_depth of tree to minimize the variance. We'll use the validation curve to select the best value of max_depth.

# In[196]:


param_name = 'max_depth'
param_range = np.arange(1, 31)
train_score, valid_score = [], []
for depth in param_range:
    rf = ens.RandomForestClassifier(n_estimators= 300,
        criterion='gini', max_features= 'auto', min_samples_split=22, 
                                    class_weight= {0: 0.5, 1: 0.5},max_depth= depth)
    rf.fit(x_train, y_train)
    train_score.append(rf.score(x_train, y_train))
    valid_score.append(rf.score(x_valid, y_valid))


# In[197]:


# Plot Regularization factor VS. cross-validated scores for training and Validation sets.
plt.figure()
plt.xlabel("Regularization factor")
plt.ylabel("Cross validated accuracy score")
plt.plot(param_range, train_score, color = 'blue')
plt.plot(param_range, valid_score, color = 'red')


# In[198]:


train_test_diff = np.array(train_score) - np.array(valid_score)

# Plot number of folds VS. difference of cross-validated scores between train and Dev sets.
plt.figure()
plt.xlabel("Regularization Factor")
plt.ylabel("Diff. Cross validated accuracy score")
plt.plot(param_range, train_test_diff)


# From the above graphs, we find that the maximum validation accuracy score is at max_depth of 3. So, we will choose max_depth of 6 because the validation accuracy score is maximum and the variance is relatively small.

# ### 6.6.4 Model Assessment

# Fit the model with best parameters that minimize both bias and variance.

# In[199]:


rf = ens.RandomForestClassifier(n_estimators= 300,
        criterion='gini', max_features= 'auto', min_samples_split=22, 
                                class_weight= {0: 0.5, 1: 0.5}, max_depth= 6)
rf.fit(features,target)


# In[200]:


# Finding the ROC curve for different threshold values.
# probability estimates of the positive class.
y_scores_rf = rf.predict_proba(x_test)[:, 1]
rf_fpr, rf_tpr, rf_thresholds = sklearn.metrics.roc_curve(y_test, y_scores_rf)
# Finding the AUC for the Decision Tree classification model.
rf_auc = sklearn.metrics.auc(x=rf_fpr, y=rf_tpr)


# In[201]:


rf_acc = rf.score(x_test, y_test)


# In[202]:


print(('Area Under Curve: {}, Accuracy: {}'.format(rf_auc, rf_acc)))


# ## 6.7 Bagging Classification Model

# In[203]:


bg = ens.BaggingClassifier()


# ### 6.7.1 Feature selection

# Here, we will try different method to select the features with the highest explainatory power. We will try the following methods, then we select the best method:
# 1. VarianceThreshold
# 2. SelectKBest
# 3. RFECV
# 4. SelectFromModel

# #### 6.7.1.1 VarianceThreshold method

# In[204]:


threshold = np.arange(1, 10, 0.5) *1e-1


# In[205]:


scores = []
for i in threshold:
    selector = sklearn.feature_selection.VarianceThreshold(threshold= i)
    selected_features = selector.fit_transform(features)
    bg.fit(selected_features, target)
    y_pred = bg.predict(features.loc[:, selector.get_support()])
    scores.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot variance threshold VS. cross-validated scores for training sets.
plt.figure()
plt.xlabel("variance threshold")
plt.ylabel("Cross validated accuracy score")
plt.plot(threshold, np.array(scores))


# In[206]:


print(('The highest accuracy score is obtained after execluding features whose variance is less than: ', 
              np.round(threshold[np.argmax(np.array(scores))],3)))


# In[207]:


print(('The highest accuracy score is:', np.max(np.array(scores))))


# #### 6.7.1.2 SelectKbest method

# In[208]:


number_of_features = list(range(1,13))


# In[209]:


scores_k = []
for i in number_of_features:
    selector = sklearn.feature_selection.SelectKBest(k=i)
    selected_features = selector.fit_transform(features, target)
    bg.fit(selected_features, target)
    y_pred = bg.predict(features.loc[:, selector.get_support()])
    scores_k.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot number of selected features VS. cross-validated scores for training sets.
plt.figure()
plt.xlabel("Number of Selected Features")
plt.ylabel("Cross validated accuracy score")    
plt.plot(number_of_features, scores_k)


# In[210]:


print(("Maximum accuracy score is :", max(scores_k)))


# In[211]:


print(("Optimal number of features :", np.argmax(np.array(scores_k)) + 1))


# #### 6.7.1.3 RFECV method

# This method can't be used with Bagging Classifier algorithm because this classifier does not expose 'coef_' or 'feature_importances_' attributes.

# #### 6.7.1.4 SelectFromModel method

# This method can't be used with Bagging Classifier algorithm because this classifier does not expose 'coef_' or 'feature_importances_' attributes.

# We conclude the highest accuracy score can be obtained while using the whole 12 features for prediction. So we will use all features to the model.

# ### 6.7.2 Bagging hyperparamters tunning

# We will use randomized search method and we will follow coarse-to-fine strategy.

# In[212]:


bg_params = {'n_estimators': [20, 25, 100], 'base_estimator': [
    None, svm], 'max_features': [0.6, 0.7, 0.8], 'oob_score' : [True, False], 
            'max_samples': [0.6,0.7,0.8]}


# In[213]:


rs_bg = RandomizedSearchCV(bg, param_distributions= bg_params,
                     scoring='accuracy', cv=StratifiedKFold(7), n_iter= 2000,refit=True)


# In[214]:


rs_bg.fit(x_train, y_train)


# In[215]:


print(('Best Parameters are:\n', rs_bg.best_params_,
      '\nTraining accuracy score is:\n', rs_bg.best_score_))


# In[216]:


print(('Validation accuracy score is:\n', rs_bg.score(x_valid, y_valid)))


# ### 6.7.3 Model Assessment

# Fit the model with best parameters that minimize both bias and variance.

# In[217]:


bg = ens.BaggingClassifier(n_estimators= 25,
        max_features= 0.8, base_estimator= svm, oob_score= True, max_samples= 0.8)
bg.fit(features,target)


# In[218]:


# Finding the ROC curve for different threshold values.
# probability estimates of the positive class.
y_scores_bg = bg.predict_proba(x_test)[:, 1]
bg_fpr, bg_tpr, bg_thresholds = sklearn.metrics.roc_curve(y_test, y_scores_bg)
# Finding the AUC for the Decision Tree classification model.
bg_auc = sklearn.metrics.auc(x=bg_fpr, y=bg_tpr)


# In[219]:


bg_acc = bg.score(x_test, y_test)


# In[220]:


print(('Area Under Curve: {}, Accuracy: {}'.format(bg_auc, bg_acc)))


# ## 6.8 Adaboost Classifier

# In[221]:


ada = ens.AdaBoostClassifier()


# ### 6.8.1 Feature selection

# Here, we will try different method to select the features with the highest explainatory power. We will try the following methods, then we select the best method:
# 1. VarianceThreshold
# 2. SelectKBest
# 3. RFECV
# 4. SelectFromModel

# #### 6.8.1.1 VarianceThreshold method

# In[222]:


threshold = np.arange(1, 10, 0.5) *1e-1


# In[223]:


scores = []
for i in threshold:
    selector = sklearn.feature_selection.VarianceThreshold(threshold= i)
    selected_features = selector.fit_transform(features)
    ada.fit(selected_features, target)
    y_pred = ada.predict(features.loc[:, selector.get_support()])
    scores.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot variance threshold VS. cross-validated scores for training sets.
plt.figure()
plt.xlabel("variance threshold")
plt.ylabel("Cross validated accuracy score")
plt.plot(threshold, np.array(scores))


# In[224]:


print(('The highest accuracy score is obtained after execluding features whose variance is less than: ', 
              np.round(threshold[np.argmax(np.array(scores))],3)))


# In[225]:


print(('The highest accuracy score is:', np.max(np.array(scores))))


# #### 6.8.1.2 SelectKbest method

# In[226]:


number_of_features = list(range(1,13))


# In[227]:


scores_k = []
for i in number_of_features:
    selector = sklearn.feature_selection.SelectKBest(k=i)
    selected_features = selector.fit_transform(features, target)
    ada.fit(selected_features, target)
    y_pred = ada.predict(features.loc[:, selector.get_support()])
    scores_k.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot number of selected features VS. cross-validated scores for training sets.
plt.figure()
plt.xlabel("Number of Selected Features")
plt.ylabel("Cross validated accuracy score")    
plt.plot(number_of_features, scores_k)


# In[228]:


print(("Maximum accuracy score is :", max(scores_k)))


# In[229]:


print(("Optimal number of features :", np.argmax(np.array(scores_k)) + 1))


# #### 6.8.1.3 RFECV method

# In[230]:


selector = sklearn.feature_selection.RFECV(ada, step= 1, cv= 5)
selector.fit(features, target)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(list(range(1, len(selector.grid_scores_) + 1)), selector.grid_scores_)


# In[231]:


print(("Optimal number of features : %d" % selector.n_features_))


# In[232]:


print(("Maximum accuracy score is :", np.max(selector.grid_scores_)))


# #### 6.8.1.4 SelectFromModel method

# In[233]:


threshold = [0.001, 0.0025, 0.005, 0.01, 0.025 ,0.05, 0.1, 0.15]


# In[234]:


scores_sfm = []
for i in threshold:
    selector = sklearn.feature_selection.SelectFromModel(ada, threshold= i)
    selector.fit(features, target)
    selected_features = features.loc[:, selector.get_support()]
    ada.fit(selected_features, target)
    y_pred = ada.predict(selected_features)
    scores_sfm.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Threshold Value")
plt.ylabel("Cross validation score")    
plt.plot(threshold, scores_sfm)


# In[235]:


print(("Maximum accuracy score is :", np.max(np.array(scores_sfm))))


# In[236]:


print(("Optimal threshold :", threshold[np.argmax(np.array(scores_sfm))]))


# We conclude the highest accuracy score can be also obtained while using the whole 12 features for prediction. So we will use all features to the model.

# ### 6.8.2 Adaboost hyperparamters tunning

# We will use randomized search method and we will follow coarse-to-fine strategy.

# In[237]:


ada_params = {'n_estimators': [90, 100, 110], 'base_estimator': [None, svm],
             'learning_rate': [0.09 ,0.1, 0.11]}


# In[238]:


rs_ada = RandomizedSearchCV(ada, param_distributions= ada_params,
                     scoring='accuracy', cv= StratifiedKFold(7), refit=True, n_iter= 500)


# In[239]:


rs_ada.fit(x_train, y_train)


# In[240]:


print(('Best Parameters are:\n', rs_ada.best_params_,
      '\nTraining accuracy score is:\n', rs_ada.best_score_))


# In[241]:


print(('Validation accuracy score is:\n', rs_ada.score(x_valid, y_valid)))


# ### 6.8.3 Model Assessment

# Fit the model with best parameters that minimize both bias and variance.

# In[242]:


ada = ens.AdaBoostClassifier(n_estimators= 110, learning_rate= 0.09)
ada.fit(features, target)


# In[243]:


# Finding the ROC curve for different threshold values.
# probability estimates of the positive class.
y_scores_ada = ada.predict_proba(x_test)[:, 1]
ada_fpr, ada_tpr, ada_thresholds = sklearn.metrics.roc_curve(y_test, y_scores_ada)
# Finding the AUC for the Decision Tree classification model.
ada_auc = sklearn.metrics.auc(x=ada_fpr, y=ada_tpr)


# In[244]:


ada_acc = ada.score(x_test, y_test)


# In[245]:


print(('Area Under Curve: {}, Accuracy: {}'.format(ada_auc, ada_acc)))


# ## 6.9 Gradient Boost Classifier

# ### 6.9.1 Feature selection

# Here, we will try different method to select the features with the highest explainatory power. We will try the following methods, then we select the best method:
# 1. VarianceThreshold
# 2. SelectKBest
# 3. RFECV
# 4. SelectFromModel

# In[246]:


gb = ens.GradientBoostingClassifier()


# #### 6.9.1.1 VarianceThreshold method

# In[247]:


threshold = np.arange(1, 10, 0.5) *1e-1


# In[248]:


scores = []
for i in threshold:
    selector = sklearn.feature_selection.VarianceThreshold(threshold= i)
    selected_features = selector.fit_transform(features)
    gb.fit(selected_features, target)
    y_pred = gb.predict(features.loc[:, selector.get_support()])
    scores.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot variance threshold VS. cross-validated scores for training sets.
plt.figure()
plt.xlabel("variance threshold")
plt.ylabel("Cross validated accuracy score")
plt.plot(threshold, np.array(scores))


# In[249]:


print(('The highest accuracy score is obtained after execluding features whose variance is less than: ', 
              np.round(threshold[np.argmax(np.array(scores))],3)))


# In[250]:


print(('The highest accuracy score is:', np.max(np.array(scores))))


# #### 6.9.1.2 SelectKbest method

# In[251]:


number_of_features = list(range(1,13))


# In[252]:


scores_k = []
for i in number_of_features:
    selector = sklearn.feature_selection.SelectKBest(k=i)
    selected_features = selector.fit_transform(features, target)
    gb.fit(selected_features, target)
    y_pred = gb.predict(features.loc[:, selector.get_support()])
    scores_k.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot number of selected features VS. cross-validated scores for training sets.
plt.figure()
plt.xlabel("Number of Selected Features")
plt.ylabel("Cross validated accuracy score")    
plt.plot(number_of_features, scores_k)


# In[253]:


print(("Maximum accuracy score is :", max(scores_k)))


# In[254]:


print(("Optimal number of features :", np.argmax(np.array(scores_k)) + 1))


# #### 6.9.1.3 RFECV method

# In[255]:


selector = sklearn.feature_selection.RFECV(gb, step= 1, cv= 5)
selector.fit(features, target)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(list(range(1, len(selector.grid_scores_) + 1)), selector.grid_scores_)


# In[256]:


print(("Optimal number of features : %d" % selector.n_features_))


# In[257]:


print(("Maximum accuracy score is :", np.max(selector.grid_scores_)))


# #### 6.9.1.4 SelectFromModel method

# In[258]:


threshold = [0.001, 0.0025, 0.005, 0.01, 0.025 ,0.05, 0.1, 0.15]


# In[259]:


scores_sfm = []
for i in threshold:
    selector = sklearn.feature_selection.SelectFromModel(gb, threshold= i)
    selector.fit(features, target)
    selected_features = features.loc[:, selector.get_support()]
    gb.fit(selected_features, target)
    y_pred = gb.predict(selected_features)
    scores_sfm.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Threshold Value")
plt.ylabel("Cross validation score")    
plt.plot(threshold, scores_sfm)


# In[260]:


print(("Maximum accuracy score is :", np.max(np.array(scores_sfm))))


# In[261]:


print(("Optimal threshold :", threshold[np.argmax(np.array(scores_sfm))]))


# We conclude that SelectKBest method results in the highest accuracy score when K = 11 features.

# In[262]:


# Fit the model with features selected by SelectFromModel method and the training set
selector = sklearn.feature_selection.SelectKBest(k= 11)
selector.fit(features, target)
gb_selected_features = selector.get_support()


# ### 6.9.2 Gradient Boost hyperparamters tunning

# We will use randomized search method and we will follow coarse-to-fine strategy.

# In[263]:


gb_params = {'n_estimators': [150, 160, 170], 'loss': ['deviance', 'exponential'],
             'subsample': [0.7, 0.8, 0.9], 'max_features': ['auto', 'log2', None]}


# In[264]:


rs_gb = RandomizedSearchCV(gb, param_distributions= gb_params,
                     scoring='accuracy', cv= StratifiedKFold(7), refit=True, n_iter= 2000)


# In[265]:


rs_gb.fit(x_train.loc[:,gb_selected_features], y_train)


# In[266]:


print(('Best Parameters are:\n', rs_gb.best_params_,
      '\nTraining accuracy score is:\n', rs_gb.best_score_))


# In[267]:


print(('Validation accuracy score is:\n',
      rs_gb.score(x_valid.loc[:,gb_selected_features], y_valid)))


# ### 6.9.3 Variance check

# We will use the max_depth of tree to minimize the variance. We'll use the validation curve to select the best value of max_depth.

# In[268]:


param_name = 'max_depth'
param_range = np.arange(1, 31)
train_score, valid_score = [], []
for depth in param_range:
    gb = ens.GradientBoostingClassifier(n_estimators= 170,
        subsample= 0.9, max_features= 'auto', loss= 'exponential',max_depth= depth)
    gb.fit(x_train.loc[:,gb_selected_features], y_train)
    train_score.append(gb.score(x_train.loc[:,gb_selected_features], y_train))
    valid_score.append(gb.score(x_valid.loc[:,gb_selected_features], y_valid))


# In[269]:


# Plot Regularization factor VS. cross-validated scores for training and Validation sets.
plt.figure()
plt.xlabel("Maximum Depth")
plt.ylabel("Cross validated accuracy score")
plt.plot(param_range, train_score, color = 'blue')
plt.plot(param_range, valid_score, color = 'red')


# In[270]:


train_test_diff = np.abs(np.array(train_score) - np.array(valid_score))

# Plot number of folds VS. difference of cross-validated scores between train and Dev sets.
plt.figure()
plt.xlabel("Maximum depth")
plt.ylabel("Diff. Cross validated accuracy score")
plt.plot(param_range, train_test_diff)


# From the above graphs, we find that the maximum validation accuracy score is at max_depth of 1. but, we will choose max_depth of 4 because the validation accuracy score is good and the variance is relatively small.

# ### 6.9.4 Model Assessment

# Fit the model with best parameters that minimize both bias and variance.

# In[271]:


gb = ens.GradientBoostingClassifier(n_estimators= 170, subsample= 0.9, max_features= 'auto',
                                    loss= 'exponential',max_depth= 4)
gb.fit(features.loc[:, gb_selected_features],target)


# In[272]:


# Finding the ROC curve for different threshold values.
# probability estimates of the positive class.
y_scores_gb = gb.predict_proba(x_test.loc[:, gb_selected_features])[:, 1]
gb_fpr, gb_tpr, gb_thresholds = sklearn.metrics.roc_curve(y_test, y_scores_gb)
# Finding the AUC for the Decision Tree classification model.
gb_auc = sklearn.metrics.auc(x=gb_fpr, y=gb_tpr)


# In[273]:


gb_acc = gb.score(x_test.loc[:, gb_selected_features], y_test)


# In[274]:


print(('Area Under Curve: {}, Accuracy: {}'.format(gb_auc, gb_acc)))


# ## 6.10 XGBoost Classifier

# ### 6.10.1 Feature selection for XGBoost

# In[275]:


xgboost = xgb.XGBClassifier()


# #### 6.10.1.1 VarianceThreshold method

# In[276]:


threshold = np.arange(1, 10, 0.5) *1e-2


# In[277]:


scores = []
for i in threshold:
    selector = sklearn.feature_selection.VarianceThreshold(threshold= i)
    selected_features = selector.fit_transform(features)
    xgboost.fit(selected_features, target)
    y_pred = xgboost.predict(selected_features)
    scores.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot variance threshold VS. cross-validated scores for training sets.
plt.figure()
plt.xlabel("variance threshold")
plt.ylabel("Cross validated accuracy score")
plt.plot(threshold, np.array(scores))


# In[278]:


print(('The highest accuracy score is obtained after execluding features whose variance is less than: ', 
              np.round(threshold[np.argmax(np.array(scores))],3)))


# In[279]:


print(('The highest accuracy score is:', np.max(np.array(scores))))


# #### 6.10.1.2 SelectKbest method

# In[280]:


number_of_features = list(range(1,13))


# In[281]:


scores_k = []
for i in number_of_features:
    selector = sklearn.feature_selection.SelectKBest(k=i)
    selected_features = selector.fit_transform(features, target)
    xgboost.fit(selected_features, target)
    y_pred = xgboost.predict(selected_features)
    scores_k.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot number of selected features VS. cross-validated scores for training sets.
plt.figure()
plt.xlabel("Number of Selected Features")
plt.ylabel("Cross validated accuracy score")    
plt.plot(number_of_features, scores_k)


# In[282]:


print(("Maximum accuracy score is :", max(scores_k)))


# In[283]:


print(("Optimal number of features :", np.argmax(np.array(scores_k)) + 1))


# #### 6.10.1.3 RFECV method

# In[284]:


selector = sklearn.feature_selection.RFECV(xgboost, step= 1, cv= 5)
selector.fit(features, target)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(list(range(1, len(selector.grid_scores_) + 1)), selector.grid_scores_)


# In[285]:


print(("Optimal number of features : %d" % selector.n_features_))


# In[286]:


print(("Maximum accuracy score is :", np.max(selector.grid_scores_)))


# #### 6.10.1.4 SelectFromModel method

# In[287]:


threshold = [0.001, 0.0025, 0.005, 0.01, 0.025 ,0.05, 0.1, 0.15]


# In[288]:


scores_sfm = []
for i in threshold:
    selector = sklearn.feature_selection.SelectFromModel(xgboost, threshold= i)
    selector.fit(features, target)
    selected_features = features.loc[:, selector.get_support()]
    xgboost.fit(selected_features, target)
    y_pred = xgboost.predict(selected_features)
    scores_sfm.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Threshold Value")
plt.ylabel("Cross validation score")    
plt.plot(threshold, scores_sfm)


# In[289]:


print(("Maximum accuracy score is :", np.max(np.array(scores_sfm))))


# In[290]:


print(("Optimal threshold :", threshold[np.argmax(np.array(scores_sfm))]))


# We conclude the highest accuracy score can be also obtained while using the whole 12 features for prediction. So we will use all features to the model.

# ### 6.10.2 XGBoost hyperparameter optimization

# By experimentation, we found that the best hyperparameters for the XGBoost classifiers are the default.

# ### 6.10.3 Model Assessment

# Fit the model with best parameters that minimize both bias and variance.

# In[291]:


xgboost = xgb.XGBClassifier()
xgboost.fit(features, target)


# In[292]:


# Finding the ROC curve for different threshold values.
# probability estimates of the positive class.
y_scores_xgb = xgboost.predict_proba(x_test)[:, 1]
xgb_fpr, xgb_tpr, xgb_thresholds = sklearn.metrics.roc_curve(y_test, y_scores_xgb)
# Finding the AUC for the Decision Tree classification model.
xgb_auc = sklearn.metrics.auc(x=xgb_fpr, y=xgb_tpr)


# In[293]:


xgb_acc = xgboost.score(x_test, y_test)


# In[294]:


print(('Area Under Curve: {}, Accuracy: {}'.format(xgb_auc, xgb_acc)))


# ## 6.11 LightGBM Classifier

# ### 6.11.1 Feature selection for LightGBM

# In[295]:


lgboost = lgb.LGBMClassifier()


# #### 6.11.1.1 VarianceThreshold method

# In[296]:


threshold = [0.001, 0.01,0.1,0.5]


# In[297]:


scores = []
for i in threshold:
    selector = sklearn.feature_selection.VarianceThreshold(threshold= i)
    selected_features = selector.fit_transform(features)
    lgboost.fit(selected_features, target)
    y_pred = lgboost.predict(selected_features)
    scores.append(sklearn.metrics.accuracy_score(target, y_pred))
plt.plot([0.001, 0.01,0.1,0.5], np.array(scores))


# In[298]:


np.max(np.array(scores))


# The highest accuracy is obtained after execluding features whose variance is less than 0.001

# #### 6.11.1.2 SelectKbest method

# In[299]:


number_of_features = list(range(1,13))


# In[300]:


scores_k = []
for i in number_of_features:
    selector = sklearn.feature_selection.SelectKBest(k=i)
    selected_features = selector.fit_transform(features, target)
    lgboost.fit(selected_features, target)
    y_pred = lgboost.predict(selected_features)
    scores_k.append(sklearn.metrics.accuracy_score(target, y_pred))
plt.plot(list(range(1,13)), scores_k)


# In[301]:


max(scores_k)


# In[302]:


print(("Optimal number of features :", np.argmax(np.array(scores_k)) + 1))


# The highest accuracy score is obtained after selecting the best 11 features.

# #### 6.11.1.3 RFECV method

# In[303]:


selector = sklearn.feature_selection.RFECV(lgboost, step= 1, cv= 7)
selector.fit(features, target)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(list(range(1, len(selector.grid_scores_) + 1)), selector.grid_scores_)


# In[304]:


print(("Optimal number of features : %d" % selector.n_features_))


# In[305]:


np.max(selector.grid_scores_)


# #### 6.11.1.4 SelectFromModel method

# In[306]:


threshold = [0.001, 0.01, 0.05, 0.1 , 0.5]


# In[307]:


scores_sfm = []
for i in threshold:
    selector = sklearn.feature_selection.SelectFromModel(lgboost, threshold= i)
    selector.fit(features, target)
    selected_features = features.loc[:, selector.get_support()]
    lgboost.fit(selected_features, target)
    y_pred = lgboost.predict(selected_features)
    scores_sfm.append(sklearn.metrics.accuracy_score(target, y_pred))

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Threshold Value")
plt.ylabel("Cross validation score")    
plt.plot([0.001, 0.01, 0.05, 0.1 , 0.5], scores_sfm)


# In[308]:


print(("Maximum accuracy score is :", np.max(np.array(scores_sfm))))


# In[309]:


print(("Optimal threshold :", threshold[np.argmax(np.array(scores_sfm))]))


# We conclude that SelectKBest method results in the highest accuracy score with K = 11.

# In[310]:


# Fit the model with the best 11 features selected.
selector = sklearn.feature_selection.SelectKBest(k= 11)
selector.fit(features, target)
lgb_selected_features = selector.get_support()


# ### 6.11.2 Model Assessment

# Fit the model with best parameters that minimize both bias and variance.

# In[311]:


lgboost = lgb.LGBMClassifier()
lgboost.fit(features.loc[:,lgb_selected_features], target)


# In[312]:


# Finding the ROC curve for different threshold values.
# probability estimates of the positive class.
y_scores_lgb = lgboost.predict_proba(x_test.loc[:,lgb_selected_features])[:, 1]
lgb_fpr, lgb_tpr, lgb_thresholds = sklearn.metrics.roc_curve(y_test, y_scores_lgb)
# Finding the AUC for the Decision Tree classification model.
lgb_auc = sklearn.metrics.auc(x=lgb_fpr, y=lgb_tpr)


# In[313]:


lgb_acc = lgboost.score(x_test.loc[:,lgb_selected_features], y_test)


# In[314]:


print(('Area Under Curve: {}, Accuracy: {}'.format(lgb_auc, lgb_acc)))


# ## 6.12 Voting Classifier

# In[315]:


v = ens.VotingClassifier(estimators=[
    ('lr', lr),('NB', nb),('KNN', knn),('SVM', svm),('DT', dt),
    ('RF', rf), ('BG', bg),('AdaBoost', ada),('GBM', gb),
    ('XGBM', xgboost),('LightGBM', lgboost)], 
                         voting='soft', 
                         weights= [1,1,1, 1.25, 1.25, 1.25, 1.25, 1.25, 1.75, 1.5, 1.5])


# ### 6.12.1 Feature selection for Voting Classifier

# We've noticed that, the SelectKBest method with K = 11 works very well with most of classifiers. So, we will use this method with voting classifier.

# In[316]:


# Fit the model with the best 11 features selected.
selector = sklearn.feature_selection.SelectKBest(k= 11)
selector.fit(features, target)
voting_selected_features = selector.get_support()


# ### 6.12.2 Model Assessment

# Fit the model with best parameters that minimize both bias and variance.

# In[317]:


v.fit(features.loc[:, voting_selected_features], target)


# In[318]:


# Finding the ROC curve for different threshold values.
# probability estimates of the positive class.
y_scores_v = v.predict_proba(features.loc[:, voting_selected_features])[:, 1]
v_fpr, v_tpr, v_thresholds = sklearn.metrics.roc_curve(target, y_scores_v)
# Finding the AUC for the Voting classification model.
v_auc = sklearn.metrics.auc(x=v_fpr, y=v_tpr)


# In[319]:


v_acc = v.score(x_test.loc[:,voting_selected_features], y_test)


# In[320]:


print(('Area Under Curve: {}, Accuracy: {}'.format(v_auc, v_acc)))


# # 7. Models Comaprison

# ## 7.1 Models score

# In[321]:


pd.DataFrame([(lr_auc, lr_acc), (nb_auc, nb_acc), (knn_auc, knn_acc), (dt_auc, dt_acc),
              (rf_auc, rf_acc), (svm_auc, svm_acc), (bg_auc, bg_acc), (ada_auc, ada_acc),
              (v_auc, v_acc), (gb_auc, gb_acc), (xgb_auc, xgb_acc), (lgb_auc, lgb_acc)],
             columns=['AUC', 'Accuracy'],
             index=['Logistic Regression', 'Naive Bayes', 'KNN', 'Decision Tree',
                    'Random Forest', 'SVM', 'Bagging', 'AdaBoost', 'Voting',
                   'Gradient Boost', 'XGBoost', 'Light Boost'])


# ## 7.2 Plotting the ROC curve

# In[322]:


plt.figure(figsize=(8, 5))
plt.title('Receiver Operating Characteristic Curve')
plt.plot(lr_fpr, lr_tpr, label='LR_AUC = %0.2f' % lr_auc)
plt.plot(nb_fpr, nb_tpr, label='NB_AUC = %0.2f' % nb_auc)
plt.plot(knn_fpr, knn_tpr, label='KNN_AUC = %0.2f' % knn_auc)
plt.plot(svm_fpr, svm_tpr, label='SVM_AUC = %0.2f' % svm_auc)
plt.plot(dt_fpr, dt_tpr, label='DT_AUC = %0.2f' % dt_auc)
plt.plot(rf_fpr, rf_tpr, label='RF_AUC = %0.2f' % rf_auc)
plt.plot(bg_fpr, bg_tpr, label='BG_AUC = %0.2f' % bg_auc)
plt.plot(ada_fpr, ada_tpr, label='Ada_AUC = %0.2f' % ada_auc)
plt.plot(v_fpr, v_tpr, label='Voting_AUC = %0.2f' % v_auc)
plt.plot(lgb_fpr, lgb_tpr, label='LBoost_AUC = %0.2f' % lgb_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve')
plt.show()


# Make Prediction for test data

# In[323]:


y_pred_v = pd.DataFrame(v.predict(unlabelled.loc[:, voting_selected_features]), columns=[
                        'Survived'], dtype='int64')


# In[324]:


v_model = pd.concat([passengerID, y_pred_v], axis=1)


# In[325]:


v_model.to_csv('voting.csv', index= False)

