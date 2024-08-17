#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# Any results you write to the current directory are saved as output.


# In[2]:


import pandas as pd
df = pd.read_csv("../input/weather-dataset-rattle-package/weatherAUS.csv")
df.head()


# In[3]:


df.shape


# In[4]:


col_names = df.columns
col_names


# In[5]:


df.drop(['RISK_MM'], axis = 1, inplace = True)


# In[6]:


df.info()


# In[7]:


# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print(('There are {} categorical variables\n'.format(len(categorical))))

print(('The categorical variables are :', categorical))


# In[8]:


df[categorical].head()


# In[9]:


df[categorical].isnull().sum()


# In[10]:


cat1 = [var for var in categorical if df[var].isnull().sum()!=0]

print((df[cat1].isnull().sum()))


# In[11]:


# view frequency of categorical variables

for var in categorical: 
    
    print((df[var].value_counts()))


# In[12]:


# view frequency distribution of categorical variables

for var in categorical:
    
    print((df[var].value_counts()/np.float(len(df))))
    


# In[13]:


for var in categorical:
    
    print((var, 'contains' , len(df[var].unique()),'labels'))


# In[14]:


df['Date'].dtypes


# In[15]:


# parse the dates, currently coded as strings, into datetime format

df['Date'] = pd.to_datetime(df['Date'])


# In[16]:


df['Year'] = df['Date'].dt.year
df['Year'].head()


# In[17]:


df['Month'] = df['Date'].dt.month
df['Month'].head()


# In[18]:


df['Day'] = df['Date'].dt.day

df['Day'].head()


# In[19]:


df.info()


# In[20]:


df.drop('Date', axis = 1, inplace = True)


# In[21]:


df.head()


# In[22]:


# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print(('There are {} categorical variables\n'.format(len(categorical))))

print(('The categorical variables are :', categorical))


# In[23]:


# check for missing values in categorical variables
df[categorical].isnull().sum()


# In[24]:


#print the number of location variables

print(('Location counts',len(df.Location.unique()), 'labels'))


# In[25]:


df.Location.unique()


# In[26]:


df.Location.value_counts()


# In[27]:


pd.get_dummies(df.Location, drop_first =True).head()


# In[28]:


print(('WindGustDir contains', len(df['WindGustDir'].unique()), 'labels'))


# In[29]:


df['WindGustDir'].unique()


# In[30]:


df.WindGustDir.value_counts()


# In[31]:


pd.get_dummies(df.WindGustDir, drop_first = True, dummy_na = True).head()


# In[32]:


pd.get_dummies(df.WindGustDir, drop_first = True, dummy_na = True).sum(axis=0)


# In[33]:


print(('WindDir9am contains', len(df['WindDir9am'].unique()), 'labels'))


# In[34]:


df.WindDir9am.unique()


# In[35]:


df.WindDir9am.value_counts()


# In[36]:


pd.get_dummies(df.WindDir9am, drop_first =True, dummy_na = True).head()


# In[37]:


pd.get_dummies(df.WindDir9am, drop_first = True, dummy_na = True).sum(axis=0)


# In[38]:


print(('WindDir3pm contains', len(df['WindDir9am'].unique()), 'labels'))


# In[39]:


df.WindDir3pm.unique()


# In[40]:


df.WindDir3pm.value_counts()


# In[41]:


pd.get_dummies(df.WindDir3pm, drop_first =True, dummy_na = True).head()


# In[42]:


pd.get_dummies(df.WindDir3pm, drop_first = True, dummy_na = True).sum(axis=0)


# In[43]:


print(('RainToday contains', len(df['RainToday'].unique()), 'labels'))


# In[44]:


df['RainToday'].unique()


# In[45]:


df.RainToday.value_counts()


# In[46]:


pd.get_dummies(df.RainToday, drop_first =True, dummy_na = True).head()


# In[47]:


pd.get_dummies(df.RainToday, drop_first =True, dummy_na = True).sum(axis=0)


# Explore Numerical Variables
# 

# In[48]:


numerical = [var for var in df.columns if df[var].dtype!='O']

print(('There are {} numerical variables \n'.format(len(numerical))))

print(('The numerical variables are:', numerical))


# In[49]:


df[numerical].head()


# In[50]:


# check missing values in numerical variables

df[numerical].isnull().sum()


# Outliers in numerical variables
# 

# In[51]:


# view summary statistics in numerical variables

print((round(df[numerical].describe()),2))


# In[52]:


# draw boxplots to visualize outliers
import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
fig = df.boxplot(column = 'Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')


plt.subplot(2,2,2)
fig = df.boxplot(column = 'Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')

plt.subplot(2,2,3)
fig = df.boxplot(column = 'WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')

plt.subplot(2,2,4)
fig = df.boxplot(column = 'WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')


# In[53]:


# plot histogram to check distribution

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('Rain Tomorrow')

plt.subplot(2,2,2)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('Rain Tommorow')

plt.subplot(2,2,3)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('Rain Tommorow')

plt.subplot(2,2,4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('Rain Tommorow')


# In[54]:


# find outliers for Rainfall variable
IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)
print(('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence)))


# In[55]:


IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
Lower_fence = df.Evaporation.quantile(0.25) - (IQR*3)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR*3)
print(('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary = Lower_fence, upperboundary = Upper_fence)))


# In[56]:


IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR*3)
Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR*3)
print(('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary = Lower_fence, upperboundary = Upper_fence)))


# In[57]:


IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR*3)
Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR*3)
print(('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary = Lower_fence, upperboundary = Upper_fence)))


# Declare feature vector and target variable 
# 

# In[58]:


X = df.drop(['RainTomorrow'], axis=1)

y = df['RainTomorrow']


# In[59]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[60]:


X_train.shape, X_test.shape


# In[61]:


X_train.dtypes


# In[62]:


# display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical


# In[63]:


numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical


# In[64]:


# check missing values in numerical variables in X_train

X_train[numerical].isnull().sum()


# In[65]:


X_test[numerical].isnull().sum()


# In[66]:


for col in numerical:
    if X_train[col].isnull().mean()>0:
        print((col, round(X_train[col].isnull().mean(), 4)))


# In[67]:


# impute missing values in X_train and X_test with respective column median in X_train

for df1 in [X_train, X_test]:
    for col in numerical:
        col_median=X_train[col].median()
        df1[col].fillna(col_median, inplace=True)  


# In[68]:


# check again missing values in numerical variables in X_train

X_train[numerical].isnull().sum()


# In[69]:


X_test[numerical].isnull().sum()


# In[70]:


X_train[categorical].isnull().mean()


# In[71]:


for col in categorical:
    if X_train[col].isnull().mean()>0:
        print((col, (X_train[col].isnull().mean())))
        


# In[72]:


# impute missing categorical variables with most frequent value

for df2 in [X_train, X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)


# In[73]:


# check missing values in categorical variables in X_train

X_train[categorical].isnull().sum()


# In[74]:


X_test[categorical].isnull().sum()


# In[75]:


X_train.isnull().sum()


# In[76]:


X_test.isnull().sum()


# In[77]:


def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3,'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)


# In[78]:


X_train.Rainfall.max(), X_test.Rainfall.max()


# In[79]:


X_train.Evaporation.max(), X_test.Evaporation.max()


# In[80]:


X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max()


# In[81]:


X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max()


# In[82]:


X_train[numerical].describe()


# In[83]:


categorical


# In[84]:


X_train[categorical].head()


# In[85]:


import category_encoders as ce

encoder = ce.BinaryEncoder(cols = ['RainToday'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)


# In[86]:


X_train.head()


# In[87]:


X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location), 
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)], axis=1)


# In[88]:


X_train.head()


# In[89]:


X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_test.Location), 
                     pd.get_dummies(X_test.WindGustDir),
                     pd.get_dummies(X_test.WindDir9am),
                     pd.get_dummies(X_test.WindDir3pm)], axis=1)


# In[90]:


X_test.head()


# Feature Scaling

# In[91]:


X_train.describe()


# In[92]:


cols = X_train.columns


# In[93]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)


# In[94]:


X_train = pd.DataFrame(X_train, columns = [cols])



# In[95]:


X_test = pd.DataFrame(X_test, columns = [cols])


# In[96]:


X_train.describe()


# Model training

# In[97]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver = 'liblinear', random_state = 0)

logreg.fit(X_train, y_train)


# Predict results

# In[98]:


y_pred_test = logreg.predict(X_test)

y_pred_test


# In[99]:


logreg.predict_proba(X_test)[:,0]


# In[100]:


logreg.predict_proba(X_test)[:,1]


# Check accuracy score

# In[101]:


from sklearn.metrics import accuracy_score

print(('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred_test))))


# In[102]:


y_pred_train = logreg.predict(X_train)

y_pred_train


# In[103]:


print(('Training set accuracy score: {0:0.4f}'.format(accuracy_score(y_train, y_pred_train))))


# In[104]:


# print the scores on training and test set

print(('Training set accuracy score: {:.4f}'.format(accuracy_score(y_train, y_pred_train))))

print(('Test set score: {:.4f}'.format(logreg.score(X_test, y_test))))


# In[105]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

print(('Confusion matrix\n\n', cm))

print(('\nTrue Positives(TP) = ', cm[0,0]))

print(('\nTrue Negatives(TN) = ', cm[1,1]))

print(('\nFalse Positives(FP) = ', cm[0,1]))

print(('\nFalse Negatives(FN) = ', cm[1,0]))


# In[106]:


import seaborn as sns
sns.heatmap(cm, annot=True)


# In[107]:


sns.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.2%', cmap='Blues')


# In[108]:


from sklearn.metrics import classification_report

print((classification_report(y_test, y_pred_test)))


# Classification accuracy

# In[109]:


TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]


# In[110]:


classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print(('Classification accuracy:{0:0.4f}'.format(classification_accuracy)))


# In[111]:


classification_error = (FP + FN) / float(TP + TN + FP + FN)

print(('Classification error:{0:0.4f}'.format(classification_error)))


# Adjusting the threshold level

# In[112]:


y_pred_prob = logreg.predict_proba(X_test)[0:10]

y_pred_prob


# In[113]:


# store the probabilities in dataframe

y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - No rain tomorrow (0)', 'Prob of - Rain tomorrow (1)'])

y_pred_prob_df


# In[114]:


logreg.predict_proba(X_test)[0:10, 1]


# In[115]:


y_pred1 = logreg.predict_proba(X_test)[:,1]


# In[116]:


# plot histogram of predicted probabilities


# adjust the font size 
plt.rcParams['font.size'] = 12


# plot histogram with 10 bins
plt.hist(y_pred1, bins = 10)


# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of rain')


# set the x-axis limit
plt.xlim(0,1)


# set the title
plt.xlabel('Predicted probabilities of rain')
plt.ylabel('Frequency')


# k-Fold Cross Validation

# In[117]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg, X_train, y_train, cv = 5, scoring = 'accuracy')

print(('Cross validation scores:{}'.format(scores)))


# In[118]:


# compute Average cross-validation score

print(('Cross validation score:{}'.format(scores.mean())))


# Hyperparameter Optimization using GridSearch CV

# In[119]:


from sklearn.model_selection import GridSearchCV

scores = []

parameters = [{'penalty': ['l1','l2']},
             {'C': [0.1 , 1 , 10 , 100, 1000]}]

grid_search = GridSearchCV(estimator = logreg,
                          param_grid = parameters, 
                          scoring = 'accuracy',
                          cv = 5,
                          verbose = 0)

grid_search.fit(X_train, y_train)
scores.append({
        'model':model_name,
        'best_score':grid_search.best_score_,
        'best_params':grid_search.best_params_
    })


# In[ ]:




