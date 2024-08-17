#!/usr/bin/env python
# coding: utf-8

# # Salary Predictions Based on Job Descriptions

# # Part 1 - DEFINE

# ### ---- 1 Define the problem ----

# Data has been gathered from different types of job listings. We have 3 different databases: 
# - train_features, where each row represents metadata for an individual job posting
# - train_salaries, where the job ID matches the one in train_features and assigns it a salary
# - test_features, similar to train_features, we are going to use it to test our model
# 
# The goal of this project is to use all the data we have to most accuretely predict the salary for the job descriptions in the test_features as well as generally predict salary for all types of job descriptions in the real world.

# In[79]:


__author__ = "Said Mrad"
__email__ = "saidmrad98@gmail.com"


# In[226]:


#import libraries
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ## Part 2 - DISCOVER

# ### ---- 2 Load the data ----

# In[81]:


#load the data into a Pandas dataframe
filepath = 'data/train_features.csv'
filepath1 = 'data/train_salaries.csv'
filepath2 = 'data/test_features.csv'
df_raw_train_feature = pd.read_csv(filepath)
df_raw_train_salaries = pd.read_csv(filepath1)
df_raw_test_feature = pd.read_csv(filepath2)


# #### Examine Data

# Examine the first 3 rows of each dataset to get a genreal overview of the training features provided. 

# In[82]:


df_raw_train_feature.head(3)


# For each job posting, we also get the position ID, the job type, the minimum degree required (can be NONE), the major required, the industry, years of experience and miles From Metropolis.
# We can already observe that job descriptions that require no degree or a high school degree for the degree field will not require a specific major which makes sense.
# 
# To observe salary we theorize that all of these features would be able to influence that variable with probably more emphasis on some specific ones like years of experience or job type.

# In[83]:


df_raw_train_salaries.head(3)


# This is the same Job ID as the train features dataset and assigns a salary to it, this means that it would be very easy to merge both tables.

# In[84]:


df_raw_test_feature.head(3)


# Similar dataset to the first one we observed however it does not have an accompanying salary dataset, we will try to predict it as we have the same features here as the training data does.

# ### ---- 3 Clean the data ----

# First we check if any of the data is null, we create a new dataset after applying the isnull() function to each of our datasets. If there is a null value it will appear as True

# In[85]:


missing_data = df_raw_train_feature.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print((missing_data[column].value_counts()))
    print("")


# In[86]:


missing_data = df_raw_train_salaries.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print((missing_data[column].value_counts()))
    print("")


# In[87]:


missing_data1 = df_raw_test_feature.isnull()
for column in missing_data1.columns.values.tolist():
    print(column)
    print((missing_data1[column].value_counts()))
    print("")


# Thus we do not have any null data.

# Here we will use the method .info() it will give us, for each datasets, the length of its columns and its data types.

# In[88]:


df_raw_train_feature.info()


# We do not have null data the data types seems to be appropriate. I want to see the amount of unique values for companyID. We have 1,000,000 x 8 for the shape of the datasets as we have 1,000,000 entries and 8 columns

# In[89]:


df_raw_train_salaries.info()


# The shape here is 1,000,000 x 2 we do not see any missing values

# In[90]:


df_raw_test_feature.info()


# In[91]:


df_raw_train_feature["companyId"].describe()


# We only have 63 unique companies, I thought it would be possible that we had 1 000 000 different companies and thus it would have been more appropriate to change it to a company ID and numerical data but we have 63 companies and it seems like they all have around 15-16k listings as if they all had the same number of listings each company would have (1,000,000/ 63 = 15873 listings) 15973 listings and that is not far from our top frequency 
# 
# Lets examine the jobId column

# In[92]:


df_raw_train_feature["jobId"]


# It seems like they are in chronological order. i want to see if I need to **check for duplicate data** I will check out the column's decribe
# The Id's start from "JOB1362684407687" and finish at "JOB1362685407686" incrementing by 1 for every new job description

# In[93]:


df_raw_train_feature["jobId"].describe()


# 1 000 000 unique values which means there are no duplicate job listings in the training dataset, there can be duplicate job description but it would be for different jobs. We confirm this is the case in all datasets by summing up the amount of duplicates found in each dataset with the duplicated() and sum() methods

# #### Check for duplicates

# In[94]:


df_raw_train_feature.duplicated().sum()


# In[95]:


df_raw_train_salaries.duplicated().sum()


# In[96]:


df_raw_test_feature.duplicated().sum()


# All jobId's are unique so different jobs so no duplicates.

# #### Summarize numerical and categorical variables of training and testing features

# In[97]:


df_raw_train_feature.describe(include = ['O'])


# In[98]:


df_raw_train_feature.describe()


# The categorical values will have the number of unique categories, the most frequent one as well as the frequency in which it is represented. 
# We notice by doing some quick math that most categories have around the same number of entries for the categorical data (except for major) (we divide the coount by the amouont of uniques, the top frequency is usually close to the result we find which leads us to give the assumption above (was already explained for companyID).
# The numerical data, we have the mean the standard deviation and 5 number summary (min, Q1, median, Q3, max)
# We will analyze this further with abalyzing the value counts of each category per categorical data

# In[99]:


df_raw_test_feature.describe(include = 'all')


# We notice that the test data is very similar to the training data. For the categorical data we have the same amount of unique categories per categorical data and for the numerical data the mean the standard deviation and the 5 number summary are all very similar

# In[100]:


df_raw_train_salaries.describe()


# We immediately notice something unusual here with the minimum salary being 0, also it seems like their is a skew to the right as the gap between the 3rd quartile and the max is bigger than any other gap. Also the mean is after the median but slighlty so there would be a very slight skew if anything
# We are going to probably remove some outliers in the salary. Since the salaries dataset shares a common unique identifier in JobId with the training features dataset we will merge them so that the jobId's we remove from one will be removed from the other.
# Also it is best practice before applying a model to it that the training and target (salary) variables are in the same dataset

# In[101]:


#Merge features and the target and examine the first 3 rows of new dataframe
df_train = pd.merge(df_raw_train_feature, df_raw_train_salaries, on='jobId')
df_train.head(3)


# We now have our complete training dataset with all the feature variables as well as our target which is salary. We make sure that all our data is there as well that all the columns data types are intact

# In[102]:


df_train.info()


# #### We now want to find potential outliers in our target variable - salaries

# In[103]:


salary_info = df_train.salary.describe()
salary_info


# As we pointed out above, we have some irregularities as we find jobs with 0 salary. We will use the IQR rule to find potential outliers - IQR = Q3-Q1, the upper bound is 1.5 * IQR + Q3 and the lower bound Q1 - 1.5 * IQR 

# In[104]:


IQR = salary_info["75%"] - salary_info["25%"]
upper_bound = salary_info["75%"] + 1.5 * IQR
lower_bound = salary_info["25%"] - 1.5 * IQR
print(("The upper and lower bounds for suspected outliers are", lower_bound, "and", upper_bound))


# #### Examine outliers

# In[105]:


df_train.loc[df_train.salary < 8.5, "salary"].value_counts()


# There are a total of 5 outliers under the lower bound all of 0, lets investigate them further

# In[106]:


# check potential outlier below lower bound
df_train[df_train.salary < 8.5]


# All the entries below the lower bound are oof 0 salary and they do not appear to be volunteer positions. These are all very likely missing or corrupt data and we are going to remove them from the training set.
# Looking at the 5 number summary for salary, we can say that its distribution will likely have a right skew, thus we assume that the count of upper bound outlier will be big. We will the upper bound outliers by jobType as better job type usually earn more salary.

# In[107]:


df_train.loc[df_train.salary > 222.5, 'jobType'].value_counts()


# Better job type do yield a better salary. We will take a closer look at the Junior category as it would be the lleast likely to earn more than the upper bound.

# In[108]:


# Check most suspicious potential outliers above upper bound
df_train[(df_train.salary > 222.5) & (df_train.jobType == 'JUNIOR')]


# There does not seem to be outliers on the upperbound as all the job description seeking Juniors paying salaries higher than the upper bound are all in high paying industries "Finance", "Oil" with high earning majors "Engineering", "Business" as well high earning degrees "Masters" "Doctoral".
# 
# Thus we will only remove the salaries under the lower bound which are all at 0.

# In[109]:


df_train = df_train[df_train.salary > 8.5]
df_train


# We notice that we now have 999,995 rows as we deleted the 5 columns with 0 salary
# we want to reset the index and then our database will be cleaned

# In[110]:


df_train = df_train.reset_index() #reset index after deleting multiple rows
df_train = df_train.drop(df_train.columns[0], axis = 'columns') #delete index columns dans comes from resetting index
df_train


# ### ---- 4 Explore the data (EDA) ----

# We will have another describe to see the types of the categories too (can also be optained with df_train.dtypes())

# In[111]:


df_train.describe()


# In[112]:


df_train.describe(include = 'O')


# In[113]:


#Analyze numerical features individually
viz = df_train[['yearsExperience','milesFromMetropolis']]
viz.hist()
plt.show()


# For our two numerical features, we notice first at milesFromMetropolis, binned by 10s, that their are as many job descriptions 100k for all the bins from the metropolis while yearExperidnce in bins of 2 ranges from 120k to 80k per bin for each one.
# We will now look at the frequency tables for some of our category variables - jobType, degree, major, industry as company ID as to many unique values

# In[114]:


#Analyze categeroical features individually
#jobtype
jobType_counts = df_train['jobType'].value_counts().to_frame()
jobType_counts.rename(columns={'jobType': 'value_counts'}, inplace=True)
jobType_counts.index.name = 'jobType'
jobType_counts


# In[115]:


#Analyze categeroical features individually
#degree
degree_counts = df_train['degree'].value_counts().to_frame()
degree_counts.rename(columns={'degree': 'value_counts'}, inplace=True)
degree_counts.index.name = 'degree'
degree_counts


# In[116]:


#Analyze categeroical features individually
#major
major_counts = df_train['major'].value_counts().to_frame()
major_counts.rename(columns={'major': 'value_counts'}, inplace=True)
major_counts.index.name = 'major'
major_counts


# In[117]:


industry_counts = df_train['industry'].value_counts().to_frame()
industry_counts.rename(columns={'industry': 'value_counts'}, inplace=True)
industry_counts.index.name = 'industry'
industry_counts


# As previously theorized, categories for our categorical features have around the same value_counts by category for job types around 125k job descriptions per. It is the same for the industry as we have around 143k job describtion per industry. 
# For the other two categories there are some differences but most of the counts are the same per category and will create different groups (the degree category, one group has no degree or only a high school degree requirements(232k each while the other group requires you at minimum to go to college (175k each) -  the first group will not require a major as a result as well for the major requirement and thus that specific category is higher than the other majors that are all at 50k job descriptions per.

# #### Visualize target variable - salary

# In[118]:


sns.boxplot(df_train.salary)


# We can see we that we got rid of all our outliers under the lower bound and we theorize a right skew with the data seeming to elongate to the right

# In[119]:


sns.distplot(df_train.salary, bins=20)


# There is a slight right skew

# #### Analyze feature variable with target variable

# First numerical variables, i want to see hoow the variables intereact with the target variable, we will then get pearsons correlation for both of them. Since they are continuous integers over a range and it is a massive database instead of doing a scatterplot and fit a line through it, i will be plotting their means  and fit a line through that as well as their standars deviations which will be filled in.

# In[120]:


mean = df_train.groupby("milesFromMetropolis")['salary'].mean()
std = df_train.groupby("milesFromMetropolis")['salary'].std()
mean.plot()
plt.fill_between(list(range(len(std.index))), mean.values-std.values, mean.values + std.values, \
                 alpha = 0.1)
plt.xticks(rotation=45)
plt.ylabel('Salaries')
plt.show()


# It seems like it is a moderate/ weak negative correlation. Let's see what the Pearson coefficient is as well as the p-value to see if that correlation is statistically siginificant.

# In[121]:


pearson_coef, p_value = stats.pearsonr(df_train['milesFromMetropolis'], df_train['salary'])
print(("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value))


# Since the p-value is < 0.001, the correlation between milesFromMetropolis and salary is statistically significant, although the linear relationship isn't extremely strong (~-0.297)

# In[122]:


mean = df_train.groupby('yearsExperience')['salary'].mean()
std = df_train.groupby('yearsExperience')['salary'].std()
mean.plot()
plt.fill_between(list(range(len(std.index))), mean.values-std.values, mean.values + std.values, \
                         alpha = 0.1)
plt.xticks(rotation=45)
plt.ylabel('Salaries')
plt.show()


# It seems like it is a positive correlation. 

# In[123]:


pearson_coef, p_value = stats.pearsonr(df_train['yearsExperience'], df_train['salary'])
print(("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value))


# Since the p-value is < 0.001, the correlation between yearsExperience and salary is statistically significant, although the linear relationship isn't extremely strong (~0.375)

# We now do the same for companyId as it has too many unique categories to be observed by category 
# 

# In[124]:


mean = df_train.groupby('companyId')['salary'].mean()
std = df_train.groupby('companyId')['salary'].std()
mean.plot()
plt.fill_between(list(range(len(std.index))), mean.values-std.values, mean.values + std.values, \
                         alpha = 0.1)
plt.xticks(rotation=45)
plt.ylabel('Salaries')
plt.show()


# In[125]:


mean


# It does noot seem to be statistically significant if we look at the means they are all similar so we assume there is a very weak correlation between companyID and salary.

# We can do an analysis of Variance (ANOVA) to confirm that more, since we have 63 unique values doing it with 10 will still yield a statistical significant result

# In[126]:


grouped_test=df_train[['companyId', 'salary']].groupby(['companyId'])
grouped_test.head(2)


# In[127]:


# ANOVA
f_val, p_val = stats.f_oneway(grouped_test.get_group('COMP0')['salary'], 
                              grouped_test.get_group('COMP1')['salary'], 
                              grouped_test.get_group('COMP2')['salary'],
                              grouped_test.get_group('COMP3')['salary'],
                              grouped_test.get_group('COMP4')['salary'],
                              grouped_test.get_group('COMP5')['salary'],
                              grouped_test.get_group('COMP6')['salary'],
                              grouped_test.get_group('COMP7')['salary'],
                              grouped_test.get_group('COMP8')['salary'],
                              grouped_test.get_group('COMP9')['salary'] )  
 
print(( "ANOVA results: F=", f_val, ", P =", p_val))


# The F value being so small as well as the p value being that high means that they are not statcally significant

# Now we look at the categorical features and compare each category to the target variable salary - jobType, degree, major, industry

# ### jobType

# In[128]:


plt.figure(figsize = (14, 6))
mean = df_train.groupby('jobType')['salary'].mean()
df_train['jobType'] = df_train['jobType'].astype('category')
levels = mean.sort_values().index.tolist()
df_train['jobType'].cat.reorder_categories(levels, inplace=True)
sns.boxplot(x = 'jobType', y = 'salary', data=df_train)


# JobType clearly influences salary with CFO, CT0 and CEO earning the most, this makes sense as they run the company. There is a clearly a positive relationship between both of them

# In[129]:


grouped_test1=df_train[['jobType', 'salary']].groupby(['jobType'])
f_val, p_val = stats.f_oneway(grouped_test1.get_group('JANITOR')['salary'], 
                              grouped_test1.get_group('JUNIOR')['salary'], 
                              grouped_test1.get_group('SENIOR')['salary'],
                              grouped_test1.get_group('MANAGER')['salary'],
                              grouped_test1.get_group('VICE_PRESIDENT')['salary'],
                              grouped_test1.get_group('CFO')['salary'],
                              grouped_test1.get_group('CTO')['salary'],
                              grouped_test1.get_group('CEO')['salary'] )  
 
print(( "ANOVA results: F=", f_val, ", P =", p_val))


# This is a great result, with a large F test score showing a strong correlation and a P value of 0 implying almost certain statistical significance.

# ## degree

# In[130]:


#degree
plt.figure(figsize = (14, 6))
mean = df_train.groupby('degree')['salary'].mean()
df_train['degree'] = df_train['degree'].astype('category')
levels = mean.sort_values().index.tolist()
df_train['degree'].cat.reorder_categories(levels, inplace=True)
sns.boxplot(x = 'degree', y = 'salary', data=df_train)


# college degrees tend to correlate to a higher salary

# In[131]:


grouped_test2=df_train[['degree', 'salary']].groupby(['degree'])
f_val, p_val = stats.f_oneway(grouped_test2.get_group('NONE')['salary'], 
                              grouped_test2.get_group('HIGH_SCHOOL')['salary'], 
                              grouped_test2.get_group('BACHELORS')['salary'],
                              grouped_test2.get_group('MASTERS')['salary'],
                              grouped_test2.get_group('DOCTORAL')['salary'])  
 
print(( "ANOVA results: F=", f_val, ", P =", p_val))


# A large F test score showing a strong correlation and a P value of 0 implying almost certain statistical significance.

# ## major

# In[132]:


#major
plt.figure(figsize = (14, 6))
mean = df_train.groupby('major')['salary'].mean()
df_train['major'] = df_train['major'].astype('category')
levels = mean.sort_values().index.tolist()
df_train['major'].cat.reorder_categories(levels, inplace=True)
sns.boxplot(x = 'major', y = 'salary', data=df_train)


# Engineering, business and math pay better

# In[133]:


grouped_test3=df_train[['major', 'salary']].groupby(['major'])
f_val, p_val = stats.f_oneway(grouped_test3.get_group('NONE')['salary'], 
                              grouped_test3.get_group('LITERATURE')['salary'], 
                              grouped_test3.get_group('BIOLOGY')['salary'],
                              grouped_test3.get_group('CHEMISTRY')['salary'],
                              grouped_test3.get_group('PHYSICS')['salary'],
                              grouped_test3.get_group('COMPSCI')['salary'],
                              grouped_test3.get_group('MATH')['salary'],
                              grouped_test3.get_group('BUSINESS')['salary'],
                              grouped_test3.get_group('ENGINEERING')['salary'])  
 
print(( "ANOVA results: F=", f_val, ", P =", p_val))


# A large F test score showing a strong correlation and a P value of 0 implying almost certain statistical significance.

# ## Industry

# In[134]:


#industry
plt.figure(figsize = (14, 6))
mean = df_train.groupby('industry')['salary'].mean()
df_train['industry'] = df_train['industry'].astype('category')
levels = mean.sort_values().index.tolist()
df_train['industry'].cat.reorder_categories(levels, inplace=True)
sns.boxplot(x = 'industry', y = 'salary', data=df_train)


# Industries of oil, finance and web tend to pay better (we did observe that when we were trying to find the corrupt data)

# In[135]:


grouped_test4=df_train[['industry', 'salary']].groupby(['industry'])
f_val, p_val = stats.f_oneway(grouped_test4.get_group('EDUCATION')['salary'], 
                              grouped_test4.get_group('SERVICE')['salary'], 
                              grouped_test4.get_group('AUTO')['salary'],
                              grouped_test4.get_group('HEALTH')['salary'],
                              grouped_test4.get_group('WEB')['salary'],
                              grouped_test4.get_group('FINANCE')['salary'],
                              grouped_test4.get_group('OIL')['salary'])  
 
print(( "ANOVA results: F=", f_val, ", P =", p_val))


# A large F test score showing a strong correlation and a P value of 0 implying almost certain statistical significance.

# We will discard jobId and companyId as a training feature as it is a unique value for each row for the first and we proved that the latter was not statiscally significant. MilesfromMetropolis, WorkExperience, Job_Type, Degree, Major and Industry are going to be our training features and salary our target variable
# We now will want to one hot encode the categorical data - we need to be able to see it as numerical data, each category will have a column to do that - even though we observed a very weak relationship between companyID and salary, we want to make sure that our assumptions are correct

# In[136]:


def cat_to_num(data):
    # encode the categories using average salary for each category to replace label
    for col in data.columns:
        if data[col].dtype.name == 'category':
            feature_dict = {}
            feature_list = data[col].cat.categories.tolist()
            for ft in feature_list:
                feature_dict[ft] = data[data[col] == ft]['salary'].mean()
            data[col] = data[col].map(feature_dict)
            data[col] = data[col].astype('float64')


# In[137]:


df_train1 = df_train.copy()
cat_to_num(df_train1)
df_train1


# In[138]:


fig = plt.figure(figsize=(12, 9))
features = ['jobType', 'degree', 'major', 'industry', 'yearsExperience', 'milesFromMetropolis']
sns.heatmap(df_train1[features + ['salary']].corr(), cmap='magma',vmin=-1, vmax=1, annot=True, linewidths=1)
plt.title('Heatmap of Correlation Matrix')
plt.xticks(rotation=45)
plt.show()


# We see that jobType is most strongly correlated with salary, followed by degree, major, and yearsExperience.

# The features also have some correlation, degree and major have a strong correlation while major/degree and jobType have a weak one

# ### ---- 5 Establish a baseline ----

# We are going to evaluate our models using the Mean Squared error which is geared more towards large errors as it squares each error. It is the reasonable metric to select as the problem is a regression one where the numerical, continuous target depends on a set of features.

# Before we build our complex models, let's start with a baseline metric. We'll compare our other models to the baseline to show how much they improve over the baseline. A common metric for making salary comparisons is the average job type salary.

# In[139]:


df_baseline = df_train.groupby('jobType', as_index = False).mean()
df_baseline.rename(columns = {'salary':'avg_salary'}, inplace = True)
df_baseline = df_baseline[['jobType', 'avg_salary']]
df_baseline


# In[140]:


df_baseline= pd.merge(df_train, df_baseline, on = 'jobType')
df_baseline.head(3)


# In[141]:


from sklearn.metrics import mean_squared_error

mean_squared_error(df_baseline['salary'], df_baseline['avg_salary']).round(2)


# It is traditional for HR departments to make salary decesions based on the mean of the specific type of job the professional will have. We'll use the MSE of 963 for the average salaries compared to the actually salaries as our baseline. Our goal is to lower the MSE below 360 with at least one of our models so that we have a decrease of 600.

# In[ ]:





# ### ---- 6 Hypothesize solution ----

# It's understandable that just using the average salary by Job Type is not the most accurate model. Almost all of the features demostrate some kind of predictive behavior, with jobType and degree being strong indicators. CompanyID is the least predictive, and I don't think it would add value to consider it as a factor.

# The prediction of our baseline model is based on the data itself without any fitting, feature engineering or tuning, so the result is not ideal. Our goal is to find the best model to beat this baseline model, i.e. get a model with much lower mse score.

# We are dealing here with a continuous variable as a target variable (salary) which means that using Regression will be ideal here. We have the labels as well so a Supervised Machine Learning algorithms will be used to improve the prediction results. Regression and Ensembles of Regression Algorithms suit our data and goal of predicting new salaries:
# 4 models that should improve results over the baseline model given the above EDA, we will want the best one:
# - LinearRegression - Sometimes simple is best
# - RandomForestRegressor - Improved accuracy and control over-fittings
# - GradientBoostingRegressor - Can optimise on Least Squares regression.
# - XGBoost - Similar to Gradient boosting with more optimization

# 

# ## Part 3 - DEVELOP

# You will cycle through creating features, tuning models, and training/validing models (steps 7-9) until you've reached your efficacy goal
# 
# #### Your metric will be MSE and your goal is:
#  - <360 for entry-level data science roles
#  - <320 for senior data science roles

# ### ---- 7 Engineer features  ----

# "jobID" is unique for each record in the data, thus it shouldn't be considered into modeling.
# "companyId" is a categorical data with low Anova F score so not statistically significant for the prediction of salary
# "jobType", "degree", "major", "industry" are categorical variables, they have significant F values and p-values, they should be applied with one-hot encoding.
# "yearExperience & "milesFromMetropolis" are numerical features with a moderate correlation so they should be kept for modeling.

# In[143]:


num_var = ['yearsExperience', 'milesFromMetropolis']

cat_var = ['jobType', 'degree', 'major', 'industry']

target_var = ['salary']


# We seperate numerical and categorical variables as well as our target variable

# In[146]:


def df_one_hot_encode_feature(df, cat_vars=None, num_vars=None):
    '''performs one-hot encoding on all categorical variables and combines result with continous variables'''
    cat_df = pd.get_dummies(df[cat_vars])
    num_df = df[num_vars].apply(pd.to_numeric)
    return pd.concat([cat_df, num_df], axis=1)#,ignore_index=False)


# In[155]:


X_tr = df_one_hot_encode_feature(df_train, cat_var, num_var)
X_tr.info()


# Now all of our data is numerical and ready for modelling. We store our target variable and then do the same for the test variables

# In[188]:


y_tr = df_train[target_var]
y_tr.info()


# In[161]:


df_test = df_raw_test_feature


# In[162]:


X_te = df_one_hot_encode_feature(df_test, cat_var, num_var)
X_te.info()


# We do not have a target for the test data

# In[260]:


#make sure that data is ready for modeling
#create any new features needed to potentially enhance model


# ### ---- 8 Create models ----

# Before creating the actual models I'll define functions to loop through the model testing.

# In[220]:


def train_model(model, model_df, target_df, num_procs, mean_mse, cv_std):
    neg_mse = cross_val_score(model, model_df, target_df, cv = 5, scoring = 'neg_mean_squared_error')
    mean_mse[model] = -1.0*np.mean(neg_mse)
    cv_std[model] = np.std(neg_mse)

def print_summary(model, mean_mse, cv_std):
    print(('\nModel:\n', model))
    print(('Average MSE:\n', mean_mse[model]))
    print(('Standard deviation during CV:\n', cv_std[model]))


# In[221]:


#initialize model list and dicts
models = []
mean_mse = {}
cv_std = {}


# In[236]:


lr = LinearRegression()
rfr = RandomForestRegressor(n_estimators=60, max_depth=15, min_samples_split=80, max_features=8)

gbr = GradientBoostingRegressor(n_estimators=40, max_depth=7, loss='ls', verbose=0) 
models.extend([lr, rfr, gbr])


# In[237]:


for model in models:
        
    train_model(model, X_tr, y_tr.values.ravel(), mean_mse, cv_std)
    print_summary(model, mean_mse, cv_std)


# In[243]:


from sklearn.preprocessing import PolynomialFeatures 
kfold = KFold(n_splits=5, shuffle=True, random_state= 7)
param_grid = {'polynomialfeatures__degree': [1],
              'gradientboostingregressor__learning_rate': [0.1, 0.2, 0.3, 0.4], 
              'gradientboostingregressor__max_depth': [4, 5, 6]}

pipe = make_pipeline(PolynomialFeatures(interaction_only=True, include_bias=False), GradientBoostingRegressor())
    
grid = GridSearchCV(pipe, param_grid=param_grid, cv=kfold, n_jobs = -1, scoring='neg_mean_squared_error')
grid.fit(X_tr, y_tr.values.ravel())

print(('model:', model))
# determine the best parameters trained on the whole training set for each model
print(("Best parameters: {}".format(grid.best_params_)))
# evaluate how well the best found parameters generalize
print(("Test-set score: {:.3f}\n\n".format(-1*grid.score(features_validation, target_validation))))


# In[195]:


neg_mse = cross_val_score(LinearRegression(), X_tr, y_tr, cv=5, scoring='neg_mean_squared_error')
mean_mse[model] = -1.0 * np.mean(neg_mse)
mean_mse


# ### ---- 9 Test models ----

# In[200]:


for model in models:
    neg_mse = cross_val_score(model, X_tr, y_tr.values.ravel(), cv=2, scoring='neg_mean_squared_error')
    Model_Training(model, X_tr, y_tr, mean_mse, cv_std)


# ### ---- 10 Select best model  ----

# In[ ]:


#select the model with the lowest error as your "prodcuction" model


# ## Part 4 - DEPLOY

# ### ---- 11 Automate pipeline ----

# In[ ]:


#write script that trains model on entire training set, saves model to disk,
#and scores the "test" dataset


# ### ---- 12 Deploy solution ----

# In[16]:


#save your prediction to a csv file or optionally save them as a table in a SQL database
#additionally, you want to save a visualization and summary of your prediction and feature importances
#these visualizations and summaries will be extremely useful to business stakeholders


# ### ---- 13 Measure efficacy ----

# We'll skip this step since we don't have the outcomes for the test data

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




