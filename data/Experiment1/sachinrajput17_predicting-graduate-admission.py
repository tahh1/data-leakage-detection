#!/usr/bin/env python
# coding: utf-8

# Importing all the necessary libraries.

# In[1]:


import numpy as np 
import pandas as pd 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt # Visualization 
import seaborn as sns
plt.style.use('fivethirtyeight')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

import warnings      # to ignore warnings
warnings.filterwarnings("ignore")



# **Load the data.**
# > Load the data and look into  sample of the data.

# In[2]:


path="/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv"
df=pd.read_csv(path)
#df1=pd.read_csv(path1)
df.head()  # Top 5 rows
df.tail()  # Bottom 5 rows
df.sample(5)  # Random 5 rows
df.sample(5) # random fractional numbers rows  of total no of rows


# # Quick EDA with pandas_profiling

# In[3]:


data=df.copy()


# In[4]:


import pandas_profiling as pp
data=df.copy()
report=pp.ProfileReport(data, title='Pandas Profiling Report')  # overview and quick data analysis
report


# 
# 
# 
# All columns in the data and shape of the data(columns,rows).We will list all the columns for all data by
# *df.columns*. We will check all columns, are there any spelling mistake?
# If we found any spelling mistake we will correct it.
# 

# In[5]:


print(("Columns of the data are:",df.columns))


#  Rename the columns name which have to required change.

# In[6]:


df=df.rename(columns={'Serial No.':'SerialNo', 'GRE Score':'GRE', 'TOEFL Score':'TOEFL',
                      'University Rating':'UniversityRating','LOR ':'LOR','Chance of Admit ':'ChanceOfAdmit'})
df.columns


# In[7]:


#Drop the column "Serial No." 
df=df.drop(columns="SerialNo")


# 
# 
# Check the sum of null values in every column.

# In[8]:


df.isnull().sum()


# There are no missing values in the dataset. It makes our data pre-processing very much easier.
# It can also be found in the following way.
#                   
# > df.info() --> give the information about the data as Index columns, datatypes of the variables,null values,memory used by data.

# In[9]:


df.info()


# In[10]:


df.isnull().values.any() # check the null values in whole of the data set if any.


# In[11]:


missing_data=df.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print((missing_data[column].value_counts()))


# In[12]:


def missing_percentage(data):
    total=data.isnull().sum().sort_values(ascending=False)
    percent=np.round(total/len(data)*100,2)
    return pd.concat([total,percent],axis=1,keys=["Total","Percent"])
missing_percentage(df)


# In[13]:


#check any duplicates data in dataframe.
df.duplicated().any()


# Lets found shape of the data 

# In[14]:


print(("Shape of the data",df.shape))


# Some important statistical summary of the data e.g. Mean,std,minimum , maximum value, 25,50,75 pecentiles of the data.

# In[15]:


#statistical summary of the data
df.describe()


#  average requirements of all features  to get admission for all universities on the basis of their Ratings.

# In[16]:


# Groupby the data by "University rating".
df.groupby("UniversityRating").mean()


# In[17]:


print((" Minimum requirements for more than 85% chance to get admission.\n",df[(df['ChanceOfAdmit']>0.85)].min()))


#  Minimum requirements for more than 85% chance of the admission.
#  
# 
# 	GRE Score	        320.00
# 	TOEFL Score 	    108.00
#     University Rating	2.00
# 	SOP	               3.00
# 	LOR                3.00
#    	CGPA	          8.94
# 	Research        	0.00
# 	Chance of Admit 	0.86

# **Pivot Table**

# In[18]:


df.pivot_table(values=['GRE','TOEFL'],index=['UniversityRating'],columns='Research',aggfunc=np.median)


# **Bar Plot**

# In[19]:


plt.figure(figsize=(15,15))
df['ChanceOfAdmit'].value_counts().plot.bar()
plt.show()


# In[20]:


#relashionship between the variables of the data in scatter form.
pd.plotting.scatter_matrix(df,figsize=(15,20)) # Scatter matrix for the data.
plt.show()


# Same matrix plot can be plot as below:-

# In[21]:


sns.pairplot(df,hue="ChanceOfAdmit")
plt.show()


# **Histogram**
# > Distribution Of the data by visualise the histograms for all features of the data.

# In[22]:


df.hist(figsize=(10,10),edgecolor="k")
plt.tight_layout()
plt.show()


# In[23]:


plt.figure(figsize=(15,12))
col_list=df.columns
 
for i in range(len(df.columns)):
    plt.subplot(3,3,i+1)
    plt.hist(df[col_list[i]],edgecolor="w")
    plt.title(col_list[i],color="g",fontsize=15)


plt.show()


# **Box Plots for all features**

# In[24]:


#Boxplot for all variables.
"""for col in df.columns:
    df[[col]].boxplot()
    plt.show()"""
df.plot(kind='box',subplots=True,layout=(3,3),grid=True,figsize=(8,8))
plt.tight_layout()

plt.show()




# **Correlation Between the data features**

# In[25]:


# Correlation Between the data features.
df.corr()


# **Heatmap of the correlation**

# In[26]:


#heatmap of the correlation of the data variables.
mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),mask=mask,annot=True,linewidths=1.0)
plt.show()


# **Pairplot**
# 
# Scatter diagram between the variables of the data and "Chance of Admit".

# In[27]:


sns.pairplot(df,x_vars=['GRE','TOEFL','UniversityRating','CGPA','SOP','LOR','Research'],
             y_vars='ChanceOfAdmit')
plt.tight_layout()


# Import the important libraries for machine learning.

# In[28]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression


# ### Data Preprocessing-
# 
# * SPLIT the data into train and test data.
# 

# In[29]:


X=df.iloc[:,:-1]
Y=df.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=2)


# In[30]:


sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)


# ### Linear Regression Model
# * Fit the model
# * Find the predicted Values with the test data  applied on model.
# * R2 value and Mean Square Error with test data and predicted data.

# In[31]:


lr=LinearRegression()
lr.fit(X_train,y_train)

#Training data score
ytrain_pred=lr.predict(X_train)
r2_score(y_train,ytrain_pred),mean_squared_error(y_train,ytrain_pred)



# In[32]:


#Testdata score
y_pred=lr.predict(X_test)
r2_score(y_test,y_pred),mean_squared_error(y_test,y_pred)


# In[33]:


print(("Intercept of Linear Regression is:\n,",lr.intercept_,"Coefficients of Linear Regression are:\n,",lr.coef_))


#  ### Random Forest Regressor

# In[34]:


from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators = 150,max_depth=4,random_state = 42,criterion="mse")
rf_model.fit(X_train,y_train)
y_pred_rf=rf_model.predict(X_test)
r2_score(y_test,y_pred_rf),mean_squared_error(y_test,y_pred_rf)


# In[35]:


feature_importance = pd.DataFrame(rf_model.feature_importances_, X.columns)
feature_importance


# CGPA  is the most important feature to determine Chance of admission.

# ### DecisionTreeRegressor

# In[36]:


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state = 4,max_depth=4)
dt.fit(X_train,y_train)
y_pred_dt = dt.predict(X_test) 
print((r2_score(y_test,y_pred_dt),mean_squared_error(y_test,y_pred_dt)))
    


# * Decision tree model without standardised data.

# In[37]:


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state = 1)
dt.fit(x_train,y_train)
y_pred_dt = dt.predict(x_test) 
r2_score(y_test,y_pred_dt),mean_squared_error(y_test,y_pred_dt)


# **Most useful Machine Learning Libraries import**

# In[38]:


#classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
 

#regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

#model selection
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#preprocessing
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Imputer,LabelEncoder

#evaluation metrics
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score 


# We will fit the various models with training data ,predict target feature values with test data and will find the which model provide the best R2 Value and MSE value, so that our prediction are more likely to true values.

# In[39]:


models=[LinearRegression(),
        RandomForestRegressor(n_estimators=150,max_depth=4),
        DecisionTreeRegressor(random_state=42,max_depth=4),GradientBoostingRegressor(),AdaBoostRegressor(),
        KNeighborsRegressor(n_neighbors=35),
        BaggingRegressor(),Ridge(alpha=1.0),RidgeCV(),SVR()]
model_names=['LinearRegression','RandomForestRegressor','DecisionTree','GradientBoostingRegressor','AdaBoost','kNN',
             'BaggingReg','Ridge','RidgeCV',"SVR"]

R2_SCORE=[]
MSE=[]
      
for model in range(len(models)):
    print(("*"*35,"\n",model_names[model]))
    reg=models[model]
    reg.fit(X_train,y_train)
    pred=reg.predict(X_test)
    r=r2_score(y_test,pred)
    mse=mean_squared_error(y_test,pred)
    R2_SCORE.append(r)
    MSE.append(mse)
    print(("R2 Score",r))
    print(("MSE",mse))


# In[40]:


df_model=pd.DataFrame({'Modelling Algorithm':model_names,'R2_score':R2_SCORE,"MSE":MSE})
df_model=df_model.sort_values(by="R2_score",ascending=False).reset_index()
print(df_model)


plt.figure(figsize=(10,10))
sns.barplot(y="Modelling Algorithm",x="R2_score",data=df_model)

plt.xlim(0.35,0.95)
plt.grid()
plt.tight_layout()


# In[41]:


df_model.head(5)


# So Top 5 models which have Best R2 Score :-
# 1. Linear Regressioin
# 2. Ridge Regression
# 3. RidgeCV Regression
# 4. Random Forest Regressor
# 5. Gradient Boosting Regressor
# 
#       

# In[42]:


lr=LinearRegression()
lr.fit(X_train,y_train)
y_lr_pred=lr.predict(X_test)
print(("MSE:",mean_squared_error(y_test,y_lr_pred),"R2 SCORE:",r2_score(y_test,y_lr_pred)))


# In[43]:


ridge=Ridge()
ridge.fit(X_train,y_train)
y_ridge=ridge.predict(X_test)
print(("MSE:",mean_squared_error(y_test,y_ridge),"R2 SCORE:",r2_score(y_test,y_ridge)))


# In[44]:


r_CV=RidgeCV()
r_CV.fit(X_train,y_train)
y_rCV=r_CV.predict(X_test)
print(("MSE:",mean_squared_error(y_test,y_rCV),"R2 SCORE:",r2_score(y_test,y_rCV)))


# In[45]:


tp=pd.DataFrame({"TEST_value":y_test,"LR_predict_value": y_lr_pred,"RIDGE_predict_value": y_ridge,"RCV_predict_value": y_rCV,"DIFF(TEST_value-LR_predict_value)": (y_test-y_lr_pred)})
tp.head()


# In[46]:


plt.figure(figsize=(10,10),dpi=75)
x=np.arange(len(tp["TEST_value"]))
y=tp["TEST_value"]
z=tp["LR_predict_value"]
plt.plot(x,y)
plt.plot(x,z,color='r')


# In[47]:


print(("Score of Linear Regression:",r2_score(y_test,y_lr_pred)))


# **If you find this kernal helpful and useful, Kindly comments your suggestion and upvote the kernal.
# **
# 
# ## THANKYOU

# In[ ]:




