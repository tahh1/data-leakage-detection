#!/usr/bin/env python
# coding: utf-8

# # Evaluation Project - 7
# Loan Application Status Prediction
# Problem Statement:
# 
# This dataset includes details of applicants who have applied for loan. The dataset includes details like credit history, loan amount, their income, dependents etc. 
# 
# Independent Variables:
# 
# - Loan_ID
# 
# - Gender
# 
# - Married
# 
# - Dependents
# 
# - Education
# 
# - Self_Employed
# 
# - ApplicantIncome
# 
# - CoapplicantIncome
# 
# - Loan_Amount
# 
# - Loan_Amount_Term
# 
# - Credit History
# 
# - Property_Area
# 
# Dependent Variable (Target Variable):
# 
# - Loan_Status
# 
# You have to build a model that can predict whether the loan of the applicant will be approved or not on the basis of the details provided in the dataset. 

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[2]:


df=pd.read_csv("loan_prediction.csv")
df


# In[3]:


df.isnull().sum()


# In[4]:


(df.loc[:,:]==" ").sum()


# In[5]:


(df.loc[:,:]==0).sum()


# In[6]:


df.dtypes


# In[7]:


df["Gender"].unique()


# In[8]:


df["Married"].unique()


# In[9]:


df["Dependents"].unique()


# In[10]:


df["Education"].unique()


# In[11]:


df["Self_Employed"].unique()


# In[12]:


df["Property_Area"].unique()


# In[13]:


df["Loan_Status"].unique()


# In[14]:


df["Loan_Amount_Term"].unique()


# In[15]:


df["Credit_History"].unique()


# In[16]:


import statistics as stats
df["Gender"]=df["Gender"].fillna(stats.mode(df["Gender"]))
df["Married"]=df["Married"].fillna(stats.mode(df["Married"]))
df["Dependents"]=df['Dependents'].fillna(stats.mode(df["Dependents"]))
df["Self_Employed"]=df["Self_Employed"].fillna(stats.mode(df["Self_Employed"]))
df["LoanAmount"]=df["LoanAmount"].fillna(np.mean(df["LoanAmount"]))
df["Credit_History"]=df["Credit_History"].fillna(stats.mode(df["Credit_History"]))
df["Loan_Amount_Term"]=df["Loan_Amount_Term"].fillna(stats.mode(df["Loan_Amount_Term"]))
df.isnull().sum()


# All nan values are filled.

# # EDA

# In[17]:


count=df["Gender"].value_counts()
explode = [0, 0.1]
plt.figure(figsize=(7,7))
plt.pie(data=count,x=count.values,labels=count.index,colors=["purple","yellow"],autopct="%0.2f%%",explode=explode)
plt.title("Gender",size=20)


# 
#     
# Most male have applied for a loan than female.

# In[18]:


count=df["Education"].value_counts()
explode = [0, 0.1]
plt.figure(figsize=(7,7))
plt.pie(data=count,x=count.values,labels=count.index,colors=["0.7","0.5"],autopct="%0.2f%%",explode=explode)
plt.title("Education",size=20)


# 
#     
# Most individuals applying for a loan are graduates.

# In[19]:


count=df["Married"].value_counts()
explode = [0, 0.1]
plt.figure(figsize=(7,7))
plt.pie(data=count,x=count.values,labels=count.index,colors=["green","0.8"],autopct="%0.2f%%",explode=explode)
plt.title("Married",size=20)


# 
#     
# Most applicants are married.

# In[20]:


count=df["Loan_Status"].value_counts()
explode = [0, 0.1]
plt.figure(figsize=(7,7))
plt.pie(data=count,x=count.values,labels=count.index,colors=["orange","pink"],autopct="%0.2f%%",explode=explode)
plt.title("Loan_Status",size=20)


# 
#     
# Most applicants had their loans approved.

# In[21]:


count=df["Property_Area"].value_counts()
explode = [0, 0.1,0.2]
plt.figure(figsize=(7,7))
plt.pie(data=count,x=count.values,labels=count.index,colors=["grey","cyan","lime"],autopct="%0.2f%%",explode=explode)
plt.title("Property_Area",size=20)


# 
#     
# 1. There higher number of semiurban applicants than urban or rural. 
# 
# 
# 2. Semiurban and rural account to close to 67% of the total applicants.

# In[22]:


plt.figure(figsize=(7,7))

sns.lineplot(y="LoanAmount",x="Loan_Amount_Term",hue="Education",data=df,ci=None,palette ="CMRmap")


# 
#     Appplicants with graduation apply for a higher loan amount for the same tenure. 

# In[23]:


plt.figure(figsize=(7,7))

sns.lineplot(y="LoanAmount",x="Loan_Amount_Term",hue="Self_Employed",data=df,ci=None,palette ="viridis")


# 
#     
# Self employed applicants apply for a  higher  loan amounts for the same tenure.

# In[24]:


plt.figure(figsize=(7,7))

sns.lineplot(y="LoanAmount",x="Loan_Amount_Term",hue="Property_Area",data=df,ci=None,palette ="Accent")


# 
#     
# Applicants from urban apply for higher loan amount for the tenure when compare to semiurban. Likewise, semiurbans apply for higher loan amount than rural.

# In[25]:


plt.figure(figsize=(7,7))

sns.lineplot(y="LoanAmount",x="Loan_Amount_Term",hue="Married",data=df,ci=None,palette ="PuBu")


# 
#     
# Most married applicants apply for higher loan amount than the ones not married.

# In[26]:


plt.figure(figsize=(7,7))

sns.lineplot(y="LoanAmount",x="Loan_Amount_Term",hue="Dependents",data=df,ci=None,palette ="RdBu_r")


# 
#     
# Most individuals with 3+ dependents apply for higher loan amount than 0,1 and 2 dependents.

# In[27]:


plt.figure(figsize=(7,7))

sns.stripplot(y="LoanAmount",data=df,x="Gender",palette="YlOrRd_r",hue="Married",dodge=True)


# 
# 
#     
# 1. Most male that are married apply for higher loan amount than the one that aren't.
# 
# 
# 2. Fewer number of women that are married apply for a loan amount higher than the one that aren't.

# In[28]:


plt.figure(figsize=(7,7))

sns.stripplot(y="LoanAmount",data=df,x="Gender",palette="gnuplot2",hue="Loan_Status",dodge=True)


# Observation:
#     
# 1. Fewer number of women when compared to men faced rejection. This could also be because there were fewer female applicants. 

# In[29]:


plt.figure(figsize=(7,7))

sns.lineplot(x="ApplicantIncome",y="LoanAmount",hue="Loan_Status",data=df,ci=None,palette ="YlOrRd_r")


# 
#     
# Applicants with lower income have chances to getting higher loan amounts approved.

# In[30]:


plt.figure(figsize=(7,7))

sns.stripplot(y="ApplicantIncome",data=df,x="Property_Area",palette="gnuplot2",hue="Loan_Status",dodge=True)


# 
#     
# 1. Urban dwellers with higer income get their loans approved. 
# 
# 
# 2. A rural dweller even with high income faced  rejection.

# In[31]:


df_org = df.loc[df["Loan_Status"] == "Y"]
df_r = df_org.groupby(["Property_Area"])[["CoapplicantIncome"]].mean().sort_values(by = "CoapplicantIncome", ascending = False)
plt.figure(figsize = (8, 10))
sns.barplot(data = df_r, x = "CoapplicantIncome", y = df_r.index)
plt.title("Approved loans", size = 16)


# 
#     
# 1. Highest number of rural applicants got their loans approved with higher coapplicant's income. 
# 
# 
# 2. least number of urban applicants got their loan approved with lower coapplicant's income. 

# In[32]:


df_org = df.loc[df["Loan_Status"] == "N"]
df_r = df_org.groupby(["Property_Area"])[["CoapplicantIncome"]].mean().sort_values(by = "CoapplicantIncome", ascending = False)
plt.figure(figsize = (8, 10))
sns.barplot(data = df_r, x = "CoapplicantIncome", y = df_r.index)
plt.title("Rejected loans", size = 16)


# 
#     
# 1. Higher number of urban applicants faced rejection with higher coapplicant's income.
# 
# 
# 2. Least number of rural applicants faced rejection with lower coapplicant's income.

# In[33]:


plt.figure(figsize=(7,7))

sns.stripplot(y="LoanAmount",data=df,x="Credit_History",palette="gnuplot2",hue="Loan_Status",dodge=True)


# 
#     
# 1. barely any applicants with 0 credit history got their loan approved.
# 
# 
# 2. Most applicants with credit history got their loans approved even with higher loan amounts.

# # Label encoding 

# In[34]:


import sklearn 
from sklearn.preprocessing import LabelEncoder 
lencode=LabelEncoder()


# In[35]:


df["Gender"]=lencode.fit_transform(df["Gender"])
df["Married"]=lencode.fit_transform(df["Married"])
df["Dependents"]=lencode.fit_transform(df["Dependents"])
df["Education"]=lencode.fit_transform(df["Education"])
df["Self_Employed"]=lencode.fit_transform(df["Self_Employed"])
df["Property_Area"]=lencode.fit_transform(df["Property_Area"])
df["Loan_Status"]=lencode.fit_transform(df["Loan_Status"])
df


# In[36]:


df=df.drop(["Loan_ID"],axis=1)  #all unique values


# In[37]:


df


# In[38]:


df.corr()


# In[39]:


plt.subplots(figsize=(15,15))
sns.heatmap(df.corr(),annot=True)


# 
#     
# 1. Applicants income and self employed have correlation with loan status close to zero.
# 
# 
# 2. Credit history has the highest correlation with loan status.

# In[40]:


df.describe()


# 
#     
# Applicant's income, coapplicant's income and loan amount are not normally distributed. 

# In[41]:


df.info()


# 
#     No null values present.

# # Checking for outliers and skewness

# In[42]:


collist=df.columns.values
ncol=20 #no.of columns and rows to display the graphs i.e max col and max row
nrows=15
plt.figure(figsize=(ncol,3*ncol))
for i in range (0, len(collist)):
    plt.subplot(nrows,ncol,i+1)
    sns.boxplot(y=df[collist[i]],color="orange",orient="v")
    plt.tight_layout()


# 
#     
# 1. applicant income, coapplicant income and loan amount have outliers.

# In[43]:


df.skew()


# 
#     
# applicant income, coapplicant income and loan amount have skewness.

# # Data Cleaning

# In[44]:


df=df.drop(["ApplicantIncome","Self_Employed"],axis=1)  #correlation is almost equal to zero with the target.


# In[45]:


x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
from statsmodels.stats.outliers_influence import variance_inflation_factor 
def c_vif(x):
    vif=pd.DataFrame()
    vif["variables"]=x.columns
    vif["VIF"]=[variance_inflation_factor(x.values,i)for i in range(x.shape[1])]
    return(vif)


# In[46]:


c_vif(x)


# In[47]:


#df["diff"]=(df["LoanAmount"]-df["Loan_Amount_Term"])**2
#df


# In[48]:


df=df.drop(["Loan_Amount_Term"],axis=1)


# In[49]:


x=df.drop(["Loan_Status"],axis=1)
y=df["Loan_Status"]
c_vif(x)


# In[50]:


import scipy 
from scipy.stats import zscore 
z=np.abs(zscore(df))  #removes outliers
z.shape


# In[51]:


threshold=3
df_new=df[(z<3).all(axis=1)]
print((df.shape))
print((df_new.shape))


# In[52]:


#data loss 
(614-594)/614*100


# data loss is less than 10% we can proceed.

# In[53]:


x=df_new.drop(["Loan_Status"],axis=1)
y=df_new["Loan_Status"]


# In[54]:


print((x.shape))
print((y.shape))


# In[55]:


from sklearn.preprocessing import power_transform 
x=power_transform(x,method="yeo-johnson") #removing skewness 


# # Preprocessing 

# In[56]:


import sklearn 
from sklearn.preprocessing import MinMaxScaler 


# In[57]:


ms=MinMaxScaler()
x=ms.fit_transform(x)
x


# # SMOTE

# In[58]:


y.value_counts()


# In[59]:


from imblearn.over_sampling import SMOTE 
smt=SMOTE()
trainx,trainy=smt.fit_resample(x,y)


# In[60]:


trainy.value_counts()


# # Logistic Regression

# In[61]:


from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split


# In[62]:


lr=LogisticRegression()
for i in range(0,100):
    x_train,x_test,y_train,y_test=train_test_split(trainx,trainy,test_size=0.2,random_state=i)
    lr.fit(x_train,y_train)
    
    pred_test_lr=lr.predict(x_test)
    
    print(("At random state=",i,'testing accuracy =',accuracy_score(y_test,pred_test_lr)))
    print("\n")


# In[63]:


x_train,x_test,y_train,y_test=train_test_split(trainx,trainy,test_size=0.2,random_state=50)
lr.fit(x_train,y_train)
    
pred_test_lr=lr.predict(x_test)
    
print(('testing accuracy =',accuracy_score(y_test,pred_test_lr)))


# In[64]:


from sklearn.model_selection import cross_val_score 
for i in range(2,11):
    rf_cv=cross_val_score(lr,trainx,trainy,cv=i)
    rfs=rf_cv.mean()
    print(("Score =",rfs*100,"at cv =",i))


# In[65]:


rf_cv=cross_val_score(lr,trainx,trainy,cv=8)
rfs=rf_cv.mean()
print(("CV Score =",rfs*100))
print(("Accuracy_score =",accuracy_score(y_test,pred_test_lr)))


# In[66]:


x_train,x_test,y_train,y_test=train_test_split(trainx,trainy,test_size=0.2)


# # DecisionTree Classifier

# In[67]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree


# In[68]:


dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)

pred_test_dt=dt.predict(x_test)
acc_test=accuracy_score(pred_test_dt,y_test)

print(("acc_test =",acc_test))


# In[69]:


for i in range(2,11):
    rf_cv=cross_val_score(dt,trainx,trainy,cv=i)
    rfs=rf_cv.mean()
    print(("Score =",rfs*100,"at cv =",i))


# In[70]:


rf_cv=cross_val_score(dt,trainx,trainy,cv=9)
rfs=rf_cv.mean()
print(("CV Score =",rfs*100))
print(("Accuracy_score =",accuracy_score(pred_test_dt,y_test)*100))


# # RandomForest Classifier

# In[71]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)

pred_test_rf=rf.predict(x_test)
acc_test=accuracy_score(pred_test_rf,y_test)

print(("acc_test =",acc_test))


# In[72]:


for i in range(2,11):
    rf_cv=cross_val_score(rf,trainx,trainy,cv=i)
    rfs=rf_cv.mean()
    print(("Score =",rfs*100,"at cv =",i))


# In[73]:


rf_cv=cross_val_score(rf,trainx,trainy,cv=10)
rfs=rf_cv.mean()
print(("CV Score =",rfs*100))
print(("Accuracy_score =",accuracy_score(pred_test_rf,y_test)*100))


# # SVC

# In[74]:


from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
y_pred_svc = svc.predict(x_test)
accuracy_score(y_test,y_pred_svc)


# In[75]:


for i in range(2,11):
    rf_cv=cross_val_score(svc,trainx,trainy,cv=i)
    rfs=rf_cv.mean()
    print(("Score =",rfs*100,"at cv =",i))


# In[76]:


rf_cv=cross_val_score(svc,trainx,trainy,cv=5)
rfs=rf_cv.mean()
print(("CV Score =",rfs*100))
print(("Accuracy_score =",accuracy_score(y_test,y_pred_svc)*100))


# # Hyper parameter tuning

# In[77]:


from sklearn.model_selection import GridSearchCV
param = {"criterion":["gini","entropy"],"max_features":["auto","sqrt","log2"],"bootstrap":[True,False],"n_estimators":[50,100,150,200]}
clf = GridSearchCV(RandomForestClassifier(),param_grid=param)


# In[78]:


clf.fit(x_train,y_train)


# In[79]:


print((clf.best_params_))
print((clf.best_score_))


# In[85]:


rf=RandomForestClassifier(bootstrap=True,criterion="gini",max_features="sqrt",n_estimators=150)
rf.fit(x_train,y_train)

pred_test_rf=rf.predict(x_test)
acc_test=accuracy_score(pred_test_rf,y_test)

print(("acc_test =",acc_test*100))


# In[86]:


rf_cv=cross_val_score(rf,trainx,trainy,cv=4)
rfs=rf_cv.mean()
print(("CV Score =",rfs*100))
print(("Accuracy_score =",accuracy_score(pred_test_rf,y_test)*100))


# In[87]:


from sklearn.metrics import roc_curve, auc
fpr,tpr,thresholds=roc_curve(pred_test_rf,y_test)
roc_auc=auc(fpr,tpr)

plt.figure()
plt.plot(fpr,tpr,color="teal",lw=10,label="ROC curve (area=%0.2f)"%roc_auc)
plt.plot([0,1],[0,1],color="red",lw=10,linestyle="--")
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.show()


# The best fit model is this having accuracy score=79.2 , CV score=78.1 and ROC=80

# # Model saving

# In[88]:


import pickle 
filename= "Loan.pkl"
pickle.dump(rf, open(filename,"wb"))


# # Conclusion 

# In[89]:


a=np.array(y_test)
predicted = np.array(rf.predict(x_test))
df_com=pd.DataFrame({"original":a,"predicted":predicted}, index=list(range(len(a))))
df_com

