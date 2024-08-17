#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st


# So basically it is a Classification Problem as we have to know if employee will be promoted or not

# In[2]:


train=pd.read_csv("../input/train.csv")


# In[3]:


test=pd.read_csv("../input/test.csv")


# In[4]:


train.shape


# In[5]:


test.shape


# ##### Problem Statement
# 
#   Your client is a large MNC and they have 9 broad verticals across the organisation. One of the problem your client is facing is around identifying the right people for promotion (only for manager position and below) and prepare them in time. Currently the process, they are following is:
# 
#     . They first identify a set of employees based on recommendations/ past performance
#     
#     . Selected employees go through the separate training and evaluation program for each vertical. These programs are based on the required skill of each vertical
#         
#     . At the end of the program, based on various factors such as training performance, KPI completion (only employees with KPIs completed greater than 60% are considered) etc., employee gets promotion
#     
#     . For above mentioned process, the final promotions are only announced after the evaluation and this leads to delay in transition to their new roles. Hence, company needs your help in identifying the eligible candidates at a particular checkpoint so that they can expedite the entire promotion cycle. 
# 
# They have provided multiple attributes around Employee's past and current performance along with demographics. 
# 
# Now, The task is to predict whether a potential promotee at checkpoint in the test set will be promoted or not after the evaluation process.

# In[6]:


combined=pd.concat([train,test],ignore_index=True,sort=True)


# In[7]:


combined_backup=combined.copy()


# In[8]:


combined.head()


# In[9]:


combined.info()


# In[10]:


combined.isnull().sum()


# In[11]:


combined.employee_id.nunique()


# In[12]:


combined.shape


# #### Age Analysis :
# People with more than54 years of age are outliers

# In[13]:


sns.boxplot(combined.age)


# In[14]:


sns.distplot(combined.age)
#Highly right skewed 


# In[15]:


combined["KPIs_met >80%"].value_counts().plot(kind="bar")


# In[16]:


sns.boxplot(combined["avg_training_score"])


# In[17]:


sns.distplot(combined["avg_training_score"])


# average training score is grouped we can clearly see it
# 
#     most people lie on 45-50 bracket with higher intensity
#     a few people lie on 70 bracket
#     50,60,70,85

# In[18]:


combined.columns


# In[19]:


sns.countplot(combined["awards_won?"])

#maximum poeple has won no awards. so obviously maximum people has got no awrds,

# we can figure out which are the poeple who got awards and were they promoted or  


# In[20]:


combined[(combined["awards_won?"]==1) & (combined["is_promoted"]==1.0)]["age"]


#          559 PEOPLE WHO GOT PROMOTED AND WON AWARDS

# In[21]:


sns.distplot(combined[(combined["awards_won?"]==1) & (combined["is_promoted"]==1.0)]["age"])


# Most people who got promoted and won awards are of 30 years of age 

# In[22]:


combined[(combined["awards_won?"]==1) & (combined["is_promoted"]==0.0)]


# In[23]:


sns.distplot(combined[(combined["awards_won?"]==1) & (combined["is_promoted"]==0.0)]["age"])


# In[24]:


combined[(combined["awards_won?"]==0) & (combined["is_promoted"]==1.0)]


# In[25]:


#0 awards won and is getting promoted is 4109


# In[26]:


sns.countplot(combined.department)
plt.xticks(rotation=90)


# In[27]:


combined.department.value_counts().plot(kind="bar")


# Sales & Marketing
# 
# Operations
# 
# Procurement
# 

# In[28]:


combined.education.value_counts().plot(kind="bar")


# In[29]:


#most people are bachelors


# In[30]:


combined.gender.value_counts().plot(kind="bar")


# In[31]:


plt.figure(figsize=(10,5))
combined.length_of_service.value_counts().plot(kind="bar")
plt.xticks(rotation=90)


# #### Max people with 3 years of work ex so its quiet definited that they got promoted the most
# 

# In[32]:


sns.countplot(combined["no_of_trainings"])


# #### most poeple have done 1 Year of training 

# In[33]:


sns.countplot(combined.previous_year_rating)


# ##### Maximum people with rating = 3 
# ##### Minimum people with rating = 2.0

# In[34]:


sns.countplot(combined.recruitment_channel)


# In[35]:


combined.region.value_counts().plot(kind="bar",figsize=(17,6))


# Maximum Region id is region 2

# #####  Bi-Variate Analysis

# In[36]:


combined.head()


# In[37]:


#Going for Boxplots


# In[38]:


sns.boxplot(combined["awards_won?"],combined.age)


# In[39]:


sns.boxplot(combined["awards_won?"],combined.length_of_service)


# In[40]:


sns.boxplot(combined["awards_won?"],combined.no_of_trainings)


# In[41]:


sns.boxplot(combined["awards_won?"],combined.previous_year_rating)


# maximum poeple who won awards have 3 to 5 previous year rating

# In[42]:


sns.boxplot(combined["awards_won?"],combined.recruitment_channel.value_counts())


# In[43]:


sns.boxplot(combined["is_promoted"],combined.age)


# In[44]:


sns.boxplot(combined["is_promoted"],combined.length_of_service)


# In[45]:


sns.boxplot(combined["is_promoted"],combined.previous_year_rating)


# maximum poeple who got promoted had previous year rating of 3.5 to 5 
# 
# one poeple is having a rating of 1.0 who os not promoted which is true

# In[46]:


sns.boxplot(combined["is_promoted"],combined.no_of_trainings)


# In[47]:


#Numerical vs Numerical


# In[48]:


plt.scatter(combined.age,combined.avg_training_score)


# In[49]:


plt.scatter(combined.age,combined.length_of_service)
# we can clearly see with age length of service is increasing


# In[50]:


#categorical vs categorical analysis
combined.head()


# In[51]:


combined.groupby(["education","department"]).describe().plot(kind="bar",figsize=(20,10))


# In[52]:


combined.groupby(["education","department"])["age"].describe().plot(kind="bar",figsize=(20,10))


# Inferences : 
#     
#     Bachelors sales and marketing with age count is very high
#     Bachleors with opetrations
#     then, bachelors in tech and procurement, analytics
#     
#     masters, S adn Marketing
#     then operations
#     tech,procurement, analytics
#     
#     Bachelors sales and marketing are most
#     

# In[53]:


combined.groupby(["education","department","gender"]).describe().plot(kind="bar",figsize=(20,10))


# In[54]:


combined.groupby(["education","department","gender"])["age"].describe().plot(kind="bar",figsize=(20,10))


# Sales and Marketing Male are very high
# 
# Bachelors Operations Female are very high after that procurement,technology

# In[55]:


combined.groupby(["department","is_promoted"])["age"].describe().plot(kind="bar",figsize=(15,7))


# In[56]:


combined.groupby(["department","education","is_promoted"])["age"].describe().plot(kind="bar",figsize=(15,7))


# People who are getiting maximum promotions
#     
#     Sales adn markerting bachelors
#     technology bachelors
#     technology masters abd above
#     Analytics, bachelors
#     operations, bachelors
#     operation s masters and above
# 

# In[57]:


combined.groupby(["department","awards_won?","is_promoted"])["age"].describe().plot(kind="bar",figsize=(15,7))


# In[58]:


combined.head()


# In[59]:


pd.DataFrame(combined.groupby(["no_of_trainings","KPIs_met >80%","is_promoted"])["age"].describe())


# In[60]:


combined.groupby(["KPIs_met >80%","is_promoted"])["age"].describe()


# In[61]:


sns.boxplot(combined["KPIs_met >80%"],combined["is_promoted"])


# In[62]:


combined.groupby(["KPIs_met >80%","is_promoted"])["age"].describe().plot(kind="bar")


# In[63]:


combined.groupby(["length_of_service","is_promoted"])["age"].describe()


# In[64]:


#Perfroming Feature Engineering


# In[65]:


combined.head()


# In[66]:


train.shape


# In[67]:


combined.iloc[54807]


# In[68]:


combined.isnull().sum()


# In[69]:


combined.previous_year_rating.mean()


# In[70]:


combined.previous_year_rating.mode()


# In[71]:


combined.previous_year_rating.median()


# In[72]:


combined.previous_year_rating.skew()


# In[73]:


combined.previous_year_rating.kurt()


# In[74]:


combined.loc[combined.previous_year_rating.isnull(),"previous_year_rating"]=3.0


# In[75]:


combined.isnull().sum()


# In[76]:


combined.education.mode()


# In[77]:


combined.education.dropna(inplace=True)


# In[78]:


train=combined[:54808]


# In[79]:


test=combined[54808:]


# In[80]:


train.head()


# In[81]:


train.corr()


# In[82]:


plt.figure(figsize=(10,6))
sns.heatmap(train.corr(),annot=True)


# In[83]:


train.drop("region",axis=1,inplace=True)
test.drop("region",axis=1,inplace=True)


# In[84]:


train.drop("employee_id",axis=1,inplace=True)
test.drop("employee_id",axis=1,inplace=True)


# In[85]:


d={"m":1,"f":0}
train.gender=train.gender.map(d)


# In[86]:


test.gender=test.gender.map(d)


# In[87]:


train.head()


# In[88]:


dummytrain=pd.get_dummies(train).drop("recruitment_channel_other",axis=1)
dummytest=pd.get_dummies(test).drop("recruitment_channel_other",axis=1)


# X and Y split

# In[89]:


X=dummytrain.drop(["is_promoted"],axis=1)


# In[90]:


y=dummytrain.is_promoted


# In[91]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=123,test_size=0.2)


# In[92]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
scaled_train=pd.DataFrame(sc.fit_transform(X_train,y_train),columns=X_train.columns)
scaled_test=pd.DataFrame(sc.transform(X_test),columns=X_test.columns)


# In[93]:


scaled_train.shape


# In[94]:


scaled_test.shape


# In[95]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
model_rf=rf.fit(scaled_train,y_train).predict(scaled_test)


# In[96]:


from sklearn.metrics import r2_score ,mean_squared_error,mean_absolute_error
print(("The R sqaure of the model is ",r2_score(y_test,model_rf)))
print(("The RMSE IS", np.sqrt(mean_squared_error(y_test,model_rf))))


# In[97]:


features = pd.DataFrame(rf.feature_importances_, index = scaled_test.columns,
            columns = ["Features"])


# In[98]:


features.sort_values(by = "Features").plot(kind = "barh", color = "red")


# In[99]:


from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
model_gb=gb.fit(scaled_train,y_train).predict(scaled_test)
print(("The R sqaure of the model is ",r2_score(y_test,model_gb)))
print(("The RMSE IS", np.sqrt(mean_squared_error(y_test,model_gb))))


# In[100]:


from xgboost import XGBRFRegressor
xg=XGBRFRegressor()
model_xg=xg.fit(scaled_train,y_train).predict(scaled_test)
print(("The RMSE IS ",np.sqrt(mean_squared_error(y_test,model_xg))))
print(("tHE R SQAURE IS ",r2_score(y_test,model_xg)))


# In[101]:


from sklearn.ensemble import AdaBoostClassifier
ad=AdaBoostClassifier(random_state=123)
model_ad=ad.fit(scaled_train,y_train).predict(scaled_test)
print(("The R sqaure of the model is ",r2_score(y_test,model_ad)))
print(("The RMSE IS", np.sqrt(mean_squared_error(y_test,model_ad))))


# In[102]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.min_rows', 1000)


# In[103]:


combined.head()


# In[104]:


#https://stackoverflow.com/questions/11707586/how-do-i-expand-the-output-display-to-see-more-columns-of-a-pandas-dataframe


# In[105]:


combined.age.head()


# In[106]:


combined.head()


# In[107]:


combined.drop(["employee_id","region"],axis=1,inplace=True)


# In[108]:


sns.distplot(np.sqrt(combined.age))


# In[109]:


sns.distplot(np.log1p(combined.age))


# In[110]:


sns.distplot(np.log1p(combined.length_of_service))


# In[111]:


combined.length_of_service=np.log1p(combined.length_of_service)


# In[112]:


combined.age=np.log1p(combined.age)


# In[113]:


combined.head()


# In[114]:


d={"f":0,"m":1}
combined.gender=combined.gender.map(d)


# In[115]:


combined.head()


# In[116]:


train.shape


# In[117]:


test.shape


# In[118]:


newtrain=combined[:54808]
newtest=combined[54808:]


# In[119]:


newtest=combined[54808:]


# In[120]:


newtest.drop("is_promoted",axis=1,inplace=True)


# In[121]:


dummytrain=pd.get_dummies(newtrain)
dummytest=pd.get_dummies(newtest)


# In[122]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
cols = dummytrain.columns[dummytrain.columns!="is_promoted"]
scaled_train = pd.DataFrame(sc.fit_transform(dummytrain.drop("is_promoted", axis = 1)), 
             columns=cols)
scaled_test = pd.DataFrame(sc.transform(dummytest),
                          columns = dummytest.columns)


# In[123]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
model = rf.fit(scaled_train, dummytrain.is_promoted).predict(scaled_test)


# In[124]:


features = pd.DataFrame(rf.feature_importances_, index = scaled_test.columns,
            columns = ["Features"])
features.sort_values(by = "Features").plot(kind = "barh", color = "red")


# In[125]:


from sklearn.metrics import r2_score ,mean_squared_error,mean_absolute_error
print(("The R sqaure of the model is ",r2_score(dummytrain.is_promoted[:23490],model)))
print(("The RMSE IS", np.sqrt(mean_squared_error(dummytrain.is_promoted[:23490],model))))


# In[126]:


y_test.shape


# In[127]:


model.shape


# In[128]:


solution = pd.DataFrame({"employee_id":pd.read_csv("../input/test.csv").employee_id, 
                        "is_promoted":model})
solution.to_csv("RF MODEL.csv", index =False)


# In[129]:


test.head()


# In[130]:


x=pd.read_csv("RF MODEL.csv")


# In[131]:


x.is_promoted=x.is_promoted.astype("int64")


# In[132]:


x.head()


# In[133]:


solution = pd.DataFrame({"employee_id":x.employee_id, 
                        "is_promoted":x.is_promoted})
solution.to_csv("RF MODEL2.csv", index =False)


# 0.4385

# 
# Applying GRADIENT BOOSTING

# In[134]:


from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
model_gb = gb.fit(scaled_train, dummytrain.is_promoted).predict(scaled_test)


# In[135]:


solution = pd.DataFrame({"employee_id":pd.read_csv("../input/test.csv").employee_id, 
                        "is_promoted":model_gb})


# In[136]:


solution.is_promoted=solution.is_promoted.astype("int64")


# In[137]:


solution.to_csv("GB MODEL.csv", index =False)


# 0.458

# ##### Applying Xtreme GRdaient boosting

# In[138]:


from xgboost import XGBRFClassifier
xg=XGBRFClassifier(n_estimators=3,max_depth=500)
model_xg = xg.fit(scaled_train, dummytrain.is_promoted).predict(scaled_test)


# In[139]:


solution = pd.DataFrame({"employee_id":pd.read_csv("../input/test.csv").employee_id, 
                        "is_promoted":model_xg})


# In[140]:


solution.is_promoted=solution.is_promoted.astype("int64")


# In[141]:


solution.to_csv("xg MODEL.csv", index =False)


# 0.446

# #### Adaboostikng 

# In[142]:


from sklearn.ensemble import AdaBoostRegressor
ad=AdaBoostRegressor(random_state=123)
model_ada = ad.fit(scaled_train, dummytrain.is_promoted).predict(scaled_test)


# In[143]:


solution = pd.DataFrame({"employee_id":pd.read_csv("../input/test.csv").employee_id, 
                        "is_promoted":model_xg})


# In[144]:


solution.is_promoted=solution.is_promoted.astype("int64")


# In[145]:


solution.to_csv("ADA MODEL.csv", index =False)


# : 0.44629629629629636.

# In[146]:


features = pd.DataFrame(ad.feature_importances_, index = scaled_test.columns,
            columns = ["Features"])
features.sort_values(by = "Features").plot(kind = "barh", color = "blue")


# In[147]:


features.sort_values(by = "Features")


# In[148]:


combined.head()


# #### Feature Engineering
# 
#     sum_performance = addition of the important factors for the promotion (awards_won;KpIs_met & previous_year_rating).
#     
#     Total nmber of training hours = avg_training_score * no_of_training
#     
#     recruitment_channel have no impact on the promotion so removed that.

# In[149]:


combined.head()


# ### Feature Engineering 

# Recruitment Channel : 

# In[150]:


combined.groupby(["recruitment_channel","is_promoted"]).describe().plot(kind="bar",figsize=(10,5))


# This column does have much impact on other columns

# In[151]:


combined.drop(["recruitment_channel"],axis=1,inplace=True)


# In[152]:


combined.head()


# Combining to make a new feature : Average Trarining Score & No of Trainings = Total hours

# In[153]:


combined["total_hours"]=combined.avg_training_score*combined.no_of_trainings


# In[154]:


combined.head()


# Combining Awards Won, KPI,Previous year rating

# In[155]:


combined["total_sum"]=combined["KPIs_met >80%"]+combined["awards_won?"]+combined["no_of_trainings"]


# In[156]:


combined.head()


# In[157]:


plt.figure(figsize=(10,5))
sns.heatmap(combined.corr(),annot=True,cmap="ocean")


# In[158]:


combined.drop(["total_score"],axis=1,inplace=True)


# In[159]:


combined.head()


# In[160]:


combined.education.unique()


# In[161]:


combined.isnull().sum()


# In[162]:


combined.education.mode()


# In[163]:


combined[combined.education.isnull()]["education"]=combined.education.mode()


# In[164]:


combined.loc[combined.education.isnull(),"education"]="Bachelor's"


# In[165]:


combined[combined.education.isnull()]["education"]


# In[166]:


combined.isnull().sum()


# In[167]:


combined.head()


# In[168]:


newtrain=combined[:54808]
newtest=combined[54808:]
newtest.drop("is_promoted",axis=1,inplace=True)
dummytrain=pd.get_dummies(newtrain)
dummytest=pd.get_dummies(newtest)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
cols = dummytrain.columns[dummytrain.columns!="is_promoted"]
scaled_train = pd.DataFrame(sc.fit_transform(dummytrain.drop("is_promoted", axis = 1)), 
             columns=cols)
scaled_test = pd.DataFrame(sc.transform(dummytest),
                          columns = dummytest.columns)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
model = rf.fit(scaled_train, dummytrain.is_promoted).predict(scaled_test)
features = pd.DataFrame(rf.feature_importances_, index = scaled_test.columns,
            columns = ["Features"])
features.sort_values(by = "Features").plot(kind = "barh", color = "red")


# In[169]:


from sklearn.metrics import r2_score ,mean_squared_error,mean_absolute_error
print(("The R sqaure of the model is ",r2_score(dummytrain.is_promoted[:23490],model)))
print(("The RMSE IS", np.sqrt(mean_squared_error(dummytrain.is_promoted[:23490],model))))
solution = pd.DataFrame({"employee_id":pd.read_csv("../input/test.csv").employee_id, 
                        "is_promoted":model})
solution.is_promoted=solution.is_promoted.astype("int64")
solution.to_csv("RF MODEL4_FEATURE ENG.csv", index =False)


# #### Model Accuracy on Analytics Vidhya : 0.4516129032258064.
# 

# In[170]:


from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
model_gb = gb.fit(scaled_train, dummytrain.is_promoted).predict(scaled_test)
solution = pd.DataFrame({"employee_id":pd.read_csv("../input/test.csv").employee_id, 
                        "is_promoted":model_gb})
solution.is_promoted=solution.is_promoted.astype("int64")
solution.to_csv("GB MODEL_feature reng.csv", index =False)


# In[171]:


features = pd.DataFrame(gb.feature_importances_, index = scaled_test.columns,
            columns = ["Features"])
features.sort_values(by = "Features").plot(kind = "barh", color = "green")


# In[172]:


get_ipython().system('pip install catboost')


# In[173]:


from catboost import CatBoostClassifier
cb=CatBoostClassifier()


# In[174]:


model_cb = cb.fit(scaled_train, dummytrain.is_promoted).predict(scaled_test)
solution = pd.DataFrame({"employee_id":pd.read_csv("test.csv").employee_id, 
                        "is_promoted":model_gb})
solution.is_promoted=solution.is_promoted.astype("int64")
solution.to_csv("CAT BOOST MODEL_feature reng.csv", index =False)


# In[175]:


features = pd.DataFrame(cb.feature_importances_, index = scaled_test.columns,
            columns = ["Features"])
features.sort_values(by = "Features").plot(kind = "barh", color = "magenta")


# Accuracy :
#         
#         0.4385964912280702.

# In[ ]:





# In[176]:


import pandas as pd
test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")

