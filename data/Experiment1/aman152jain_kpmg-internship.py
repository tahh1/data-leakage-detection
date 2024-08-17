#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/econdse'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


transactions = pd.read_excel("/kaggle/input/econdse/B1863B00.xlsx", sheet_name='Transactions')
new_customer_lists = pd.read_excel("/kaggle/input/econdse/B1863B00.xlsx", sheet_name='NewCustomerList')
customer_demographic = pd.read_excel("/kaggle/input/econdse/B1863B00.xlsx", sheet_name='CustomerDemographic')
customer_add = pd.read_excel("/kaggle/input/econdse/B1863B00.xlsx", sheet_name='CustomerAddress')


# In[3]:


transactions.info()


# In[4]:


transactions.isnull().sum()


# In[5]:


new_customer_lists.head()


# MERGE ALL THE INPUT VARIABLE i.e. TRANSACTIONS, CUSTOMER DEMOGRAPHIC AND CUSTOMER ADDRESS INTO ONE SEPARATE FILE NAMED AS 'df' ON THE BASIS OF CUSTOMER ID.
# 

# In[6]:


temp1 = pd.merge(customer_add,customer_demographic, how = 'outer', on = 'customer_id')
df = pd.merge(transactions,temp1, how ='outer', on = 'customer_id')


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


df.columns


# In[11]:


new_customer_lists.columns


# DROPING FOR SOME UNUSEFUL VARIABLE FROM df AND new_customer_lists SO THAT WE CAN RUN THE CLASSIFICATION TECHNIQUE ON THE MODEL

# In[12]:


df= df.drop(['online_order', 'address', 'postcode', 'country','first_name', 'last_name','DOB','job_title','deceased_indicator','transaction_id', 'product_id','transaction_date',
       'order_status', 'brand', 'product_line', 'product_class',
       'product_size', 'list_price', 'standard_cost',
       'product_first_sold_date','Unnamed: 15',
       'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19'] , axis= 1)


# In[13]:


new_customer_lists= new_customer_lists.drop([ 'address', 'postcode', 'country','first_name', 'last_name','DOB','job_title','deceased_indicator', 'Rank', 'Value'] , axis= 1)


# In[14]:


new_customer_lists.columns


# In[15]:


df.columns


# In[16]:


df.head()


# In[17]:


new_customer_lists['state']= new_customer_lists['state'].replace('NSW','New South Wales')
new_customer_lists['state']= new_customer_lists['state'].replace('VIC','Victoria')


# In[18]:


new_customer_lists.head()


# NOW WE ARE UPLOADING NEW DATA SET OR YOU CAN SAY BETTER VERSION OF DATA SET
# THINGS DONE IN THAT NEW DATA SET
# 1. REMOVAL OF ALL THE MISSING VALUES IN THE df
# 2. Provide CATAGORICAL VARTIABLE A SEPARATE VALUE E.G. MALE =1 , FEMALE = 0, YES=1, NO= 0 
# 3. REMOVAL OF UNDEFINED CATAGARY FROM GENDER
# 4. ROUNDED OFF THE AGE

# In[19]:


df1= pd.read_excel ("../input/econdsekpmg-virtual-internship-2020/export_df1 (9).xlsx", sheet_name= 'final')
aa= pd.read_excel (r"../input/econdsekpmg-virtual-internship-2020/new (2).xlsx", sheet_name= 'Sheet2')


# In[20]:


df2= pd.get_dummies(df1, drop_first= True)
nc= pd.get_dummies(aa, drop_first= True)


# In[21]:


df2.head()


# In[22]:


nc.head()


# In[23]:


X = df2.drop('Range', axis=1)
y = df2['Range']


# In[24]:


nc1= nc.drop('customer_id', axis=1)


# In[25]:


y.head()


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[27]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


# In[28]:


logreg = LogisticRegression()
a= logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, y_train) * 100, 2)
cm = confusion_matrix(y_test, Y_pred)
print(cm)
acc_log


# In[29]:


y_prob_train= logreg.predict_proba(X_train)[:, 1]
y_prob_train.reshape(1,-1)
y_prob_train


# In[30]:


y_prob= logreg.predict_proba(X_test)[:, 1]
y_prob.reshape(1,-1)
y_prob


# In[31]:


from sklearn.metrics import classification_report
print((classification_report(y_test,Y_pred)))

tn,fp,fn,tp = cm.ravel()
print(('TRUE NEGATIVE:', tn))
print(('FALSE POSITIVE:', fp))
print(('FALSE NEGATIVE:', fn))
print(('TRUE POSITIVE:', tn))

specificity = tn/(tn+fp)
print(('specificity {:0.2f}'.format(specificity)))
sensitivity = tp/(tp+fn)
print(('sensitivity {:0.2f}'.format(sensitivity)))


# In[32]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
log_roc_auc= roc_auc_score(y_train,y_prob_train )
fpr, tpr, thresholds = roc_curve(y_train, y_prob_train )
roc_auc= auc(fpr,tpr)


# In[33]:


plt.figure()
plt.plot(fpr, tpr, color= "blue" , label = 'ROC CURVE (area= %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0 ,1.0])
plt.ylim([0.0 ,1.05])  
plt.xlabel('FALSE POSITIVE RATE')
plt.ylabel('TRUE POSITIVE RATE')
plt.title('ROC CURVE')
plt.legend(loc= 'lower right')
plt.show()


# In[34]:


log_roc_auc= roc_auc_score(y_test,y_prob)
fpr1, tpr1, thresholds = roc_curve(y_test,y_prob)
roc_auc= auc(fpr1,tpr1)


# In[35]:


plt.figure()
plt.plot(fpr1, tpr1, color= "blue" , label = 'ROC CURVE (area= %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0 ,1.0])
plt.ylim([0.0 ,1.05])  
plt.xlabel('FALSE POSITIVE RATE')
plt.ylabel('TRUE POSITIVE RATE')
plt.title('ROC CURVE')
plt.legend(loc= 'lower right')
plt.show()


# In[36]:


svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train) * 100, 2)
cm = confusion_matrix(y_test, Y_pred)
print(cm)
acc_svc


# In[37]:


from sklearn.metrics import classification_report
print((classification_report(y_test,Y_pred)))

tn,fp,fn,tp = cm.ravel()
print(('TRUE NEGATIVE:', tn))
print(('FALSE POSITIVE:', fp))
print(('FALSE NEGATIVE:', fn))
print(('TRUE POSITIVE:', tn))

specificity = tn/(tn+fp)
print(('specificity {:0.2f}'.format(specificity)))
sensitivity = tp/(tp+fn)
print(('sensitivity {:0.2f}'.format(sensitivity)))


# In[38]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
cm = confusion_matrix(y_test, Y_pred)
print(cm)
acc_knn


# In[39]:


from sklearn.metrics import classification_report
print((classification_report(y_test,Y_pred)))

tn,fp,fn,tp = cm.ravel()
print(('TRUE NEGATIVE:', tn))
print(('FALSE POSITIVE:', fp))
print(('FALSE NEGATIVE:', fn))
print(('TRUE POSITIVE:', tn))

specificity = tn/(tn+fp)
print(('specificity {:0.2f}'.format(specificity)))
sensitivity = tp/(tp+fn)
print(('sensitivity {:0.2f}'.format(sensitivity)))


# In[40]:


linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)
cm = confusion_matrix(y_test, Y_pred)
print(cm)
acc_linear_svc


# In[41]:


from sklearn.metrics import classification_report
print((classification_report(y_test,Y_pred)))

tn,fp,fn,tp = cm.ravel()
print(('TRUE NEGATIVE:', tn))
print(('FALSE POSITIVE:', fp))
print(('FALSE NEGATIVE:', fn))
print(('TRUE POSITIVE:', tn))

specificity = tn/(tn+fp)
print(('specificity {:0.2f}'.format(specificity)))
sensitivity = tp/(tp+fn)
print(('sensitivity {:0.2f}'.format(sensitivity)))


# In[42]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
cm = confusion_matrix(y_test, Y_pred)
print(cm)
acc_decision_tree


# In[43]:


from sklearn.metrics import classification_report
print((classification_report(y_test,Y_pred)))

tn,fp,fn,tp = cm.ravel()
print(('TRUE NEGATIVE:', tn))
print(('FALSE POSITIVE:', fp))
print(('FALSE NEGATIVE:', fn))
print(('TRUE POSITIVE:', tn))

specificity = tn/(tn+fp)
print(('specificity {:0.2f}'.format(specificity)))
sensitivity = tp/(tp+fn)
print(('sensitivity {:0.2f}'.format(sensitivity)))


# In[44]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
cm = confusion_matrix(y_test, Y_pred)
print(cm)
acc_random_forest


# In[45]:


from sklearn.metrics import classification_report
print((classification_report(y_test,Y_pred)))

tn,fp,fn,tp = cm.ravel()
print(('TRUE NEGATIVE:', tn))
print(('FALSE POSITIVE:', fp))
print(('FALSE NEGATIVE:', fn))
print(('TRUE POSITIVE:', tn))

specificity = tn/(tn+fp)
print(('specificity {:0.2f}'.format(specificity)))
sensitivity = tp/(tp+fn)
print(('sensitivity {:0.2f}'.format(sensitivity)))


# In[46]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[47]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(nc1)


# In[48]:


y_pred


# In[49]:


submission = pd.DataFrame({
        "customer_id": nc["customer_id"],
        "Target": y_pred
    })


# In[50]:


submission


# In[51]:


submission['Target'].value_counts()


# **NOW WE ARE PLANNING TO ANSWER ONE MORE QUESTION I.E. IS IT CORRECT TO DROP SO MUCH VARIABLES AND IS THIS VARIABLES PLAY A SIGNIFICANT ROLL IN IT ?**

# In[52]:


lr = pd.read_excel("../input/econdsekpmg-virtual-internship-2020/linear regression.xlsx")


# In[53]:


import matplotlib as mpl
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression
from scipy import stats
import seaborn as sns


# In[54]:


lr.columns


# In[55]:


df= pd.get_dummies(lr, drop_first= True)


# In[56]:


df.columns


# In[57]:


df = df.rename(columns = {'brand_Norco Bicycles':'brand_Norco_Bicycles'})
df = df.rename(columns = {'brand_OHM Cycles':'brand_OHM_Cycles'})
df = df.rename(columns = {'brand_Trek Bicycles':'brand_Trek_Bicycles'})


# In[58]:


df.columns


# In[59]:


df.head(10)


# In[60]:


df.info()


# In[61]:


df.describe()


# In[62]:


X = df[['online_order', 'brand_Norco_Bicycles', 'brand_OHM_Cycles',
       'brand_Solex', 'brand_Trek_Bicycles', 'brand_WeareA2B',
       'product_line_Road', 'product_line_Standard', 'product_line_Touring',
       'product_class_low', 'product_class_medium', 'product_size_medium','product_size_small']]
y = df['profit']


# In[63]:


plt.figure(figsize= (10,10))
sns.heatmap(df.corr(),annot= True, cmap= "coolwarm")


# In[64]:


from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols


# In[65]:


reg= ols(formula= "profit ~ online_order + brand_Norco_Bicycles + brand_OHM_Cycles + brand_Solex+ brand_Trek_Bicycles + brand_WeareA2B + product_line_Road + product_line_Standard+ product_line_Touring  + product_class_low + product_class_medium + product_size_medium + product_size_small", data= df)
fit1= reg.fit()
print((fit1.summary()))


# In[66]:


reg= ols(formula= "profit ~ brand_Norco_Bicycles + brand_OHM_Cycles + brand_Solex+ brand_Trek_Bicycles + brand_WeareA2B + product_line_Road + product_line_Standard+ product_line_Touring  + product_class_low + product_class_medium + product_size_medium + product_size_small", data= df)
fit1= reg.fit(cov_type='HC1')
print((fit1.summary()))


# In[67]:


reg= ols(formula= "profit ~ brand_Norco_Bicycles + brand_OHM_Cycles + brand_Solex+ brand_Trek_Bicycles + brand_WeareA2B + product_line_Road + product_line_Standard+ product_line_Touring  + product_class_low + product_class_medium + product_size_medium + product_size_small", data= df)
fit1= reg.fit()
print((fit1.summary()))


# In[68]:


from statsmodels.formula.api import gls
reg= gls(formula= "profit ~ online_order + brand_Norco_Bicycles + brand_OHM_Cycles + brand_Solex+ brand_Trek_Bicycles + brand_WeareA2B + product_line_Road + product_line_Standard+ product_line_Touring  + product_class_low + product_class_medium + product_size_medium + product_size_small", data= df)
fit1= reg.fit()
print((fit1.summary()))


# In[69]:


reg= gls(formula= "profit ~ online_order + brand_Norco_Bicycles + brand_OHM_Cycles + brand_Solex+ brand_Trek_Bicycles + brand_WeareA2B + product_line_Road + product_line_Standard+ product_line_Touring  + product_class_low + product_class_medium + product_size_medium + product_size_small", data= df)
fit1= reg.fit(cov_type='HC1')
print((fit1.summary()))


# In[70]:


reg= ols(formula= "profit ~ brand_Norco_Bicycles + brand_OHM_Cycles + brand_Solex+ brand_Trek_Bicycles + brand_WeareA2B + product_line_Road + product_line_Standard+ product_line_Touring  + product_class_low + product_class_medium + product_size_medium + product_size_small", data= df)
fit1= reg.fit(cov_type='HC1')
print((fit1.summary()))


# ***OTHER THAN ONLINE ORDER, EVERY VARIABLE PROVIDE A SIGNIFICANT ROLE IN DETERMINE PROFIT***

# In[ ]:




