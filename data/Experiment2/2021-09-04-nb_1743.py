#!/usr/bin/env python
# coding: utf-8

# In[1367]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
from matplotlib import colors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)


# In[1368]:


test = pd.read_csv("test.csv", sep=";")
train = pd.read_csv("train.csv", sep=";")


# In[1369]:


test.head()


# In[1370]:


train.head()


# In[1371]:


print((train.shape))
print((test.shape))


# In[1372]:


#check the numbers of samples and features
print(("The test data size before dropping Id feature is : {} ".format(test.shape)))

#Save the 'Id' column
test_ID = test['Id']

#Drop Id ở tập test
test.drop("Id", axis = 1, inplace = True)

print(("The test data size after dropping Id feature is : {} ".format(test.shape)))


# In[1373]:


train.columns


# In[1374]:


#kiểm tra có cái nào bị nan trong khung dữ liệu
train.isna().any()


# In[1375]:


train.info()


# In[1376]:


print(("Có {} duplicated value".format(train.duplicated().sum())))


# In[1377]:


print(("Có {} duplicated value".format(test.duplicated().sum())))


# In[1378]:


train.info()


# In[1379]:


test.info()


# In[1380]:


train.isna().any()


# ## EDA

# In[1381]:


#Tìm số lượng giá trị duy nhất có trong mỗi cột
train.nunique()


# In[1382]:


train.year.value_counts()


# In[1383]:


sns.countplot(x='fuel',data=train)
print((train.fuel.value_counts()))


# Hai loại fuel được sử dụng nhiều là Diesel và Petrol. Hai loại fuel rất ít dòng xe sử dụng (chiếm số lượng rất nhỏ) là LPG và CNG.

# In[1384]:


sns.countplot(x='transmission',data=train)
print((train.transmission.value_counts(normalize=True)*100))


# Nhiều loại xe sử dụng hộp số sàn hơn là hộp số tự động (hơn gấp 4 lần).

# In[1385]:


sns.countplot(x='owner',data=train)
print((train.owner.value_counts(normalize=True)*100))


# In[1386]:


sns.countplot(x='seats',data=train)
print((train.seats.value_counts()))


# Đa số các xe có 5 chỗ ngồi (4590/6000 chiếc ở tập train)

# In[1387]:


sns.countplot(x='seats',data=test)
print((test.seats.value_counts()))


# ## Outliers

# In[1388]:


#standardizing data
sellingprice_scaled = StandardScaler().fit_transform(train['selling_price'][:,np.newaxis]);
low_range = sellingprice_scaled[sellingprice_scaled[:,0].argsort()][:10]
high_range= sellingprice_scaled[sellingprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


# - Low range values khá là tương tự nhau và không quá xa 0.
# - High range values khá là xa không, có giá trị là 6.55 đến 11.6, đặc biệt chú ý đến range 7.295 và 11.605

# ## Bivariate analysis

# In[1389]:


train.describe()


# In[1390]:


# Selling_price và 'km_driven'
var = 'km_driven'
data = pd.concat([train['selling_price'], train[var]], axis=1)
data.plot.scatter(x=var, y='selling_price', ylim=(0,10000000));


# In[1391]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots
trace0 = go.Box(y=train['selling_price'],name='Selling_price')
trace1 = go.Box(y=data['km_driven'],name='km_driven')
trace2 = go.Box(y=train['seats'],name='seats')
trace3 = go.Box(y=train['year'],name='year')
fig = make_subplots(rows=2, cols=3)
fig.append_trace(trace0, row = 1, col = 1)
fig.append_trace(trace1, row = 1, col = 2)
fig.append_trace(trace2, row = 1, col = 3)
fig.append_trace(trace3, row = 2, col = 1)
fig.update_layout(width=800, height=400, title='Box Plot to check for outliers')

fig.show()


# In[1392]:


#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['km_driven'], train['selling_price'])
plt.ylabel('selling_price', fontsize=13)
plt.xlabel('km_driven', fontsize=13)
plt.show()


# km_driven: số km mà xe đã đi.
# xe đã đi càng nhiều (từ 200000 đổ ra) giá xe càng thấp.

# In[1393]:


var = 'seats'
data = pd.concat([train['selling_price'], train[var]], axis=1)
data.plot.scatter(x=var, y='selling_price', ylim=(0,800000));


# In[1394]:


var = 'year'
data = pd.concat([train['selling_price'], train[var]], axis=1)
data.plot.scatter(x=var, y='selling_price', ylim=(0,10000000));


# Max giá của những năm 2016 đến 2019 cao hơn hẳn max giá của những năm khác.

# In[1280]:


var = 'year'
data = pd.concat([train['selling_price'], train[var]], axis=1)
data.plot.scatter(x=var, y='selling_price', ylim=(0,800000));


# Min giá của xe từ năm 2015 (đặc biệt từ năm 2017) đến năm 2020 cao hơn hẳn min giá của những năm khác.

# In[1281]:


#transmission & selling_price
sns.barplot(y="transmission", x="selling_price", data=train)


# Giá của xe dùng hộp số tự động có khoảng rộng hơn nhiều so với hộp số sàn (Manual). Giá của xe dùng hộp số sàn có khoảng khá hẹp và có mức giá thấp.

# In[1282]:


#owner vs selling_price
sns.barplot(y="owner", x="selling_price", data=train)


# Test Drive Car có các xe có mức giá cao hơn hẳn những phân loại owner khác.

# In[1283]:


plt.figure(figsize=(10,6))
dc=train.copy()
plt.scatter(dc['seats'],dc['selling_price'],alpha=0.3)
plt.xlabel("seats")
plt.ylabel('selling_price')
plt.show()


# Có nhiều xe 5 chỗ ngồi và có khoảng giá khá rộng và có một số xe 5 chỗ có phân khúc giá khá cao.
# Không có nhiều xe 4 chỗ nhưng hầu hết đều ở phân khúc giá thấp, nhưng có một số xe ngoại lệ lại ở phân khúc giá cao.
# Xe 2 chỗ và 14 chỗ không có nhiều và hai loại xe đấy đều có phân khúc giá thấp.

# In[1284]:


#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# Vì không có giá trị |cor| >0.8 nên không có hiện tượng tự tương quan giữa các biến với nhau.
# Vậy không cần bỏ bất kì biến nào.
# 

# In[1285]:


#scatterplot
sns.set()
cols = ['selling_price', 'km_driven', 'year', 'seats']
sns.pairplot(train[cols], size = 2.5)
plt.show();


# Phân tích selling_price

# In[1192]:


sns.distplot(train['selling_price'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['selling_price'])
print(( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma)))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('selling_price distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['selling_price'], plot=plt)
plt.show()


# In[1193]:


#skewness and kurtosis
print(("Skewness: %f" % train['selling_price'].skew()))
print(("Kurtosis: %f" % train['selling_price'].kurt()))


# 
# The target variable bị lệch phải (right skewness). Vì các mô hình (tuyến tính) mong muốn dữ liệu được phân phối bình thường, chúng ta cần transform the varible và làm cho nó được phân phối bình thường hơn.

# In[1314]:


train.info()


# ### Missing data

# In[1395]:


#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_train = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_train


# In[1396]:


train = train.dropna()


# In[1397]:


y_train = train["selling_price"]


# In[1398]:


y_train


# In[1399]:


train.drop(['selling_price'], axis=1, inplace=True)


# In[1400]:


train.info()


# In[1401]:


test.info()


# In[1402]:


train.drop(['name'], axis=1, inplace=True)
test.drop(['name'], axis=1, inplace=True)


# In[1403]:


train[['mileage', 'unit']] = train['mileage'].str.split(' ', expand=True)
train[['mileage']]
train["mileage"] = train["mileage"].astype((float))


# In[1404]:


test['mileage'] = test['mileage'].str.replace(' kmpl','',True)
test['mileage'] = test['mileage'].str.replace(' km/kg','',True)
test['mileage'] = test['mileage'].astype(float)
test['mileage'].fillna(test['mileage'].mean(),inplace = True)


# In[1405]:


test.info()


# In[1406]:


train[['engine', 'unit']] = train['engine'].str.split(' ', expand=True)
train[['engine']]
train["engine"] = train["engine"].astype((float))


# In[1407]:


test['engine']


# In[1408]:


test['engine'] = test['engine'].str.replace(' CC','',True)
test['engine'] = test['engine'].astype(float)
test['engine'].fillna(test['engine'].mean(),inplace = True)


# In[1409]:


test.info()


# In[1410]:


train.info()


# In[1411]:


train.drop(['unit'], axis=1, inplace=True)


# In[1412]:


train[['max_power', 'unit']] = train['max_power'].str.split(' ', expand=True)
train['max_power'] = train['max_power'].astype((float))


# In[1413]:


test['max_power']


# In[1414]:


test['max_power'] = test['max_power'].str.replace(' bhp','',True)
test['max_power'] = test['max_power'].astype(float)
test['max_power'].fillna(test['max_power'].mean(),inplace = True)


# In[1415]:


test['max_power'] = test['max_power'].astype((float))


# In[1416]:


test.info()


# In[1417]:


train.info()


# In[1418]:


train.drop(['unit'], axis=1, inplace=True)


# In[1419]:


#scatterplot
sns.set()
cols = ['km_driven', 'year', 'seats','mileage','engine','max_power']
sns.pairplot(train[cols], size = 5.0)
plt.show();


# In[1420]:


train.drop(['torque'], axis=1, inplace=True)
test.drop(['torque'], axis=1, inplace=True)


# In[1421]:


test['seats'] = test['seats'].astype(float)
test['seats'].fillna(test['seats'].mean(),inplace = True)


# In[1422]:


test.info()


# In[1423]:


train.info()


# In[1424]:


#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# Vì không có giá trị |cor| >0.8 nên không có hiện tượng tự tương quan giữa các biến với nhau. Vậy không cần bỏ bất kì biến nào.

# Label Encoding

# In[1425]:


# Nhận danh sách các biến phân loại
s = (train.dtypes == 'object')
object_cols = list(s[s].index)

print(("Các biến phân loại trong tập bộ liệu:", object_cols))


# In[1426]:


from sklearn.preprocessing import LabelEncoder
#Label Encoding the object dtypes
LE=LabelEncoder()
for i in object_cols:
  train[i]=train[[i]].apply(LE.fit_transform)


# In[1427]:


# Nhận danh sách các biến phân loại
s = (test.dtypes == 'object')
object_cols = list(s[s].index)

print(("Các biến phân loại trong tập bộ liệu:", object_cols))



# In[1428]:


from sklearn.preprocessing import LabelEncoder
#Label Encoding the object dtypes
LE=LabelEncoder()
for i in object_cols:
  test[i]=test[[i]].apply(LE.fit_transform)


# In[1429]:


test.info()


# In[1430]:


numeric_feats = train.dtypes[train.dtypes != "object"].index
from scipy.stats import skew

# Check the skew of all numerical features
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
print(skewness)

from scipy.stats import kurtosis
kurt_feats = train[numeric_feats].apply(lambda x: kurtosis(x.dropna())).sort_values(ascending=False)
print("\nKurt in numerical features: \n")
kurtosis = pd.DataFrame({'Kurt' :kurt_feats})
print(kurtosis)


# In[1431]:


#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# ### Training Model

# In[1432]:


y_train 


# In[1433]:


#Getting dummy categorical features
train = pd.get_dummies(train)
print((train.shape))


# In[1434]:


train.head()


# In[1435]:


train.info()


# In[1436]:


test.info()


# # Modelling

# In[1437]:


from sklearn.ensemble import ExtraTreesRegressor


# In[1438]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(train, y_train, test_size = 0.2, random_state = 25)


# In[1439]:


rf = ExtraTreesRegressor()
rf.fit(X_train, Y_train)
pred_test = rf.predict(X_test)
print((rf.score(X_train,Y_train)))
print((rf.score(X_test,Y_test)))



# In[1444]:


from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

print(('Mean Absolute Error      : ', metrics.mean_absolute_error(Y_test, pred_test)))
print(('Mean Squared  Error      : ', metrics.mean_squared_error(Y_test, pred_test)))
print(('Root Mean Squared  Error : ', np.sqrt(metrics.mean_squared_error(Y_test, pred_test))))
print(('R Squared Error          : ', metrics.r2_score(Y_test, pred_test)))


# In[1441]:


sns.distplot(Y_test-pred_test)
plt.show()


# In[1442]:


pred = rf.predict(test)


# In[1443]:


pre = pd.DataFrame()
pre['Id'] = test_ID
pre['Predicted'] = pred
pre.to_csv('prediction.csv',index=False)


# In[ ]:




