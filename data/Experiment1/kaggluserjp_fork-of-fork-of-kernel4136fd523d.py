#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


#DS_307_b 仮説2


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


#データセットの読み込み
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sample_submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')


# In[5]:


#データセットの確認
train


# In[6]:


test


# In[7]:


sample_submission


# In[8]:


#'SalePrice'のヒストグラムを確認
plt.figure(figsize = (20,10))
sns.distplot(train['SalePrice'])


# In[9]:


#'SalePrice'のヒストグラムを確認　対数変換してみる
plt.figure(figsize = (20,10))
sns.distplot(np.log1p(train['SalePrice']))


# In[10]:


train['SalePrice'] = np.log1p(train['SalePrice'])


# In[11]:


train


# In[12]:


#カラムの表示
train.columns


# In[13]:


#ひとまず、相関をとってみる
tc = train.corr()
#tc


# In[14]:


tc1 = tc['SalePrice']
tc1


# In[15]:


#ケース１          そのまま   対数変換
#広さ
#1stFlrSF         0.605852 0.596981
#2ndFlrSF         0.319334 0.319300
#TotalBsmtSF      0.613581 0.612134

#築年数
#YearBuilt        0.522897 0.586570
#YearRemodAdd     0.507101 0.565608

#立地


# In[16]:


#可視化してみる
#散布図を描いてみる
#'1stFlrSF'1Fの広さ
plt.figure(figsize = (20,10))
sns.scatterplot(x = '1stFlrSF', y = 'SalePrice',data = train)


# In[17]:


#'2ndFlrSF'2Fの広さ
plt.figure(figsize = (20,10))
sns.scatterplot(x = '2ndFlrSF', y = 'SalePrice',data = train)


# In[18]:


#'TotalBsmtSF'地下の広さ
plt.figure(figsize = (20,10))
sns.scatterplot(x = 'TotalBsmtSF', y = 'SalePrice',data = train)


# In[19]:


#'YearBuilt'建築年
plt.figure(figsize = (20,10))
sns.scatterplot(x = 'YearBuilt', y = 'SalePrice',data = train)


# In[20]:


#'YearRemodAdd'改修年
plt.figure(figsize = (20,10))
sns.scatterplot(x = 'YearRemodAdd', y = 'SalePrice',data = train)


# In[21]:


#'1stFlrSF'と'2ndFlrSF'の相関を見てみる
tc_1st = tc['1stFlrSF']
tc_1st['2ndFlrSF']


# In[22]:


#弱い負の相関がある


# In[23]:


#'1stFlrSF'と'TotalBsmtSF'の相関を見てみる
tc_1st['TotalBsmtSF']


# In[24]:


#強い正の相関がある＞片方を削除してみる＞仮に'TotalBsmtSF'を採用してみる(パート1)


# In[25]:


#'YearBuilt'と'YearRemodAdd'の相関を見てみる
tc_year = tc['YearBuilt']
tc_year['YearRemodAdd']


# In[26]:


#正の相関がある＞片方を削除してみる＞仮に'YearBuilt'を採用してみる(パート1&2)


# In[27]:


#新しい説明変数を追加(パート2)
#延べ床面積を追加
train['TotalFloorArea'] = train['1stFlrSF'] + train['2ndFlrSF'] + train['TotalBsmtSF']
train


# In[28]:


#'TotalFloorArea'延べ床面積
plt.figure(figsize = (20,10))
sns.scatterplot(x = 'TotalFloorArea', y = 'SalePrice',data = train)


# In[29]:


#多重比較法で立地の水準間に有意差があるかを確認
#'Neighborhood'
from statsmodels.stats.multicomp import (pairwise_tukeyhsd,MultiComparison)
res = pairwise_tukeyhsd(train['SalePrice'], groups = train['Neighborhood'], alpha=0.01)
print((res.summary()))


# In[30]:


res.plot_simultaneous()


# In[31]:


#多重比較法で立地の水準間に有意差があるかを確認
#'Condition1'
from statsmodels.stats.multicomp import (pairwise_tukeyhsd,MultiComparison)
res = pairwise_tukeyhsd(train['SalePrice'], groups = train['Condition1'], alpha=0.01)
print((res.summary()))


# In[32]:


res.plot_simultaneous()


# In[33]:


#多重比較法で立地の水準間に有意差があるかを確認
#'Condition2'
from statsmodels.stats.multicomp import (pairwise_tukeyhsd,MultiComparison)
res = pairwise_tukeyhsd(train['SalePrice'], groups = train['Condition2'], alpha=0.01)
print((res.summary()))


# In[34]:


res.plot_simultaneous()


# In[35]:


#Conditionの各水準間に有意差はほぼない


# In[36]:


#仮説2の場合
tc2 = tc1[abs(tc1) >= 0.5]
tc2


# In[37]:


#相関係数が>0.5のもの
#OverallQual     0.790982
#YearBuilt       0.522897
#YearRemodAdd    0.507101
#TotalBsmtSF     0.613581
#1stFlrSF        0.605852
#GrLivArea       0.708624
#FullBath        0.560664
#TotRmsAbvGrd    0.533723
#GarageYrBlt     0.541073
#GarageCars      0.640409
#GarageArea      0.623431


# In[38]:


train[tc2.index].corr()


# In[39]:


#特にTotalBsmtSFと1stFlrSF、GarageCarsとGarageAreaに相関がある


# In[40]:


#TotalBsmtSFとGarageCarsを説明変数として採用してみる


# In[41]:


#散布図を描いてみる
#'OverallQual'家の全体的な素材と仕上げを評価
plt.figure(figsize = (20,10))
sns.scatterplot(x = 'OverallQual', y = 'SalePrice',data = train)


# In[42]:


#'YearBuilt'築年数
plt.figure(figsize = (20,10))
sns.scatterplot(x = 'YearBuilt', y = 'SalePrice',data = train)


# In[43]:


#'YearRemodAdd'改装年
plt.figure(figsize = (20,10))
sns.scatterplot(x = 'YearRemodAdd', y = 'SalePrice',data = train)


# In[44]:


#'TotalBsmtSF'地下の広さ
plt.figure(figsize = (20,10))
sns.scatterplot(x = 'TotalBsmtSF', y = 'SalePrice',data = train)


# In[45]:


#'1stFlrSF'1階の広さ
plt.figure(figsize = (20,10))
sns.scatterplot(x = '1stFlrSF', y = 'SalePrice',data = train)


# In[46]:


#'GrLivArea'グレード（地上）のリビングエリアの平方フィート
plt.figure(figsize = (20,10))
sns.scatterplot(x = 'GrLivArea', y = 'SalePrice',data = train)


# In[47]:


#'FullBath'グレードを超えるフルバスルーム
plt.figure(figsize = (20,10))
sns.scatterplot(x = 'FullBath', y = 'SalePrice',data = train)


# In[48]:


#'TotRmsAbvGrd'グレードを超える部屋の合計
plt.figure(figsize = (20,10))
sns.scatterplot(x = 'TotRmsAbvGrd', y = 'SalePrice',data = train)


# In[49]:


#'GarageYrBlt'ガレージが建てられた年
plt.figure(figsize = (20,10))
sns.scatterplot(x = 'GarageYrBlt', y = 'SalePrice',data = train)


# In[50]:


#'GarageCars'ガレージの大きさ
plt.figure(figsize = (20,10))
sns.scatterplot(x = 'GarageCars', y = 'SalePrice',data = train)


# In[51]:


#'GarageArea'ガレージの大きさ
plt.figure(figsize = (20,10))
sns.scatterplot(x = 'GarageArea', y = 'SalePrice',data = train)


# In[52]:


#仮説2ここまで


# In[53]:


#訓練データとテストデータを行方向に結合
test['TotalFloorArea'] = test['1stFlrSF'] + test['2ndFlrSF'] + test['TotalBsmtSF']
data = pd.concat([train,test], sort = False)
data


# In[54]:


#目的変数'SalePrice'と'Id'削除
x = data.drop(['Id','SalePrice'], axis = 1)
x


# In[55]:


#欠損の状況を確認
x_na = x.isnull().sum()[x.isnull().sum()>0]
x_na


# In[56]:


#欠損値の型を確認
x[x_na.index].dtypes


# In[57]:


#説明変数を分類
#説明変数が数字のものだけを抽出
x_i_i = x.dtypes[x.dtypes=='int64'].index
x_i = x[x_i_i]
x_i


# In[58]:


#説明変数が少数を含むものだけを抽出
x_f_i = x.dtypes[x.dtypes=='float64'].index
x_f = x[x_f_i]
x_f


# In[59]:


#説明変数が文字のものだけを抽出
x_o_i = x.dtypes[x.dtypes=='object'].index
x_o = x[x_o_i]
x_o


# In[60]:


#少数を含む説明変数の欠損値を中央値で置き換える
#for i in x_f_i:
#    x_f.loc[x_f[i].isnull(),i] = 0
#x_f

for i in x_f_i:
    x_f[i].fillna(x_f[i].median(skipna = True), inplace = True)
x_f


# In[61]:


#再度欠損状況を確認
x_f.isnull().sum()


# In[62]:


#文字を含む説明変数の欠損値を"NA"で置き換える
for i in x_o_i:
    x_o.loc[x_o[i].isnull(),i] = 'NA'
x_o


# In[63]:


#再度欠損状況を確認
x_o.isnull().sum()


# In[64]:


#数字の説明変数を標準化
x_i_train = x_i[:len(train)]
x_i_test = x_i[len(train):]

from sklearn.preprocessing import StandardScaler
stds = StandardScaler()
x_i_train = pd.DataFrame(stds.fit_transform(x_i_train))
x_i_test = pd.DataFrame(stds.transform(x_i_test))
x_i[x_i.columns.tolist()] = pd.concat([x_i_train,x_i_test],sort = False)
x_i_s = x_i
x_i_s

#from sklearn.preprocessing import StandardScaler
#train_standard = StandardScaler()
#train_copied = x_i.copy()
#train_standard.fit(x_i)
#train_std = pd.DataFrame(train_standard.transform(x_i))
#x_i[x_i.columns.tolist()] = train_std
#x_i_s = x_i
#x_i_s


# In[65]:


#少数を含む数字の説明変数を標準化
x_f_train = x_f[:len(train)]
x_f_test = x_f[len(train):]

from sklearn.preprocessing import StandardScaler
stds = StandardScaler()
x_f_train = pd.DataFrame(stds.fit_transform(x_f_train))
x_f_test = pd.DataFrame(stds.transform(x_f_test))
x_f[x_f.columns.tolist()] = pd.concat([x_f_train,x_f_test],sort = False)
x_f_s = x_f
x_f_s

#from sklearn.preprocessing import StandardScaler
#train_standard = StandardScaler()
#train_copied = x_f.copy()
#train_standard.fit(x_f)
#train_std = pd.DataFrame(train_standard.transform(x_f))
#x_f[x_f.columns.tolist()] = train_std
#x_f_s = x_f
#x_f_s


# In[66]:


#文字を含む説明変数をダミー変数化
x_o_neighbor = x_o['Neighborhood']#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<ケース１
x_o_d = pd.get_dummies(x_o, drop_first = True)#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<ケース１_neighbor_neighbor
x_o_d


# In[67]:


#数字の説明変数を結合
x_n = pd.concat([x_i_s,x_f_s],axis=1,sort = False)
x_n


# In[68]:


#使用する説明変数を抽出してカテゴリ変数を追加
x_n_c1 = ["OverallQual","GrLivArea","GarageArea","GarageCars","1stFlrSF","TotalBsmtSF"]
x_n_c2 = ["OverallQual","GrLivArea","GarageCars","TotalBsmtSF"]
x_n_c3 = ["2ndFlrSF","TotalBsmtSF","YearBuilt"]#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<ケース１
x_n_c4 = ["TotalFloorArea","YearBuilt"]#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<ケース１
x_n_c5 = ["TotalFloorArea","YearBuilt","OverallQual","GrLivArea","GarageCars"]#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<ケース１
x_n_c6 = ["GarageCars","TotalFloorArea","GarageYrBlt","OverallQual","GrLivArea","FullBath","TotRmsAbvGrd"]#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<ケース１
x_final = pd.concat([x_n[x_n_c6],x_o_d],axis = 1,sort = False)##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<ケース１
x_final


# In[69]:


#訓練データとテストデータを分割
x_train = x_final[:len(train)]
x_train


# In[70]:


x_test = x_final[len(train):]
x_test


# In[71]:


#正解確認
y_train = train['SalePrice']
#y_train = np.log1p(y_train)
y_train


# In[72]:


#訓練データを分割し80%を学習データに、20%をテストデータにする
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)


# In[73]:


#学習する
#sklearn通常解析
#from sklearn.linear_model import LinearRegression
#model = LinearRegression()

#ロジスティック回帰の場合
#from sklearn.linear_model import LogisticRegression
#model = LogisticRegression()

#Lasso
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.001, max_iter=1000000)

model.fit(X_train,Y_train)


# In[74]:


#学習データを予想
train_predicted = model.predict(X_train)
train_predicted


# In[75]:


#正解確認
Y_train


# In[76]:


#テストデータを予想
#test_predicted = model.predict(X_test)
#test_predicted


# In[77]:


#正解確認
Y_test


# In[78]:


print(f"学習データの予想精度: {model.score(X_train, Y_train):.2}")
print(f"テストデータの予想精度: {model.score(X_test, Y_test):.2f}")


# In[79]:


#提出用テストデータを予想
test_predicted2 = model.predict(x_test)
test_predicted2


# In[80]:


#提出用ファイルの作成
sample_submission['SalePrice'] = list(map(int,np.exp(test_predicted2)))
sample_submission.to_csv('submission.csv', index = False)
sample_submission


# In[81]:


#プログラムここまで

