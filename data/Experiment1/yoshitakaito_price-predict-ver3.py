#!/usr/bin/env python
# coding: utf-8

# # (1) 内容  
# * Ver2では、欠損値補完を「KNN」と「中央値・最頻値」の２種類で比較したが、そんなに違いなかったので、欠損値は「中央値・最頻値」で埋める。
# * 外れ値は「5%tile値で置き換え」と「対数変換で外れ値を均す」２種類を比較したが、全般的に「5%tile値で置き換え」の方がスコアが高かったので「5%tile値で置き換え」を使う。
# * 欠損値の埋め方は、「中央値・最頻値で埋める方法」と「KNNで埋める方法」の両方試す
# * 欠損値が２割以上の変数は削除する(Ver2のまま)
# * Categoricalの名義変数は、Binary Encodingで変換する(Ver2のまま)
# * Categoricalの順序変数は、Ordinal Encodingで変換する(Ver2のまま)  
# * 正規化する(Ver2のまま)
#   
# 以下Ver3での追加
# * SalePriceと相関の低い変数の削除。
# * 独立変数間で相関の高い変数の削除。
# * 使うモデルは、Random Forest、Gradient Boosting、lightgbm　の３種
# 

# In[337]:


import numpy as np
import pandas as pd
from scipy import stats
import collections
import itertools
import math
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
sns.set_style('whitegrid')

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from category_encoders import BinaryEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import optuna.integration.lightgbm as lgb_optuna
import optuna


# In[338]:


# トレインデータ取り出し
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
# テストデータ取り出し
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# データ合体
combined_df = pd.concat([train_df, test_df], ignore_index=True)
combined_df.drop('SalePrice',axis=1, inplace=True)

train_SalePrice = train_df['SalePrice'] # target value
train_df.drop('SalePrice',axis=1, inplace=True)


# # (2) 前処理 
# 
# ### 以下の順番で前処理実施  
# 1. 変数(特徴量)の確認
# 1. 定数と変わらない変数や内容が同じ変数の削除
# 1. 各変数の分析
# 1. ２変数間の分析 -> 独立変数間で相関高い変数の削除。SalePriceと関係無い変数の削除。
# 1. 外れ値の処理 
# 1. 欠損値の処理：欠損値が２割以上の変数は削除
# 1. 変数の創出  
# 1. 変数変換  
# 

# ## 1. 変数の確認  

# In[339]:


#train_df.info()


# In[340]:


# 変数削除処理用のdf用意
# 簡易的にラベルエンコーディングするので、ここで分ける。
del_col_train = train_df.copy()


# In[341]:


train_df.tail()


# ### ・欠損値データの確認  

# In[342]:


null_col_train = ''
null_col_test = ''
null_train_list = [] # train_dfの欠損値有り変数のリスト
null_test_list = [] # test_dfの欠損値有り変数のリスト

# train_dfの欠損値有り変数取り出し
for col in train_df.columns:
    check = train_df[col].isnull().value_counts()
    if len(check) != 1:
        # 欠損値有り変数
        null_col_train = null_col_train + col + ', '
        null_train_list.append(col)

# test_dfの欠損値有り変数取り出し
for col in test_df.columns:
    check = test_df[col].isnull().value_counts()
    if len(check) != 1:
        # 欠損値有り変数
        null_col_test = null_col_test + col + ', '
        null_test_list.append(col)

print(('訓練データの欠損値有り変数の数： ',len(null_train_list)))
print(('変数名：', null_col_train, '\n'))
print(('テストデータの欠損値有り変数の数： ',len(null_test_list)))
print(('変数名：', null_col_test))


# ### 欠損値が２割以上の変数削除  
# + データが８割以上ある変数のみモデリングに利用する
# + 訓練・テストデータ合わせたデータにおいて、欠損値が多いデータは、補完が難しいので削除する

# In[343]:


# 欠損値数のボーダー（訓練データ・テストデータ合計の８割以上ある変数のみ利用）
null_border = 2919 * 0.2

# trainデータで、欠損値が8割以上の変数抽出
null_count = combined_df.isnull().sum()
null_col_list = null_count[null_count >= null_border]
null_col_list = null_col_list.index.to_list()
print(('削除対象変数：', null_col_list))

# 欠損値が8割以上の変数削除
del_col_train = del_col_train.drop(null_col_list, axis=1)
train_df = train_df.drop(null_col_list, axis=1)
combined_df = combined_df.drop(null_col_list, axis=1)


# In[344]:


# 残りの変数の欠損値状況
combined_df.isnull().sum().sort_values(ascending=False)[:30]


# ## 2. 定数と変わらない変数や、内容が同じ変数の削除  
# Categorical変数はlabel encodingを行い、実質的に定数の変数を削除する
# 

# #### ・Label Encoding用に欠損値を最頻値で一時的に埋めて、Label Encodingで数値化

# In[345]:


# 各変数の欠損値を最頻値で埋める
# mode()はdfで返すので、先頭データを取り出す。
del_col_train.fillna(del_col_train.mode().iloc[0], inplace=True)
# del_col_test.fillna(del_col_test.mode().iloc[0], inplace=True)

# Label Encoding対象列取り出し
object_list = del_col_train.select_dtypes(include='object').columns.to_list()
object_df = del_col_train[object_list]

# Label Encoding
object_df = object_df.apply(LabelEncoder().fit_transform)
del_col_train[object_df.columns] = object_df


# #### ・定数と変わらない変数の削除

# In[346]:


# 分散0.1以下の変数削除
constant_filter = VarianceThreshold(threshold=0.1)
constant_filter.fit(del_col_train)
constant_list = [not temp for temp in constant_filter.get_support()]
# constant_listの中身は[False,True,False...]と各変数が対象かどうかを示すリスト
x_train_filter = constant_filter.transform(del_col_train)
x_train_filter.shape, del_col_train.shape


# ８変数がほぼ定数みたいな変数

# In[347]:


# 削除対象変数抽出
del_list = del_col_train.columns[constant_list]
print(('削除対象変数：',del_list))

# 変数削除
del_col_train = del_col_train.drop(del_list, axis=1) # 同じ変数削除処理用
train_df = train_df.drop(del_list, axis=1)
combined_df = combined_df.drop(del_list, axis=1)


# #### ・内容がほぼ同じ変数の削除  
# 同じ内容の変数が２つ以上ある場合は、それら変数を削除する

# In[348]:


# データを転置してduplicatedメソッドで重複行削除する
del_col_train_T = del_col_train.T
del_col_train_T = pd.DataFrame(del_col_train_T)
del_col_train_T.head()


# In[349]:


del_col_train_T.shape


# In[350]:


del_col_train_T.duplicated().sum()


# ー＞ 内容が同じ変数は無かった

# 'Id'列は、変数として利用しないので、ここで削除しておく。

# In[351]:


# 不要な列('ID')の削除
combined_df.drop('Id', axis=1, inplace=True)
train_df.drop('Id', axis=1, inplace=True)


# 欠損値が多い変数５つと、定数みたいな変数８つに、不要な'Id'変数で、計１４変数を削除した

# ## 3. 各変数の分析  
# 各変数の分布状況(平均、中央値、最頻度、分散、外れ値の状況など)を確認する  
#   
# -> 省略

# ・Categorical変数とNumerical変数に分ける。  Categorical変数はさらに、順序変数と名義変数に分ける

# In[352]:


# object型変数(Categorical)の抽出
object_list = combined_df.select_dtypes(include='object').columns.to_list()

# int型変数(Categorical変数とNumerical変数が混ざっている)の抽出
int_list = combined_df.select_dtypes(include='int').columns.to_list()

# float型変数(Numerical変数)の抽出
numeric_list = combined_df.select_dtypes(include='float').columns.to_list()


# object型変数をnominalとordinal feature に分ける

# In[353]:


# ojbect_listを名義変数リストと順序変数リストに分ける

# 名義変数
nominal_list = ['MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'Neighborhood',
                'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
                'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
                'GarageType', 'PavedDrive', 'SaleType', 'SaleCondition']
# 順序変数
ordinal_list = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'Electrical',
                'KitchenQual', 'Functional', 'GarageFinish','GarageQual', 'GarageCond']


# int型変数を、CategoricalとNumericに分ける

# In[354]:


# Categorical-name変数
to_nominal_list = ['MSSubClass']

# Categorical-ordinal変数
to_ordinal_list = ['OverallQual', 'OverallCond']

# numerical変数
to_num_list = ['LotArea', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF',
               'LowQualFinSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr',
               'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF',
               'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
               'MiscVal', 'MoSold', 'YrSold']


# int型変数をfloatとCategoricalにまとめる

# In[355]:


# 名義変数
all_nominal_list = nominal_list + to_nominal_list
# 順序変数
all_ordinal_list = ordinal_list + to_ordinal_list
# 量的変数
all_numeric_list = numeric_list + to_num_list

# 上記リストには、Id変数は除いている


# ## 4. 2変数間の分析  
# 1. SalePriceと相関ない変数の削除
# 1. 独立変数間で相関高い変数の削除

# ### 4-1 SalePriceと相関ない変数の削除  
# 

# + Numeric変数：SalePriceと相関が-0.2より大きく、0.2未満の変数を削除する

# In[356]:


# 相関係数計算し、相関係数が0.2未満の変数を削除する
no_corr_list = []
for col in all_numeric_list:
    corr = train_SalePrice.corr(train_df[col])
    if np.abs(corr) < 0.2:
        no_corr_list.append(col)

print(('相関係数が-0.2より大きく0.2未満のnumeric変数： ',len(no_corr_list)))
# print('削除される変数の数： ', len(all_numeric_list) - len(corr_list))
    


# In[357]:


# 相関係数が低い変数の削除
train_df.drop(no_corr_list, axis=1, inplace=True)
combined_df.drop(no_corr_list, axis=1, inplace=True)

# all_numeric_listの更新
all_numeric_list = set(all_numeric_list)
no_corr_list = set(no_corr_list)
all_numeric_list = all_numeric_list - no_corr_list
all_numeric_list = list(all_numeric_list)


# In[358]:


# 可視化して、相関が高い変数10個確認
train_df['SalePrice'] = train_SalePrice
corrmat = train_df.corr()
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# * Categorical変数：クラスカル=ウォリス検定(有意水準0.95)かけて、効果量が中以上の変数のみ抽出

# In[359]:


categorical_list = all_nominal_list + all_ordinal_list

# 効果量計算関数
def effect_size(H, k, n):
    '''
    H: H統計量
    k: カテゴリ数
    n: データ数
    戻り値: 効果量
    '''
    eta2 = (H - k + 1)/(n - k)
    
    return eta2

# クラスカル＝ウォリス検定かけて有意水準と効果量で振るい分け
signi_list = []
for i in range(len(categorical_list)):
    grouped = train_df.groupby(categorical_list[i])
    result = stats.kruskal(*(x[1]['SalePrice'] for x in grouped))
    
    if result.pvalue < 0.05:
        H = result.statistic # H統計量
        k = len(list(grouped.indices.keys())) # 変数内のカテゴリ数
        n = train_df[categorical_list[i]].notnull().value_counts()[True] # データ数
        e_size = effect_size(H, k, n) # 効果量計算
        
        if e_size >= 0.14:
            # 効果量が大以上の変数抽出
            print((categorical_list[i], '効果量大 = ', e_size))
            signi_list.append(categorical_list[i])
        elif 0.06 < e_size < 0.14:
            # 効果量が中以上の変数抽出
            print((categorical_list[i], '効果量中 = ', e_size))
            signi_list.append(categorical_list[i])

print(('\n削除対象変数の数： ', len(categorical_list) - len(signi_list)))


# In[360]:


# データから変数を削除
del_cat_list = set(categorical_list) - set(signi_list)
del_cat_list = set(del_cat_list)
categorical_list = signi_list

train_df.drop(del_cat_list, axis=1, inplace=True)
combined_df.drop(del_cat_list, axis=1, inplace=True)

print(('削除対象変数: ', del_cat_list))


# In[361]:


# all_nominal_list と all_ordinal_list の更新 (categorical_listを分ける)

# 名義変数
all_nominal_list = ['MSZoning', 'LotShape', 'Neighborhood', 'HouseStyle', 'Exterior1st', 'Exterior2nd',
                'MasVnrType', 'Foundation', 'GarageType', 'PavedDrive', 'SaleType', 'SaleCondition',
                'MSSubClass']

# 順序変数
all_ordinal_list = ['ExterQual', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'HeatingQC', 'Electrical',
                'KitchenQual', 'GarageFinish', 'OverallQual', 'OverallCond']


# ### 4-2 独立変数間で相関高い変数の削除

# In[362]:


# 可視化
corrmat = train_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corrmat)


# In[363]:


# 指定した相関係数以上の変数のインデックスを取り出す関数
def get_correlation(data, threshold):
    corr_col = set()
    corrmat = data.corr()
    for i in range(len(corrmat.columns)):
        for j in range(i):
            if abs(corrmat.iloc[i, j]) > threshold:
                colname = corrmat.columns[i]
                corr_col.add(colname)
    return corr_col


# In[364]:


# 他の変数と相関係数が0.85以上の変数取り出し
corr_features = get_correlation(train_df, 0.85)
print(('相関高くて削除する変数: ', corr_features))

# 変数の削除
train_df.drop(corr_features, axis=1, inplace=True)
combined_df.drop(corr_features, axis=1, inplace=True)


# In[365]:


# 変数リスト更新
all_numeric_list = set(all_numeric_list) - set(corr_features)
all_numeric_list = list(all_numeric_list)

all_nominal_list = set(all_nominal_list) - set(corr_features)
all_nominal_list = list(all_nominal_list)


# ここでは、SalePriceと関係ないと考えられる変数１２個、他変数と相関の強い変数２個の計１４個削除した。

# ## 5. 外れ値の処理  
# + all_numeric_listのみ対象  
# + 1.5IQRを超える変数は、5%tile値、95%tile値で置き換え

# In[366]:


# boxplotで分布確認
fig = plt.figure(figsize=(15,20))
plt.subplots_adjust(hspace=0.2, wspace=0.8)
for i in range(len(all_numeric_list)):
    ax = fig.add_subplot(4, 5, i+1)
    sns.boxplot(y=train_df[all_numeric_list[i]], data=train_df, ax=ax)
plt.show()


# In[367]:


out_train_df = train_df.copy()

for col in all_numeric_list:
    # 置き換え値
    upper_lim = out_train_df[col].quantile(.95)
    lower_lim = out_train_df[col].quantile(.05)
    
    # IQR
    Q1 = out_train_df[col].quantile(.25)
    Q3 = out_train_df[col].quantile(.75)
    IQR = Q3 - Q1
    outlier_step = 1.5 * IQR
    
    # 1.5IQRを超える数値は95%tile値で置き換え、下回る数値は5%tile値で置き換える
    out_train_df.loc[(out_train_df[col] > (Q3 + outlier_step)), col] = upper_lim
    out_train_df.loc[(out_train_df[col] < (Q1 - outlier_step)), col] = lower_lim


# In[368]:


# boxplotで分布確認
fig = plt.figure(figsize=(15,20))
plt.subplots_adjust(hspace=0.2, wspace=0.8)
for i in range(len(all_numeric_list)):
    ax = fig.add_subplot(4, 5, i+1)
    sns.boxplot(y=out_train_df[all_numeric_list[i]], data=out_train_df, ax=ax)
plt.show()


# In[369]:


# 合体データに外れ値処理データ反映
out_combined_df = combined_df.copy()
out_combined_df.loc[:1459, out_train_df.drop('SalePrice', axis=1).columns] = out_train_df.drop('SalePrice', axis=1)


# ## 6. 欠損値の処理  
# * Categorical変数は最頻値、Continuous変数は中央値で埋める

# In[370]:


# 欠損値の確認
combined_df.isnull().sum().sort_values(ascending=False)[:30]


# In[371]:


# Categorical変数 - 最頻値
Cat_out_df = out_combined_df[all_nominal_list + all_ordinal_list].copy()
Cat_df = combined_df[all_nominal_list + all_ordinal_list].copy()
Cat_out_df.fillna(Cat_out_df.mode().iloc[0], inplace=True)
Cat_df.fillna(Cat_df.mode().iloc[0], inplace=True)

# Continuous変数 - 中央値
Num_out_df = out_combined_df[all_numeric_list].copy()
Num_df = combined_df[all_numeric_list].copy()
Num_out_df.fillna(Num_out_df.median(), inplace=True)
Num_df.fillna(Num_df.median(), inplace=True)

# dfに反映
out_combined_df[all_nominal_list + all_ordinal_list] = Cat_out_df
combined_df[all_nominal_list + all_ordinal_list] = Cat_df

out_combined_df[all_numeric_list] = Num_out_df 
combined_df[all_numeric_list] = Num_df


# ## 7. 変数の創出

# ## 8. 変数変換  
# + 名義変数は、Binary Encoding
# + 順序変数は、Ordinal Encoding

# ### ・名義変数 -> Binary Encoding

# In[372]:


# 処理用データ
BE_out_df = out_combined_df[all_nominal_list].copy()
BE_df = combined_df[all_nominal_list].copy()

# Binary Encoding
BE_out_df = BinaryEncoder(cols=all_nominal_list).fit_transform(BE_out_df)
BE_df = BinaryEncoder(cols=all_nominal_list).fit_transform(BE_df)

# 変換前の変数削除
out_combined_df.drop(all_nominal_list, axis=1, inplace=True)
combined_df.drop(all_nominal_list, axis=1, inplace=True)

# BE後のデータ反映
out_combined_df[BE_out_df.columns] = BE_out_df
combined_df[BE_df.columns] = BE_df


# ### ・順序変数 -> Ordinal Encoding

# In[373]:


# ExterQual
ExterQual = {'TA': 3, 'Gd': 4, 'Ex': 5, 'Fa': 2}
out_combined_df['ExterQual'] = out_combined_df['ExterQual'].map(ExterQual)
combined_df['ExterQual'] = combined_df['ExterQual'].map(ExterQual)

# BsmtQual
BsmtQual = {'TA': 3, 'Gd': 4, 'Ex': 5, 'Fa': 2}
out_combined_df['BsmtQual'] = out_combined_df['BsmtQual'].map(BsmtQual)
combined_df['BsmtQual'] = combined_df['BsmtQual'].map(BsmtQual)

# BsmtExposure
BsmtExposure = {'No': 2, 'Av': 4, 'Gd': 5, 'Mn': 3}
out_combined_df['BsmtExposure'] = out_combined_df['BsmtExposure'].map(BsmtExposure)
combined_df['BsmtExposure'] = combined_df['BsmtExposure'].map(BsmtExposure)

# BsmtFinType1
BsmtFinType1 = {'Unf': 2, 'GLQ': 7, 'ALQ': 6, 'Rec': 4, 'BLQ': 5, 'LwQ': 3}
out_combined_df['BsmtFinType1'] = out_combined_df['BsmtFinType1'].map(BsmtFinType1)
combined_df['BsmtFinType1'] = combined_df['BsmtFinType1'].map(BsmtFinType1)

# HeatingQC
HeatingQC = {'Ex': 5, 'TA': 3, 'Gd': 4, 'Fa': 2, 'Po': 1}
out_combined_df['HeatingQC'] = out_combined_df['HeatingQC'].map(HeatingQC)
combined_df['HeatingQC'] = combined_df['HeatingQC'].map(HeatingQC)

# Electrical
Electrical = {'SBrkr': 5, 'FuseA': 4, 'FuseF': 3, 'FuseP': 2, 'Mix': 1}
out_combined_df['Electrical'] = out_combined_df['Electrical'].map(Electrical)
combined_df['Electrical'] = combined_df['Electrical'].map(Electrical)

# KitchenQual
KitchenQual = {'TA': 3, 'Gd': 4, 'Ex': 5, 'Fa': 2}
out_combined_df['KitchenQual'] = out_combined_df['KitchenQual'].map(KitchenQual)
combined_df['KitchenQual'] = combined_df['KitchenQual'].map(KitchenQual)

# GarageFinish
GarageFinish = {'Unf': 2, 'RFn': 3, 'Fin': 4}
out_combined_df['GarageFinish'] = out_combined_df['GarageFinish'].map(GarageFinish)
combined_df['GarageFinish'] = combined_df['GarageFinish'].map(GarageFinish)

# OverallQual - 不要
# OverallCond - 不要


# # モデリング
# * Random Forest
# * lightgbm

# In[374]:


# モデリング用、提出用にデータ分割
out_train = out_combined_df.loc[:1459, :]
out_test = out_combined_df.loc[1460:, :] # 提出用

train = combined_df.loc[:1459, :]
test = combined_df.loc[1460:, :] #提出用


# ### ・モデリング用データを提出前のファイナルテスト用と、訓練・検証用に分ける

# In[384]:


# モデリング用データを提出前のファイナルテスト用と、訓練・検証用に分ける
out_train_x, out_test_x, out_train_y, out_test_y = train_test_split(out_train, train_SalePrice, test_size=0.20, random_state=1)
train_x, test_x, train_y, test_y =  train_test_split(train, train_SalePrice, test_size=0.20, random_state=1)


# ファイナルテスト用データ  
# + out_test_x, out_test_y  
# + test_x, test_y

# ・Random Forest

# 外れ値処理不要なので、外れ値未処理データで行う

# In[378]:


# クロスバリデーション用に分割
train_xxx, test_xxx, train_yyy, test_yyy = train_test_split(train_x, train_y, test_size=0.25, random_state=1)


# In[379]:


n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)] # 10個
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(start=10, stop=200, num = 10)]
min_samples_split = [2,5,10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[380]:


get_ipython().run_cell_magic('time', '', "rf = RandomForestRegressor()\nrf_random = RandomizedSearchCV(rf, random_grid, n_iter=100, cv=5, n_jobs=-1, scoring='r2')\nrf_random.fit(train_xxx, train_yyy)\nprint('score: ', rf_random.score(test_xxx, test_yyy))\nprint('best parameter: ', rf_random.best_params_)\nprint('score for train_data: ', rf_random.best_score_, '\\n')\n")


# In[381]:


params = rf_random.best_params_
params


# In[ ]:


{'n_estimators': 400,
 'min_samples_split': 2,
 'min_samples_leaf': 1,
 'max_features': 'sqrt',
 'max_depth': 52,
 'bootstrap': False}


# In[382]:


rf_model = RandomForestRegressor(
                n_estimators = params['n_estimators'],
                min_samples_split = params['min_samples_split'],
                min_samples_leaf = params['min_samples_leaf'],
                max_features = params['max_features'],
                max_depth = params['max_depth'],
                bootstrap = params['bootstrap'])

rf_model.fit(train_x, train_y)
predicted = rf_model.predict(test_x)
r2 = r2_score(test_y, predicted)

print(('r2_score: ', r2, '\n'))


# In[383]:


rf_model = RandomForestRegressor(
                n_estimators = params['n_estimators'],
                min_samples_split = params['min_samples_split'],
                min_samples_leaf = params['min_samples_leaf'],
                max_features = params['max_features'],
                max_depth = params['max_depth'],
                bootstrap = params['bootstrap'])

rf_model.fit(train, train_SalePrice)
predicted = rf_model.predict(test)

predicted_rf = pd.DataFrame({'Id':test_df['Id'], 'SalePrice': predicted})
predicted_rf.to_csv('/kaggle/working/ver_rf_NO_minmax_0603.csv', index=False)


# スコア：0.14455

# lightgbm

# ・外れ値処理データのモデリング

# Cross validationのために、訓練・検証用データを分割

# In[ ]:


out_train_xxx, out_test_xxx, out_train_yyy, out_test_yyy = train_test_split(out_train_x, out_train_y, test_size=0.25, random_state=1)


# In[ ]:


# 正規化
ob = MinMaxScaler()
ob = ob.fit(out_train_xxx.values)
out_train_xxx.iloc[:, :] = ob.transform(out_train_xxx.values)
out_test_xxx.iloc[:, :] = ob.transform(out_test_xxx.values)

# 従属変数の正規化
ob_y = MinMaxScaler()
ob_y = ob_y.fit(out_train_yyy.values.reshape(-1, 1))
out_train_yyy = ob_y.transform(out_train_yyy.values.reshape(-1, 1)).reshape(876,)
out_test_yyy = ob_y.transform(out_test_yyy.values.reshape(-1, 1)).reshape(292,)


# In[ ]:


# lgb用データ準備
lgb_train = lgb.Dataset(out_train_xxx, out_train_yyy)
lgb_test = lgb.Dataset(out_test_xxx, out_test_yyy)


# In[ ]:


def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    }
    gbm = lgb_optuna.train(params,
                           lgb_train,
                           valid_sets=(lgb_train, lgb_test),
                           num_boost_round=10000,
                           early_stopping_rounds=100,
                           verbose_eval=50)
    predicted = gbm.predict(out_test_xxx)
    r2 = r2_score(out_test_yyy, predicted)
    
    return r2


# In[ ]:


study_gbm = optuna.create_study()
study_gbm.optimize(objective, timeout=600)


# In[ ]:


print(('score: ', study_gbm.best_value))
print(('parms: ', study_gbm.best_params))


# ファイナルテスト用データでスコア確認

# In[ ]:


out_train_x, out_test_x, out_train_y, out_test_y


# In[ ]:


out_test_y.shape


# In[ ]:


# 正規化
ob = MinMaxScaler()
ob = ob.fit(out_train_x.values)
out_train_x.iloc[:, :] = ob.transform(out_train_x.values)

ob_f = MinMaxScaler()
ob_f = ob_f.fit(out_test_x.values)
out_test_x.iloc[:, :] = ob_f.transform(out_test_x.values)

# 従属変数の正規化
ob_y = MinMaxScaler()
ob_y = ob_y.fit(out_train_y.values.reshape(-1, 1))
out_train_y =  ob_y.transform(out_train_y.values.reshape(-1, 1)).reshape(1168,)
# 提出用の予測値は、train_yで作ったMinMaxScalerオブジェクトでもとに戻すので、従属変数側は、trainのオブジェクトでtransformする。
out_test_y = ob_y.transform(out_test_y.values.reshape(-1, 1)).reshape(292,)


# In[ ]:


# lgb用データ作成
lgb_train = lgb.Dataset(out_train_x, out_train_y)
lgb_test = lgb.Dataset(out_test_x, out_test_y)


# In[ ]:


# parameter
params = study_gbm.best_params


# In[ ]:


params['objective'] = 'regression'
params['metric'] = 'rmse'

lgb_regressor = lgb.train(params,
                          lgb_train,
                          valid_sets=(lgb_train, lgb_test),
                          num_boost_round=10000,
                          early_stopping_rounds=100,
                          verbose_eval=50)
predicted = lgb_regressor.predict(out_test_x)
r2 = r2_score(out_test_y, predicted)

print(('score: ', r2))


# In[ ]:


out_train
train_SalePrice
out_test  # 提出用


# 提出用

# In[ ]:


# 正規化
ob = MinMaxScaler()
ob = ob.fit(out_train.values)
out_train.iloc[:, :] = ob.transform(out_train.values)

ob_f = MinMaxScaler()
ob_f = ob_f.fit(out_test.values)
out_test.iloc[:, :] = ob_f.transform(out_test.values)

# 従属変数の正規化
ob_y = MinMaxScaler()
ob_y = ob_y.fit(train_SalePrice.values.reshape(-1, 1))
out_train_y =  ob_y.transform(train_SalePrice.values.reshape(-1, 1)).reshape(1460,)


# In[ ]:


# lgbデータ準備
lgb_train = lgb.Dataset(out_train, out_train_y)


# In[ ]:


lgb_regressor = lgb.train(params,
                          lgb_train,
                          num_boost_round=10000,
                          verbose_eval=50)

predicted = lgb_regressor.predict(out_test)
predicted_price = ob_y.inverse_transform(predicted.reshape(-1, 1)).reshape(1459,) # 提出データは1459レコード
# 提出データ吐き出し
predicted_df = pd.DataFrame({'Id':test_df['Id'], 'SalePrice': predicted_price})
file_name = '/kaggle/working/ver3_lgb_outlier_minmax_0603.csv'
predicted_df.to_csv(file_name, index=False)


# スコア：0.29634 Ver2より大分悪くなった

# 外れ値未処理データのモデリング

# In[ ]:


# クロスバリデーション用に分割
train_xxx, test_xxx, train_yyy, test_yyy = train_test_split(train_x, train_y, test_size=0.25, random_state=1)


# In[ ]:


# 正規化
ob = MinMaxScaler()
ob = ob.fit(train_xxx.values)
train_xxx.iloc[:, :] = ob.transform(train_xxx.values)
test_xxx.iloc[:, :] = ob.transform(test_xxx.values)

# 従属変数の正規化
ob_y = MinMaxScaler()
ob_y = ob_y.fit(train_yyy.values.reshape(-1, 1))
train_yyy = ob_y.transform(train_yyy.values.reshape(-1, 1)).reshape(876,)
test_yyy = ob_y.transform(test_yyy.values.reshape(-1, 1)).reshape(292,)


# In[ ]:


# lgb用データ準備
lgb_train = lgb.Dataset(train_xxx, train_yyy)
lgb_test = lgb.Dataset(test_xxx, test_yyy)


def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    }
    gbm = lgb_optuna.train(params,
                           lgb_train,
                           valid_sets=(lgb_train, lgb_test),
                           num_boost_round=10000,
                           early_stopping_rounds=100,
                           verbose_eval=50)
    predicted = gbm.predict(test_xxx)
    r2 = r2_score(test_yyy, predicted)
    
    return r2


# In[ ]:


study_gbm = optuna.create_study()
study_gbm.optimize(objective, timeout=600)


# In[ ]:


print(('score: ', study_gbm.best_value))
print(('parms: ', study_gbm.best_params))


# ファイナルテスト用データでスコア確認

# In[ ]:


# 正規化
ob = MinMaxScaler()
ob = ob.fit(train_x.values)
train_x.iloc[:, :] = ob.transform(train_x.values)

ob_f = MinMaxScaler()
ob_f = ob_f.fit(test_x.values)
test_x.iloc[:, :] = ob_f.transform(test_x.values)

# 従属変数の正規化
ob_y = MinMaxScaler()
ob_y = ob_y.fit(train_y.values.reshape(-1, 1))
train_y =  ob_y.transform(train_y.values.reshape(-1, 1)).reshape(1168,)
# 提出用の予測値は、train_yで作ったMinMaxScalerオブジェクトでもとに戻すので、従属変数側は、trainのオブジェクトでtransformする。
test_y = ob_y.transform(test_y.values.reshape(-1, 1)).reshape(292,)


# In[ ]:


# lgb用データ作成
lgb_train = lgb.Dataset(train_x, train_y)
lgb_test = lgb.Dataset(test_x, test_y)


# In[ ]:


# parameter
params = study_gbm.best_params


# In[ ]:


params['objective'] = 'regression'
params['metric'] = 'rmse'

lgb_regressor = lgb.train(params,
                          lgb_train,
                          valid_sets=(lgb_train, lgb_test),
                          num_boost_round=10000,
                          early_stopping_rounds=100,
                          verbose_eval=50)
predicted = lgb_regressor.predict(test_x)
r2 = r2_score(test_y, predicted)

print(('score: ', r2))


# 提出用

# In[ ]:


# 正規化
ob = MinMaxScaler()
ob = ob.fit(train.values)
train.iloc[:, :] = ob.transform(train.values)

ob_f = MinMaxScaler()
ob_f = ob_f.fit(test.values)
test.iloc[:, :] = ob_f.transform(test.values)

# 従属変数の正規化
ob_y = MinMaxScaler()
ob_y = ob_y.fit(train_SalePrice.values.reshape(-1, 1))
train_y =  ob_y.transform(train_SalePrice.values.reshape(-1, 1)).reshape(1460,)


# In[ ]:


# lgbデータ準備
lgb_train = lgb.Dataset(train, train_y)


# In[ ]:


lgb_regressor = lgb.train(params,
                          lgb_train,
                          num_boost_round=10000,
                          verbose_eval=50)

predicted = lgb_regressor.predict(test)
predicted_price = ob_y.inverse_transform(predicted.reshape(-1, 1)).reshape(1459,) # 提出データは1459レコード
# 提出データ吐き出し
predicted_df = pd.DataFrame({'Id':test_df['Id'], 'SalePrice': predicted_price})
file_name = '/kaggle/working/ver3_lgb_minmax_0603.csv'
predicted_df.to_csv(file_name, index=False)


# スコア：0.19766

# 正規化せずにモデリング

# In[385]:


# クロスバリデーション用に分割
train_xxx, test_xxx, train_yyy, test_yyy = train_test_split(train_x, train_y, test_size=0.25, random_state=1)


# In[ ]:


# # 正規化
# ob = MinMaxScaler()
# ob = ob.fit(train_xxx.values)
# train_xxx.iloc[:, :] = ob.transform(train_xxx.values)
# test_xxx.iloc[:, :] = ob.transform(test_xxx.values)

# # 従属変数の正規化
# ob_y = MinMaxScaler()
# ob_y = ob_y.fit(train_yyy.values.reshape(-1, 1))
# train_yyy = ob_y.transform(train_yyy.values.reshape(-1, 1)).reshape(876,)
# test_yyy = ob_y.transform(test_yyy.values.reshape(-1, 1)).reshape(292,)


# In[386]:


# lgb用データ準備
lgb_train = lgb.Dataset(train_xxx, train_yyy)
lgb_test = lgb.Dataset(test_xxx, test_yyy)


def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    }
    gbm = lgb_optuna.train(params,
                           lgb_train,
                           valid_sets=(lgb_train, lgb_test),
                           num_boost_round=10000,
                           early_stopping_rounds=100,
                           verbose_eval=50)
    predicted = gbm.predict(test_xxx)
    r2 = r2_score(test_yyy, predicted)
    
    return r2


# In[387]:


study_gbm = optuna.create_study()
study_gbm.optimize(objective, timeout=600)


# In[388]:


print(('score: ', study_gbm.best_value))
print(('parms: ', study_gbm.best_params))


# 
# ファイナルテスト用データでスコア確認

# In[ ]:


# # 正規化
# ob = MinMaxScaler()
# ob = ob.fit(train_x.values)
# train_x.iloc[:, :] = ob.transform(train_x.values)

# ob_f = MinMaxScaler()
# ob_f = ob_f.fit(test_x.values)
# test_x.iloc[:, :] = ob_f.transform(test_x.values)

# # 従属変数の正規化
# ob_y = MinMaxScaler()
# ob_y = ob_y.fit(train_y.values.reshape(-1, 1))
# train_y =  ob_y.transform(train_y.values.reshape(-1, 1)).reshape(1168,)
# # 提出用の予測値は、train_yで作ったMinMaxScalerオブジェクトでもとに戻すので、従属変数側は、trainのオブジェクトでtransformする。
# test_y = ob_y.transform(test_y.values.reshape(-1, 1)).reshape(292,)


# In[389]:


# lgb用データ作成
lgb_train = lgb.Dataset(train_x, train_y)
lgb_test = lgb.Dataset(test_x, test_y)


# In[390]:


# parameter
params = study_gbm.best_params
params


# In[391]:


params['objective'] = 'regression'
params['metric'] = 'rmse'

lgb_regressor = lgb.train(params,
                          lgb_train,
                          valid_sets=(lgb_train, lgb_test),
                          num_boost_round=10000,
                          early_stopping_rounds=100,
                          verbose_eval=50)
predicted = lgb_regressor.predict(test_x)
r2 = r2_score(test_y, predicted)

print(('score: ', r2))


# 提出用

# In[ ]:


# # 正規化
# ob = MinMaxScaler()
# ob = ob.fit(train.values)
# train.iloc[:, :] = ob.transform(train.values)

# ob_f = MinMaxScaler()
# ob_f = ob_f.fit(test.values)
# test.iloc[:, :] = ob_f.transform(test.values)

# # 従属変数の正規化
# ob_y = MinMaxScaler()
# ob_y = ob_y.fit(train_SalePrice.values.reshape(-1, 1))
# train_y =  ob_y.transform(train_SalePrice.values.reshape(-1, 1)).reshape(1460,)


# In[392]:


# lgbデータ準備
lgb_train = lgb.Dataset(train, train_SalePrice)


# In[393]:


lgb_regressor = lgb.train(params,
                          lgb_train,
                          num_boost_round=10000,
                          verbose_eval=50)

predicted = lgb_regressor.predict(test)

# 提出データ吐き出し
predicted_df = pd.DataFrame({'Id':test_df['Id'], 'SalePrice': predicted})
file_name = '/kaggle/working/ver3_lgb_No_minmax_0603_2.csv'
predicted_df.to_csv(file_name, index=False)


# スコア：0.18110

# In[ ]:




