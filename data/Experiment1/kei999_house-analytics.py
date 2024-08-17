#!/usr/bin/env python
# coding: utf-8

# # House Sales in King County, USA
# Author：Kei Osanai

# #### 1. データ読み込み

# In[ ]:


import pandas as pd
df = pd.read_csv("../input/kc_house_data.csv")
df.head()


# | #  | 項目         | 説明                                                                                                |
# |----|--------------|:-----------------------------------------------------------------------------------------------------|
# | 1  | id           |家の表記                                                                                             |
# | 2  | date         |デートハウスは売却された                                                                             |
# | 3  | price        |価格は予測対象です                                                                                   |
# | 4  | bedrooms     |寝室数/戸数                                                                                          |
# | 5  | bathrooms    |バスルーム/ベッドルームの数                                                                          |
# | 6  | sqft_living  |家の平方フィート                                                                                     |
# | 7  | sqft_lot     |ロットの平方フィート                                                                                 |
# | 8  | floors       |家の中の全フロア（レベル）                                                                           |
# | 9  | waterfront   |ウォーターフロントを望む家                                                                           |
# | 10 | view         |見たことがある                                                                                       |
# | 11 | condition    |どのくらいの状態が良いか（全体的）                                                                   |
# | 12 | grade        |キング郡格付け制度に基づいて、住宅部門に与えられた全体的な等級                                       |
# | 13 | sqft_above   |地下から離れた家の広場                                                                               |
# | 14 | sqft_basement|地下の広場                                                                                           |
# | 15 | yr_built     |築年                                                                                                 |
# | 16 | yr_renovated |家が改築された年                                                                                     |
# | 17 | zipcode      |ジップ                                                                                               |
# | 18 | lat          |緯度座標                                                                                             |
# | 19 | long         |経度座標                                                                                             |
# | 20 | sqft_living15|2015年のリビングルームエリア（いくつかの改装を含む）これはロットサイズエリアに影響を及ぼしたかどうか |
# | 21 | sqft_lot15   |2015年のlotSize領域（一部の改装を含む）                                                              |
# 

# #### 2 散布行列を表示してみる

# In[ ]:


import matplotlib.pyplot as plt

y_var = "price"
X_var = ["bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront"]
df_tmp = df[[y_var]+X_var]
pd.plotting.scatter_matrix(df_tmp,alpha=0.3,s=10, figsize=(10,10))#散布図の作成
plt.show()#グラフをここで描画させるための行


# In[ ]:


y_var = "price"
X_var = ["view","condition","grade","sqft_above","sqft_basement","yr_built"]
df_tmp = df[[y_var]+X_var]
pd.plotting.scatter_matrix(df_tmp,alpha=0.3,s=10, figsize=(10,10))
plt.show()


# In[ ]:


y_var = "price"
X_var = ["yr_renovated","zipcode","lat","long","sqft_living15","sqft_lot15"]
df_tmp = df[[y_var]+X_var]
pd.plotting.scatter_matrix(df_tmp,alpha=0.3,s=10, figsize=(10,10))
plt.show()


# #### 3 要約統計量を出力する。
# 

# In[ ]:


df.describe ()


# #### 4 線形回帰分析を行ってみる

# #### 5 テストデータとトレーニングデータを分割する。

# In[ ]:


from sklearn.model_selection import train_test_split

# 利用できるデータのうち、学習用を8割、テスト用を2割にする
X_train, X_test, y_train, y_test = train_test_split(df[["sqft_living"]], df["price"], test_size=0.2, random_state=1234)


# #### 6 学習する。

# In[ ]:


from sklearn import linear_model 
# 学習
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(X_train, y_train)


# #### 7 精度を求める

# In[ ]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

y_pred = regr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(("MSE=%s"%round(mse,3) ))
print(("RMSE=%s"%round(np.sqrt(mse), 3) ))
print(("MAE=%s"%round(mae,3) ))


# In[ ]:





# In[ ]:




