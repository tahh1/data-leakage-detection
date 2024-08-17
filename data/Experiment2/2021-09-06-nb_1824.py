#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/minassy/ISID_2021/blob/main/210918_%E3%83%A2%E3%83%87%E3%83%AA%E3%83%B3%E3%82%B0_LSTM.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


import os
import zipfile
import glob
import pandas as pd
import numpy as np
import pathlib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#作業基本ディレクトリ
DIR = 'drive/MyDrive/00_Competition/ISID/2021'
os.listdir(DIR)
#os.mkdir(DIR+'/data')


# In[ ]:


DIR_DATA_TRAIN = os.path.join(DIR, 'data', 'Train_Unit_20210903')
DIR_DATA_TEST = os.path.join(DIR, 'data', 'Test_Unit_20210903')


# In[ ]:


#一連のデータ処理
def get_some_values(file_path):
  #ファイルパスから、ユニット名を取得。
  dirname = os.path.dirname(file_path)
  dirname_split = dirname.split('/')#区切り文字'/'で分割
  unit = dirname_split[-2]
  charge_mode = dirname_split[-1]
  #ファイルパスから、サイクル数を取得。
  basename = os.path.basename(file_path)
  basename_split = basename.split('_')[4]#区切り文字'_'で分割
  cycle_num = basename_split.split('.')[0]
  return unit, charge_mode, cycle_num


# In[ ]:


#%time
#DATA_DIR = DIR_DATA_TRAIN
#DATA_DIR = DIR_DATA_TEST

#C = 'Charge'
#C = 'Discharge'

def data_reading(DATA_DIR, C):

  #カラム名変更前後の辞書
  col_list = {'Time (s)' : 'Time',
              'Voltage Measured (V)' : 'VM',
              'Voltage Charge (V)' : 'VC',
              'Current Measured (Amps)' : 'CM', 
              'Current Charge (Amps)' : 'CC',
              'Temperature (degree C)': 'Temp', 
              'unit' : 'unit_name', 
              'charge_mode' : 'mode', 
              'Cycle_num' : 'Cycle',
              'Voltage Load (V)' : 'VL',
              'Current Load (Amps)' : 'CL'
  }

  df_list = []
  for folder in tqdm(os.listdir(DATA_DIR)[:3]):
    files = glob.glob(os.path.join(DATA_DIR, folder, C, '*.csv'))
    df_list_0 = []
    for file in files:
      tmp_df = pd.read_csv(file, encoding='utf-8')
      tmp_df = tmp_df.rename(columns=col_list)
      #ユニット名、充・放電モード、サイクル数の取得
      unit, charge_mode, cycle_num = get_some_values(file)
      #特徴量のデータフレームの作成
      if C == 'Charge':
        df_exp = pd.DataFrame([{'unit':unit,
                          'charge_mode' : charge_mode,
                          'Cycle_num' : int(cycle_num),
                          'feature_c_vm' : featured_c_vm(tmp_df),
                          'feature_c_cm' : featured_c_cm(tmp_df),
                          'feature_c_vc' : featured_c_vc(tmp_df)}])
      else:
        df_exp = pd.DataFrame([{'unit':unit,
                          'charge_mode' : charge_mode,
                          'Cycle_num' : int(cycle_num),
                          'feature_d_vm' : featured_d_vm(tmp_df),
                          'feature_d_vl' : featured_d_vl(tmp_df)}])
      df_list_0.append(df_exp)
    df_0 = pd.concat(df_list_0)
    df_list.append(df_0)
  df = pd.concat(df_list)

  #ユニットとサイクルでソート。
  df = df.sort_values(['unit', 'Cycle_num'])
  #インデックスの振り直し
  df = df.reset_index(drop=True)

  return df


# In[ ]:


def featured_c_vm(df):
  #LAG=5が0.001未満が続くところを、飽和領域とみなす。
  p_c_vm = df.loc[:, 'VM'][df.loc[:, 'VM'].diff(5) < 0.001]
  #飽和領域のなかで、最初の点を点P_C_VMとする。
  p_c_vm = p_c_vm.reset_index(inplace=False)
  p_c_vm = p_c_vm['index'].iloc[0]
  #充電VMの特徴量は、点P_C_VMにおける時間
  feature_c_vm = df['Time'].loc[p_c_vm]
  return feature_c_vm

def featured_c_cm(df):
  #飽和領域からの立ち下がり点(LAG=5の勾配が最小となる点)を点P_C_CMとする。
  p_c_cm = np.gradient(df.loc[:, 'CM'].diff(5)[10:]).argmin()
  #充電CMの特徴量は、点P_C_CMにおける時間
  feature_c_cm = df['Time'].loc[p_c_cm]
  return feature_c_cm

def featured_c_vc(df):
  #VCが最大値を示す点を、点P_C_VCとする。
  p_c_vc = df.VC.argmax()
  #充電VCの特徴量は、点P_C_VCにおける時間
  feature_c_vc = df['Time'].loc[p_c_vc]
  return feature_c_vc

def featured_d_vm(df):
  #VMが最小値を示す点を、点P_D_VMとする。
  p_d_vm = df.VM.argmin()
  #放電VMの特徴量は、点P_D_VMにおける時間
  feature_d_vm = df['Time'].loc[p_d_vm]
  return feature_d_vm

def featured_d_vl(df):
  #VLの最大値点以降の領域において、最小値を示す点を、点P_D_VLとする。
  VL_max = df.VL.argmax()
  p_d_vl = df.VL[VL_max: ].argmin()
  #放電VLの特徴量は、点P_D_VLにおける時間
  feature_d_vl = df['Time'].loc[p_d_vl]
  return feature_d_vl


# # 学習データ、テストデータの読み込み

# In[ ]:


get_ipython().run_line_magic('time', '')
#データの読み込み(学習)
#充電

DATA_DIR = DIR_DATA_TRAIN
#DATA_DIR = DIR_DATA_TEST

C = 'Charge'
#C = 'Discharge'

df_train_charge = data_reading(DATA_DIR, C)


# In[ ]:


get_ipython().run_line_magic('time', '')
#データの読み込み(学習)
#放電

DATA_DIR = DIR_DATA_TRAIN
#DATA_DIR = DIR_DATA_TEST

#C = 'Charge'
C = 'Discharge'

df_train_discharge = data_reading(DATA_DIR, C)


# In[ ]:


#充電と放電の特徴量のデータセット作成
df_train_feature = pd.merge(df_train_charge, df_train_discharge,
                            how = 'inner',
                            on = ['unit', 'Cycle_num'])
df_train_feature.head()


# In[ ]:


get_ipython().run_line_magic('time', '')
#データの読み込み(学習)
#充電

#DATA_DIR = DIR_DATA_TRAIN
DATA_DIR = DIR_DATA_TEST

C = 'Charge'
#C = 'Discharge'

df_test_charge = data_reading(DATA_DIR, C)


# In[ ]:


get_ipython().run_line_magic('time', '')
#データの読み込み(学習)
#放電

#DATA_DIR = DIR_DATA_TRAIN
DATA_DIR = DIR_DATA_TEST

#C = 'Charge'
C = 'Discharge'

df_test_discharge = data_reading(DATA_DIR, C)


# In[ ]:


#充電と放電の特徴量のデータセット作成
df_test_feature = pd.merge(df_test_charge, df_test_discharge,
                            how = 'inner',
                            on = ['unit', 'Cycle_num'])
df_test_feature.head()


# # 学習データの作成

# In[ ]:


import math
from keras.models import Sequential
from keras.layers import Dense


# まとめ  
# テストユニット1：訓練ユニット1と似ている。 feature_d_vm と feature_d_vm or feature_d_vlで、ほぼ説明できそう。
# 
# テストユニット2：訓練ユニット2と似ている。 feature_c_vm と feature_d_vm or feature_d_vlで、ほぼ説明できそう。　　
# 
# テストユニット3：訓練ユニット1と似ている。 feature_d_vm feature_d_vl と feature_d_vm feature_d_vlで、ほぼ説明できそう。　

# In[ ]:


num_train_unit_1 = 124
num_train_unit_2 = 40
num_train_unit_3 = 97

num_test_unit_1 = 70
num_test_unit_2 = 12
num_test_unit_3 = 55


# テストユニット1

# In[ ]:


#ラグを定義し、LSTM用のデータセットを作る
def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


# In[ ]:


#学習データと検証データ
train_unit_1 = df_train_feature[df_train_feature['unit'] == 'Train_Unit_1' ].reset_index()
dataset = train_unit_1['feature_d_vm'].values.astype('float32')


# In[ ]:


print((dataset.ndim))
print((dataset.shape))
print((dataset.size))


# In[ ]:


#次元数の変更
dataset = dataset.reshape(124, 1)
print((dataset.ndim))
print((dataset.shape))
print((dataset.size))


# In[ ]:


# split into train and test sets
train_size = num_test_unit_1
val_size = len(dataset) - train_size
train, val = dataset[0:train_size], dataset[train_size:len(dataset)]


# In[ ]:


# reshape dataset
look_back = 2
trainX, trainY = create_dataset(train, look_back)
valX, valY = create_dataset(val, look_back)


# In[ ]:


trainX


# In[ ]:


# create and fit Multilayer Perceptron model
model = Sequential()
model.add(Dense(12, input_dim=look_back, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=2, verbose=2)


# In[ ]:


# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print(('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore))))
valScore = model.evaluate(valX, valY, verbose=0)
print(('Validation Score: %.2f MSE (%.2f RMSE)' % (valScore, math.sqrt(valScore))))
# generate predictions for training
trainPredict = model.predict(trainX)
valPredict = model.predict(valX)


# In[ ]:


# shift train predictions for plotting
plt.rcParams['figure.figsize'] = [6, 5]
#datasetと同じ空配列リストを作成
trainPredictPlot = np.empty_like(dataset)
#リストをnanに置き換え
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
valPredictPlot = np.empty_like(dataset)
valPredictPlot[:, :] = np.nan
valPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = valPredict
# plot baseline and predictions
plt.plot(dataset)
plt.plot(trainPredictPlot,linewidth =1, linestyle = '--',color='k')
plt.plot(valPredictPlot,linestyle= ':',color='r')

plt.show()


# # テストデータによる推論

# 学習データ(1～70)で、テストデータ(1～70)を予測するモデルを作成する。  
# 推論も同様に?  
# テストデータでモデル作って、未知部分を予想するタスク？

# In[ ]:





# In[ ]:


#テストデータ
test_unit_1 = df_test_feature[df_test_feature['unit'] == 'Test_Unit_1' ].reset_index()
dataset_test = test_unit_1['feature_d_vm'].values.astype('float32')


# In[ ]:


print((dataset_test.ndim))
print((dataset_test.shape))
print((dataset_test.size))


# In[ ]:


#次元数の変更
dataset_test = dataset_test.reshape(70, 1)
print((dataset_test.ndim))
print((dataset_test.shape))
print((dataset_test.size))


# In[ ]:


# reshape dataset
look_back = 2
test = dataset_test
testX, testY = create_dataset(test, look_back)


# In[ ]:


testPredict = model.predict(testX)


# In[ ]:


testPredict.shape


# In[ ]:


plt.plot(testPredict)


# In[ ]:


testX.ndim


# In[ ]:


hoge = trainX.shape[0] - 1
future_test = trainX[hoge]
n = testX.shape[0]
test_data = future_test.reshape(2,1)


# In[ ]:


#テストデータの最終行を取得
hoge = trainX.shape[0] - 1
future_test = trainX[hoge]

#予測結果をy_pred_summary に入れる
y_pred_summary = np.empty((1))
 
n = testX.shape[0]
 
for step in range(n):
    
    #最新のテストデータで予測値を算出
    test_data = future_test.reshape(2,1)
    batch_predict = model.predict(test_data)
    
    #予測値を最新のテストデータにしてテストデータを更新
    future_test = np.delete(future_test, 0)
    future_test = np.append(future_test, batch_predict)
    
    #予測結果を順番に保存
    y_pred_summary = np.append(y_pred_summary, batch_predict)

predicted = model.predict(testX) 
dataf =  pd.DataFrame(predicted)
dataf.columns = ["predict(with test data)"]
dataf["Stock price"] = testY

#教師データなしの予測値を追加
dataf["predict(No test data)"] = y_pred_summary
 
dataf.plot(figsize=(15, 5)).legend(loc='upper left')


# In[ ]:




