#!/usr/bin/env python
# coding: utf-8

# # Working with MNIST data and Visualize using t-SNE
# 
# ## Background
# MNIST data is most famous dataset available for us which has 60,000 samples of hand-written digits (0 to 9). Out of which, 42,000 in the training set and 28,000 is testing test. 
# The digits are already converted to vectors of 784 data points. Each data point will  be considered as feature in this dataset
# 
# 
# ## Problem statement
# 
# ### Higher Dimension
# In this kernel I would like to focus on visualization of data points of these samples more than actual classifications. As we all know the task of representation of datapoints with features more than 2 or 3 is not new. It is commom problem to plot the graphs with higher dimentional features. Here we have 784 features. 
# 
# ### Expensive TSNE Computation
# Whoever working with SKLearn might have already figured out, TSNE computation is very expensive. For example, fitting 10,000 MNIST data with perplexity=30 and 1000 iteration would take around 3/4 min to complete. 
# 
# ## Available solutions 
# Best solution to represent the higher dimensions data is to perform Dimensionality reduction exercise. Following are solutions available
# 
# 1. **Feature Eliminations**: Each and every feature is removed and/or added to the solution and check the error results. Which ever feature had least positive impact on results or most negative impact, are eliminated from the set of features. 
# 
# 2. **Factor Analysis**: Among the features, check the correlations between the features themselves. If we find correlated features, we can keep one among them, rest can be removed 
# 
# 3. **Principal Component Analysis PCA**:  Most of common approach is PCA, where, we can project the data points orthogonally on principal component and reduce dimensions
# 
# 4. **T Distributed Stochastic Neighbourhood Embedding (T-SNE)**: This is most advanced and latest technology where data points are embedded on lower dimesion by keeping local topology and not concerning with gloabl configuration. Also this is similar to graph based technic. 
# 
# ### GPU Solution
# To tackle the problem of slow TSNE computation, we need to find a library which uses GPU effectively. One of the best available library which uses GPU for tsne computation is  ```tsnecuda``` https://github.com/CannyLab/tsne-cuda. However, Kaggle does not comes with tsnecuda and other pre-requisites installed in GPU enabled kernels. This notebook helps to install required libraries and pre-prequisites for running ```tsnecuda```
# 
# 
# ## Basic checks
# Let us practice on TSNE techniche in this notebook and try to reduce 784 dimensions of MNIST to 2 dimensions and plot it on 2D graphics. Also let us perform those tsne computations using GPU using ```tsnecuda``` library

# First of all, let us see if GPU is present

# In[1]:


get_ipython().system('nvidia-smi')


# Next check what is the version of cuda in this GPU 

# In[2]:


get_ipython().system('cat /usr/local/cuda/version.txt')


# ## Installation of Pre-requisites
# 
# 1. One of the pre-requisite library for tsnecuda to run is ```faiss```. Let us isntall the same for the version of *cuda*

# In[3]:


## Passing Y as input while conda asks for confirmation, we use yes command
get_ipython().system('yes Y | conda install faiss-gpu cudatoolkit=10.0 -c pytorch')


# 2. Now let us install *tsnecuda* from the sources
# Also we found *libfaiss.so* was not found while running, but this file comes as part of sources. So, we move that to ```/usr/local/cuda/lib64```

# In[4]:


get_ipython().system('wget https://anaconda.org/CannyLab/tsnecuda/2.1.0/download/linux-64/tsnecuda-2.1.0-cuda100.tar.bz2')
get_ipython().system("tar xvjf tsnecuda-2.1.0-cuda100.tar.bz2 --wildcards 'lib/*'")
get_ipython().system("tar xvjf tsnecuda-2.1.0-cuda100.tar.bz2 --wildcards 'site-packages/*'")
get_ipython().system('cp -r site-packages/* /opt/conda/lib/python3.6/site-packages/')
# !export LD_LIBRARY_PATH="/kaggle/working/lib/" 
get_ipython().system('cp /kaggle/working/lib/libfaiss.so /usr/local/cuda/lib64/')


# 3. We found another library missing, *openblas*, we install that now.

# In[5]:


get_ipython().system('apt search openblas')
get_ipython().system('yes Y | apt install libopenblas-dev')
# !printf '%s\n' 0 | update-alternatives --config libblas.so.3 << 0
# !apt-get install libopenblas-dev 
get_ipython().system('rm -rf  ./*')


# ## Start the main objective of notebook

# In[6]:


import numpy as np
import pandas as pd
from keras.datasets import mnist
import matplotlib.pyplot as plt
from tsnecuda import TSNE
from sklearn.decomposition import PCA
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train=X_train.reshape(60000,-1)
X_test=X_test.reshape(len(X_test),-1)
X_train.shape


# In[8]:


# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train=X_train.reshape(60000,-1)/255
X_test=X_test.reshape(len(X_test),-1)/255
X=np.concatenate((X_train, X_test), axis=0)
y=np.concatenate((y_train, y_test), axis=0)


# In[9]:


plt.imshow(X[1116].reshape(28,28))


# ### 接下來使用t-SNE做降維，sklearn t-SNE中較重要的參數:
# - n_components: 降維之後的維度
# - perpexity: 最佳化過程中考慮鄰近點的多寡，default 30，原始paper建議5-50
# - n_iter: 迭代次數，預設1000

# In[10]:


tsne = TSNE(n_components=2, perplexity=42, n_iter=200000)
train_reduced = tsne.fit_transform(X_train)


# In[11]:


plt.figure(figsize=(8,6))
plt.scatter(train_reduced[:, 0], train_reduced[:, 1], c=y_train, alpha=0.5,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))

plt.colorbar()
plt.show()


# ## XGBoost(regression) Fit t-SNE model
# > 這一步驟是學出一個模型可以直接將784維的(input)資料預測t-sne過後的2D資料。

# In[12]:


from sklearn.multioutput import MultiOutputRegressor
import xgboost

xgb = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.09,
                 max_depth=6,
                 min_child_weight=1.5,
                 n_estimators=5000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42,
                 objective ='reg:squarederror',
                 tree_method='gpu_hist',
                 predictor='cpu_predictor')
xgbModel=MultiOutputRegressor(xgb)

xgbModel.fit(X_train, train_reduced)
trainPred=xgbModel.predict(X_train)


# ### 內部測試

# In[13]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
print(("Score: ",xgbModel.score(X_train, train_reduced)))
print(("MAE: ",mean_absolute_error(trainPred,train_reduced)))
print(("MSE: ",(mean_squared_error(trainPred,train_reduced))))
print(("RMSE: ",sqrt(mean_squared_error(trainPred,train_reduced))))


# In[14]:


plt.figure(figsize=(8,6))
plt.scatter(trainPred[:, 0], trainPred[:, 1], c=y_train, alpha=0.5,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))

plt.colorbar()
plt.show()


# ### 外部測試

# In[15]:


testPred=xgbModel.predict(X_test)
plt.figure(figsize=(8,6))
plt.scatter(testPred[:, 0], testPred[:, 1], c=y_test, alpha=0.5,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))

plt.colorbar()
plt.show()


# ### 儲存XGBoost(regression) Model

# In[16]:


import pickle
import gzip
with gzip.GzipFile('./xgb(regression)-42-12000-scale-all.pgz', 'w') as f:
    pickle.dump(xgbModel, f)


# ### 輸出KNN(784D->2D)預測

# In[17]:


X=np.concatenate((X_train, X_test), axis=0)
y=np.concatenate((y_train, y_test), axis=0)
pred=xgbModel.predict(X)

pdData = pd.DataFrame(pred, columns = ["x1", "x2"])
pdData["y"]=y
pdData.to_csv('./Result-xgb-42-12000-scale-all.csv',index=False)


# ## XGboost(classfication) Model
# 這一部分是利用KNN所預測出來的2D資料去學習一個預測0~9數字的分類模型

# In[18]:


#讀取資料
data = pd.read_csv("./Result-xgb-42-12000-scale-all.csv") #load the dataset
X=data[['x1','x2']].values
y=data[['y']].values.reshape(-1)


#  ### 切割訓練集與測試集

# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=100)


# ### 模型初始化與學習

# In[20]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# fit model no training data
xgbModel = XGBClassifier(tree_method='gpu_hist',predictor='cpu_predictor')
xgbModel.fit(X_train, y_train)


# In[21]:


# make predictions for test data
y_pred = xgbModel.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print(("Accuracy: %.2f%%" % (accuracy * 100.0)))


# ## 儲存XGboost(classfication) Model

# In[22]:


import pickle
import gzip
with gzip.GzipFile('./xgb(classfication)-42-12000-scale-all.pgz', 'w') as f:
    pickle.dump(xgbModel, f)


# In[ ]:




