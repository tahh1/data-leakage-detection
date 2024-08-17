#!/usr/bin/env python
# coding: utf-8

# In[38]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle


# ### 点集实验bagging

# In[57]:


def plot_decision_boundary(X, model):
    h = .02 

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))


    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)


# #### Bagging基于决策树  
# 
# - 实现fit, predict, score三个函数  
# 
# 
# sklearn model template:
# 
# class Mymodel(BaseEstimator):  
# 
#     def __init__(self, ...):
#     
#     
#     def fit(self, ...):
#     
#     
#     def predict(self, ...):
#     
#     
#     def score(self, ...):
#     

# In[58]:


class BaggedTreeClassifier:
    def __init__(self, M):
        # M个决策树
        # TODO
        self.M = M

        
    def fit(self, X, Y):
        # TODO
        N = len(X)
        # 袋子
        self.models = list()
        # 有放回的采样N个数据
        for m in range(self.M):
            idx = np.random.choice(N, size=N, replace=True)
            Xb = X[idx]
            Yb = Y[idx]
            model = DecisionTreeClassifier(max_depth=5)
            model.fit(Xb, Yb)
            self.models.append(model)
        # 创建深度为5的决策树

    
    def predict(self,X):
        # TODO
        # 二分类预测
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += model.predict(X)
        return np.round(predictions / self.M)
    
    def score(self,X,Y):
        #TODO
        result = self.predict(X)
        return np.mean(result == Y)
        


# #### 在线性不可分上的实验 XOR异或数据点

# In[60]:


## 决策树的代码 

np.random.seed(10)

N = 500
D = 2
X = np.random.randn(N, D)

delta = 1.75
X[:125] += np.array([delta, delta])
X[125:250] += np.array([delta, -delta])
X[250:375] += np.array([-delta, delta])
X[375:] += np.array([-delta, -delta])
Y = np.array([0] * 125 + [1]*125 + [1]*125 + [0] * 125)

plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()

# 基础树模型
model = DecisionTreeClassifier()
# use different model and compare the boundaries.
#model = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
model.fit(X, Y)
print(("score for basic tree:", model.score(X, Y)))

# plot data with boundary
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model)
plt.show()


# 树的深度为3的模型
model_depth_3 = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model_depth_3.fit(X, Y)

print(("score for tree of depth 3:", model_depth_3.score(X, Y)))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_3)
plt.show()

# 树的深度为5的模型
model_depth_5 = DecisionTreeClassifier(criterion='entropy', max_depth=5)
model_depth_5.fit(X, Y)

print(("score for tree of depth 5:", model_depth_5.score(X, Y)))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_5)
plt.show()


#logistic regression模型
model_logistic = LogisticRegression()
model_logistic.fit(X, Y)

print(("score for logistic regression model:", model_logistic.score(X, Y)))

# 可视化
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_logistic)
plt.show()  

# bagged tree 模型
#TODO
baggedTree = BaggedTreeClassifier(200)
baggedTree.fit(X, Y)
print(("score for bagged tree model:", baggedTree.score(X, Y)))

plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, baggedTree)
plt.show()


# #### 在线性不可分上的实验 甜甜圈数据点

# In[61]:


## 决策树代码

np.random.seed(10)

N = 500
D = 2
X = np.random.randn(N, D)

R_smaller = 5
R_larger = 10

R1 = np.random.randn(N//2) + R_smaller
theta = 2 * np.pi * np.random.random(N//2)
X[:250] = np.concatenate([[R1 * np.cos(theta)], [R1*np.sin(theta)]]).T


R2 = np.random.randn(N//2) + R_larger
theta = 2 * np.pi * np.random.random(N//2)
X[250:] = np.concatenate([[R2 * np.cos(theta)], [R2*np.sin(theta)]]).T

Y = np.array([0] * (N//2) + [1] * (N//2))


# plot the data
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()

# 基础树模型
model = DecisionTreeClassifier()
model.fit(X, Y)
print(("score for basic tree:", model.score(X, Y)))

plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model)
plt.show()


# 树的深度为3的模型
model_depth_3 = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model_depth_3.fit(X, Y)

print(("score for tree of depth 3:", model_depth_3.score(X, Y)))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_3)
plt.show()

# 树的深度为5的模型
model_depth_5 = DecisionTreeClassifier(criterion='entropy', max_depth=5)
model_depth_5.fit(X, Y)

print(("score for tree of depth 5:", model_depth_5.score(X, Y)))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_5)
plt.show()


# logistic regression 模型
model_logistic = LogisticRegression()
model_logistic.fit(X, Y)

print(("score for logistic regression model:", model_logistic.score(X, Y)))

plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_logistic)
plt.show()

# bagged tree 模型
#TODO
baggedTree = BaggedTreeClassifier(200)
baggedTree.fit(X,Y)
print(("score for bagged tree:", baggedTree.score(X, Y)))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, baggedTree)
plt.show()


# In[ ]:




