#!/usr/bin/env python
# coding: utf-8

# # CSCE 421 HW1
# ### Sahil Palnitkar
# 
# 
# 
# #### Problem 1
# 1) 
# 
#    $
#        \nabla(f)= [\frac{\partial f(x,y)}{\partial x}, \frac{\partial f(x,y)}{\partial y}]
#    $
#    
#    $
#        \frac{\partial f(x,y)}{\partial x}(x^2+ ln(x)+xy+y^3)=2x+\frac{1}{x}+y
#    $
#    
#    $
#        \frac{\partial f(x,y)}{\partial y}(x^2+ ln(x)+xy+y^3)=x+3y^2
#    $
#    
#    $
#        = \begin{bmatrix}\left(2x+\frac{1}{x}+y\right)\\ \left(x+3y^2\right)\end{bmatrix}
#    $
#    
#    Evaluating gradient value at (x,y) = (1, -1), we get
#    
#    $
#       = \begin{bmatrix}\left(2\right)\\ \left(4\right)\end{bmatrix}
#    $
#    
#    
#    
# 2)
#    
#    $
#       \nabla(f)= [\frac{\partial f(x,y,z)}{\partial x}, \frac{\partial f(x,y,z)}{\partial y}, \frac{\partial f(x,y,z)}{\partial z}]
#    $
# 
#    $
#       \frac{\partial f(x,y,z)}{\partial x}\left(\tanh \left(x^3y^3\right)+\sin \left(z\right)\right) = 3x^2y^3sech^2\left(x^3y^3\right)
#    $
#    
#    $
#       \frac{\partial f(x,y,z)}{\partial y}\left(\tanh \left(x^3y^3\right)+\sin \left(z\right)\right) = 3x^3y^2sech^2\left(x^3y^3\right)
#    $
#    
#    $
#       \frac{\partial f(x,y,z)}{\partial y}\left(\tanh \left(x^3y^3\right)+\sin \left(z\right)\right) = cos \left(z\right)
#    $
#    
#    Evaluating gradient value at (x,y,z) = (-1, 0, $\frac{\pi}{2}$), we get
#    
#    $
#       = \begin{bmatrix}\left(0\right)\\ \left(0\right)\\ \left(0\right)\end{bmatrix}
#    $
#    
#    
# #### Problem 2
# 1)
#     $
#     \begin{pmatrix}1\cdot 6+\left(-1\right)\cdot 0+6\left(-3\right)+7\cdot 11&1\cdot 2+\left(-1\right)\left(-1\right)+6\cdot 0+7\cdot 4\\ 9\cdot 6+0\cdot 0+8\left(-3\right)+1\cdot 11&9\cdot 2+0\cdot \left(-1\right)+8\cdot 0+1\cdot 4\\ \left(-8\right)\cdot 6+5\cdot 0+2\left(-3\right)+3\cdot 11&\left(-8\right)\cdot 2+5\left(-1\right)+2\cdot 0+3\cdot \:4\\ 10\cdot \:6+4\cdot \:0+0\cdot \left(-3\right)+1\cdot 11&10\cdot 2+4\left(-1\right)+0\cdot 0+1\cdot 4\end{pmatrix}
#     $
#         $
#     = \begin{pmatrix}65&31\\ 41&22\\ -21&-9\\ 71&20\end{pmatrix}
#     $
#     
# 2)
#     $
#     \begin{pmatrix}10\cdot 7&10\cdot 3&10\cdot 0&10\cdot 1\\ 4\cdot 7&4\cdot 3&4\cdot 0&4\cdot 1\\ \left(-1\right)\cdot 7&\left(-1\right)\cdot 3&\left(-1\right)\cdot 0&\left(-1\right)\cdot 1\\ 8\cdot 7&8\cdot 3&8\cdot 0&8\cdot 1\end{pmatrix}
#     $
#     $
#     = \begin{pmatrix}70&30&0&10\\ 28&12&0&4\\ -7&-3&0&-1\\ 56&24&0&8\end{pmatrix}
#     $
#     
# 3)
#     $
#     \begin{pmatrix}9\left(-3\right)+\left(-3\right)\cdot 4+1\cdot \left(-9\right)+6\cdot 0\end{pmatrix}
#     $
#     $
#     = \left(-48\right)
#     $
#     
# 
# #### Problem 3
# 1) 
#     Vector 
#     $
#     = \begin{bmatrix}\left(5-7\right)\\ \left(0-9\right)\\ \left(-1-5\right)\\ \left(4-2\right)\end{bmatrix} = \begin{bmatrix}\left(-2\right)\\ \left(-9\right)\\ \left(-6\right)\\ \left(2\right)\end{bmatrix}
#     $
#     So we get
#     $
#     \ell_0 = 4
#     $
#     as there are 4 non-zero terms.
#     
# 2)
#     Vector
#     $
#     = \begin{bmatrix}\left(5-7\right)\\ \left(0-9\right)\\ \left(-1-5\right)\\ \left(4-2\right)\end{bmatrix} = \begin{bmatrix}\left(-2\right)\\ \left(-9\right)\\ \left(-6\right)\\ \left(2\right)\end{bmatrix}
#     $
#     So we get
#     $
#     \ell_1 = |(-2 + -9 + -6 + 2)| = (15)
#     $
#     
# 3)
#     $
#     \ell_2 = \sqrt{\left(5-7\right)^2 + \left(0-9\right)^2 + \left(-1-5\right)^2 + \left(4-2\right)^2}
#            = \sqrt{\left(-2\right)^2 + \left(-9\right)^2 + \left(-6\right)^2 + \left(2\right)^2}
#            = \sqrt{\left(4\right) + \left(81\right) + \left(36\right) + \left(4\right)}
#            = \sqrt{\left(125\right)}
#            = 11.1803
#     $
#     
# 4)  
#     Vector 
#     $
#     = \begin{bmatrix}\left(5-7\right)\\ \left(0-9\right)\\ \left(-1-5\right)\\ \left(4-2\right)\end{bmatrix} = \begin{bmatrix}\left(-2\right)\\ \left(-9\right)\\ \left(-6\right)\\ \left(2\right)\end{bmatrix}
#     $
#     So we get
#     $
#     \ell_\infty = 9
#     $
#     as this is the absolute value of the max term of the vector.
#     
# #### Problem 4
# 
# 1) The sample space is $6 \cdot 6 = 36$
# 
# 2) The sum 10 can be achieved by getting either $\begin{pmatrix}4,6\end{pmatrix}$ or $\begin{pmatrix}5,5\end{pmatrix}$ or $\begin{pmatrix}6,4\end{pmatrix}$. This is 3 ways out of 36. So the probability of getting a sum of 10 when 2 dice are rolled is $\frac{3}{36} = \frac{1}{12}$
# 
# 3) The sum 6 can be achieved by getting either $\begin{pmatrix}1,5 \end{pmatrix}$ or $\begin{pmatrix}2,4\end{pmatrix}$ or $\begin{pmatrix}3,3\end{pmatrix}$ or $\begin{pmatrix}4,2\end{pmatrix}$ or $\begin{pmatrix}5,1\end{pmatrix}$. This is 5 ways out of 36. So the probability of getting a sum 6 when 2 dice are rolled is $\frac{5}{36}$
# 
# 
# #### Problem 5
# 1) The mean of X will be $ \frac{\left(a + b\right)}{2} $
# 
# 2) The standard deviation of X will be $ \sqrt{\frac{\left(b-a\right)^2}{12}} $
# 
# 
# #### Problem 6
# 1) The accuracy of the detector will be $ \frac{\left(FP+FN\right)}{Total} = \frac{\left(37+55\right)}{160} = 0.575 $
# 
# 
# 2) The sensitivity of the detector is $ \frac{TP}{\left(TP+FN\right)} = \frac{37}{\left(37+45\right)} = 0.4512 $
# 
#    The specificity of the detector is $ \frac{TN}{\left(FP+TN\right)} = \frac{55}{\left(23+55\right)} = 0.7051 $
#    
#    Balanced accuracy = $ \frac{\left(Sensitivity + Specifity\right)}{2} = \frac{\left(0.4512 + 0.7051\right)}{2} = 0.5782 $
# 
# 
# 3) The precision of the detector is $ \frac{TP}{\left(TP+FP\right)} = 0.6167 $
# 
# 
# 4) The recall is the same as the sensitivity which is 0.4512
# 
# 
# 5) The F-measure of the detector is $ \frac{2\cdot Precision\cdot Recall}{\left(Precision + Recall\right)} = 0.5211 $
# 
# 
# #### Problem 7
# 
# 1) ROC value for threshold = 0 will be $\begin{pmatrix}1,1\end{pmatrix}$
# 
# 2) ROC value for threshold = 0.25 will be $\begin{pmatrix}0.4,0.8\end{pmatrix}$
# 
# 3) ROC value for threshold = 0.5 will be $\begin{pmatrix}0.4,0.6\end{pmatrix}$
# 
# 4) ROC value for threshold = 0.75 will be $\begin{pmatrix}0,0.2\end{pmatrix}$
# 
# 5) ROC value for threshold = 1 will be $\begin{pmatrix}0,0\end{pmatrix}$
# 
# 6) AUROC using left endpoint approximation will be 
# 
#    $\int_{a}^b f(x)dx \approx \sum_{i=1}^{n-1}{x_{i+1} - x_i}\cdot f(x_i) $
#    
#    $\therefore \int_{a}^b f(x)dx \approx \left(\left(0 - 0\right)\cdot 0\right) + \left(\left(0.4 - 0\right)\cdot 0.2\right) + \left(\left(0.4 - 0.4\right)\cdot 0.6\right) + \left(\left(1 - 0.4\right)\cdot 0.8\right) \approx 0.56 $
# 

# In[3]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# In[4]:


dataframe = pd.read_csv("Smarket.csv")


# In[5]:


print((dataframe.head()))


# In[6]:


print((dataframe.shape))


# In[7]:


X = dataframe[['Lag1','Lag2']]
y = dataframe['Direction']
print(X)
print(y)


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[9]:


plot_y = []
plot_x = []
for i in range(1,11):
    plot_y.append(i)
    print((str(i) + ":"))
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train,y_train) #Trains model
    score_test = neigh.score(X_test, y_test) #.score uses .predict as an underlying function
    plot_x.append(score_test)
    print(("test_" + str(i) + ":"))
    print(score_test)
    


# In[10]:


plt.plot(plot_x)
plt.ylabel(plot_y)
plt.show()

