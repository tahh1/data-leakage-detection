#!/usr/bin/env python
# coding: utf-8

# # Exercice : Zalando Fashion MNIST

# ## Librairies et fonctions utiles

# In[1]:


# Directive pour afficher les graphiques dans Jupyter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Pandas : librairie de manipulation de données
# NumPy : librairie de calcul scientifique
# MatPlotLib : librairie de visualisation et graphiques
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split

from IPython.core.display import HTML # permet d'afficher du code html dans jupyter


# Fonction pour tracer les courbes d'apprentissage sur l'ensemble d'apprentissage et l'ensemble de validation :

# In[3]:


from sklearn.model_selection import learning_curve
def plot_learning_curve(est, X_train, y_train) :
    train_sizes, train_scores, test_scores = learning_curve(estimator=est, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10),
                                                        cv=5,
                                                        n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.figure(figsize=(8,10))
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean,color='green', linestyle='--',marker='s', markersize=5,label='validation accuracy')
    plt.fill_between(train_sizes,test_mean + test_std,test_mean - test_std,alpha=0.15, color='green')
    plt.grid(b='on')
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.6, 1.0])
    plt.show()


# Fonction pour tracer la courbe ROC :

# In[4]:


def plot_roc_curve(est,X_test,y_test) :
    probas = est.predict_proba(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,probas[:, 1])
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.figure(figsize=(8,8))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')        # plus mauvaise courbe
    plt.plot([0,0,1],[0,1,1],'g:')     # meilleure courbe
    plt.xlim([-0.05,1.2])
    plt.ylim([-0.05,1.2])
    plt.ylabel('Taux de vrais positifs')
    plt.xlabel('Taux de faux positifs')
    plt.show


# ## Zalando Fashion MNIST

# Le dataset a été constitué par Zalando :  
# https://github.com/zalandoresearch/fashion-mnist  
#   
# On a un ensemble d'apprentissage de 60 000 images 28x28 pixels en niveaux de gris, et 10 classes de vêtements : jupes, pantalons, baskets, ...

# <img src="https://github.com/zalandoresearch/fashion-mnist/blob/master/doc/img/fashion-mnist-sprite.png?raw=true">

# In[5]:


df = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')


# In[6]:


df.head()


# In[7]:


labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt",
          "Sneaker","Bag","Ankle boot"]


# La première image du dataset est un pull :

# In[8]:


print((labels[df.label[0]]))


# **Afficher les 50 premiers éléments du dataset avec leur label**  
# 

# In[9]:


y = df['label']
X = df.drop(['label'], axis=1)


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[11]:


X1 = np.array(X) 


# In[12]:


image = X1[0].reshape(28,28)
plt.imshow(image)


# In[13]:


plt.imshow(image, cmap="gray_r")
plt.axis('off')
plt.title(y[0])


# In[14]:


n_samples = len(df.index)
images =X1.reshape(n_samples,28,28)


# In[15]:


plt.figure(figsize=(10,20))
for i in range(50) :
    plt.subplot(10,5,i+1)
    plt.axis('off')
    plt.imshow(images[i], cmap="gray_r")  
    plt.title(labels[y[i]])


# **Appliquer des méthodes de machine learning à la reconnaissance des objets (forêts aléatoires, xgboost, ...)**  

# In[16]:


from sklearn import ensemble
rf = ensemble.RandomForestClassifier()
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)


# In[17]:


plot_learning_curve(rf, X, y)


# In[18]:


print((classification_report(y_test, y_rf)))


# In[19]:


cm = confusion_matrix(y_test, y_rf)
print(cm)


# # XGBoost

# In[20]:


import xgboost as XGB
xgb  = XGB.XGBClassifier()
xgb.fit(X_train, y_train)
y_xgb = xgb.predict(X_test)
cm = confusion_matrix(y_test, y_xgb)
print(cm)
print((classification_report(y_test, y_xgb)))

