#!/usr/bin/env python
# coding: utf-8

# This is my first kernel on Kaggle. All comments welcome :)
# 
# Some parts of the code (modfied and used as is) has been taken or was inspired from the original work by Dr Mehdi Mohammadi and improvisation work by ryches. Links below.
# 
# https://www.kaggle.com/mehdimka/localization-in-waldo-library-using-keras-dnn
# 
# https://www.kaggle.com/ryches/better-indoor-localization-wip
# 
# 
# ## About
# From source:
# The dataset was created using the RSSI readings of an array of 13 ibeacons in the first floor of Waldo Library, Western Michigan University. Data was collected using iPhone 6S. The dataset contains two sub-datasets: a labeled dataset (1420 instances) and an unlabeled dataset (5191 instances). The recording was performed during the operational hours of the library. For the labeled dataset, the input data contains the location (label column), a timestamp, followed by RSSI readings of 13 iBeacons. RSSI measurements are negative values. Bigger RSSI values indicate closer proximity to a given iBeacon (e.g., RSSI of -65dBm represent a closer distance to a given iBeacon compared to RSSI of -85dBm). For out-of-range iBeacons, the RSSI is indicated by -200dBm. The locations related to RSSI readings are combined in one column consisting a letter for the column and a number for the row of the position.
# 
# ## Importing Data

# In[1]:


import pandas as pd
pd.set_option('display.max_columns', None)

import numpy as np
from datetime import datetime as dt
import seaborn as sns


# In[2]:


path='../input/ble-rssi-dataset/iBeacon_RSSI_Labeled.csv'
data = pd.read_csv(path, index_col=None)
data.head(5)


# ## Data Exploration

# In[3]:


plots = data.hist(bins=15, figsize=(20,20))

for ax in plots.flatten():
    ax.set_xlabel("Signal Strength")
    ax.set_ylabel("count")


# From the above plot, we can understand that the strength of signal from each of the beacons is on either ends of the strength spectrum i.e. either good signal or no signal at all. This can be confirmed by the values of the signal strengths. -200 means no signal at all while the higher numbers mean good signal. 
# 
# I realized as a matter of fact that a signal strength can not be 0 after I tested the signal strength from my wifi on my phone and touching my phone to the wifi router gave me a strength of -49dBm. 
# 
# Edit: After further research I can't say for sure that the signal strengths can not be 0. Some people argue to have achieved signal strengths in positive values. I'll leave that for now.
# 
# Source: https://www.metageek.com/training/resources/wifi-signal-strength-basics.html
# 
# Though the above link talks about Wi-Fi signals and not Bluetooth as in the case with the dataset but it gives us an understanding of the signal strengths.
# 
# In the dataset, the max signal strength observed is -55 dBm from Beacon 9 which is a strength good enough of majority of the real world applications and usages.

# In[4]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(data.corr(method='kendall'), ax=ax)


# From the above co-relation heat-map, we can observe that the difference in the colors is high on the top left and on the bottom right while the bottom left and the top right are fairly the same. If we observe the layout of the library (supplied) where the data was collected, we can see that the overlap of the signals is directly proportional to the distance by which they are kept apart. 
# ![title](../input/ble-rssi-dataset/iBeacon_Layout.jpg)
# 
# ## Data Preparation

# In[5]:


data.max()


# Max signal strength has been observed from beacon #9 with -55dBm. It is to be noted that in real life, the signal strength might not be a constant and depends on various factors like temperature, humidity, etc. in the medium. 
# 
# For the preparation part, we shall do the following:
# 
# 1) Bin the data between -50dBm and -200dBm in bin ranges of 10. It is to be noted that there we will no values for some bins as observed from the plots in the data exploration phase.<br>
# 2) The location feature is a combination of both the x-axis and y-axis information. i.e. The alphabet in the location is an axis point on the x-coordinate and the number an axis point in the y-coordinate. We split them into two features: x and y. Then, drop the location feature.<br>
# 3) Label encode the split location features (x and y) using sklearn's LabelEncoder thus transforming the values between 0 and n_classes-1.<br>
# 4) Drop the date feature.

# In[6]:


label = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"]
for col in data.select_dtypes(include="number").columns:
    data[col] = pd.cut(data[col], bins = 15, labels = label)
    
# Splitting the location:
data['x'] = data['location'].str[0]
data['y'] = data['location'].str[1:]

# Label Encoding
from sklearn.preprocessing import LabelEncoder
data['x'] = LabelEncoder().fit_transform(data['x'])
data['y'] = LabelEncoder().fit_transform(data['y'])

# Dropping the columns
data = data.drop(columns=["date","location"])

data.head(5)


# One-hot encoding the descriptive features and separating the target features

# In[7]:


data = pd.get_dummies(data, columns=data.columns[0:-2])
data.head(5)


# In[8]:


target_x = data['x']
target_y = data['y']
data.drop(columns=['x','y'], inplace=True)
data.head(5)


# ## Modeling
# 
# For modeling, since we have two target features, a model each has to be chosen for each of the target features. Since it is classification, KNN and Decision Tree classifiers have been selected. The models are fit with the best hyper parameters using GridSearchCV from scikit learn for both the target features.
# 
# Then the accuracies can be plotted by mixing and matching these models on the data and selecting the best model(s) for the data.
# 
# First, split the data set into test and train set for both the target features.

# In[9]:


from sklearn.model_selection import train_test_split

D_train, D_test, t_train_x, t_test_x = train_test_split(data, 
                                                    target_x, 
                                                    test_size = 0.3,
                                                    random_state=999)

D_train, D_test, t_train_y, t_test_y = train_test_split(data, 
                                                    target_y, 
                                                    test_size = 0.3,
                                                    random_state=999)


# Next, import the classifiers, k-fold method and GridSearchCV for fitting the model with the best params and training with k-folds.

# In[10]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV

k_fold_method = RepeatedStratifiedKFold(n_splits=5, 
                                    n_repeats=3, 
                                    random_state=8)

################################## KNN #####################################################

parameters_knn = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15], 
              'p': [1, 2, 5]}

knn = KNeighborsClassifier()

gs_knn_x = GridSearchCV(estimator=knn, 
                      param_grid=parameters_knn, 
                      cv=k_fold_method,
                      verbose=1, 
                      n_jobs=-2,
                      scoring='accuracy',
                      return_train_score=True)

gs_knn_y = GridSearchCV(estimator=knn, 
                      param_grid=parameters_knn, 
                      cv=k_fold_method,
                      verbose=1, 
                      n_jobs=-2,
                      scoring='accuracy',
                      return_train_score=True)

################################### DT ########################################################

parameters_dt = {'criterion':['gini','entropy'],'max_depth':[2,3,4]}

dt = DecisionTreeClassifier()

gs_dt_y = GridSearchCV(estimator=dt,
                    param_grid=parameters_dt,
                    cv = k_fold_method,
                    verbose=1,
                    n_jobs=-2,
                    scoring='accuracy',
                    return_train_score=True)

gs_dt_x = GridSearchCV(estimator=dt,
                    param_grid=parameters_dt,
                    cv = k_fold_method,
                    verbose=1,
                    n_jobs=-2,
                    scoring='accuracy',
                    return_train_score=True)

####################################### SVC ####################################################

parameters_svc = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}

svc = SVC()

gs_svc_x = GridSearchCV(estimator=svc,
                    param_grid=parameters_svc,
                    cv = k_fold_method,
                    verbose=1,
                    n_jobs=-2,
                    scoring='accuracy',
                    return_train_score=True)

gs_svc_y = GridSearchCV(estimator=svc,
                    param_grid=parameters_svc,
                    cv = k_fold_method,
                    verbose=1,
                    n_jobs=-2,
                    scoring='accuracy',
                    return_train_score=True)


# Fitting the data to the model.

# In[11]:


gs_dt_y.fit(D_train, t_train_y)
gs_dt_x.fit(D_train, t_train_x)

gs_knn_y.fit(D_train, t_train_y)
gs_knn_x.fit(D_train, t_train_x)

gs_svc_y.fit(D_train, t_train_y)
gs_svc_x.fit(D_train, t_train_x)


# Defining a function to return us the array containing Euclidean distances between the actual and predicted points and a dictionary to hold our fitted models' predictions.

# In[12]:


#function to return an array with distances between the actual and predicted points
def distance(x_actual, y_actual, x_predicted, y_predicted):
    d_x = x_actual - x_predicted
    d_y = y_actual - y_predicted
    dist = d_x**2 + d_y**2
    dist = np.sqrt(dist)
    #dist = np.sort(dist)
    return dist


# Predictions for each model for both x and y

# In[13]:


models_predictions_x = {'KNN_x': gs_knn_x.predict(D_test), 'DT_x': gs_dt_x.predict(D_test), 'SVC_x': gs_svc_x.predict(D_test)}

models_predictions_y = {'KNN_y': gs_knn_y.predict(D_test), 'DT_y': gs_dt_y.predict(D_test), 'SVC_y': gs_svc_y.predict(D_test)}


# Plotting the prediction probabilities

# In[14]:


fig, axs = plt.subplots(3, 3, figsize=(15,15))
(ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) = axs

for ax, px in zip(axs,models_predictions_x):
    for axes, py in zip(ax,models_predictions_y):
        distances = distance(t_test_x, t_test_y, models_predictions_x[px], models_predictions_y[py])
        sorted_distances = np.sort(distances)
        probabilites = 1. * np.arange(len(sorted_distances))/(len(sorted_distances) - 1)
        axes.plot(sorted_distances, probabilites)
        axes.set_title(f'CDF: Euclidean dist. error: {px}|{py}')
        axes.set(xlabel = 'Distance (m)', ylabel = 'Probability')
        axes.text(2,0.05,f"Mean Error dist.: {np.mean(distances)}")
        axes.grid(True)
        gridlines = axes.get_xgridlines() + axes.get_ygridlines()
        for line in gridlines:
            line.set_linestyle(':')

fig.tight_layout()
plt.show()
plt.close()


# From the above plots, it can be concluded that using Decision Trees to predict y target feature was not a good idea. The single best model to use is SVC to predict both the x-target feature and y-target feature. It can predict with 100% probability within ~6m radius of the actual location while having only ~1.69m mean error distance. Whereas, KNN predicts with a 100% probability within a radius of ~7m and having a mean error distance of ~1.9m There is no need to use DT at all.<br>
# The worst option is to choose only DT to predict both the features.
# 
# Below is the plotting of the actual and predicted points on the image using SVC. (Taken and modified from ryches notebook)

# In[15]:


from plotly.offline import init_notebook_mode, iplot
from IPython.display import display, HTML
import numpy as np
from PIL import Image

image = Image.open("../input/ble-rssi-dataset/iBeacon_Layout.jpg")
init_notebook_mode(connected=True)

xm=np.min(t_test_x)-1.5
xM=np.max(t_test_x)+1.5
ym=np.min(t_test_y)-1.5
yM=np.max(t_test_y)+1.5

data=[dict(x=[0], y=[0], 
           mode="markers", name = "Predictions",
           line=dict(width=2, color='green')
          ),
      dict(x=[0], y=[0], 
           mode="markers", name = "Actual",
           line=dict(width=2, color='blue')
          )
      
    ]

layout=dict(xaxis=dict(range=[xm, 24], autorange=False, zeroline=False),
            yaxis=dict(range=[ym, 21], autorange=False, zeroline=False),
            title='Predictions for SVC', hovermode='closest',
            images= [dict(
                  source= image,
                  xref= "x",
                  yref= "y",
                  x= -3.5,
                  y= 22,
                  sizex= 36,
                  sizey=25,
                  sizing= "stretch",
                  opacity= 0.5,
                  layer= "below")]
            )

frames=[dict(data=[dict(x=[models_predictions_x['SVC_x'][k]], 
                        y=[models_predictions_y['SVC_y'][k]], 
                        mode='markers',
                        
                        marker=dict(color='red', size=10)
                        ),
                   dict(x=[t_test_x.iloc[k]], 
                        y=[t_test_y.iloc[k]], 
                        mode='markers',
                        
                        marker=dict(color='blue', size=10)
                        )
                  ]) for k in range(int(len(t_test_x))) 
       ]    
          
figure1=dict(data=data, layout=layout, frames=frames)          
iplot(figure1)

