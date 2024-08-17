#!/usr/bin/env python
# coding: utf-8

# # Car Accident Severity Project

# In[1]:


import pandas as pd #load pandas library
import numpy as np #load numpy library 

car_accidents=pd.read_csv("https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Data-Collisions.csv")


# Below is a sample of the data:

# In[2]:


car_accidents.head()


# In[3]:


print((car_accidents.columns))
print((car_accidents.dtypes))


# In[4]:


car_accidents.describe(include ="all")


# In[5]:


car_accidents[['LIGHTCOND','WEATHER','ROADCOND','JUNCTIONTYPE','COLLISIONTYPE','SPEEDING']].describe(include ="object")


# In[6]:


car_accidents[['SEVERITYCODE','SEVERITYDESC']]


# As indicated above, the severity code consists of two values: 
# - SEVERITYCODE=1 means Property Damage Only Collision
# - SEVERITYCODE=2 means Injury Collision

# In[7]:


df=car_accidents[['LIGHTCOND','WEATHER','ROADCOND','JUNCTIONTYPE','COLLISIONTYPE','SPEEDING','SEVERITYCODE']]
df.head(5)


# In[8]:


missing_data = df.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print((missing_data[column].value_counts()))
    print("") 


# In[9]:


df["LIGHTCOND"].dropna(axis=0,inplace=True)
df["ROADCOND"].dropna(axis=0,inplace=True)
df["WEATHER"].dropna(axis=0,inplace=True)
df["JUNCTIONTYPE"].dropna(axis=0,inplace=True)
df["COLLISIONTYPE"].dropna(axis=0,inplace=True)
df["SPEEDING"].dropna(axis=0,inplace=True)


# In[10]:


light=df['LIGHTCOND'].value_counts().to_frame()
light


# In[11]:


weather=df['WEATHER'].value_counts().to_frame()
weather


# In[12]:


road=df['ROADCOND'].value_counts().to_frame()
road


# In[13]:


junction=df['JUNCTIONTYPE'].value_counts().to_frame()
junction


# In[14]:


collision=df['COLLISIONTYPE'].value_counts().to_frame()
collision


# In[15]:


speeding=df['SPEEDING'].value_counts().to_frame()
speeding


# In[16]:


df=df[['LIGHTCOND','WEATHER','ROADCOND','JUNCTIONTYPE','COLLISIONTYPE','SEVERITYCODE']] #dropped SPEEDING
df.head(5)


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


light.plot(kind='bar', figsize=(10, 6))

plt.xlabel('Light Conditions') # add to x-label to the plot
plt.ylabel('No. of Car Collisions') # add y-label to the plot
plt.title('Car Collisions vs. Light Conditions') # add title to the plot

plt.show()


# In[19]:


weather.plot(kind='bar', figsize=(10, 6))

plt.xlabel('Weather') # add to x-label to the plot
plt.ylabel('No. of Car Collisions') # add y-label to the plot
plt.title('Car Collisions vs. Weather') # add title to the plot

plt.show()


# In[20]:


road.plot(kind='bar', figsize=(10, 6))

plt.xlabel('Road Condition') # add to x-label to the plot
plt.ylabel('No. of Car Collisions') # add y-label to the plot
plt.title('Car Collisions vs. Road Condition') # add title to the plot

plt.show()


# In[21]:


junction.plot(kind='bar', figsize=(10, 6))

plt.xlabel('Junction Type') # add to x-label to the plot
plt.ylabel('No. of Car Collisions') # add y-label to the plot
plt.title('Car Collisions vs. Junction Type') # add title to the plot

plt.show()


# In[22]:


collision.plot(kind='bar',figsize=(10, 6))

plt.xlabel('Collision Type') # add to x-label to the plot
plt.ylabel('No. of Car Collisions') # add y-label to the plot
plt.title('Car Collisions vs. Collision Type') # add title to the plot

plt.show()


# In[23]:


df.columns


# In[24]:


#Encoding Light Conditions(0 = Light, 1 = Medium, 2 = Dark)
df["LIGHTCOND"].replace("Daylight", 0, inplace=True)
df["LIGHTCOND"].replace("Dark - Street Lights On", 1, inplace=True)
df["LIGHTCOND"].replace("Dark - No Street Lights", 2, inplace=True)
df["LIGHTCOND"].replace("Dusk", 1, inplace=True)
df["LIGHTCOND"].replace("Dawn", 1, inplace=True)
df["LIGHTCOND"].replace("Dark - Street Lights Off", 2, inplace=True)
df["LIGHTCOND"].replace("Dark - Unknown Lighting", 2, inplace=True)
df["LIGHTCOND"].replace("Other",999, inplace=True)
df["LIGHTCOND"].replace("Unknown",999, inplace=True)


# In[25]:


df["LIGHTCOND"].head()


# In[26]:


#Encoding Weather Conditions(0 = Clear, 1 = Overcast and Cloudy, 2 = Windy, 3 = Rain and Snow
df["WEATHER"].replace("Clear", 0, inplace=True)
df["WEATHER"].replace("Raining", 3, inplace=True)
df["WEATHER"].replace("Overcast", 1, inplace=True)
df["WEATHER"].replace("Other",999, inplace=True)
df["WEATHER"].replace("Unknown",999, inplace=True)
df["WEATHER"].replace("Snowing", 3, inplace=True)
df["WEATHER"].replace("Fog/Smog/Smoke", 2, inplace=True)
df["WEATHER"].replace("Sleet/Hail/Freezing Rain", 3, inplace=True)
df["WEATHER"].replace("Blowing Sand/Dirt", 2, inplace=True)
df["WEATHER"].replace("Severe Crosswind", 2, inplace=True)
df["WEATHER"].replace("Partly Cloudy", 1, inplace=True)


# In[27]:


df["WEATHER"].head()


# In[28]:


#Encoding Road Conditions(0 = Dry, 1 = Mushy, 2 = Wet)
df["ROADCOND"].replace("Dry", 0, inplace=True)
df["ROADCOND"].replace("Wet", 2, inplace=True)
df["ROADCOND"].replace("Ice", 2, inplace=True)
df["ROADCOND"].replace("Snow/Slush", 1, inplace=True)
df["ROADCOND"].replace("Other",999, inplace=True)
df["ROADCOND"].replace("Unknown",999, inplace=True)
df["ROADCOND"].replace("Standing Water", 2, inplace=True)
df["ROADCOND"].replace("Sand/Mud/Dirt", 1, inplace=True)
df["ROADCOND"].replace("Oil", 2, inplace=True)


# In[29]:


df["ROADCOND"].head()


# In[30]:


"""Encoding Junction type (10 = Mid-Block (not related to intersection), 
21= At intersection (intersection related), 11=Mid-Block (intersection related), 3= Driveway Junction, 
20=Intersection (not intersection related), 4=Ramp Junction)"""
df["JUNCTIONTYPE"].replace("Mid-Block (not related to intersection)",10, inplace=True)
df["JUNCTIONTYPE"].replace("Mid-Block (but intersection related)",11, inplace=True)
df["JUNCTIONTYPE"].replace("At Intersection (but not related to intersection)",20, inplace=True)
df["JUNCTIONTYPE"].replace("At Intersection (intersection related)",21, inplace=True)
df["JUNCTIONTYPE"].replace("Driveway Junction",3, inplace=True)
df["JUNCTIONTYPE"].replace("Ramp Junction",4, inplace=True)
df["JUNCTIONTYPE"].replace("Unknown",999,inplace=True) 


# In[31]:


df["JUNCTIONTYPE"].head()


# In[32]:


"""Encoding Collision type (0=Parked Car, 1=Angles, 2=Rear Ended, 3=Sideswipe, 
4=Left Turn, 5=Pedestrian, 6=Cycles, 7=Right Turn, 8=Head On"""
df["COLLISIONTYPE"].replace("Parked Car",0, inplace=True)
df["COLLISIONTYPE"].replace("Angles",1, inplace=True)
df["COLLISIONTYPE"].replace("Rear Ended",2, inplace=True)
df["COLLISIONTYPE"].replace("Sideswipe",3, inplace=True)
df["COLLISIONTYPE"].replace("Left Turn",4, inplace=True)
df["COLLISIONTYPE"].replace("Pedestrian",5, inplace=True)
df["COLLISIONTYPE"].replace("Cycles",6, inplace=True)
df["COLLISIONTYPE"].replace("Right Turn",7, inplace=True)
df["COLLISIONTYPE"].replace("Head On",8, inplace=True)
df["COLLISIONTYPE"].replace("Other",999,inplace=True)


# In[33]:


df["COLLISIONTYPE"].head()


# In[34]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[35]:


X=df[["LIGHTCOND","WEATHER","ROADCOND","JUNCTIONTYPE","COLLISIONTYPE"]].values
X[0:5]


# In[36]:


np.isnan(X)

np.where(np.isnan(X)) 

X=np.nan_to_num(X)


# In[37]:


y = df['SEVERITYCODE'].values
y[0:5]


# In[38]:


np.isnan(y)

np.where(np.isnan(y)) 

y=np.nan_to_num(y)


# In[39]:


X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]


# In[40]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print(('Train set:', X_train.shape,  y_train.shape))
print(('Test set:', X_test.shape,  y_test.shape))


# In[41]:


from sklearn.tree import DecisionTreeClassifier


# In[42]:


severityTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
severityTree


# In[43]:


severityTree.fit(X_train,y_train)


# In[44]:


predTree = severityTree.predict(X_test)
print((predTree [0:5]))
print((y_test [0:5]))


# In[45]:


from sklearn import metrics
import matplotlib.pyplot as plt
print(("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree)))


# In[46]:


from sklearn.neighbors import KNeighborsClassifier


# In[47]:


k = 5
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh


# In[48]:


yhat = neigh.predict(X_test)
yhat[0:5]


# In[49]:


print(("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train))))
print(("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat)))


# In[50]:


import pylab as pl
import scipy.optimize as opt
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[51]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR


# In[52]:


yhat = LR.predict(X_test)
yhat


# In[53]:


yhat_prob = LR.predict_proba(X_test)
yhat_prob


# In[54]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)


# In[55]:


from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)


# In[56]:


df.corr()


# In[ ]:




