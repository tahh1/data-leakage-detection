#!/usr/bin/env python
# coding: utf-8

# # Analysing car mpg data set using clustering and PCA

# In[1]:


# To enable plotting graphs in Jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Numerical libraries
import numpy as np   

from sklearn.model_selection import train_test_split

# Import Linear Regression machine learning library
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# to handle data in form of rows and columns 
import pandas as pd    

# importing ploting libraries
import matplotlib.pyplot as plt   

#importing seaborn for statistical plots
import seaborn as sns


# In[3]:


# reading the CSV file into pandas dataframe
mpg_df = pd.read_csv("../input/carmpg/car-mpg.csv")  


# In[4]:


# Check top few records to get a feel of the data structure
mpg_df.head()


# In[5]:


mpg_df.describe().transpose()     # horsepower is missing


# In[6]:


temp = pd.DataFrame(mpg_df.hp.str.isdigit()) 
temp[temp['hp'] == False]


# In[7]:


mpg_df = mpg_df.replace('?', np.nan)


# In[8]:


mpg_df.info()


# In[9]:


mpg_df['hp'] = mpg_df['hp'].astype('float64')


# In[10]:


numeric_cols = mpg_df.drop('car_name', axis=1)

# Copy the 'mpg' column alone into the y dataframe. This is the dependent variable
car_names = pd.DataFrame(mpg_df[['car_name']])


numeric_cols = numeric_cols.apply(lambda x: x.fillna(x.median()),axis=0)
mpg_df = numeric_cols.join(car_names)   # Recreating mpg_df by combining numerical columns with car names

mpg_df.info()


# ##  Step 4 Let us do a pair plot analysis to visually check number of likely clusters

# In[11]:


# This is done using scatter matrix function which creates a dashboard reflecting useful information about the dimensions
# The result can be stored as a .png file and opened in say, paint to get a larger view 

mpg_df_attr = mpg_df.iloc[:, 0:9]
mpg_df_attr['dispercyl'] = mpg_df_attr['disp'] / mpg_df_attr['cyl']
sns.pairplot(mpg_df_attr, diag_kind='kde')   # to plot density curve instead of histogram

#sns.pairplot(mpg_df_attr)  # to plot histogram, the default


# # Step 5 LINEAR MODEL BUILT ON ORIGINAL RAW DATA

# In[12]:


from scipy.stats import zscore

mpg_df_attr = mpg_df.loc[:, 'mpg':'origin']
mpg_df_attr_z = mpg_df_attr.apply(zscore)

mpg_df_attr_z.pop('origin')      # Remove "origin" and "yr" columns
mpg_df_attr_z.pop('yr')

array = mpg_df_attr_z.values
X = array[:,1:5] # select all rows and first 7 columns which are the attributes
y = array[:,0]   # select all rows and the 8th column which is the classification "Yes", "No" for diabeties
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)


# In[13]:


from sklearn import svm
clr = svm.SVR()  
clr.fit(X_train , y_train)


# In[14]:


y_pred = clr.predict(X_test)


# In[15]:


import seaborn as sns
import seaborn as sns
sns.set(style="darkgrid", color_codes=True)

            
#g = sns.jointplot("y_actuals", "y_predicted", data=tips, kind="reg",
#                  xlim=(0, 60), ylim=(0, 12), color="r", size=7)
            
with sns.axes_style("white"):
    sns.jointplot(x=y_test, y=y_pred, kind="reg", color="k");


# ##  ITERATION 2 
# 
# 
# 
# 

# In[16]:


# 1. Drop acc column based on the above visual analysis

mpg_df_attr_z.pop('acc')

array = mpg_df_attr_z.values
X = array[:,1:5] # select all rows and first 7 columns which are the attributes
y = array[:,0]   # select all rows and the 8th column which is the classification "Yes", "No" for diabeties
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
clr.fit(X_train , y_train)
y_pred = clr.predict(X_test)

            
          


# In[17]:


import seaborn as sns
import seaborn as sns
sns.set(style="darkgrid", color_codes=True)

            
#g = sns.jointplot("y_actuals", "y_predicted", data=tips, kind="reg",
#                  xlim=(0, 60), ylim=(0, 12), color="r", size=7)
            
with sns.axes_style("white"):
    sns.jointplot(x=y_test, y=y_pred, kind="reg", color="k");


# In[18]:


# Achieved 89% on the test data. The low score is due to the large spread and apparent mix of gaussians
# Let us explore the data for hidden clusters


# # Step 6 KMeans Clustering

# In[19]:


cluster_range = list(range( 2, 6))   # expect 3 to four clusters from the pair panel visual inspection hence restricting from 2 to 6
cluster_errors = []
for num_clusters in cluster_range:
  clusters = KMeans( num_clusters, n_init = 5)
  clusters.fit(mpg_df_attr)
  labels = clusters.labels_
  centroids = clusters.cluster_centers_
  cluster_errors.append( clusters.inertia_ )
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )
clusters_df[0:15]


# In[20]:


# Elbow plot

plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )


# In[21]:


# The elbow plot confirms our visual analysis that there are likely 3 or 4 good clusters
# Let us start with 3 clusters


# In[22]:


from sklearn.cluster import KMeans
from scipy.stats import zscore

mpg_df_attr = mpg_df.loc[:, 'mpg':'origin']
mpg_df_attr_z = mpg_df_attr.apply(zscore)

cluster = KMeans( n_clusters = 3, random_state = 2354 )
cluster.fit(mpg_df_attr_z)

prediction=cluster.predict(mpg_df_attr_z)
mpg_df_attr_z["GROUP"] = prediction     # Creating a new column "GROUP" which will hold the cluster id of each record

mpg_df_attr_z_copy = mpg_df_attr_z.copy(deep = True)  # Creating a mirror copy for later re-use instead of building repeatedly


# In[23]:


centroids = cluster.cluster_centers_
centroids


# In[24]:


centroid_df = pd.DataFrame(centroids, columns = list(mpg_df_attr) )
centroid_df


# In[25]:


## Instead of interpreting the neumerical values of the centroids, let us do a visual analysis by converting the 
## centroids and the data in the cluster into box plots.


# In[26]:


import matplotlib.pylab as plt

mpg_df_attr_z.boxplot(by = 'GROUP',  layout=(2,4), figsize=(15, 10))



# In[27]:


# There are many outliers on each dimension  (indicated by the black circles)
# Spread of data on each dimension (indivated by the whiskers is long ... due to the outliers)
# If the outliers are addressed, the clusters will overlap much less than right now (except in year dimension which has no outlier)


# ## Identifying and handling outliers
# 
# 

# In[28]:


# Addressing outliers at group level

data = mpg_df_attr_z   # lazy to type long names. Renaming it to data. Remember data is not a copy of the dataframe
       
def replace(group):
    median, std = group.median(), group.std()  #Get the median and the standard deviation of every group 
    outliers = (group - median).abs() > 2*std # Subtract median from every member of each group. Take absolute values > 2std
    group[outliers] = group.median()       
    return group

data_corrected = (data.groupby('GROUP').transform(replace)) 
concat_data = data_corrected.join(pd.DataFrame(mpg_df_attr_z['GROUP']))


# In[29]:


concat_data.boxplot(by = 'GROUP', layout=(2,4), figsize=(15, 10))


# ### Note: When we remove outliers and replace with median or mean, the distribution shape changes, the standard deviation becomes tighter creating new outliers. The new outliers would be much closer to the centre than original outliers so we accept them without modifying them
# 

# # Let us analyze the mpg column vs other columns group wise. 

# In[30]:


# mpg Vs hp

var = 'hp'

with sns.axes_style("white"):
    plot = sns.lmplot(var,'mpg',data=concat_data,hue='GROUP')
plot.set(ylim = (-3,3))


# In[31]:


var = 'disp'
with sns.axes_style("white"):
    plot = sns.lmplot(var,'mpg',data=concat_data,hue='GROUP')
plot.set(ylim = (-3,3))


# In[32]:


var = 'acc'
with sns.axes_style("white"):
    plot = sns.lmplot(var,'mpg',data=concat_data,hue='GROUP')
plot.set(ylim = (-3,3))


# In[33]:


var = 'wt'
with sns.axes_style("white"):
    plot = sns.lmplot(var,'mpg',data=concat_data,hue='GROUP')
plot.set(ylim = (-3,3))


# # STEP 7 - Break clusters into separate datasets and analyze individually.

# In[34]:


# Let us break the data into largecar and smallcar segments

largecar = concat_data[concat_data['GROUP']==0]
smallcar = concat_data[concat_data['GROUP']==1]
sedancar = concat_data[concat_data['GROUP']==2]


# In[35]:


# Let us look at largecar pair panel

mpg_df_attr = sedancar.iloc[:, 0:8]   # CHANGE THE CARTYPE AT THIS POINT TO CHECK HOW THE MODEL PERFORMS FOR EACH GROUP

sns.pairplot(mpg_df_attr, diag_kind='kde')   # to plot density curve instead of histogram

#sns.pairplot(mpg_df_attr)  # to plot histogram, the default


# In[36]:


mpg_df_attr.shape


# In[37]:


from sklearn import svm
clr = svm.SVR()  


array = mpg_df_attr.values
X = array[:,1:5] # select all rows and first 7 columns which are the attributes
y = array[:,0]   # select all rows and the 8th column which is the classification "Yes", "No" for diabeties
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
clr.fit(X_train , y_train)
y_pred = clr.predict(X_test)


# In[38]:


sns.set(style="darkgrid", color_codes=True)
       
with sns.axes_style("white"):
    sns.jointplot(x=y_test, y=y_pred, kind="reg", color="k");
    
# Discuss the accuracy score for each group in background of the visual distribution of data in the individual plots above


# # STEP 8 Alternative clustering strategy by origin. Since each record belongs to one origin, the cluster based on origin can be obtained using "group" 

# In[39]:


mpg_df_attr_z.boxplot(by = 'origin',  layout=(2,4), figsize=(15, 10))


# In[40]:


# Let us analyze origin wise. Do the origins form natural groups

# Looking at the countrywise spread, maybe subclusters i.e. within origin cluster sub clusters will give better results

var = 'acc'


with sns.axes_style("white"):
    plot = sns.lmplot( var,'mpg',data=mpg_df_attr,hue='origin')
plot.set(ylim = (0,50))


# In[41]:


# Observations -  

# 1. Origin based clustering is not clearly distinguishing the clusters on any dimension

# 2. Two groups of cars based on origin ( America and Europ + Asia) is unlikely to give better results as they overlap on
# all dimensions. 

# 3. Instead, grouping into large and small cars may give better clusters.


# # Step 9  Repeat Step 6 and 7 with K = 4

# In[42]:


from sklearn.cluster import KMeans
from scipy.stats import zscore

mpg_df_attr = mpg_df.loc[:, 'mpg':'origin']
mpg_df_attr_z = mpg_df_attr.apply(zscore)

cluster = KMeans( n_clusters = 4, random_state = 2354 )
cluster.fit(mpg_df_attr_z)

prediction=cluster.predict(mpg_df_attr_z)
mpg_df_attr_z["GROUP"] = prediction

mpg_df_attr_z_copy = mpg_df_attr_z.copy(deep = True)


# In[43]:


centroids = cluster.cluster_centers_
centroids


# In[44]:


centroid_df = pd.DataFrame(centroids, columns = list(mpg_df_attr) )
centroid_df


# In[45]:


# Addressing outliers at group level

data = mpg_df_attr_z   # lazy to type long names. Renaming it to data. Remember data is not a copy of the dataframe
       
def replace(group):
    median, std = group.median(), group.std()  #Get the median and the standard deviation of every group 
    outliers = (group - median).abs() > 2*std # Subtract median from every member of each group. Take absolute values > 2std
    group[outliers] = group.median()       
    return group

data_corrected = (data.groupby('GROUP').transform(replace)) 
concat_data = data_corrected.join(pd.DataFrame(mpg_df_attr_z['GROUP']))


# In[46]:


import matplotlib.pylab as plt

mpg_df_attr_z.boxplot(by = 'GROUP',  layout=(2,4), figsize=(15, 10))


# In[47]:


# mpg Vs hp

var = 'hp'

with sns.axes_style("white"):
    plot = sns.lmplot(var,'mpg',data=concat_data,hue='GROUP')
plot.set(ylim = (-3,3))


# In[48]:


var = 'disp'
with sns.axes_style("white"):
    plot = sns.lmplot(var,'mpg',data=concat_data,hue='GROUP')
plot.set(ylim = (-3,3))


# In[49]:


var = 'wt'
with sns.axes_style("white"):
    plot = sns.lmplot(var,'mpg',data=concat_data,hue='GROUP')
plot.set(ylim = (-3,3))


# In[50]:


# Let us break the data into largecar and smallcar segments

largecar = concat_data[concat_data['GROUP']==1]
smallcar = concat_data[concat_data['GROUP']==0]
sedancar = concat_data[concat_data['GROUP']==2]
minicar  = concat_data[concat_data['GROUP']==3]


# In[51]:


# Let us look at largecar pair panel

mpg_df_attr = sedancar.iloc[:, 0:8]

sns.pairplot(mpg_df_attr, diag_kind='kde')   # to plot density curve instead of histogram

#sns.pairplot(mpg_df_attr)  # to plot histogram, the default


# In[52]:


from sklearn import svm
clr = svm.SVR()  


array = mpg_df_attr.values
X = array[:,1:5] # select all rows and first 7 columns which are the attributes
y = array[:,0]   # select all rows and the 8th column which is the classification "Yes", "No" for diabeties
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
clr.fit(X_train , y_train)
y_pred = clr.predict(X_test)


# In[53]:


sns.set(style="darkgrid", color_codes=True)
       
with sns.axes_style("white"):
    sns.jointplot(x=y_test, y=y_pred, kind="reg", color="k");


# In[54]:


# Most of the attributes are now showing almost gaussian distribution 
# The interaction between dimensions too is relatively more linear
# But there is a lot of spread.


# # STEP 10 Kmeans clustering is not helping at all. Try PRINCIPAL COMPONENT ANALYSIS

# In[55]:


# PCA should be used when the relations are linear. Looking at the pairplot, "Cyl", "yr" and "Origin" 
# are likely to be ineffective. So let us remove them before doing the PCA.

# Apply PCA for each group of cars


# In[56]:


cols_to_drop = ["cyl", "origin", "GROUP" , "acc"]

car_attr = smallcar.drop(cols_to_drop , axis = 1)

car_mpg = np.array(car_attr.pop('mpg'))


# In[57]:


from sklearn.decomposition import PCA
# pca = PCA(4)
# largecar_projected = pca.fit_transform(largecar_attr)  Reason for avoiding this is the PCA will 
# automatically convert data to z scores which is already done. Hence doing the steps of PCA one by one

cov_matrix = np.cov(car_attr, rowvar=False)

np.linalg.eig(cov_matrix)

eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)

eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]

# Sort the (eigenvalue, eigenvector) pairs from lowest to highest with respect to eigenvalue
eig_pairs.sort()
eig_pairs.reverse()    # reverses the sorted pairs from increasing value of eigenvalue to lowest

# Extract the descending ordered eigenvalues and eigenvectors
eigvalues_sort = [eig_pairs[index][0] for index in range(len(eigenvalues))]
eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]


# In[58]:


tot = sum(eigenvalues)

var_explained = [(i / tot) for i in sorted(eigenvalues, reverse=True)]  # an array of variance explained by each 
# eigen vector... there will be 4 entries as there are 4 eigen vectors)

cum_var_exp = np.cumsum(var_explained)  # an array of cumulative variance. There will be 8 entries with 8 th entry 
# cumulative reaching almost 100%

cum_var_exp


# In[59]:


plt.bar(list(range(0, 4)), var_explained, alpha=0.5, align='center', label='individual explained variance')
plt.step(list(range(0,4)),cum_var_exp, where= 'mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc = 'best')
plt.show()


# In[60]:


# Transform the data into PC dimensions

car_mpg = car_mpg.reshape(len(car_mpg), 1)

eigen_space = np.array(eigvectors_sort[0:2]).transpose()

proj_data_3D = np.dot(car_attr, eigen_space)


# names = ['PC1', 'PC2' , 'PC3' , 'mpg']      Try the PCA based model with different number of PCs in the same group
names = ['PC1', 'pc2', 'mpg']

mpg_pca_array = np.concatenate((proj_data_3D, car_mpg), axis=1)

mpg_pca_df = pd.DataFrame(mpg_pca_array ,columns=names )


X = mpg_pca_array[:,0:1] # select only the PCAs
y = mpg_pca_array[:,1]   # select only the mpg column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)


# In[61]:


car_mpg.shape


# In[62]:


from sklearn import svm
clr = svm.SVR()  
clr.fit(X_train , y_train)


# In[63]:


y_pred = clr.predict(X_test)


# In[64]:


import seaborn as sns
import seaborn as sns
sns.set(style="darkgrid", color_codes=True)

            
#g = sns.jointplot("y_actuals", "y_predicted", data=tips, kind="reg",
#                  xlim=(0, 60), ylim=(0, 12), color="r", size=7)
            
with sns.axes_style("white"):
    sns.jointplot(x=y_test, y=y_pred, kind="reg", color="k");


# # STEP 11 Feature Engineering
# 
# 

# # Let us group by car type (large cars = 0 and others = 1)
# 

# In[65]:


# Create a separate column in the CSV file where cars with 8 and 6 cylinders are large cars (code 0) and 
# all other cars i.e. with other cylinder numbers are grouped into others (code = 1)


# In[66]:


mpg_df_attr = mpg_df.loc[:, 'mpg':'car_type']
mpg_df_attr_z = mpg_df_attr.apply(zscore)
mpg_df_attr_z.boxplot(by = 'car_type',  layout=(2,4), figsize=(15, 10))


# In[67]:


# Large cars (higher horsepower, higher displacement, higher wt) seem to have a better inverse relation with mileage than small 
# cars. Let us see if we can cluster on these dimensions


# In[68]:


# mpg Vs hp

var = 'hp'

with sns.axes_style("white"):
    plot = sns.lmplot(var,'mpg',data=mpg_df_attr_z,hue='car_type')
plot.set(ylim = (-3,3))


# In[69]:


# mpg Vs wt

var = 'wt'

with sns.axes_style("white"):
    plot = sns.lmplot(var,'mpg',data=mpg_df_attr_z,hue='car_type')
plot.set(ylim = (-3,3))


# In[70]:


# mpg Vs disp

var = 'disp'

with sns.axes_style("white"):
    plot = sns.lmplot(var,'mpg',data=mpg_df_attr_z,hue='car_type')
plot.set(ylim = (-3,3))


# In[71]:


# Let us break the data into largecar and smallcar segments

largecar = mpg_df_attr_z[mpg_df_attr_z['car_type'] < 0]   # note : largecar indicated by 0 has become negative zscore
othercar = mpg_df_attr_z[mpg_df_attr_z['car_type'] > 0]   #        othercar indicated by 1 has become positive value zscore


# In[72]:


# Let us look at largecar pair panel

mpg_df_attr = largecar.iloc[:, 0:9]

sns.pairplot(mpg_df_attr, diag_kind='kde')   # to plot density curve instead of histogram

#sns.pairplot(mpg_df_attr)  # to plot histogram, the default


# In[73]:


array = mpg_df_attr.values
X = array[:,2:5] # select all rows and desired columns 
y = array[:,0]   # select all rows and the 8th column which is the classification "Yes", "No" for diabeties
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)


# In[74]:


from sklearn import svm
clr = svm.SVR()  
clr.fit(X_train , y_train)


# In[75]:


y_pred = clr.predict(X_test)


# In[76]:


sns.set(style="darkgrid", color_codes=True)

            
#g = sns.jointplot("y_actuals", "y_predicted", data=tips, kind="reg",
#                  xlim=(0, 60), ylim=(0, 12), color="r", size=7)
            
with sns.axes_style("white"):
    sns.jointplot(x=y_test, y=y_pred, kind="reg", color="k");


# In[77]:


# Try PCA to improve performance


# In[78]:


cols_to_drop = ["cyl", "origin", "acc" , "yr", "car_type"]

car_attr = mpg_df_attr.drop(cols_to_drop , axis = 1)

car_mpg = np.array(car_attr.pop('mpg'))


# In[79]:


from sklearn.decomposition import PCA
# pca = PCA(4)
# largecar_projected = pca.fit_transform(largecar_attr)  Reason for avoiding this is the PCA will 
# automatically convert data to z scores which is already done. Hence doing the steps of PCA one by one

cov_matrix = np.cov(car_attr, rowvar=False)

np.linalg.eig(cov_matrix)

eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)

eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]

# Sort the (eigenvalue, eigenvector) pairs from lowest to highest with respect to eigenvalue
eig_pairs.sort()
eig_pairs.reverse()    # reverses the sorted pairs from increasing value of eigenvalue to lowest

# Extract the descending ordered eigenvalues and eigenvectors
eigvalues_sort = [eig_pairs[index][0] for index in range(len(eigenvalues))]
eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]


# In[80]:


tot = sum(eigenvalues)

var_explained = [(i / tot) for i in sorted(eigenvalues, reverse=True)]  # an array of variance explained by each 
# eigen vector... there will be 4 entries as there are 4 eigen vectors)

cum_var_exp = np.cumsum(var_explained)  # an array of cumulative variance. There will be 8 entries with 8 th entry 
# cumulative reaching almost 100%

cum_var_exp


# In[81]:


# Transform the data into PC dimensions

car_mpg = car_mpg.reshape(len(car_mpg), 1)

eigen_space = np.array(eigvectors_sort[0:1]).transpose()

proj_data_3D = np.dot(car_attr, eigen_space)


#names = ['PC1', 'PC2' , 'PC3' , 'mpg']

names = ['pc1', 'mpg']

mpg_pca_array = np.concatenate((proj_data_3D, car_mpg), axis=1)

mpg_pca_df = pd.DataFrame(mpg_pca_array ,columns=names )


X = mpg_pca_array[:,0:1] # select only the PCAs
y = mpg_pca_array[:,1]   # select only the mpg column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)



# In[82]:


from sklearn import svm
clr = svm.SVR()  
clr.fit(X_train , y_train)


# In[83]:


y_pred = clr.predict(X_test)

            
with sns.axes_style("white"):
    sns.jointplot(x=y_test, y=y_pred, kind="reg", color="k");


# # Observations
# 
# # 1. The mpg column for the different brand names are a suspect. Found values much larger than the factory values for those cars! Definition of mpg too may have to be looked at. 
# 
# # 2. The weight of the car too is a suspect as they differed from the specifications for those models. There are different types of weights. Was the data collected consistently
# 
# # 3. The HP column too had values different from the factory specifications. There are different types of HP values. Was the a standard definition followed
# 
# # Suggestions
# 
# # 1. For those instances where the declared mpg is greater than factory mpg, replace with factory mpg. Similarly for other columns. When this was doen the standard distribution for the mpg column fell by 50%
