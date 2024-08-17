#!/usr/bin/env python
# coding: utf-8

# ### Overview of the problem
# 
# #### What is the 311 service.
# 311 is the online service that provides the public with quick, easy access to all New York City government non-emergency services and information while offering the best customer service. Also, 311 service do an analysis of service delivery. 
# We can consider it as a link between usual people and government non-emergency services.
# The Department of Housing Preservation and Development of New York City is one of these services that is available under 311 and help to solve issues with housing and buildings.

# #### What is the issue with the Department of Housing Preservation and Development of New York City.
# The number of requests is growing tremendously every year. We need to help analyze the current situation and give the recommendations.

# #### Tasks that should be answered in this project.

# <div class="alert alert-block alert-warning">  
# 1) Analyze the current numbers of requests:<br>
# 1.1)identify top priority types of complaints on which Department of Housing Preservation and Development of New York City should concentrate first.<br>
# 1.2)identify the most problematic area(districts) with the highest amount of the top priority complaints(same for streets,ZIPs). Show it in on a map.<br>        
# 1.3)determine whether there is a relationship between the top problematic distrirct with the most common complaint and the buildings.<br>
# 2)Predict in advance the most problematic streets/zip in order to fix the largest number of problems for the smallest number of departures.<br>
# </div>
# 

# ### Import modules

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px
from IPython.display import display, HTML
import numpy as np


# how to install SMOTE
#!pip install imbalanced-learn
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE


# ###  Step: Data acquisition
# 

# We can take data from the opendata database. We can send the request with a help of the SODA Api.
# 
# What is SODA API?<br>
# SODA API = The Socrata Open Data API (SODA) provides programmatic access to this dataset including the ability to filter, query, and aggregate data. More info(https://dev.socrata.com/docs/queries/)

# 1)Read more about 311 database:  
# Database "311 Service Requests from 2010 to Present" - link (https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9)
# 
# 2)Get data from 311 database https://data.cityofnewyork.us/resource/erm2-nwe9.csv But we don't need all 26.6M of rows and all 41 columns. We will take 10000000rows and not all the columns since it is a pet project.
# Also, we don't need all agencies, only where "Department of Housing Preservation and Development of New York City"=HPD

# In[2]:


#%%time
##download data in csv format after 2018+HPD only+limit of rows
#NY311_df=pd.read_csv("https://data.cityofnewyork.us/resource/erm2-nwe9.csv?$limit=10000000&agency=HPD&$where=created_date>'2018-01-01'",low_memory=False)


# In[3]:


#to make it faster I did previous step localy
#download data in csv format after 2018+HPD only+limit of rows +put it localy
NY311_df=pd.read_csv('downloaded_NY311_df.csv')


# In[4]:


#grep only these columns
NY311_df=NY311_df[['created_date','unique_key','agency','complaint_type','descriptor','location_type','incident_zip', 'incident_address','street_name','address_type','city','resolution_description','borough','latitude','longitude','closed_date','location_type','status']]


# In[5]:


#Do I need to save db csv in local file? ->if yes -> uncommend
#NY311_df.to_csv('NY311_df',index=False)


# In[6]:


#example of rows
NY311_df.head(2)


# In[7]:


#number of rows, columns
NY311_df.shape


# In[8]:


#dataset size in mb
from sys import getsizeof
(getsizeof(NY311_df))/1024


# <div class="alert alert-block alert-info">
# <b>Useful variables from Data acquisition step:  </b> <br> 
# NY311_df  $~~~~~~~~~~~$//311 database 
# </div>
# 

# ### Step: EDA

# In[9]:


#columns data type
NY311_df.info()


# <b>On EDA step we can answer our 1.1 question:  </b>   
# <b>-identify top priority types of complaints on which Department of Housing Preservation and Development of New York City should concentrate first.</b>

# In[10]:


#rating of a priority of the complaints
NY311_df['complaint_type'].value_counts()


# In[11]:


#It was found, that "HEAT/HOT WATER" was replaced by "HEATING" after 2014. So our task is to rename value"HEAT/HOT WATER" to "HEATING"
NY311_df['complaint_type']=NY311_df['complaint_type'].replace(['HEAT/HOT WATER'],'HEATING')
NY311_df['complaint_type'].value_counts()


# In[12]:


display(HTML("<b>Answer for 1.1: Top priority complaint is the first row in this table. It's across all districts. </b>"))
display(HTML("<div class='alert alert-block alert-warning'>  <br> </div>"))
print((NY311_df['complaint_type'].value_counts().to_markdown()))
display(HTML("<div class='alert alert-block alert-warning'>  <br> </div>"))


# In[13]:


#visualization of rating of complaints
plt.figure(figsize=(55,5))
suborder=NY311_df['complaint_type'].value_counts().index
sns.countplot(x=NY311_df['complaint_type'],data=NY311_df.groupby('complaint_type').count(), order=suborder)


# In[14]:


#let's check the dynamic of complaints over the time

#change column's "created_date" datatype from object->datetime
NY311_df["created_date"]=pd.to_datetime(NY311_df["created_date"])

#create a column "year" from a column "created_date"
NY311_df['Year']=NY311_df["created_date"].apply(lambda t:t.year)

#dynamic of complaints
fig,ax=plt.subplots(figsize=(15,7))
NY311_df.groupby(['Year','complaint_type']).count()['unique_key'].unstack().plot(ax=ax)


# <b>On EDA step we can answer our 1.2 question too:  
# -identify the most problematic area(districts) with the highest amount of the top priority complaints. Show in on map.</b>

# In[15]:


#choose needed columns to answer the question
NY311_df_most_problematic_area=NY311_df[['complaint_type','borough','incident_zip','latitude','longitude','unique_key']]
NY311_df_most_problematic_area.head(2)


# In[16]:


#what boroughs have the biggest amount of all types of complaints
NY311_df_most_problematic_area.groupby(by=['borough']).count()['unique_key'].sort_values(ascending=False)


# In[17]:


#specific complaints across boroughs
pd.options.display.max_rows = 1000
NY311_df_most_problematic_area.groupby(by=['complaint_type','borough']).count()['unique_key']


# ![Screenshot](curcledendrogram2.png)  

# In[18]:


#our top problematic complaint_type(or the most common type of complaints) is the :
top_problematic_complaint_type=NY311_df['complaint_type'].value_counts().head(1).index[0]
top_problematic_complaint_type


# In[19]:


#So, the district that has most of the top problematic complaint_type is
display(HTML("<b>Answer for 1.2: The district that has highest number of the top problematic complaint_type is: </b>"))
display(HTML("<div class='alert alert-block alert-warning'> </div>"))
top_problematic_district=NY311_df[NY311_df['complaint_type']==(top_problematic_complaint_type)]['borough'].value_counts().head(1).index[0]
print(top_problematic_district)
display(HTML("<div class='alert alert-block alert-warning'>  </div>"))


# In[20]:


#dataframe with a top problematic distirct  with the top problematic problem
top_problematic_district_and_top_problematic_complaint_type=NY311_df[(NY311_df['complaint_type']==top_problematic_complaint_type)&(NY311_df['borough']==top_problematic_district)]
top_problematic_district_and_top_problematic_complaint_type.head(2)


# In[21]:


#create a map of the top problematic district  with a top problematic problem across all areas
fig = px.scatter_mapbox(data_frame=top_problematic_district_and_top_problematic_complaint_type, lat=top_problematic_district_and_top_problematic_complaint_type['latitude'], lon=top_problematic_district_and_top_problematic_complaint_type['longitude'], hover_name=top_problematic_district_and_top_problematic_complaint_type['complaint_type'], hover_data=top_problematic_district_and_top_problematic_complaint_type[['borough','unique_key']],
                        color_discrete_sequence=["fuchsia"], zoom=3, height=300)


fig.update_layout(
    mapbox_style="white-bg",
    mapbox_layers=[
        {
            "below": 'traces',
            "sourcetype": "raster",
            "sourceattribution": "United States Geological Survey",
            "source": [
                "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
            ]
        }
      ])
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[22]:


#top 10 steets for a top problematic distirct and top problematic complaint type
display(HTML("<b>Answer for 1.2: Top 10 steets for a top problematic distirct and top problematic complaint type are: </b>"))
display(HTML("<div class='alert alert-block alert-warning'> </div>"))
t=top_problematic_district_and_top_problematic_complaint_type.street_name.value_counts().head(10)
print(t)
display(HTML("<div class='alert alert-block alert-warning'>  </div>"))


# In[23]:


#top 10 ZIP for a top problematic distirct and top problematic complaint type
display(HTML("<b>Answer for 1.2: Top 10 ZIP for a top problematic distirct and top problematic complaint type are: </b>"))
display(HTML("<div class='alert alert-block alert-warning'> </div>"))
t=top_problematic_district_and_top_problematic_complaint_type.incident_zip.value_counts().head(10)
print(t)
display(HTML("<div class='alert alert-block alert-warning'>  </div>"))


# #### Generalized conclusion for 1.2:

# In[24]:


display(HTML("<div class='alert alert-block alert-warning'> </div>"))
print(("The top problematic complaint_type across all distircts is:" +'  '+top_problematic_complaint_type))
print(("The district that has most of the top problematic complaint_type is:" +'  '+top_problematic_district))
print(("The most problematic street in the most problematic district with with the top common complaint is:" +'  '+ top_problematic_district_and_top_problematic_complaint_type.street_name.value_counts().head(1).index[0]))
print(("The most problematic ZIP in the most problematic district with with the top common complaint is:" +'  '+ str(top_problematic_district_and_top_problematic_complaint_type.incident_zip.value_counts().head(1).index[0])))
display(HTML("<div class='alert alert-block alert-warning'> </div>"))


# ### Data Preprocessing

# #### The 1.3 questions is:  " determine whether there is a relationship between the top problematic distirct  with the most common complaint and the buildings."  
# From above, we now know the top complaint across all districts, so now it's time to find the relationships with houses.

# To do it, we need information about American houses. Info about hoses in NYC can be taken from PLUTO.

# Read more about PLUTO database:
# Database "Primary Land Use Tax Lot Output (PLUTO)" - link(https://data.cityofnewyork.us/City-Government/Primary-Land-Use-Tax-Lot-Output-PLUTO-/64uk-42ks)
# 
# 
# WHAT IS PLUTO?  
# The Primary Land Use Tax Lot Output (PLUTO) data file contains extensive land use and
# geographic data at the tax lot level in an ASCII comma-delimited file.
# The PLUTO tax lot data files contain over seventy data fields derived from data files maintained
# by the Department of City Planning (DCP), Department of Finance (DOF), Department of
# Citywide Administrative Services (DCAS), and Landmarks Preservation Commission (LPC).
# DCP has created additional fields based on data obtained from one or more of the major data
# sources. PLUTO data files contain three basic types of data:    
# ● Tax Lot Characteristics;    
# ● Building Characteristics;   and  
# ● Geographic/Political/Administrative Districts.  
# 
# 
# In a nutshell it is info about every house in NY.

# In[25]:


#Get data from Primary Land Use Tax Lot Output (PLUTO):
#NY_Pluto_df=pd.read_csv("https://data.cityofnewyork.us/resource/64uk-42ks.csv?$limit=10000000",low_memory=False)
#NY_Pluto_df.head(2)


# In[26]:


#I did the step above but localy, I downloaded NY_Pluto_db to local pc to make it faster
NY_Pluto_df=pd.read_csv('downloaded_NY_Pluto_df.csv')
NY_Pluto_df.head(2)


# In[27]:


#db size
NY_Pluto_df.shape


# In[28]:


#dataframe with a top problematic distirct  with the top problematic problem
top_problematic_district_and_top_problematic_complaint_type=NY311_df[(NY311_df['complaint_type']==top_problematic_complaint_type)&(NY311_df['borough']==top_problematic_district)]
top_problematic_district_and_top_problematic_complaint_type.head(2)


# In[29]:


#generate dataframe of number of complaints per street, zip for a top problematic distirct  with the top problematic problem
top_problematic_district_and_top_problematic_complaint_type_onlysteetandzip = top_problematic_district_and_top_problematic_complaint_type[['street_name','incident_zip','unique_key']]#let's take only these columns
top_problematic_district_and_top_problematic_complaint_type_onlysteetandzip = top_problematic_district_and_top_problematic_complaint_type_onlysteetandzip.groupby(by=['street_name','incident_zip']).count()['unique_key'].reset_index()#let's group by street and zip
top_problematic_district_and_top_problematic_complaint_type_onlysteetandzip.columns=['address','incident_zip','Number_of_top_complaints']#rename columns "street_name" -> address
top_problematic_district_and_top_problematic_complaint_type_onlysteetandzip.head(10)


# In[30]:


#"top_problematic_district_and_top_problematic_complaint_type_onlysteetandzip" dataframe size
top_problematic_district_and_top_problematic_complaint_type_onlysteetandzip.shape


# In[31]:


#merge pluto and top_problematic_district_and_top_problematic_complaint_type_onlysteetandzip
merged=pd.merge(top_problematic_district_and_top_problematic_complaint_type_onlysteetandzip, NY_Pluto_df, how='inner',on='address')


# In[32]:


#"merge" dataframe size
merged.shape


# In[33]:


#Do we have row where the column Number_of_top_complaints=Null?
merged[merged['Number_of_top_complaints']=='Null'] #no


# In[34]:


#Pearson correlation between "Number of top complaint type" for top problematic district and parameters of the house
display(HTML("<b>Answer for 1.3: Correlation between 'Number of top complaint type' and and parameters of the house  </b>"))
display(HTML("<b>FYI: info about columns description can also be found https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwi7vfvmgIzzAhWjl4sKHbWqCYgQFnoECBYQAQ&url=https%3A%2F%2Fdata.ny.gov%2Fapi%2Fviews%2Ff888-ni5f%2Ffiles%2F4a9e79bd-9c1e-4ca7-a266-0fa1b8aa0765%3Fdownload%3Dtrue%26filename%3Dmeta_mappluto.pdf&usg=AOvVaw0w65qTeAubtPiVXZrlgbpm  </b>"))

display(HTML("<div class='alert alert-block alert-warning'> </div>"))
plt.figure(figsize=(30,5))
merged.corr()['Number_of_top_complaints'].sort_values().plot(kind='bar')


# ### Prediction.

# #### Let's try to answer question 2:
# #### -Predict in advance the most problematic streets/zip in order to fix the largest number of problems for the smallest number of departures.<br>

# In[35]:


merged.head(3)


# In[36]:


#How to understand what street is a problematic street?b
#First, let's check how many complaints we have in total
merged['Number_of_top_complaints'].value_counts().sort_index(ascending=False)
#complains #number


# In[37]:


#Let's take the 40 place from the top according to the number of complaints and use it as a trigger for a "really problematic location".
Forty_most_popular_number_of_complaints=(merged['Number_of_top_complaints'].value_counts().sort_index(ascending=False)).index[40]
Forty_most_popular_number_of_complaints


# In[38]:


#how many rows we have with the number of coplaints per street/zip that is >= Forty_most_popular_number_of_complaints
(merged[merged['Number_of_top_complaints']>=Forty_most_popular_number_of_complaints]).shape[0]


# In[39]:


#they are
merged[merged['Number_of_top_complaints']>=Forty_most_popular_number_of_complaints]


# In[40]:


#ok, done. Let's now mark street/zip where number of top complaints >  Forty_most_popular_number_of_complaints
def binary_column_Number_of_top_complaints(Number_of_top_complaints_perzip):
    if Number_of_top_complaints_perzip < Forty_most_popular_number_of_complaints:
        return 0
    elif Number_of_top_complaints_perzip >=Forty_most_popular_number_of_complaints:
        return 1


# In[41]:


#it means in our case change "Number_of_top_complaints" to binary form (< or >= than 'Forty_most_popular_number_of_complaints' )
merged['Number_of_top_complaints']=merged['Number_of_top_complaints'].apply(binary_column_Number_of_top_complaints)
merged.head(5)


# In[42]:


#let's prepare the to the alrorithm

#drop some unnecessary columns
merged=merged.drop('address',axis=1)
merged=merged.drop('incident_zip',axis=1)
merged=merged.drop('firecomp',axis=1)
merged=merged.drop('sanitsub',axis=1)
merged=merged.drop('zonedist1',axis=1)
merged=merged.drop(['zonedist2','zonedist3','zonedist4','overlay1','overlay2','spdist1','spdist2','spdist3','ltdheight','splitzone','bldgclass','ownertype','ownername','comarea','resarea','officearea','retailarea','garagearea','strgearea','factryarea','otherarea','ext','histdist','landmark','condono','xcoord','ycoord','latitude','longitude','zonemap','zmcode','edesignum','appbbl','appdate','firm07_flag','pfirm15_flag','rpaddate','dcasdate','zoningdate','landmkdate','basempdate','masdate','polidate','edesigdate','geom','dcpedited','notes'],axis=1)
merged=merged.drop(['cb2010','sanborn','version'],axis=1)



#dummies for borought
borough_dummies=pd.get_dummies(merged['borough'],drop_first=True)
merged=pd.concat([merged.drop('borough',axis=1),borough_dummies],axis=1)


merged.head(2)


# #### Delete nulls from data

# In[43]:


#yellow color represents how many nulls do we have
sns.heatmap(merged.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[44]:


#print rows with >0 nulls inside
merged[merged.isnull().sum(axis=1)>0]


# In[45]:


#in this dataset it's ok to simply drop them
index_of_the_rows_with_nulls=(merged[merged.isnull().sum(axis=1)>0]).index
index_of_the_rows_with_nulls


# In[46]:


#drop rows with nulls from the dataset
for i in index_of_the_rows_with_nulls:
    print(('we are dropping row'+' '+ str(i)))
    merged.drop(i,axis=0,inplace=True)


# In[47]:


#check if we still have nulls
sns.heatmap(merged.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#no yellow color => no nulls


# <div class="alert alert-block alert-info">
#     <b>merged</b> //on this step this variable with data inside  is prepared for algorithms to play with
# </div>

# In[48]:


#let's start with KNN

#let's normolize
from sklearn.preprocessing import StandardScaler #import normolizer tool
scaler=StandardScaler()#scaler is now our normalizer 
scaler.fit(merged.drop('Number_of_top_complaints',axis=1))#calculate normalization parameters
scaled_features=scaler.transform(merged.drop('Number_of_top_complaints',axis=1))#implement normalization parameters
scaled_features


# In[49]:


scaled_features.shape


# In[50]:


merged.drop('Number_of_top_complaints',axis=1).shape


# In[51]:


#merged_feat  is now our new normolized dataframe
merged_feat=pd.DataFrame(scaled_features,columns=merged.drop('Number_of_top_complaints',axis=1).columns)
merged_feat.head()


# In[52]:


from sklearn.model_selection import train_test_split


# In[53]:


X=merged_feat
y=merged['Number_of_top_complaints']


# In[54]:


#data that is ready to put in KNN + splitted into train/test
X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.3, random_state=101)


# In[55]:


from sklearn.neighbors import  KNeighborsClassifier


# In[56]:


knn=KNeighborsClassifier(n_neighbors=1)


# In[57]:


knn.fit(X_train,y_train)


# In[58]:


pred=knn.predict(X_test)
pred


# In[59]:


from sklearn.metrics import classification_report, confusion_matrix


# In[60]:


print((confusion_matrix(y_test,pred)))
print((classification_report(y_test,pred)))


# In[61]:


#try to find the smallest error by trying n_neighbors from 1 to 40
error_rate=[]

for i in range(1,40):
    
    knn=KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[62]:


plt.figure(figsize=(10,6))
plt.plot(list(range(1,40)),error_rate, color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[63]:


#Answer for Qeestion 2: This is the exactitude of the model
display(HTML("<b>Answer for Question 2: This is the exactitude of the model </b>"))
display(HTML("<div class='alert alert-block alert-warning'> </div>"))



#use the best K(the smaller error rate -> the better)
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)

print((confusion_matrix(y_test,pred)))
print('\n')
print((classification_report(y_test,pred)))





display(HTML("<div class='alert alert-block alert-warning'>  </div>"))








# #### Let's summ results above:

# <div class="alert alert-block alert-warning">    
#     <b>#Answer for Qeestion 2: This is the exactitude of the model </b>    <br> 
# 1)The model works - it's a good <br>    
# 2)f1 score for 0 type is 96% - it's a good<br>   
# <br>   
# 3)f1 score for 1 type is 38% - it's bad <br>  
# 4)there are no changed if we run k from 1 to 40, it can be because KNN was implemented on many features
# </div>
# 

# In[ ]:





# In[ ]:





# #### Let's try to make f1 score for 1 type better:
# #### by balancing classes

# In[64]:


#class 1 is smaller
merged['Number_of_top_complaints'].value_counts()


# In[65]:


#let's try SMOTE oversampling
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='auto')
X_sm, y_sm = smote.fit_resample(X, y)


# In[66]:


#data that is ready to put in KNN + splitted into train/test
X_train, X_test, y_train, y_test =train_test_split(X_sm,y_sm, test_size=0.3, random_state=101)


# In[67]:


from sklearn.neighbors import  KNeighborsClassifier


# In[68]:


knn=KNeighborsClassifier(n_neighbors=1)


# In[69]:


knn.fit(X_train,y_train)


# In[70]:


pred=knn.predict(X_test)
pred


# In[71]:


from sklearn.metrics import classification_report, confusion_matrix


# In[72]:


print((confusion_matrix(y_test,pred)))
print((classification_report(y_test,pred)))


# In[73]:


#try to find the smallest error by trying n_neighbors from 1 to 40
error_rate=[]

for i in range(1,40):
    
    knn=KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[74]:


plt.figure(figsize=(10,6))
plt.plot(list(range(1,40)),error_rate, color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[75]:


#Answer for Qeestion 2 with SMOTE: This is the exactitude of the model
display(HTML("<b>Answer for Question 2: This is the exactitude of the model </b>"))
display(HTML("<div class='alert alert-block alert-warning'> </div>"))



#use the best K(the smaller error rate -> the better)
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)

print((confusion_matrix(y_test,pred)))
print('\n')
print((classification_report(y_test,pred)))





display(HTML("<div class='alert alert-block alert-warning'>  </div>"))




# #### Let's summ results above:

# <div class="alert alert-block alert-warning">    
#     <b>#Answer for Qeestion 2: This is the exactitude of the model </b>    <br> 
# 1)The model works - it's a good <br>    
# 2)f1 score for 0 type is 94% - it's a good<br>   
# <br>   
# 3)f1 score for 1 type is 94% - it's good <br>  
# 4)there are no changed if we run k from 1 to 40, it can be because KNN was implemented on many features
# </div>
# 

# In[ ]:





# In[ ]:





# In[ ]:





# ### Manual tests:

# In[81]:


#Let's check how our model works on random single piece of data
#Let's take 3 rows(with 0 and 1 class) from our previous data

merged.iloc[[636]] #example where Num_of_top_compl =0


# In[123]:


merged.iloc[[700]]  #example where Num_of_top_compl =1


# In[124]:


merged.iloc[[2019]]#example where Num_of_top_compl =1


# In[144]:


#z=new manually created df with with these 2 rows of 0 and 1 type
z=pd.DataFrame( (merged.iloc[6043].values, merged.iloc[636].values,merged.iloc[2019].values,merged.iloc[611].values,merged.iloc[111].values)     ,columns=['Number_of_top_complaints', 'block', 'lot', 'cd', 'ct2010',
       'schooldist', 'council', 'zipcode', 'policeprct', 'healtharea',
       'sanitboro', 'landuse', 'easements', 'lotarea', 'bldgarea',
       'areasource', 'numbldgs', 'numfloors', 'unitsres', 'unitstotal',
       'lotfront', 'lotdepth', 'bldgfront', 'bldgdepth', 'proxcode',
       'irrlotcode', 'lottype', 'bsmtcode', 'assessland', 'assesstot',
       'exempttot', 'yearbuilt', 'yearalter1', 'yearalter2', 'builtfar',
       'residfar', 'commfar', 'facilfar', 'borocode', 'bbl', 'tract2010',
       'taxmap', 'plutomapid', 'sanitdistrict', 'healthcenterdistrict', 'BX',
       'MN', 'QN', 'SI']
)
               
z


# In[145]:


#let's drop results and predict them
z.drop('Number_of_top_complaints',axis=1,inplace=True)
z


# In[146]:


#let's normolize
from sklearn.preprocessing import StandardScaler #import normolizer tool
scaler=StandardScaler()#scaler is now our normalizer 
scaler.fit(z)#calculate normalization parameters
scaled_features=scaler.transform(z)#implement normalization parameters
scaled_features


# In[147]:


z=pd.DataFrame( scaled_features             ,columns=['block', 'lot', 'cd', 'ct2010',
       'schooldist', 'council', 'zipcode', 'policeprct', 'healtharea',
       'sanitboro', 'landuse', 'easements', 'lotarea', 'bldgarea',
       'areasource', 'numbldgs', 'numfloors', 'unitsres', 'unitstotal',
       'lotfront', 'lotdepth', 'bldgfront', 'bldgdepth', 'proxcode',
       'irrlotcode', 'lottype', 'bsmtcode', 'assessland', 'assesstot',
       'exempttot', 'yearbuilt', 'yearalter1', 'yearalter2', 'builtfar',
       'residfar', 'commfar', 'facilfar', 'borocode', 'bbl', 'tract2010',
       'taxmap', 'plutomapid', 'sanitdistrict', 'healthcenterdistrict', 'BX',
       'MN', 'QN', 'SI']
)
               
z


# In[148]:


knn.predict(z)# same as should be, nice!


# In[ ]:





# In[ ]:




