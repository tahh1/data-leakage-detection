#!/usr/bin/env python
# coding: utf-8

# # Notebook 4: EDA, Feature Engineering, and Linear Regression

# In[161]:


import os
os.chdir('/Users/beth/Documents/Metis/metis_project_2/metis_project2')
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import math


# ## Exploring the data

# Import the dataframe scraped from Discogs.com and Spotify.com

# In[927]:


with open('final_record_1000_manual_update.pickle', 'rb') as read_file:
    df = pickle.load(read_file)


# In[928]:


df.info()


# There are some null values, but only 2 nulls for the target value category (Median price).
# Lowest price and highest price will be ignored.

# ### Converting to floats

# I need to convert a lot of these columns to numbers, and since NaN values can't be converted to integers, but can be convert to floats, I converted the values to floats

# In[929]:


df['release_year'] = df['release_year'].astype('float')
df['total_artist_albums'] = df['total_artist_albums'].astype('float')
df['artist_last_years'] = df['artist_last_years'].astype('float')
df['artist_first_years'] = df['artist_first_years'].astype('float')
df['number_of_songs'] = df['number_of_songs'].astype('float')
df['users_have'] = df['users_have'].astype('float')
df['users_want'] = df['users_want'].astype('float')
df['user_rating'] = df['user_rating'].astype('float')
df['versions'] = df['versions'].astype('float')
df['spotify_monthly_listeners'] = df['spotify_monthly_listeners'].astype('float')

df['median_price'] = df['median_price'].astype('str')
df['median_price'] = df['median_price'].str.replace(',','')
df['median_price'] = df['median_price'].astype('float')

df['number_for_sale'] = df['number_for_sale'].astype('str')
df['number_for_sale'] = df['number_for_sale'].str.replace(',','')
df['number_for_sale'] = df['number_for_sale'].astype('float')

df.drop('highest_price', axis=1, inplace=True)
df.drop('lowest_price', axis=1, inplace=True)


# ### Creating new variables based on year or time differences

# I changed some of the date and time columns to datetime objects, and I calculated some differences between times as new variables. For example, years_after_first_album represents the number of years the album of interest was released after the artist's first album was. I dropped columns that would be highly correlated with these new columns.

# In[930]:


#the time since last sold as a number of days
df['last_sold'] = pd.to_datetime(df['last_sold'])
df['time_since_sold'] = datetime.today()-df['last_sold']
df = df.drop('last_sold', axis=1)
df['time_since_sold'] = df['time_since_sold'].dt.days

#the number of years the album was released after the artist's first album was 
#(measure of how established the artist was)
df['years_after_first_album']=df['release_year']-df['artist_first_years']
df=df.drop('artist_first_years',axis=1)

#the number of years since any album containing that artist's material was produced
#(measure of the relevance of the artist today)
df['years_since_any_album']=2020-df['artist_last_years']
df=df.drop('artist_last_years', axis=1)

#the number of years it has been since the album was released
df['years_since_release']=2020-df['release_year']
df=df.drop('release_year', axis=1)

#the average song length in minutes as a decimal float number
df['average_song_length'] = pd.to_datetime(df['average_song_length'])
df['average_song_length'] = df['average_song_length'].dt.hour + df['average_song_length'].dt.minute/60


# I also dropped the link columns

# In[931]:


df.drop(['artist_links', 'album_links', 'first_release_links'], axis=1, inplace=True)


# In[932]:


df.info()


# ### Looking at the distribution of the target variable (median price)

# I looked at a histogram of my target variable 'median_price' and saw it was skewed toward 0, and had a few very high outliers. I thought that transforming my target variable would be a good idea. I tried both the inverse (1/x) and the log (log(x)) of my target, and in a correlation plot of the predictors and the target below, the log of 'median_price' had the highest linear correlations with predictors.

# In[169]:


sns.distplot(df['median_price'])


# In[170]:


plt.hist(df['median_price'],bins=200);
#the warning is due to np.nan values


# In[171]:


#inverse median price
plt.hist(1/df['median_price'],bins=200);


# In[172]:


#log median price
plt.hist(np.log(df['median_price']),bins=200);


# In[173]:


plt.figure(figsize=(10,5))
sns.distplot(np.log(df['median_price']),bins=50)
             
plt.xlabel('Log Median Price', fontsize=20)
plt.xticks(fontsize=20)
plt.ylabel('Percent of Dataset', fontsize=20)
plt.yticks(fontsize=20);

plt.tight_layout()
plt.savefig('Log Median Price Dist.tiff')


# In[933]:


df['log_median_price']=np.log(df['median_price'])
df['inv_median_price']=1/df['median_price']


# ### Plotting Pearson correlations between the predictors, target, and transformed targets

# I looked at the Pearson correlation between the continous variables in my dataframe with a heatmap.

# In[934]:


plt.rcParams['font.size'] = 20

plt.figure(figsize=(20,20))

sns.heatmap(df.corr(), cmap="bwr", annot=True, vmin=-1, vmax=1, fmt='.2f')

plt.gca().set_ylim(len(df.corr())+0.5, -0.5);  # quick fix to make sure viz isn't cut off


# From the above heatmap, it looks like my target median_price has a positive correlation with the number of versions (re-pressings) of the album made, as well as the user rating and the number of users that want the album. It has a negative correlation with the number for sale, the number of users that have the album, and the index (position in top 1000 collected albums list - not sure I want to use this variable in modeling). 
# 
# The inverse of median_price and log of median_price has higher linear correlations with the predictor variables. For inverse, the sign on the relationships is inverted. The log of the median price has the highest linear correlations with the predictor variables, so I am going to predict the log of the median value.
# 
# 
# The variable 'spotify_monthly_listeners' is correlated with number of songs, and total number of artist albums, which makes sense because the more material the artist has produced, the more Spotify listens there may be by chance. However, it has a very week correlation with the target varaible. 
# 
# The variable 'number_for_sale' seems the least correlated with any other variable.
# 
# The variable 'avg_song_length' has the most missing values, and fairly low correlations with anything other than 'number_of_songs' so I am going to drop it. I could have also filled in the missing values with the average, or a value calculated from number of songs.

# In[935]:


df.drop('average_song_length', axis=1, inplace=True)


# In[936]:


df.drop(['inv_median_price','median_price'], axis=1, inplace=True)


# Based on my intial pairplots I realized there were two very incorrect values of an album being released 20 years before the artist's first album, and over a million of a record for sale. I looked back on Discogs and corrected these valuse in the dataframe, I am not sure of the source of this error.

# In[937]:


df.loc[541,'number_for_sale'] = 220
df.loc[983,'years_after_first_album']=0


# ### Pairplots to look at linearity of relationships, and for interesting patterns

# This first pair plot using the continous variables that are related to things intrinsic to the album (not related to users)

# In[938]:


plt.rcParams['font.size'] = 15
sns.pairplot(data=df[['log_median_price','total_artist_albums','number_of_songs','versions','years_after_first_album','years_since_any_album','years_since_release']])


# I can see the log_median_price has a somewhat linear positive relationship with number of songs and number of versions of that album. I interpret this as albums with more material being more valuable, and albums that earn money being re-pressed serveral times by record companies.
# 
# It has a somewhat linear negative relationshp with years_after_first_album and years_since_any_album. I interpret this as artist's early albums being more expensive, and record companies continuing to put out albums using material from artists they know will add more value to the record. Maybe the relationship with years_after_first_album is quadratic - newer compeliation albums of popular artists could also be more expensive.
# 
# The log_median_price has an unusual relationship with 'years_since_release'. It seems that album prices drop off sharply after a certain age. This may be because at that point the albums are all used and more likely damaged. I explore this plot more below.

# In[178]:


#It seems like when an album is less than 30 years old it sells for a higher price
#It might be useful to convert this into a categorical varialble (>30yr or <30yr)
#I keep the original 'years_since_release' since there seems to be a slight positive linear relationship overall

plt.scatter(df['years_since_release'], df['log_median_price'])
plt.xticks(np.arange(0, 70, step=5))
plt.xlabel('years_since_release')
plt.ylabel('log_median_price');


# In[939]:


df['less_than_30_year_old']=df['years_since_release'] <= 30
df['more_than_30_year_old']=df['years_since_release'] > 30


# In[180]:


sns.lmplot('total_artist_albums','log_median_price',data=df)


# In[941]:


df['log_total_artist_albums']=np.log(df['total_artist_albums'])
sns.lmplot('log_total_artist_albums','log_median_price',data=df)


# In[942]:


df.drop(['log_total_artist_albums', 'total_artist_albums'],axis=1,inplace=True)


# Now I plotted several predicters related to how users interact with the albums and artists on Discogs and Spotify.

# In[146]:


sns.pairplot(data=df[['log_median_price','users_have','users_want','user_rating','number_for_sale','spotify_monthly_listeners','time_since_sold']])


# Here, 'log_median_price' has a positive relationship with users_want and user_rating. The relationship with users_want is not linear. 
# 
# 'log_median_price' also has a negative non-linear relationship with number_for_sale. 

# The variable 'time_since_sold' looked skewed toward 0, so I also tried log transforming it. Here a log transformation improved the linear relationship between it and 'log_median_price'.

# In[183]:


sns.lmplot('time_since_sold','log_median_price',data=df)


# In[943]:


df['log_time_since_sold']=np.log(df['time_since_sold'])
sns.lmplot('log_time_since_sold','log_median_price',data=df)


# In[944]:


df.drop('time_since_sold',axis=1,inplace=True)


# ### Important: 
# Variables closely linked to market value like, supply (number_for_sale, time_since_sold), demand (users_want, users_have) were really strong predictors, especially once interaction terms were generated. In the end I decided to pursue a model that didn't use these variables, because it would be more challenging and the results may be more interesting.

# ## Creating the categorical variable columns

# ### Genre dummies
# 
# First, I viewed the genres and styles columns to look for popular tags. Sometimes a tag appears in genre and sometimes it appears in styles, tags are also non-exclusive because an album can be a fusion of multiple styles. I found all rows that contained a tag of interest and created a column for that tag of 1's and 0's.

# In[945]:


#df['styles'].value_counts().head(20)


# In[946]:


df['Classic_Rock'] = df['styles'].apply(lambda x: 'Classic Rock' in x)
df['Classic_Rock_2'] = df['genres'].apply(lambda y: 'Classic Rock' in y)
df['Classic_Rock'] = df['Classic_Rock'] | df['Classic_Rock_2']
df.drop('Classic_Rock_2',axis=1,inplace=True)


# In[947]:


df['Pop_Rock'] = df['styles'].apply(lambda x: 'Pop Rock' in x)
df['Pop_Rock_2'] = df['genres'].apply(lambda y: 'Pop Rock' in y)
df['Pop_Rock'] = df['Pop_Rock'] | df['Pop_Rock_2']
df.drop('Pop_Rock_2',axis=1,inplace=True)


# In[948]:


df['Prog_Rock'] = df['styles'].apply(lambda x: 'Prog Rock' in x)
df['Prog_Rock_2'] = df['genres'].apply(lambda y: 'Prog Rock' in y)
df['Prog_Rock'] = df['Prog_Rock'] | df['Prog_Rock_2']
df.drop('Prog_Rock_2',axis=1,inplace=True)


# In[949]:


df['Heavy_Metal'] = df['styles'].apply(lambda x: 'Hard Rock' in x or 'Heavy Metal' in x)
df['Heavy_Metal_2'] = df['genres'].apply(lambda y: 'Hard Rock' in y or 'Heavy Metal' in y)
df['Heavy_Metal'] = df['Heavy_Metal_2'] | df['Heavy_Metal']
df.drop('Heavy_Metal_2',axis=1,inplace=True)


# In[950]:


df['Folk'] = df['styles'].apply(lambda x: 'Folk' in x or 'Folk Rock' in x)
df['Folk_2'] = df['genres'].apply(lambda y: 'Folk' in 'Fold Rock' in y)
df['Folk'] = df['Folk_2'] | df['Folk']
df.drop('Folk_2',axis=1,inplace=True)


# In[951]:


df['New_Wave'] = df['styles'].apply(lambda x: 'New Wave' in x or 'Synth-Pop' in x)
df['New_Wave_2'] = df['genres'].apply(lambda y: 'New Wave' in y or 'Synth-Pop' in y)
df['New_Wave'] = df['New_Wave'] | df['New_Wave_2']
df.drop('New_Wave_2',axis=1,inplace=True)


# In[952]:


df['Electronic'] = df['styles'].apply(lambda x: 'Electronic' in x)
df['Electronic_2'] = df['genres'].apply(lambda y: 'Electronic' in y)
df['Electronic'] = df['Electronic'] | df['Electronic_2']
df.drop('Electronic_2',axis=1,inplace=True)


# In[953]:


df['Hip_Hop'] = df['styles'].apply(lambda x: 'Hip Hop' in x)
df['Hip_Hop_2'] = df['genres'].apply(lambda y: 'Hip Hop' in y)
df['Hip_Hop'] = df['Hip_Hop'] | df['Hip_Hop_2']
df.drop('Hip_Hop_2',axis=1,inplace=True)


# In[954]:


df['Blues'] = df['styles'].apply(lambda x: 'Blues' in x or 'Blues Rock' in x)
df['Blues_2'] = df['genres'].apply(lambda y: 'Blues' in y or 'Blues Rock' in y)
df['Blues'] = df['Blues'] | df['Blues_2']
df.drop('Blues_2',axis=1,inplace=True)


# In[955]:


df['Jazz'] = df['styles'].apply(lambda x: 'Jazz' in x)
df['Jazz_2'] = df['genres'].apply(lambda y: 'Jazz' in y)
df['Jazz'] = df['Jazz'] | df['Jazz_2']
df.drop('Jazz_2',axis=1,inplace=True)


# In[956]:


df['Funk'] = df['styles'].apply(lambda x: 'Funk / Soul' in x)
df['Funk_2'] = df['genres'].apply(lambda y: 'Funk / Soul' in y)
df['Funk'] = df['Funk'] | df['Funk_2']
df.drop('Funk_2',axis=1,inplace=True)


# In[957]:


df['Soundtrack'] = df['styles'].apply(lambda x: 'Stage & Screen' in x or 'Soundtrack' in x)
df['Soundtrack_2'] = df['genres'].apply(lambda y: 'Stage & Screen' in y or 'Soundtrack' in y)
df['Soundtrack'] = df['Soundtrack'] | df['Soundtrack_2']
df.drop('Soundtrack_2',axis=1,inplace=True)


# In[958]:


df['Classic_Rock'] = df['Classic_Rock'].astype(int)
df['Pop_Rock'] = df['Pop_Rock'].astype(int)
df['Prog_Rock'] = df['Prog_Rock'].astype(int)
df['Heavy_Metal'] = df['Heavy_Metal'].astype(int)
df['Folk'] = df['Folk'].astype(int)
df['New_Wave'] = df['New_Wave'].astype(int)
df['Electronic'] = df['Electronic'].astype(int)
df['Hip_Hop'] = df['Hip_Hop'].astype(int)
df['Jazz'] = df['Jazz'].astype(int)
df['Funk'] = df['Funk'].astype(int)
df['Soundtrack'] = df['Soundtrack'].astype(int)


# In[960]:


plt.rcParams['font.size'] = 15
sns.pairplot(df[['log_median_price','Classic_Rock','Pop_Rock','Prog_Rock','Heavy_Metal','Folk','New_Wave','Electronic','Hip_Hop','Jazz','Funk','Soundtrack']])


# From the above plot I can see some differences in the overall proportions and price distributions of the different genres/styles

# There are the most Pop_Rock albums, they are usually older and cheaper.

# In[962]:


plt.figure(figsize=(10,5))
sns.scatterplot(df['years_since_release'],df['log_median_price'],alpha=0.8, hue=df['Pop_Rock'])

plt.xlabel('Years Since Release', fontsize=20)
plt.ylabel('Log Median Price', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=15);


# Jazz albums are amoung the oldest

# In[429]:


plt.figure(figsize=(10,5))
sns.scatterplot(df['years_since_release'],df['log_median_price'],alpha=0.8, hue=df['Jazz'])

plt.xlabel('Years Since Release', fontsize=20)
plt.ylabel('Log Median Price', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=15);

plt.savefig('Jazz Years.tiff')


# Columbia is an older record label

# In[431]:


plt.figure(figsize=(10,5))
sns.scatterplot(df['years_since_release'],df['log_median_price'],hue=df['Columbia'])


plt.xlabel('Years Since Release', fontsize=20)
plt.ylabel('Log Median Price', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=15);

plt.savefig('Columbia Years.tiff')


# ### Country dummies
# I created country of album release dummy variables, and selected a few more common ones to keep.

# In[963]:


country = pd.get_dummies(df['release_country'])


# In[204]:


country.head()


# In[205]:


df['release_country'].value_counts().head(10)


# In[964]:


country = country[['US','UK','Europe','UK & Europe','Canada','Australia','Germany']]


# I also created an artists dummy variable and kept the top 10 artists as variables

# In[965]:


artist = pd.get_dummies(df['artists'])


# In[208]:


df['artists'].value_counts().head(10)


# In[966]:


top_artists=list(df['artists'].value_counts().head(10).index)


# In[967]:


artist = artist[top_artists]


# In[968]:


artist


# ### Label Dummies
# I did the same for the record labels variable

# In[969]:


label = pd.get_dummies(df['label'])


# In[213]:


label.columns


# In[970]:


df['label'].value_counts().head(10)


# In[971]:


top_labels=list(df['label'].value_counts().head(10).index)
top_labels


# In[972]:


label = label[top_labels]


# In[973]:


pd.concat([df,label,country,artist],axis=1).shape


# In[974]:


df=pd.concat([df,label,country,artist],axis=1)


# Saving an updated version of the data frame

# In[223]:


import pickle
with open('feature_engineering_df_2.pickle', 'wb') as write_file:
    pickle.dump(df, write_file)


# In[604]:


import pickle
with open('feature_engineering_df_2.pickle', 'rb') as read_file:
    df=pickle.load(read_file)


# In[975]:


df.shape


# Drop some columns that were replaced by dummies, were not meant to be analyzed or I no longer want in the analysis

# In[976]:


df.drop(['top_artists_spotify_name','top_artists_orig_name'],axis=1, inplace=True)


# In[977]:


df.drop(['label','release_country','index','styles','genres'],axis=1, inplace=True)


# In[978]:


df.drop(['users_have', 'users_want', 'number_for_sale', 'log_time_since_sold'], axis=1, inplace=True)


# ## Train/test split

# In[224]:


from sklearn.model_selection import train_test_split


# In[225]:


df.dropna(how='any',axis=0,inplace=True)


# In[226]:


df.shape


# The train test split with an 80-20 split. This 20% of the data in the test split I will not use to inform my model in any way. I used stratify to stratify across the categorical variables with the most coverage of the dataset, the country to release. 

# In[227]:


X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'log_median_price'], df['log_median_price'], test_size=0.2, random_state=111, stratify=df[['US','UK','Europe','UK & Europe','Canada','Australia','Germany']])


# In[228]:


X_train.iloc[:,2:].head()


# ## Exploration of polynomial and interaction variables

# Working with training data, I explored polynomial and interaction variables since I noticed they may be present in data visualization above. I used a naive approach to identify variables that may have predictive power, by creating all possible varialbes and then creating a model with statsmodel OLS. I looked at the terms with coefficients that had low p-value and graphed them to find interesting ones.
# 
# I could have also done a Lasso regression and looked at the highest value coefficients from that regression, but I did it this way originally. I like the idea of using the p-value, because sometimes a coefficent can be large but it's estimate has high variance so it is might not be different from 0. Lasso might change those coefficients to 0, but I'm not sure. 

# In[229]:


from sklearn.preprocessing import PolynomialFeatures


# In[230]:


p = PolynomialFeatures()
X_train_poly = p.fit_transform(X_train.iloc[:,2:])


# Linear model with continuous features from statsmodel

# In[231]:


import statsmodels.api as sm


# In[232]:


import numpy as np


# In[233]:


Y_sm = np.array(y_train)
X_sm = X_train_poly


# In[234]:


model = sm.OLS(Y_sm, sm.add_constant(X_sm)) 


# In[235]:


fit = model.fit()


# In[139]:


#fit.summary()


# In[236]:


X_train_poly.shape


# In[237]:


poly_features_names = p.get_feature_names(X_train.iloc[:,2:].columns)


# In[238]:


highly_sig_poly_features = [i for i in range(len(fit.pvalues)) if fit.pvalues[i] < 0.01]


# In[239]:


[poly_features_names[i] for i in highly_sig_poly_features]


# Some squared terms were suggested from this model:

# In[144]:


X_train["user_rating**2"]= X_train["user_rating"]**2
X_train['years_since_release**2'] = X_train['years_since_release']**2
X_test['user_rating**2']=X_test['user_rating']**2
X_test['years_since_release**2'] = X_test['years_since_release']**2


# In[147]:


plt.rcParams['font.size'] = 10
fig, axes = plt.subplots(1,2)
sns.scatterplot(ax=axes[0], x=X_train['user_rating'],y=y_train)
sns.scatterplot(ax= axes[1], x=X_train['user_rating**2'],y=y_train)


# In[148]:


fig, axes = plt.subplots(1,2)
sns.scatterplot(ax=axes[0], x=X_train['years_since_release'],y=y_train)
sns.scatterplot(ax= axes[1], x=X_train['years_since_release**2'],y=y_train)


# ### Here I plotted some interaction terms and added to my data frame. Some were from the OLS coefficients with low p-values and some were from guessing by domain knowledge.

# In[426]:


plt.figure(figsize=(7,5))
sns.scatterplot(X_train['user_rating'],y_train, hue=df['Heavy_Metal'], alpha=0.7)

plt.xlabel('User Rating', fontsize=20)
plt.ylabel('Log Median Price', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=15);

plt.savefig('Heavy Metal Rating.tiff')


# In[304]:


X_train['user_rating*HeavyMetal']=X_train['user_rating']*X_train['Heavy_Metal']
X_test['user_rating*HeavyMetal']=X_test['user_rating']*X_test['Heavy_Metal']


# In[255]:


sns.scatterplot(X_train['user_rating'],y_train, hue=df['Classic_Rock'], alpha=0.5)


# In[305]:


X_train['user_rating*ClassicRock']=X_train['user_rating']*X_train['Classic_Rock']
X_test['user_rating*ClassicRock']=X_test['user_rating']*X_test['Classic_Rock']


# In[240]:


sns.scatterplot(X_train['years_after_first_album'],y_train, hue=df['Jazz'])


# In[306]:


X_train['years_after_first*Jazz']=X_train['years_after_first_album']*X_train['Jazz']
X_test['years_after_first*Jazz']=X_test['years_after_first_album']*X_test['Jazz']


# In[241]:


sns.scatterplot(X_train['years_after_first_album'],y_train, hue=df['Funk'])


# In[307]:


X_train['years_after_first*Funk']=X_train['years_after_first_album']*X_train['Funk']
X_test['years_after_first*Funk']=X_test['years_after_first_album']*X_test['Funk']


# In[244]:


sns.scatterplot(X_train['spotify_monthly_listeners'],y_train, hue=df['versions'])


# In[310]:


X_train['spotify*versions']=X_train['spotify_monthly_listeners']*X_train['versions']
X_test['spotify*versions']=X_test['spotify_monthly_listeners']*X_test['versions']


# In[437]:


import matplotlib.ticker as tick


# In[922]:


plt.figure(figsize=(12,5))
sns.scatterplot(X_train_orig['spotify_monthly_listeners'],Y_train_orig, alpha=0.9, hue=X_train_orig['less_than_30_year_old'])

plt.xlabel('Spotify Monthly Listeners', fontsize=25)
plt.ylabel('Log Median Price', fontsize=25)
#plt.xticks(fontsize=20, ticks=[0, 5000000, 10000000, 15000000, 20000000,  25000000, 30000000], labels=['0','5M','10M','15M','20M','25M','30M'])
plt.yticks(fontsize=20)
plt.legend(fontsize=15);
plt.tight_layout()

#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#plt.savefig('Spotify Age.tiff')



# In[312]:


X_train['spotify*30year']=X_train['spotify_monthly_listeners']*X_train['more_than_30_year_old']
X_test['spotify*30year']=X_test['spotify_monthly_listeners']*X_test['more_than_30_year_old']


# In[511]:


sns.scatterplot(X_train['years_since_release'],y_train, hue=df['versions'])


# In[313]:


X_train['years_since_release*versions']=X_train['years_since_release']*X_train['versions']
X_test['years_since_release*versions']=X_test['years_since_release']*X_test['versions']


# In[267]:


sns.barplot(X_train['Classic_Rock'],y_train, hue=df['Vertigo'])


# In[314]:


X_train['Classic_Rock*Vertigo'] = X_train['Classic_Rock']*X_train['Vertigo']
X_test['Classic_Rock*Vertigo'] = X_test['Classic_Rock']*X_test['Vertigo']


# In[315]:


sns.barplot(X_train['Pop_Rock'],y_train, hue=df['A&M Records'])


# In[316]:


X_train['Pop_Rock*A&M'] = X_train['Pop_Rock']*X_train['A&M Records']
X_test['Pop_Rock*A&M'] = X_test['Pop_Rock']*X_test['A&M Records']


# In[317]:


sns.barplot(X_train['more_than_30_year_old'],y_train, hue=df['Epic'])


# In[318]:


X_train['more_than_30*Epic']=X_train['more_than_30_year_old']*X_train['Epic']
X_test['more_than_30*Epic']=X_test['more_than_30_year_old']*X_test['Epic']


# In[319]:


sns.barplot(X_train['Prog_Rock'],y_train, hue=df['Jazz'])


# In[320]:


X_train['Prog_Rock*Jazz']=X_train['Prog_Rock']*X_train['Jazz']
X_test['Prog_Rock*Jazz']=X_test['Prog_Rock']*X_test['Jazz']


# In[322]:


sns.barplot(X_train['Heavy_Metal'],y_train, hue=df['Vertigo'])


# In[323]:


X_train['Heavy_Metal*Vertigo']=X_train["Heavy_Metal"]*X_train["Vertigo"]
X_test['Heavy_Metal*Vertigo']=X_test["Heavy_Metal"]*X_test["Vertigo"]


# In[324]:


sns.barplot(X_train['Folk'],y_train, hue=df['Island Records'])


# In[325]:


X_train['Folk*Island_Records']=X_train['Folk']*X_train['Island Records']
X_test['Folk*Island_Records']=X_test['Folk']*X_test['Island Records']


# In[326]:


sns.barplot(X_train['Folk'],y_train, hue=df['UK'])


# In[327]:


X_train['Folk*UK']=X_train['Folk']*X_train["UK"]
X_test['Folk*UK']=X_test['Folk']*X_test["UK"]


# In[328]:


sns.barplot(X_train['Electronic'],y_train, hue=df['Hip_Hop'])


# In[329]:


X_train['Hip_Hop*Electronic']=X_train["Hip_Hop"]*X_train["Electronic"]
X_test['Hip_Hop*Electronic']=X_test["Hip_Hop"]*X_test["Electronic"]


# In[330]:


sns.barplot(X_train['Funk'],y_train, hue=df['Columbia'])


# In[331]:


X_train['Funk*Columbia']=X_train['Funk']*X_train["Columbia"]
X_test['Funk*Columbia']=X_test['Funk']*X_test["Columbia"]


# In[332]:


sns.barplot(X_train['Epic'],y_train, hue=df['US'])


# In[333]:


X_train['Epic*US'] = X_train['Epic'] * X_train['US']
X_test['Epic*US'] = X_test['Epic'] * X_test['US']


# In[533]:


sns.barplot(X_train_orig['Pink Floyd'],Y_train_orig, hue=df['UK'],palette='muted')
       
plt.xlabel('Pink Floyd', fontsize=15)
plt.ylabel('Log Median Price', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('PinkFloydUK.tiff')


# In[335]:


X_train['PinkFloyd*UK']=X_train["Pink Floyd"]*X_train["UK"]
X_test['PinkFloyd*UK']=X_test["Pink Floyd"]*X_test["UK"]


# In[336]:


sns.barplot(X_train['The Rolling Stones'],y_train, hue=df['US'])


# In[337]:


X_train['RollingStones*US']=X_train["The Rolling Stones"] * X_train['US']
X_test['RollingStones*US']=X_test["The Rolling Stones"] * X_test['US']


# In[339]:


sns.barplot(X_train['New_Wave'],y_train, hue=df['Pop_Rock'])


# In[345]:


X_train['New_Wave*Pop']=X_train['New_Wave']*X_train['Pop_Rock']
X_test['New_Wave*Pop']=X_test['New_Wave']*X_test['Pop_Rock']


# In[346]:


sns.barplot(X_train['Soundtrack'],y_train, hue=df['Classic_Rock'])


# In[347]:


X_train['Soundrack*ClassicRock']=X_train["Soundtrack"]*X_train["Classic_Rock"]
X_test['Soundtrack*ClassicRock']=X_test["Soundtrack"]*X_test["Classic_Rock"]


# In[352]:


sns.barplot(X_train['more_than_30_year_old'],y_train, hue=df['Heavy_Metal'])


# In[353]:


X_train['more_than_30*HeavyMetal']=X_train['more_than_30_year_old']*X_train["Heavy_Metal"]
X_test['more_than_30*HeavyMetal']=X_test['more_than_30_year_old']*X_test["Heavy_Metal"]


# In[361]:


sns.barplot(X_train['Electronic'],y_train, hue=df['Elektra'])


# In[362]:


X_train['Electronic*Elektra']=X_train['Electronic']*X_train["Elektra"]
X_test['Electronic*Elektra']=X_test['Electronic']*X_test["Elektra"]


# In[371]:


sns.barplot(X_train['Hip_Hop'],y_train, hue=df['US'])


# In[372]:


X_train['Hip_Hop*US']=X_train['Hip_Hop']*X_train["US"]
X_test['Hip_Hop*US']=X_test['Hip_Hop']*X_test["US"]


# In[373]:


sns.barplot(X_train['Jazz'],y_train, hue=df['US'])


# In[374]:


X_train['Jazz*US']=X_train['Jazz']*X_train["US"]
X_test['Jazz*US']=X_test['Jazz']*X_test["US"]


# In[530]:


sns.barplot(X_train_orig['Blues'],Y_train_orig, hue=df['US'], palette="muted")
           
plt.xlabel('Blues', fontsize=15)
plt.ylabel('Log Median Price', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('BluesUS.tiff')


# In[376]:


X_train['Blues*US']=X_train['Blues']*X_train["US"]
X_test['Blues*US']=X_test['Blues']*X_test["US"]


# 
# 
# Excluding some of the others because they apply only to a small number of samples

# In[381]:


X_train.columns


# In[382]:


X_train.iloc[:,2:].shape


# In[383]:


X_train=X_train.reindex()
y_train=y_train.reindex()


# ### First linear model in sklearn, testing score by cross-validation

# In[384]:


from sklearn.model_selection import KFold


# In[385]:


from sklearn.linear_model import LinearRegression


# In[386]:


kf = KFold(n_splits=10, shuffle=True)

cv_lm_r2s = [] 
cv_lm_MSEs = []

for train_ind, val_ind in kf.split(X_train,y_train):
    #print(train_ind)
    X_train_cv, y_train_cv = X_train.iloc[train_ind], y_train.iloc[train_ind]
    X_val, y_val = X_train.iloc[val_ind], y_train.iloc[val_ind]
    #print(X_train_cv.shape)
    #print(X_val.shape)
    #print(y_train_cv.shape)
    #print(y_val.shape)
    #simple linear regression
    lm = LinearRegression()
    
    #y_train_cv=np.array(y_train_cv).reshape(-1,1)
    #y_val=np.array(y_val).reshape(-1,1)

    lm.fit(X_train_cv.iloc[:,2:], y_train_cv)
    cv_lm_r2s.append((lm.score(X_train_cv.iloc[:,2:], y_train_cv),lm.score(X_val.iloc[:,2:], y_val)))
    cv_lm_MSEs.append((np.sum((lm.predict(X_train_cv.iloc[:,2:]) - y_train_cv)**2) / len(y_train_cv),np.sum((lm.predict(X_val.iloc[:,2:]) - y_val)**2) / len(y_val)))


# In[387]:


cv_lm_r2s


# In[388]:


cv_lm_MSEs


# The average mean square error acros the 10 folds was 0.5033 for the linear regression. This is in terms of log dollars.

# In[496]:


np.mean([cv_lm_MSEs[i][1] for i in range(10)])


# The first column of training set R^2 is higher than the test set R^2. It may be overfit, and I should use Lasso or Ridge to increase fitting on test sets

# I checked the shape of the cross validatiation splits:

# In[390]:


y_val.shape


# In[391]:


X_val.shape


# In[392]:


y_train_cv.shape


# In[393]:


X_train_cv.shape


# I made a copy the training and test sets before scaling them for regularized regression

# In[452]:


X_train_orig=X_train.copy()
Y_train_orig=y_train.copy()
X_test_orig=X_test.copy()
Y_test_orgi=y_test.copy()


# ### Scaling 

# While variables do not need to be scaled for stanard linear regression, the penalities from Lasso and Ridge regression will not be able to be applied evenly on non-scaled variables. For Lasso and Ridge to perform optimally, I am going to scale my variables with StandardScaler (Z-score transformation).

# In[453]:


from sklearn.preprocessing import StandardScaler


# I need to scale the categorical variables as well, so the variables with more 1's don't have more weight in the penalty score from Lasso or Ridge. 

# In[454]:


scaler = StandardScaler()
X_train.iloc[:,2:] = scaler.fit_transform(X_train.iloc[:,2:])
X_test.iloc[:,2:] = scaler.fit_transform(X_test.iloc[:,2:])


# In[455]:


X_train.describe()


# ### Lasso and Ridge with cross validation for identifying optimal alpha

# In[456]:


from sklearn.linear_model import LassoCV, RidgeCV, Lasso, Ridge


# LassoCV with 5 folds, to identify best alpha

# In[540]:


lm = LassoCV(cv=10, n_alphas=1000, max_iter=5000)

lm.fit(X_train.iloc[:,2:],y_train)


# The training set R^2 is 0.6362 overall, and the MSE is 0.3957, but I should look at the score in cross-fold validation

# In[541]:


lm.score(X_train.iloc[:,2:], y_train)


# In[460]:


np.sum((lm.predict(X_train.iloc[:,2:]) - y_train)**2) / len(y_train)


# In[462]:


lasso_coef= lm.coef_


# In[463]:


#The best alpha from LassoCV was 0.00378
lasso_alpha=lm.alpha_
lasso_alpha


# In[464]:


kf = KFold(n_splits=10, shuffle=True, random_state = 71)

cv_lasso_r2s = [] 
cv_lasso_MSEs = []

for train_ind, val_ind in kf.split(X_train,y_train):
    #print(train_ind)
    X_train_cv, y_train_cv = X_train.iloc[train_ind], y_train.iloc[train_ind]
    X_val, y_val = X_train.iloc[val_ind], y_train.iloc[val_ind]
    #print(X_train_cv.shape)
    #print(X_val.shape)
    #print(y_train_cv.shape)
    #print(y_val.shape)
    #simple linear regression
    lm = Lasso(alpha=lasso_alpha, max_iter=5000)
    
    #y_train_cv=np.array(y_train_cv).reshape(-1,1)
    #y_val=np.array(y_val).reshape(-1,1)

    lm.fit(X_train_cv.iloc[:,2:], y_train_cv)
    
    predictions=lm.predict(X_train_cv.iloc[:,2:])
    train_MSE=np.sum((predictions - y_train_cv)**2)/len(y_train_cv)
    
    val_predictions=lm.predict(X_val.iloc[:,2:])
    validation_MSE = np.sum((val_predictions - y_val)**2)/len(y_val)
    
    cv_lasso_r2s.append((lm.score(X_train_cv.iloc[:,2:], y_train_cv),lm.score(X_val.iloc[:,2:], y_val)))
    cv_lasso_MSEs.append((train_MSE, validation_MSE))


# Using Lasso, the R^2 values in the validation sets were closer to the training sets. The average MSE in the validation sets was 0.4926

# In[465]:


cv_lasso_r2s


# In[466]:


cv_lasso_MSEs


# In[467]:


np.sqrt(np.array(cv_lasso_MSEs))


# In[497]:


np.mean([cv_lasso_MSEs[i][1] for i in range(10)])


# RidgeCV with 5 folds, to identify the best alpha

# In[474]:


lm = RidgeCV(cv=10, alphas=(np.arange(0,2,0.001)))

lm.fit(X_train.iloc[:,2:],y_train)


# In[475]:


lm.score(X_train.iloc[:,2:], y_train)


# In[477]:


np.sum((lm.predict(X_train.iloc[:,2:]) - y_train)**2) / len(y_train)


# In[479]:


ridge_alpha=lm.alpha_


# In[480]:


ridge_coef = lm.coef_


# In[483]:


kf = KFold(n_splits=10, shuffle=True, random_state = 71)

cv_ridge_r2s = [] 
cv_ridge_MSEs = []

for train_ind, val_ind in kf.split(X_train,y_train):
    #print(train_ind)
    X_train_cv, y_train_cv = X_train.iloc[train_ind], y_train.iloc[train_ind]
    X_val, y_val = X_train.iloc[val_ind], y_train.iloc[val_ind]
    #print(X_train_cv.shape)
    #print(X_val.shape)
    #print(y_train_cv.shape)
    #print(y_val.shape)
    #simple linear regression
    lm = Ridge(alpha=ridge_alpha)
    
    #y_train_cv=np.array(y_train_cv).reshape(-1,1)
    #y_val=np.array(y_val).reshape(-1,1)

    lm.fit(X_train_cv.iloc[:,2:], y_train_cv)
    cv_ridge_r2s.append((lm.score(X_train_cv.iloc[:,2:], y_train_cv),lm.score(X_val.iloc[:,2:], y_val)))
    cv_ridge_MSEs.append((np.sum((lm.predict(X_train_cv.iloc[:,2:]) - y_train_cv)**2) / len(y_train_cv),np.sum((lm.predict(X_val.iloc[:,2:]) - y_val)**2) / len(y_val)))


# The R^2 from Ridge regression were not as close in the validation sets to the training sets as in Lasso regression

# In[484]:


cv_ridge_r2s


# In[485]:


cv_ridge_MSEs


# In[486]:


np.sqrt(np.array(cv_ridge_MSEs))


# I compared the average MSE in the 10 cross-fold validation sets to choose LinearRegression, Lasso or Ridge for the final model. Lasso had the lowest.

# In[498]:


np.mean([cv_ridge_MSEs[i][1] for i in range(10)])


# In[499]:


np.mean([cv_lm_MSEs[i][1] for i in range(10)])


# In[500]:


np.mean([cv_lasso_MSEs[i][1] for i in range(10)])


# ### Final Lasso model

# In[ ]:


lm = LassoCV(cv=10, n_alphas=1000, max_iter=5000)

lm.fit(X_train.iloc[:,2:],y_train)


# In[501]:


lasso_coef


# I identified the varaibles with the higest coefficients in the Lasso model. These are coefficients for the scaled variables, so they can be compared directly.

# In[506]:


pd.DataFrame({'column':X_train.iloc[:,2:].columns,'coef':lasso_coef}).sort_values('coef')


# I calculated the predictions and residuals from the model on the holdout test set

# In[728]:


predictions = lm.predict(X_test.iloc[:,2:])


# In[ ]:


residuals = lm.predict(X_test.iloc[:,2:]) - y_test


# Since the target variable was log transformed, these have to be exponentiated (np.log is as a default natural log)

# In[721]:


e=math.e


# In[729]:


predictions=e**predictions


# In[724]:


actual=e**y_test


# In[730]:


residual_df = pd.DataFrame({'artist':X_test['artists'],'album':X_test['albums'],'price':actual,'prediction':predictions})


# In[1008]:


residual_df.head()


# I calculated the absolute mean error between predicted price and actual price in the test set as an easily-understandable evaulation of the model. This is the average error in dollars. It is still a bit high.

# In[732]:


np.sum(np.abs(residual_df['price'] - residual_df['prediction']))/len(residual_df)


# I created some random samplings of the test set to use to compare my model to the subject-matter expert's guesses

# In[548]:


import random


# In[979]:


comparison_dataset=residual_df.loc[random.sample(list(residual_df.index),50)]


# Calculated residuals and predictions on the training dataset

# In[982]:


residuals_train = lm.predict(X_train.iloc[:,2:]) - y_train


# In[983]:


y_train_predict = lm.predict(X_train.iloc[:,2:])


# In[987]:


y_train_price = e**y_train


# In[988]:


y_train_predict_price = e**y_train_predict


# To get residuals in dollar values, rather than log dollar values, it was easiest to use the exponentiated prices

# In[990]:


train_rediuals_df=pd.DataFrame({'y_train_price':y_train_price,'y_train_predict_price':y_train_predict_price,'residual':residuals_train})


# In[994]:


train_residuals_df['y_train_predict_price'] = e**y_train_predict
train_residuals_df['y_train_price'] = e**y_train


# In[997]:


train_residuals_df['actual_residual'] = train_residuals_df['y_train_predict_price'] - train_residuals_df['y_train_price'] 


# In[1009]:


train_residuals_df.head()


# The absolute mean error on the training set was 18.9, a bit higher than on the test set, which is expected for a Lasso model

# In[716]:


np.sum(np.abs(train_residuals_df['y_train_price']- \
              train_residuals_df['y_train_predict_price']))/786


# In[1000]:


#The residuals in dollar values don't look as evenly distributed as in log dollar values (shown later)
#This is because the model was built with log dollars

sns.scatterplot(x=train_residuals_df['y_train_predict_price'], y=train_residuals_df['actual_residual'])
plt.xlabel("Fitted values ($)", fontsize=20)
plt.ylabel("Residuals ($)",fontsize=20)
plt.title('Residuals Plot - Train in dollars',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


plt.xscale('log')
#plt.yscale('log')
plt.hlines(0,0,500)
plt.tight_layout()
#plt.savefig('residuals_plot_dollarvsdollar.tiff')


# In[1001]:


#When compared to the actual prices, the model consistantly underprices the very expensive albums over 100 dollars
#it may have been a better to exclude very high price outliers from the dataset, since they cannot be modeled well

sns.scatterplot(x=train_residuals_df['y_train_price'], y=train_residuals_df['actual_residual'])
plt.xlabel("Actual Price ($)", fontsize=20)
plt.ylabel("Residuals ($)",fontsize=20)
plt.title('Residuals Plot - Train in dollars',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


plt.xscale('log')
#plt.yscale('log')
plt.hlines(0,0,500)
plt.tight_layout()


# In[1002]:


#When plotted as log dollar values, the residuals are fairly evenly distributed across the predicted values

sns.scatterplot(x=lm.predict(X_train.iloc[:,2:]), y=lm.predict(X_train.iloc[:,2:])-y_train)
plt.xlabel("Fitted values (log $)", fontsize=20)
plt.ylabel("Residuals (log $)",fontsize=20)
plt.title('Residuals Plot - Train in log dollars',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


#plt.xscale('log')
#plt.yscale('log')
plt.hlines(0,0,6)
plt.tight_layout()
plt.savefig('residuals_plot_log$vslog$.tiff')


# In[1005]:


#The log $ residuals are also fairly evenly distributed for the test set data

sns.scatterplot(x=np.log(predictions), y=residuals)
plt.xlabel("Fitted values (log $)", fontsize=15)
plt.ylabel("Residuals (log $)",fontsize=15)
plt.title('Residuals Plot - Test in log dollars',fontsize=15)

plt.hlines(0,0,6);


# In[1006]:


#Prediction plot for the training set

sns.scatterplot(x=e**y_train, y=e**y_train_predict)
plt.xlabel("Actual values ($)", fontsize=20)
plt.ylabel("Fitted values ($)",fontsize=20)
plt.title('Prediction Plot - Train',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot(2.17**y_train, 2.17**y_train)
plt.tight_layout()
plt.xscale('log')
plt.yscale('log')
plt.xlim(1,1500)
plt.ylim(1,1500)
plt.tight_layout()
plt.savefig('Prediction Plot.tiff')


# In[1007]:


#Prediction Plot - Test

plt.figure(figsize=(5,5))
sns.scatterplot(x=actual, y=predictions)
plt.xlabel("Actual values ($)", fontsize=20)
plt.ylabel("Predicted values ($)",fontsize=20)
plt.xscale("log")
plt.yscale("log")
plt.xlim(1,500)
plt.ylim(1,500)
plt.plot(2.17**y_test, 2.17**y_test)

plt.title("Prediction Plot - Test", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.savefig('Test Set predictions.tiff')


# In[572]:


from statsmodels.graphics.gofplots import qqplot


# In[682]:


#QQ_plot on log dollar residulars from the training set
#shows some curviness at the upper and lower quantiles, which is consistant with prediction plots,
#the model underprices the very high price records, and overprices the very low price records.

qqplot(residuals_train,line='45')
plt.xlabel("Theoretical Quantiles", fontsize=15)
plt.ylabel("Sample Quantiles",fontsize=15)
plt.title('QQ plot',fontsize=15)
plt.savefig('QQ.tiff')

