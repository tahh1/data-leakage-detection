#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Linked article on this data:
# [The 📈 Ups and 📉Downs of the San Francisco Construction Industry. 🏗Trends and History of the Construction
# ](https://www.linkedin.com/pulse/ups-downs-san-francisco-construction-industry-trends-history-artem
# )
# 
# This data set pertains to all types of structural permits. Data includes details on application/permit numbers, job addresses, supervisorial districts, and the current status of the applications. Data is uploaded weekly by DBI. Users can access permit information online through DBI’s Permit Tracking System which is 24/7 at www.sfdbi.org/dbipts.
# 
# Note if you need to open permits in Excel, use one of the pre-filtered datasets:
# 
# 1. Building Permits on or after January 1, 2013 https://data.sfgov.org/d/p4e4-a5a7
# 2. Building Permits before January 1, 2013 https://data.sfgov.org/d/4jpb-z4kk
# 
# 🔎 Data on more than a million building permits (records in two datasets) from the San Francisco Construction Department allow us to analyze not only the construction activity in the city, but also critically examine the latest trends and development history of the construction industry over the past 40 years, from 1980 to 2019.
# 
# 📈 The movement of activity in the construction industry in San Francisco almost completely coincides with the growth schedule for gold and bitcoin (section "The future of the San Francisco construction industry, pattern prediction")
# 
# 💡 Open data provide an opportunity to explore the main factors that influenced and will affect the development of the construction industry in the city, dividing them into “external” (economic booms and crises) and “internal” (the effect of holidays and seasonal-annual cycles).
# 
# Linked article on this data:
# [The 📈 Ups and 📉Downs of the San Francisco Construction Industry. 🏗Trends and History of the Construction
# ](https://www.linkedin.com/pulse/ups-downs-san-francisco-construction-industry-trends-history-artem
# )

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# Any results you write to the current directory are saved as output.


# # Reading the files
# 
# San Francisco building permit data - taken from the open data portal - data.sfgov.org. The portal has several datasets on the topic of construction. Two such datasets store and update data on permits issued for the construction or repair of facilities in the city:
# 
# Building permits for the period 1980-2013 (850 thousand records)
# Building permits for the period after 2013 (280 thousand records, data are downloaded and updated weekly)
# These datasets contain information on the issued building permits, with various characteristics of the facility for which the permit is issued. The total number of records (permits) received in the period 1980-2019 is 1,137,695 permits. Link to Jupyter Notebook with data and graphs.
# 
# The dataset provided is in a .csv format. This is a structured dataset, with columns representing a range of things. For dealing with structured data, pandas is the most important library. We already imported pandas as pd when we used the import* command earlier. We will now use the read_csv function of pandas to read the data:

# In[2]:


# create new df after 2013


df_after2013 = pd.read_csv('/kaggle/input/sf-building-permits/Building_Permits_on_or_after_January_1__2013.csv', low_memory=False);


# In[3]:


df_after2013.describe()


# In[4]:


# create new df after 2013
df_before2013 = pd.read_csv('/kaggle/input/permits-before-2013/Building_Permits_before_January_1__2013.csv', low_memory=False);
df_before2013.describe()


# In[5]:


frames = [df_after2013, df_before2013]
df = pd.concat(frames)
df.describe()


# Let us look at the first few rows of the data. Since the dataset is large, this command does not show us the complete column-wise data. 
# To fix this, we will define the following function, where we set max.rows and max.columns to 100.

# Let's change the names of our columns so that it can be more convenient to work with them.
# Translate to lowercase and remove characters from names

# In[6]:


pd.set_option('display.max_columns',100)
df.sample(3)


# In[7]:


# translate to lowercase
df.columns = list(map(str.lower, df.columns))
# remove characters from names
df.columns = [c.replace(' ', '_') for c in df.columns]
df.columns = [c.replace('-', '') for c in df.columns]
df.head(1)


# Now with the help of the library missingno we visualize empty values in our dataset. 
# White passes are missing data in our dataset.
# 
# **Visualising missing values for a sample of 250**

# In[8]:


import missingno as msno
msno.matrix(df.sample(250))


# The missingno correlation heatmap measures nullity correlation: how strongly the presence or absence of one variable affects the presence of another
# * **msno.bar** is a simple visualization of nullity by column:

# In[9]:


msno.bar(df.sample(1000))


# In[10]:


msno.heatmap(df);


# Nullity correlation ranges from -1 (if one variable appears the other definitely does not) to 0 (variables appearing or not appearing have no effect on one another) to 1 (if one variable appears the other definitely also does).
# 
# Variables that are always full or always empty have no meaningful correlation, and so are silently removed from the visualization—in this case for instance the datetime and injury number columns, which are completely filled, are not included.
# 
# Entries marked <1 or >-1 are have a correlation that is close to being exactingly negative or positive, but is still not quite perfectly so. This points to a small number of records in the dataset which are erroneous. For example, in this dataset the correlation between 'exiting_use' and ' is <1, indicating that, contrary to our expectation, there are a few records which have one or the other, but not both. These cases will require special attention.
# 
# The heatmap works great for picking out data completeness relationships between variable pairs, but its explanatory power is limited when it comes to larger relationships and it has no particular support for extremely large datasets.

# **Dendrogram** allows you to more fully correlate variable completion, revealing trends deeper than the pairwise ones visible in the correlation heatmap:

# In[11]:


msno.dendrogram(df)


# The dendrogram uses a hierarchical clustering algorithm (courtesy of scipy) to bin variables against one another by their nullity correlation (measured in terms of binary distance). At each step of the tree the variables are split up based on which combination minimizes the distance of the remaining clusters. The more monotone the set of variables, the closer their total distance is to zero, and the closer their average distance (the y-axis) is to zero.
# 
# To interpret this graph, read it from a top-down perspective. Cluster leaves which linked together at a distance of zero fully predict one another's presence—one variable might always be empty when another is filled, or they might always both be filled or both empty, and so on. In this specific example the dendrogram glues together the variables which are required and therefore present in every record.
# 
# Cluster leaves which split close to zero, but not at it, predict one another very well, but still imperfectly. If your own interpretation of the dataset is that these columns actually are or ought to be match each other in nullity (for example, as CONTRIBUTING FACTOR VEHICLE 2 and VEHICLE TYPE CODE 2 ought to), then the height of the cluster leaf tells you, in absolute terms, how often the records are "mismatched" or incorrectly filed—that is, how many values you would have to fill in or drop, if you are so inclined.
# 
# As with matrix, only up to 50 labeled columns will comfortably display in this configuration. However the dendrogram more elegantly handles extremely large datasets by simply flipping to a horizontal configuration.

# In[12]:


# delete columns with a name delet_
df.drop(df.filter(regex='delete').columns, axis=1, inplace=True)
df.head(5)


# Let's look at the number of unique values in each column.

# In[13]:


df.nunique()


# # Primary data analysis / Primary visual data analysis

# * We select all the "estimated cost" and "revised_cost" values by the time the order was created.
# * Using the to_datetime function, we translate the string values in the date column into a time format.

# In[14]:


data_loc = df.loc[:,['estimated_cost', 'revised_cost','permit_creation_date']]
data_cost = data_loc 
data_cost.permit_creation_date = pd.to_datetime(data_cost.permit_creation_date)
data_cost = data_cost.set_index('permit_creation_date')


# * Using the dropna function, we delete all empty values, thus deleting all lines that will contain some kind of empty value in one of the parameters.
# * And using the groupby function, we group all our data by month. In this case, the value in other columns will be summarized. 
# 

# In[15]:


data_cost = data_cost.dropna()
data_cost_m = data_cost.groupby(pd.Grouper(freq='M')).sum()
data_cost_m.head()


# # Annual Construction Activity in San Francisco
# 
# In the graph below, the data on the estimated_cost and revised_cost parameters is presented as a distribution of the total cost of work by month (in billions US dollars).
# 
# Now let's create a new chart, where on the X axis we will have the month of creating the order, and on the Y axis - the "estimated cost". We see here a slight cyclicality and a general tendency towards a decline in the total number of requests for construction.
# 

# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(19,8))
# Add title
plt.title("Estimated costs and revised costs in 2013-2020")
sns.lineplot(data=data_cost_m)


# In[17]:


data_cost_m_after2012 = data_cost_m[data_cost_m.index > "2008-01-01"] 
data_cost_m_after2012 = data_cost_m_after2012[data_cost_m_after2012.index < "2019-8-01"] 
plt.figure(figsize=(14,6))
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14) 
data_cost_m_after2012 = data_cost_m_after2012.groupby(pd.Grouper(freq='Y')).sum()
sns.lineplot(data=data_cost_m_after2012) 



#data_cost_m_after2012.plot.line(linewidth=2)


# In[18]:


data_month = data_loc.dropna()
data_month.permit_creation_date = data_month.permit_creation_date.dt.month_name()
data_month = data_month.drop(['revised_cost'], axis = 1)


# # Statistics on the total number of applications by month and day
# # 
# General statistics on the number of applications by month and day from 1980 to 2019 shows that the “quietest” months for construction departments - are spring and winter months. At the same time, the amount of investments offered in the applications varies greatly, and differs from month to month (see “Construction activity depending on the season of the year”). Among the days of the week on Monday, the department’s workload is approximately 20% less than the rest of the week.

# In[19]:


months = [ 'January', 'February', 'March', 'April', 'May','June', 'July', 'August', 'September', 'October', 'November', 'December' ]
data_month_count  = data_month.groupby(['permit_creation_date']).count().reindex(months) 

plt.figure(figsize=(20,6))
data_month_count.plot.bar(legend=None)
plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=10) 
#sns.lineplot(data=data_cost_d)


# While June and July practically do not differ in the number of applications, the difference in total estimated cost reaches 100% (4.3 billion in May and July and 8.2 billion in June).

# In[20]:


months = [ 'January', 'February', 'March', 'April', 'May','June', 'July', 'August', 'September', 'October', 'November', 'December' ]
data_month_sum  = data_month.groupby(['permit_creation_date']).sum().reindex(months) 

#data_month = data_month.drop(['revised_cost'], axis = 1)
plt.figure(figsize=(40,10))
plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=20) 
data_month_sum.plot.bar(figsize=(10,5), legend=None)
#sns.lineplot(data=data_cost_d)


# In[21]:


data_month_sum.plot.line()


# # Construction activity depending on the season of the year
# 
# Having grouped the data by calendar weeks in a year (54 weeks), you can observe the construction activity of the city of San Francisco, depending on seasonality and time of year.
# 
# By Christmas, all construction companies are trying to manage to get permission for new “large” objects (at the same time! The number! Permits in the same months are at the same level throughout the year). Investors, planning to get their property over the next year, conclude contracts in the winter months, counting on big discounts (since summer contracts, for the most part, are coming to an end by the end of the year and construction companies are interested in receiving new applications).
# 
# Before Christmas, the largest amounts are submitted in applications (an increase from an average of 1-1.5 billion per month. Up to 5 billion in December alone). At the same time, the total number of applications by month remains at the same level (see the section below: Statistics on the total number of applications by month and days)
# 
# After the winter holidays, the construction industry is actively (almost without an increase in the number of permits) planning and implementing “Christmas” orders, so that by the middle of the year (before the Independence Day) have time to free up resources before the beginning of immediately after the June holidays - a new wave of summer agreements.

# In[22]:


data_month_year = data_loc 
data_month_year = data_month_year.assign(week_year = data_month_year.permit_creation_date.dt.week)
#data_month_year.permit_creation_date = data_month_year.permit_creation_date.dt.strftime('%m-%d')
data_month_year = data_month_year.drop(['revised_cost', 'permit_creation_date'], axis = 1)


# In[23]:


data_month_year2 = data_month_year.groupby(['week_year'])['estimated_cost'].sum()


# In[24]:


plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=20) 

plt.figure(figsize=(40,7))
fig, ax = plt.subplots(1,1)
major_ticks = np.arange(0, 55, 1)

ax.set_xticks(major_ticks)

data_month_year2.plot.line(linewidth=3, figsize=(20,5))

#fig = plt.figure(figsize=(50,6))  # sets the window to 8 x 6 inches
# left, bottom, width, height (range 0 to 1)
# so think of width and height as a percentage of your window size


# In[25]:


data_month_year2.to_csv('percet.csv')


# In[26]:


data_month_year2.columns = ['cost']
data_month_year2.head(2)


# To reduce monthly “emissions”, monthly data are grouped by year. The graph of the amount of money invested over the years has received a more logical, and amenable to analysis - view.
# 
# Let's make the graph more visual and now group these same data by year. Here we will use the groupby function again, only as an argument to the function we will have not a month, but a year. And as before, the values ​​in the output columns during grouping will be summarized.
# 
# Here you can already notice the general trend and we see that the total number of permits to the construction department has fallen compared to 2016. In 2019, activity in the construction industry of San Francisco was at the level of 2014. The general trend of the last five years - activity in the construction industry is falling.

# In[28]:


import matplotlib.ticker as plticker
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib.dates as mdates




data_cost_y = data_cost.groupby(pd.Grouper(freq='Y')).sum()
plt.figure(figsize=(19,8))
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=40) 
# Add title
#ax = plt.subplots(1,1)
#g.xaxis.set_ticks(np.arange(start, end, 2))
#loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
#ax.xaxis.set_major_locator(loc)


#plt.title("Estimated costs and revised costs in 1980-2020")
plt.xticks(rotation=40)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=400))


g=sns.lineplot(data=data_cost_y, linewidth=3, size = 17)
plt.savefig('plotname.png', transparent=True)




# # Expectation and reality in drawing up estimated cost
# # 
# In the used datasets, data on the cost of permitting a building object is divided into:
# 
# initial estimated cost (estimated_cost)
# cost of work after revaluation (revised_cost)
# During the boom, the main purpose of revaluation is to increase the initial cost, when the investor (construction customer) shows an appetite after the start of construction.
# 
# During the crisis, the estimated cost, they try not to exceed, and the initial estimates practically do not undergo changes (with the exception of the 1989 earthquake).
# 
# According to the graph of the revalued and estimated cost built on the difference (revised_cost - estimated_cost), we can observe that:
# 
# The amount of cost increase during the revaluation of the volume of construction work - directly depends on the cycles of the economic boom
# 
# 
# 
# 
# By the annual movement of the sum of costs (all permits for the year) in urban facilities, Economic factors from 1980 to 2019 clearly influenced the number and cost of construction projects or in another way, on San Francisco real estate investments.
# 
# The number of building permits (the number of construction works or the number of investments) over the past 40 years has been closely related to economic activity in the silicone valley.
# 
# 

# In[29]:


data_spred = data_cost_y.assign(spred = (data_cost_y.revised_cost-data_cost_y.estimated_cost))
data_spred = data_spred.drop(['revised_cost', 'estimated_cost'], axis = 1)
plt.figure(figsize=(19,8))
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=15) 
plt.xticks(rotation=40)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=400))
g=sns.lineplot(data=data_spred, linewidth=3, size = 17)


# If you look at this table as a percentage change, the peak increase in estimates (100% or 2 times the original estimated cost) came in the year before the earthquake in 1989 near the city. I suppose that after the earthquake the construction projects that were started in 1988 required after the earthquake in 1989 - more time and money to implement.
# 
# Conversely, a downward revision of the estimated cost (which happened only once during the period from 1980 to 2019) a few years before the earthquake is presumably due to the fact that some objects started in 1986-1987 were frozen or investments in these objects were cut back. According to the schedule, on average for each object begun in 1987, the estimated cost reduction was -20% of the original plan.

# In[32]:


data_spred_percent = data_cost_y.assign(spred = ((data_cost_y.revised_cost-data_cost_y.estimated_cost)/data_cost_y.estimated_cost*100))
data_spred_percent = data_spred_percent.drop(['revised_cost', 'estimated_cost'], axis = 1)
data_spred_percent = data_spred_percent[data_spred_percent.index > "1982-8-01"] 
data_spred_percent = data_spred_percent[data_spred_percent.index < "2019-1-31"] 


plt.figure(figsize=(19,8))
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=15) 
plt.xticks(rotation=40)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=400))
g=sns.lineplot(data=data_spred_percent, linewidth=3, size = 17)


# Let's look at the number of building permits by day of the week. To do this, select the data from our previous data loc dataframe. Using the day_name function, we define for each date in the string the day of the week. Group all the data by day of the week. And display our data in a new chart.
# 

# In[33]:


data_cost_m


# In[34]:


data_cost_d = data_loc.drop(['revised_cost'], axis = 1)
data_cost_d = data_cost_d.dropna()
data_cost_d.permit_creation_date = data_cost_d.permit_creation_date.dt.day_name()

days = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
data_cost_d  = data_cost_d.groupby(['permit_creation_date']).count().reindex(days) 
plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=10) 
plt.figure(figsize=(20,6), )
data_cost_d.plot.bar(legend=None)
#data_cost_d.plot.line(legend=None)



# Now create new latitude and longitude data for each object: 
# * To do this, we will split the Location column using the split function. 
# * And create a new dataframe where we will already have new data on latitude and longitude and "estimated cost".
# * Also here we will remove all zero values ​​from our new data frame.
# 

# In[35]:


df[['lat','long']] = df.location.str.split(",",expand=True)
data_location = df.loc[:,['long','lat','zipcode','estimated_cost']]
data_location = data_location.dropna()
data_location.head()


# * Using the apply function, we remove the opening and closing quotes from the latitude and longitude columns.
# * Using the info function, we see that there are 227,000 rows in our dataframe. Large enough dataframe for analysis.
# 
# 

# In[36]:


data_location.long = data_location.long.apply(lambda x: x.replace(')',''))
data_location.lat = data_location.lat.apply(lambda x: x.replace('(',''))
data_location.info()


# Using the to_numeric function, we translate the data in the latitude and longitude columns to float64 format - that is, a floating-point number.

# In[37]:


data_location.lat = pd.to_numeric(data_location.lat)
data_location.long = pd.to_numeric(data_location.long)


# * Group all our dataframe “Data Location” by the zip code parameter. 
# * At the same time, when grouping the data in the columns latitude, longitude and cost - all values will be taken as average values.
# That is, they will not be summed up, as we already did before, but here the average values ​​will be taken.

# # Average estimated cost of an application for construction by city district
# 
# All data, as in the case of the total amount of investments, were grouped by zip code. Only in this case with the average (.mean ()) estimated cost of the application by zip code.
# 
# data_location_mean = data_location.groupby(['zipcode'])['lat','long','estimated_cost'].mean()
# In ordinary areas of the city (more than 2 km. From the city center) - the average estimated cost of an application for construction is $ 50 thousand.

# In[38]:


data_location_mean = data_location.groupby(['zipcode'])['lat','long','estimated_cost'].mean()
data_location_mean.head()


# Using the folium library, we can display the average postcode price on a San Francisco map. 
# * To do this, we will set the center of our map to a point with the long and lat values, which will be taken from our data frame by the average value in the latitude and longitude columns. 
# * Jumpstart points to the scale of the map and with the argument Stamen Toner - points to the black and white style that we will use for the map.
# * Using the for loop and the Circle function, we can specify the average estimated_cost as a circle, where each circle will point to a zip code with the center of longitude and latitude. 
# 
# From this map, we can conclude that on Treasure Island we have the highest average cost of a building permit.
# 

# In[39]:


import folium
from folium import Circle
from folium import Marker
from folium.features import DivIcon

# map folium display
lat = data_location_mean.lat.mean()
long = data_location_mean.long.mean()
map1 = folium.Map(location = [lat, int], zoom_start = 12, tiles='Stamen Toner')
param = 'estimated_cost'

text = 'Test'
circle_lat = 60
circle_lon = 10

for i in range(0,len(data_location_mean)):
    Circle(
        location = [data_location_mean.iloc[i]['lat'], data_location_mean.iloc[i]['long']],
        radius= [data_location_mean.iloc[i]['estimated_cost']/4000],
        fill = True, fill_color='#cc0000',color='#cc0000').add_to(map1)
    
    Marker(
    [data_location_mean.iloc[i]['lat'], data_location_mean.iloc[i]['long']],
    icon=DivIcon(
        icon_size=(6000,3336),
        icon_anchor=(0,0),
        html='<div style="font-size: 18pt; text-shadow: 0 0 5px #fff, 0 0 5px #fff, 0 0 10px #fff, 0 0 5px #fff, 0 0 15px #fff, 0 0 5px #fff, 0 0 5px #fff; color: #000";"">%s</div>'
        %("$ "+ str((data_location_mean.iloc[i]['estimated_cost']/1000000).round(2)) + 'mln.'))).add_to(map1)
    
map1


# The average estimated cost in the area of the city center is about three times higher ($ 150 thousand to $ 400 thousand) than in other areas ($ 30-50 thousand).
# 
# In addition to the cost of land, three factors determine the total cost of housing construction: labor, materials, and government fees. These three components are higher in California than in the rest of the country. California building codes are considered among the most comprehensive and stringent in the country (due to earthquakes and environmental regulations), often requiring more expensive materials and labor.
# 
# For example, the State requires builders to use higher quality building materials (windows, insulation, heating and cooling systems) to achieve high standards in energy efficiency.

# Let's look at the total total cost of building permits depending on the zip code. 
# * Group all the data that is in the Data Location data frame by zip code. Now the values from the longitude and latitude columns will be averaged when grouping. 
# * And display our new dataframe Data Location.

# # In which areas of San Francisco have invested more over the past 40 years
#  
# With the help of the Folium library, let's see where these $ 91.5 billion by regions were invested. To do this, grouping the data by zip code (zipcode), imagine the value obtained using circles (Circle function from the Folium library).

# In[40]:


data_location_lang_long = data_location.groupby(['zipcode'])['lat','long'].mean()
data_location_lang_long.head()


# After the first grouping, using the assign function, we add to our frame date a new column called cost, which in turn was grouped by zip code, and in which it is no longer the average value - but the sum of all the values ​​in the group.

# In[41]:


data_location_lang_long = data_location_lang_long.assign(cost = data_location.groupby(['zipcode'])['estimated_cost'].sum())
data_location_lang_long.head()


# Using the folium library, we will once again display our data on the total cost of all building permits for the postal code. From this we see that the total total cost of the appeal is mainly localized in downtown. And specifically on several of the main streets of San Francisco. At the same time, on a treasure island, where there was a very large average cost of building permits - the total cost of work at the level of suburban areas.
# 

# In[42]:


import folium
from folium import Circle
from folium import Marker
from folium.features import DivIcon

# map folium display
lat = data_location_lang_long.lat.mean()
long = data_location_lang_long.long.mean()
map1 = folium.Map(location = [lat, int], zoom_start = 12)

for i in range(0,len(data_location_lang_long)):
    Circle(
        location = [data_location_lang_long.iloc[i]['lat'], data_location_lang_long.iloc[i]['long']],
        radius= [data_location_lang_long.iloc[i]['cost']/20000000],
        fill = True, fill_color='#cc0000',color='#cc0000').add_to(map1)
    Marker(
    [data_location_mean.iloc[i]['lat'], data_location_mean.iloc[i]['long']],
    icon=DivIcon(
        icon_size=(6000,3336),
        icon_anchor=(0,0),
        html='<div style="font-size: 14pt; text-shadow: 0 0 10px #fff, 0 0 10px #fff;; color: #000";"">%s</div>'
        %("$ "+ str((data_location_lang_long.iloc[i]['cost']/1000000000).round()) + ' bn'))).add_to(map1)
map1


# 

# # Total San Francisco Real Estate Investments
# 
# Based on the data on building permits in the city:
# 
# The total investment in construction projects in San Francisco from 1980 to 2019 is $ 91.5 billion.

# In[43]:


sf_worth = data_location_lang_long.cost.sum()
sf_worth 


# In[44]:


data_location_p = data_location
data_location_p['freq']=data_location_p.groupby(by='zipcode')['lat'].transform('count')
data_location_p = data_location.groupby('zipcode')['lat','long','freq'].mean()


# In fact, the highest average claim in these two areas is associated with the lowest number of applications for this zip code (145 and 3064 respectively, construction on the island is very limited), while for the rest of the postal codes for the period 1980-2019, approximately 1300 applications were received per year (total average of 30-50 thousand applications for the entire period).
# 
# By the parameter “number of permits” is noticeable a perfectly even distribution of the number of applications per zip code throughout the city.

# In[45]:


# map folium display
lat = data_location_p.lat.mean()
long = data_location_p.long.mean()
map1 = folium.Map(location = [lat, int], zoom_start = 12)

for i in range(0,len(data_location_p)):
    Circle(
        location = [data_location_p.iloc[i]['lat'], data_location_p.iloc[i]['long']],
        radius= [data_location_p.iloc[i]['freq']/200],
        fill = True, fill_color='#DEEF34',color='#3DC0B7').add_to(map1)
    Marker(
    [data_location_p.iloc[i]['lat'], data_location_mean.iloc[i]['long']],
    icon=DivIcon(
        icon_size=(6000,3336),
        icon_anchor=(0,0),
        html='<div style="font-size: 14pt; text-shadow: 0 0 10px #fff, 0 0 10px #fff;; color: #000";"">%s</div>'
        %(int((data_location_p.iloc[i]['freq']).round())))).add_to(map1)
map1


# In[46]:


sns.countplot(x='zipcode', data=data_location)



# In[47]:


#data_location_count= data_location.groupby(['zipcode']).value_counts()
#data_location_count.head()


# In[48]:


# map folium display
lat = data_location_lang_long.lat.mean()
long = data_location_lang_long.long.mean()
map1 = folium.Map(location = [lat, int], zoom_start = 12)
#+str(data_location_lang_long.iloc[i]['cost']/100000000)+
for i in range(0,len(data_location_lang_long)):
    Circle(
        location = [data_location_lang_long.iloc[i]['lat'], data_location_lang_long.iloc[i]['long']],
        radius= [data_location_lang_long.iloc[i]['cost']/20000000],
        fill = True, fill_color='#cc0000',color='#cc0000').add_to(map1)
    Marker(
    [data_location_mean.iloc[i]['lat'], data_location_mean.iloc[i]['long']],
    icon=DivIcon(
        icon_size=(6000,3336),
        icon_anchor=(0,0),
        html='<div style="font-size: 16pt; text-shadow: 0 0 10px #fff, 0 0 10px #fff;; color: #000";"">%s</div>'
        %("$ "+ str((data_location_lang_long.iloc[i]['cost']/1000000000).round()) + ' bn.'))).add_to(map1)
map1


# For the prediction problem, we choose the "estimated cost" parameter.
# * Using the heatmap function, we display the correlations between our parameters and see that the "estimated cost" parameter has practically no correlation with other parameters, which of course greatly complicates our task of predicting estimated_cost.

# In[50]:


import seaborn as sn
sn.heatmap(df.corr());


# * In order to limit our selection, we first remove all empty values from the description column. 

# In[51]:


dfn = df.dropna(subset=['description'])
dfn.description.isnull().values.any()


# From our entire data frame, we will select only the data for which the description parameter has the value reroofing, that is, the work of deconstructing the old and creating a new roof. We select all the objects on which some changes on the roof have been made since 2014.
# 

# In[52]:


dfn = dfn[dfn['description'].str.match('kitchen')]
dfn.head()


# * Present graphically our new data. To do this, we Select from our already cleared dfn dataframe - 'estimated_cost', 'existing_use', 'existing_units', 'zipcode', 'issued_date'. 
# * And we delete all empty lines with empty values.
# 

# In[53]:


df_unit = dfn.loc[:,['estimated_cost','existing_use', 'existing_units', 'zipcode','issued_date']]
df_unit = df_unit.dropna()
df_unit.head(15)


# Here we will have large emissions due to the fact that hotels and industrial buildings are also taken into account here. 
# * Therefore, we will limit our data frame to only one-story, two-story houses, offices and apartments.

# In[54]:


df_unit[df_unit.existing_use.str.contains("family|office|apartments")]


# On the new chart we display the average "estimated cost" by type of housing. On this graph you can see how much higher the average "estimated cost" of repairing roofs in office buildings is. Here you can certainly talk about some kind of “cartel conspiracy” :). At the same time, the average cost of repairing a roof for two and one family house is practically the same.
# 

# In[55]:


fam1 = df_unit[df_unit['existing_use']=='1 family dwelling']['estimated_cost'].mean()
fam2 = df_unit[df_unit['existing_use']=='2 family dwelling']['estimated_cost'].mean()
office = df_unit[df_unit['existing_use']=='office']['estimated_cost'].mean()
apartments = df_unit[df_unit['existing_use']=='apartments']['estimated_cost'].mean()
data = {'1 family dwelling':fam1,'2 family dwelling':fam2,'Apartments':apartments}
typedf = pd.DataFrame(data = data,index=['redevelopment of the bathroom'])
typedf.plot(kind='barh', title="Average estimated cost by type", figsize=(8,6));


# Group all these data by years. 
# * We will select data for only one private house and apartment, since for the office the cost data is too high. 
# * And here, as in previous examples, we will group our data with average values.
# 
# On the graph you can see that the value for apartments varies greatly from year to year, while for two family and one family houses estimed cost do not change so much. And in the second graph we see that the cost is growing from year to year - this way you can see inflation in the construction market. We can approximately see that inflation, for example, by the average cost of repairing a roof from 2014 to 2019 was approximately 30%. That is, inflation in the construction market is 6% per year.
# 

# In[56]:


df_unit.issued_date = pd.to_datetime(df_unit.issued_date)
df_unit.issued_date = df_unit.issued_date.dt.year


# In[57]:


years = list(range(1980, 2020)) 
keywords = ['1 family dwelling','2 family dwelling','apartments']
val_data = []
for year in years:
    iss_data = []
    for word in keywords:
        v = df_unit[(df_unit['existing_use']==word) & (df_unit['issued_date']== year)]['estimated_cost'].mean()
        iss_data.append(v)
    val_data.append(iss_data)
#print(val_data)


# In[58]:


dfnew = pd.DataFrame(data=val_data, index=years, columns=keywords)
dfnew.head()


dfnew.plot.bar(figsize=(20, 8)) 
plt.xlabel("Years")
plt.ylabel("Estimated cost of reroofing")
plt.title("Estimated cost of reroofing by year");
dfnew.plot.line(figsize=(12, 6))


# In[59]:


years2 = list(range(1980, 2020)) 
keywords2 = ['2 family dwelling']

val_data2 = []
for year in years2:
    iss_data2 = []
    for word in keywords2:
        v = df_unit[(df_unit['existing_use']==word) & (df_unit['issued_date']== year)]['estimated_cost'].mean()
        iss_data2.append(v)
    val_data2.append(iss_data2)
#print(val_data)

dfnew_2 = pd.DataFrame(data=val_data2, index=years2, columns=keywords2)


# In[60]:


dfnew_2.plot.line(figsize=(12, 6))
dfnew_2.reset_index(level=0, inplace=True)


# In[61]:


sns.regplot(y=dfnew_2['2 family dwelling'],x=dfnew_2['index'],data=dfnew_2, fit_reg=True) 
#sns.jointplot(dfnew_2['index'], dfnew_2['2 family dwelling'], data=dfnew_2, fit_reg=True, stat_func=stats.pearsonr)
lines = plt.gca().lines
lower1990 = [line.get_ydata().min() for line in lines]
upper2019 = [line.get_ydata().max() for line in lines]
plt.scatter(1990, lower1990, marker='x', color='C3', zorder=3)
plt.scatter(2019, upper2019, marker='x', color='C3', zorder=3)
print(("In 1990 it cost = $" + str(lower1990[0].round()) + "; In 2019 it cost = $ " + str(upper2019[0].round())))
print(("Inflation for the period 1980-2019 = " + str(((upper2019[0]-lower1990[0])/lower1990[0]*100).round())+"%"))
all2 = [line.get_ydata() for line in lines]


# In[62]:


trendline_xy = lines[0].get_xydata()
df_cost = pd.DataFrame(data=trendline_xy, columns = ['year','trendline_xy'])
df_cost.year = df_cost.year.round()
df_cost=df_cost.groupby(['year'])['trendline_xy'].mean().reset_index()
#data_location_lang_long = data_location_lang_long.assign(cost = data_location.groupby(['zipcode'])['estimated_cost'].sum())

df_cost = df_cost.assign(diff_trend = df_cost.trendline_xy.diff())
df_cost = df_cost.assign(inflation = (df_cost.diff_trend)/(df_cost.trendline_xy)*100)
df_cost['year'] = df_cost['year'].apply(lambda f: format(f, '.0f'))
df_cost = df_cost.set_index(df_cost.year)
df_cost


# In[63]:


#ax = df_cost.plot.bar(x='year', y='inflation', rot=0)
#plt.plot(df_cost.year,df_cost.inflation)
plt.xticks(rotation='vertical');
plt.rc('xtick', labelsize=10) 
plt.ylim([0, 9])
plt.plot()


# In[64]:


#r2 = stats.pearsonr(dfnew_2['index'], dfnew_2['2 family dwelling'])
#r2


# We set ourselves the task of determining the "estimated cost" by several parameters.
# At the next stage, we need to determine all the characteristics by which we will determine the estimetet cost for roof repairs for a new facility. Unfortunately, we do not have data on the size of objects and the cost of, for example, the house itself, which would be the main parameter in determining the cost of work. We will work with those parameters that are in the public domain.
# 

# In[65]:


df_corr = dfn.dropna(subset=['existing_use'])
df_corr.description.isnull().values.any()


# *  To limit our selection and improve the prediction, we will take data for only 1 family houses.
# 

# In[66]:


df_1fam = df_corr[df_corr.existing_use.str.contains('1 family')]


# * First we find all the characteristics with numerical values in our data frame. 
# * And sort them by correlation with our desired value "estimated cost".

# In[67]:


num_feuture = df_1fam.select_dtypes(include=[np.number])
corr = num_feuture.corr()
print((corr['estimated_cost'].sort_values(ascending = False)))


# In[68]:


dfn[['lat','long']] = dfn.location.str.split(",",expand=True)


# Since we see that our "estimated cost" is little correlated with other parameters, so take 'permit_creation_date', 'zipcode', 'number_of_existing_stories', 'number_of_proposed_stories', 'current_police_districts' and other parameters that somehow correlate with the cost. 

# In[69]:


#df_pr = dfn.loc[:,['permit_creation_date', 'existing_use', 'existing_units','estimated_cost','zipcode','current_supervisor_districts', 'analysis_neighborhoods', ]]
#df_pr = dfn.loc[:,['permit_creation_date', 'zipcode', 'existing_use', 'existing_construction_type', 'estimated_cost', 'long','lat' ]]#
#df_pr = df_1fam.loc[:,['permit_creation_date', 'zipcode', 'number_of_existing_stories', 'number_of_proposed_stories',  'current_police_districts', 'existing_use', 'long','lat', 'record_id',  'estimated_cost',  ]]

df_pr = df_1fam.loc[:,['permit_creation_date', 'zipcode', 'number_of_existing_stories', 'number_of_proposed_stories',  'current_police_districts', 'long','lat', 'record_id',  'estimated_cost',  ]]
df_pr = df_pr.dropna()
#df_pr = df_pr[df_pr.existing_use.str.contains('1 family')]
df_pr.permit_creation_date = pd.to_datetime(df_pr.permit_creation_date)
df_pr.head()


# Let's look at the distribution of the "estimated cost" values ​​in the form of a histogram. Here we will see that we have some values ​​of $ 200,000 and there are very few of them. And a large number of small values.

# In[70]:


histplot = df_pr.estimated_cost.plot.hist(bins = 40)


# * We will consider this data as outliers and delete them from our dataframe.
# That is, we delete all the lines where the "estimated cost" will be more than 20,000 and less than 12,000. 
# 

# In[71]:


indexNames = df_pr[ (df_pr['estimated_cost'] > 20000)].index
df_pr.drop(indexNames , inplace=True)


# In[72]:


indexNames = df_pr[ (df_pr['estimated_cost'] < 12000)].index
df_pr.drop(indexNames , inplace=True)


# And now our distribution will not look so one-sided.

# In[73]:


histplot = df_pr.estimated_cost.plot.hist(bins = 40)


# Using the Johnson library, we can look at the "normality" of the distribution of our values

# In[74]:


import scipy.stats as st
y = df_pr['estimated_cost']
plt.figure(figsize=(7,4))
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(figsize=(7,4))
plt.figure(figsize=(7,4))
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)


# In[75]:


df_pr.describe()


# * As we did before, we remove the extra characters from the columns of longitude and latitude

# In[76]:


df_pr.long= df_pr.long.apply(lambda x: x.replace(')',''))
df_pr.lat = df_pr.lat.apply(lambda x: x.replace('(',''))
df_pr.lat = pd.to_numeric(df_pr.lat)
df_pr.long = pd.to_numeric(df_pr.long)


# Find the distance to the center of Downtown San Francisco. That is, we will take each object and from its point of latitude and longitude we will find the distance to the city center that is, to the value of the longitude and width of downtown San Francisco (37.7945742, -122.3999445).

# In[77]:


from geopy.distance import vincenty
def distance_calc (row):
    start = (row['lat'], row['long'])
    stop = (37.7945742, -122.3999445)

    return vincenty(start, stop).meters/1000

df_pr['distance'] = df_pr.apply (lambda row: distance_calc (row),axis=1)


# By district, it is clear that the majority of investments went to DownTown. Having simplified the grouping of all objects according to the distance to the city center and the time needed to get to the city center (of course, expensive houses are also being built on the coast), all permissions were divided into 4 groups: 'Downtown', '<0.5H Downtown', '< 1H Downtown ',' Outside SF '.

# In[78]:


def downtown_proximity(dist):
    '''
    < 2 -> Near Downtown,  >= 2, <4 -> <0.5H Downtown
    >= 4, <6 -> <1H Downtown, >= 8 -> Outside SF
    '''
    if dist < 2:
        return 'Downtown'
    elif dist < 4:
        return  '<0.5H Downtown'
    elif dist < 6:
        return '<1H Downtown'
    elif dist >= 6:
        return 'Outside SF'
df_pr['downtown_proximity'] = df_pr.distance.apply(downtown_proximity)


# Display the average values ​​of the "estimated cost" by category in our new Downtown proximity column. Here you can see that the average cost differ by about 6 - 7%. In principle, houses in a rich area spend on the roof about 10% more than people whose houses are farther from the center.
# 

# In[79]:


df_pr


# In[80]:


value_count=df_pr['downtown_proximity'].value_counts()
plt.figure(figsize=(12,5))
plt.title('Estimated cost of Bathroom depending on Downtown Proximity');
sns.boxplot(x="downtown_proximity", y="estimated_cost", data=df_pr);


# 
# Look at the same data on the map and see the amount of data by distance from the center. Here, we chose the distance of 3 km to the city center as the main indicator - where the yellow dots show objects on the map that are located up to 3 km from the city center.
# 

# In[81]:


sns.set()
local_coord=[-122.3999445, 37.7945742] # the point near which we want to look at our variables
euc_dist_th = 0.03 # distance treshhold

euclid_distance=df_pr[['lat','long']].apply(lambda x:np.sqrt((x['long']-local_coord[0])**2+(x['lat']-local_coord[1])**2), axis=1)

# indicate wethere the point is within treshhold or not
indicator=pd.Series(euclid_distance<=euc_dist_th, name='indicator')

print(("Data points within treshhold:", sum(indicator)));

# a small map to visualize th eregion for analysis
sns.lmplot('long', 'lat', data=pd.concat([df_pr,indicator], axis=1), hue='indicator', markers ='.', fit_reg=False, height=8);


# In[82]:


sns.set()
local_coord=[-122.3999445, 37.7945742] # the point near which we want to look at our variables
euc_dist_th = 0.03 # distance treshhold

euclid_distance=data_location[['lat','long']].apply(lambda x:np.sqrt((x['long']-local_coord[0])**2+(x['lat']-local_coord[1])**2), axis=1)

# indicate wethere the point is within treshhold or not
indicator=pd.Series(euclid_distance<=euc_dist_th, name='indicator')

print(("Data points within treshhold:", sum(indicator)));

# a small map to visualize th eregion for analysis


####sns.lmplot('long', 'lat', data=pd.concat([data_location,indicator], axis=1), hue='indicator', markers ='.', fit_reg=False, height=8);




# In[83]:


data_location_all = data_location
data_location_all['distance'] = data_location_all.apply (lambda row: distance_calc (row),axis=1)
data_location_all['downtown_proximity'] = data_location_all.distance.apply(downtown_proximity)


# In[84]:


data_investcost_dist = data_location_lang_long
data_investcost_dist['distance'] = data_location_lang_long.apply (lambda row: distance_calc (row),axis=1)
data_investcost_dist['downtown_proximity'] = data_investcost_dist.distance.apply(downtown_proximity)
data_investcost_dist


# In[85]:


#value_count=df_pr['downtown_proximity'].value_counts()
plt.figure(figsize=(12,15))
ax.set(ylim=(0, 1))
#appr = [ 'Downtown','<0.5H Downtown','<1H Downtown''Outside SF' ]
#data_investcost_dist = data_investcost_dist.reindex(appr) 
#my_colors = 'gbry'

sns.boxplot(x="downtown_proximity", y="cost", data=data_investcost_dist)
plt.savefig('plotname2.png', transparent=True)


# Of the 91.5 billion invested in the city, almost 70 billion (75% of all investments) invested in repairs and construction are in the city center (green zone) and in the city area within a 2 km radius from the center (blue zone).

# In[86]:


ax.tick_params(axis='x', labelrotation=30)
appr = [ 'Downtown','<0.5H Downtown','<1H Downtown','Outside SF' ]
#data_investcost_dist  = data_investcost_dist.reindex(months) 

df_cost_sum =data_investcost_dist.groupby(['downtown_proximity'])['cost'].sum().reindex(appr)/1000000000
#ax.set_xticklabels(rotation=30, ha='right')

plt.figure(figsize=(12,5))

#df_cost_sum.head()

my_colors = list('gbry')
#plot.setp(ax.get_xticklabels(), ha="gbr", rotation=15)
plee = df_cost_sum.plot.bar(legend=None, color=my_colors)
plt.savefig('plee.png', transparent=True)


# On the new map we can mark all our objects in our 4 categories: city center, half an hour on foot, 1 hour, or outside the city.
# 

# In[87]:


sns.lmplot('long', 'lat', data=data_location_all, markers ='.', hue='downtown_proximity', fit_reg=False, height=8)
plt.savefig('map2.png', transparent=True)

plt.show()


# In[88]:


sns.lmplot('long', 'lat', data=df_pr,markers ='.', hue='downtown_proximity', fit_reg=False, height=8)
plt.show()


# * Create a new column for the year 
# * Delete the value that is no longer needed - permit_creation_date and location which we already used, in order to find the longitude and latitude.

# In[89]:


#df_pr['month'] = df_pr.permit_creation_date.dt.month
df_pr['year'] = df_pr.permit_creation_date.dt.year
df_pr = df_pr.drop(columns=['permit_creation_date', 'long', 'lat'])


# get_dummies - Convert categorical variable downtown_proximity into dummy/indicator variables (0,1).

# In[90]:


#df_pr = pd.concat([df_pr, pd.get_dummies(df_pr.existing_use, prefix='existing_use')], axis=1)
df_pr = pd.concat([df_pr, pd.get_dummies(df_pr.downtown_proximity, prefix='dt_pr')], axis=1)

#df_pr = df_pr.drop(columns=['existing_use'])
df_pr = df_pr.drop(columns=['downtown_proximity'])


# In[91]:


df_pr.describe()


# 
# We can escalate the value in our data 
# 
# * Or we can simply subtract the minimum values from the data with large values.
# * For example, from the values of the year we subtract the minimum value - thereby we reduce the total value and these values will less affect our predictions.
# 

# In[92]:


#df_pr.existing_units = df_pr.existing_units.apply(lambda x: 10 if x > 10 else x)
df_pr.zipcode = df_pr.zipcode - df_pr.zipcode.min()
df_pr.year = df_pr.year - df_pr.year.min()
df_pr.record_id = df_pr.record_id - df_pr.record_id.min()
#df_pr.head()


# In[93]:


df_pr.head()


# In[94]:


df_pr.hist(bins=50, figsize=(10, 10));


# Now we visualize our data using the technique of nonlinear dimensionality reduction and visualization of multidimensional variables. TSNE **
# 

# In[95]:


from sklearn.manifold import TSNE
tsne=TSNE(perplexity = 3)
tsne.fit(df_pr)


# **
# t-Distributed Stochastic Neighbor Embedding** (t-SNE) is an unsupervised, non-linear technique primarily used for data exploration and visualizing high-dimensional data. In simpler terms, t-SNE gives you a feel or intuition of how the data is arranged in a high-dimensional space. 

# In[96]:


plt.scatter(tsne.embedding_[:,0], tsne.embedding_[:,1])


# * Separate the points, that is, our objects. And we will find projects in which the cost of construction work to create a new roof is more than "$"13,000 = orange and less than 13,000 in blue.

# In[97]:


df_pr['proofcost'] = df_pr.estimated_cost.apply(lambda x: True if x>=13000 else False )


# In[98]:


plt.scatter(tsne.embedding_[df_pr.proofcost.values, 0], tsne.embedding_[df_pr.proofcost.values, 1], color='orange')
plt.scatter(tsne.embedding_[~df_pr.proofcost.values, 0], tsne.embedding_[~df_pr.proofcost.values, 1], color='blue')


# * Delete the column 'proofcost' we do not need

# In[99]:


df_pr = df_pr.drop(columns = 'proofcost')


# In[100]:


#df_pr = df_pr.drop(['existing_construction_type'], axis = 1)


# # Creating, Training, Evaluating, Validating, and Testing ML Models

# Now we can start testing our model. First, we import all the libraries we need from the main sklearn library.
# 
# **sklearn** is a Python module integrating classical machine learning algorithms in the tightly-knit world of scientific Python packages (numpy, scipy, matplotlib).
# 
# It aims to provide simple and efficient solutions to learning problems that are accessible to everybody and reusable in various contexts: machine-learning as a versatile tool for science and engineering.

# In[101]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold


# **Defining Training/Test Sets**
# 
# * divide all our data into training and validation data. In X we will have all the values ​​except the value. And in Y the value is only value.

# In[102]:


X_training = df_pr.drop(['estimated_cost'], axis = 1)
y_training = df_pr['estimated_cost']


# **Splitting into Validation**
# 
# * Using the train_test_split function, we will separate our data regarding 80% training data and 20% validation data.

# In[103]:


from sklearn.model_selection import train_test_split #to create validation data set
X_train, X_valid, y_train, y_valid = train_test_split(X_training, y_training, test_size=0.2, random_state=0) 
#X_valid and y_valid are the validation sets


# **Linear Regression Model
# **
# 
# Let's look at the indicators of the model that the Linear Regression module will build for us. 
# 
# *  Train our data using the fit function.
# 
# Each machine learning algorithm has a basic set of parameters that can be changed to improve its accuracy. During the fitting process, you run an algorithm on data for which you know the target variable, known as “labeled” data, and produce a machine learning model. 
# 
# * Compare the outcomes to real, observed values of the target variable to determine their accuracy.
# 
# Then we predict based on our new cost model for the validation data frame X_valid. And we compare our obtained data with the initial data y_valid, calculating the determination coefficient for these values ​​and the standard deviation - the RMSE coefficient.
# 
# R2 = 1 - sum of (valid value for each row - prediction) ^ 2 / sum of (valid value for each prediction - mean) ^ 2
# RMSE = sqrt (np.mean (np.square (y - y_pred)))
# 
# We got the value of RMSE = 2000 dollar.
# Those. When predicting the value using linear regression, our accuracy will be + + - 2000 dollar.
# We can also see that due to the lack of important parameters in calculating the value, we obtained small values ​​of the determination coefficient near zero. This means that, now the forecasts do not match the actual values.
# 
# 

# In[104]:


linreg = LinearRegression()
linreg.fit(X_train, y_train)
lin_pred = linreg.predict(X_valid)
r2_lin = r2_score(y_valid, lin_pred)
rmse_lin = np.sqrt(mean_squared_error(y_valid, lin_pred))
print(("R^2 Score: " + str(r2_lin)))
print(("RMSE Score: " + str(rmse_lin)))


# Try to predict the "estimated cost" for arbitrary parameters. Here I took an arbitrary zip code, and arbitrary values ​​for the rest as a parameter, and got values ​​with an accuracy of + - $ 2000.
# 

# In[105]:


lin_pred = linreg.predict([[20.0, 1.0, 3.0, 4.0, 1163316512454, 4.825703, 0, 0, 0, 0, 1]])
print(("Prediction for data with arbitrary values: " + str(lin_pred[0])))


# We do the same for other regressors: DecisionTreeRegressor.

# In[106]:


linsvc = DecisionTreeRegressor()
linsvc.fit(X_train, y_train)
lin_pred = linsvc.predict(X_valid)
r2_lin = r2_score(y_valid, lin_pred)
rmse_lin = np.sqrt(mean_squared_error(y_valid, lin_pred))
print(("R^2 Score: " + str(r2_lin)))
print(("RMSE Score: " + str(rmse_lin)))


# In[107]:


linsvc = linsvc.predict([[20.0, 1.0, 3.0, 4.0, 1163316512454, 4.825703, 0, 0, 0, 0, 1]])
print(("Prediction for data with arbitrary values: " + str(linsvc[0])))


# **Decision Tree Regressor Model**
# 
# Here, when training the model, we use standard hyperparameters. In order to configure these hyperparameters and to search for the best parameters specifically for our data, we will use GridSearchCV
# 
# GridSearchCV is a library function that is a member of sklearn’s model_selection package. It helps to loop through predefined hyperparameters and fit your estimator (model) on your training set. So, in the end, you can select the best parameters from the listed hyperparameters.
# In addition to that, you can specify the number of times for the cross-validation for each set of hyperparameters.
# 

# In[108]:


dtr = DecisionTreeRegressor()
parameters_dtr = {"criterion" : ["mse", "friedman_mse", "mae"], "splitter" : ["best", "random"], "min_samples_split" : [2, 3, 5, 10], 
                  "max_features" : ["auto", "log2"]}
grid_dtr = GridSearchCV(dtr, parameters_dtr, verbose=1, scoring="r2")
grid_dtr.fit(X_train, y_train)

print(("Best DecisionTreeRegressor Model: " + str(grid_dtr.best_estimator_)))
print(("Best Score: " + str(grid_dtr.best_score_)))


# In[109]:


dtr = grid_dtr.best_estimator_
dtr.fit(X_train, y_train)
dtr_pred = dtr.predict(X_valid)
r2_dtr = r2_score(y_valid, dtr_pred)
rmse_dtr = np.sqrt(mean_squared_error(y_valid, dtr_pred))
print(("R^2 Score: " + str(r2_dtr)))
print(("RMSE Score: " + str(rmse_dtr)))


# In[110]:


#scores_dtr = cross_val_score(dtr, X_train, y_train, cv=10, scoring="r2")
#print("Cross Validation Score: " + str(np.mean(scores_dtr)))


# Next, we will test the model using the following other machine algorithms: Random Forest Regressor, Lasso, Ridge.

# In[111]:


rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_valid)
r2_rf = r2_score(y_valid, rf_pred)
rmse_rf = np.sqrt(mean_squared_error(y_valid, rf_pred))
print(("R^2 Score: " + str(r2_rf)))
print(("RMSE Score: " + str(rmse_rf)))


# In[112]:


scores_rf = cross_val_score(rf, X_train, y_train, cv=10, scoring="r2")
print(("Cross Validation Score: " + str(np.mean(scores_rf))))


# In[113]:


lasso = Lasso()
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_valid)
r2_lasso = r2_score(y_valid, lasso_pred)
rmse_lasso = np.sqrt(mean_squared_error(y_valid, lasso_pred))
print(("R^2 Score: " + str(r2_lasso)))
print(("RMSE Score: " + str(rmse_lasso)))


# In[114]:


scores_lasso = cross_val_score(lasso, X_train, y_train, cv=10, scoring="r2")
print(("Cross Validation Score: " + str(np.mean(scores_lasso))));


# In[115]:


ridge = Ridge()
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_valid)
r2_ridge = r2_score(y_valid, ridge_pred)
rmse_ridge = np.sqrt(mean_squared_error(y_valid, ridge_pred))
print(("R^2 Score: " + str(r2_ridge)))
print(("RMSE Score: " + str(rmse_ridge)))


# In[116]:


scores_ridge = cross_val_score(ridge, X_train, y_train, cv=10, scoring="r2");
print(("Cross Validation Score: " + str(np.mean(scores_ridge))));


# The obtained data on the coefficient of determination R ^ 2 and the standard error are written in the general resulting table. We see that the best results show Lasso algorithm.
# 

# In[117]:


model_performances = pd.DataFrame({
    "Model" : ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor","Ridge", "Lasso"],
    "R Squared" : [str(r2_lin)[0:5], str(r2_dtr)[0:5], str(r2_rf)[0:5], str(r2_ridge)[0:5], str(r2_lasso)[0:5]],
    "RMSE" : [str(rmse_lin)[0:8], str(rmse_dtr)[0:8], str(rmse_rf)[0:8], str(rmse_ridge)[0:8], str(rmse_lasso)[0:8]]
})
model_performances.round(4)


# In[118]:


X_train_v = X_train.values
y_train_v = y_train.values


# In[119]:


from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier()
decisiontree.fit(X_train,y_train)


# In[ ]:


from sklearn.tree import export_graphviz
export_graphviz(dtr_pred, out_file='tree.dot', feature_names = X_trai.columns, filled = True)
get_ipython().system('dot -Tpng tree.dot -o tree.png -Gdpi = 600')
from IPython.display import Image
Image(filename = 'tree.png' )


# # Model Building
# We have dealt with the categorical columns and the date values. We have also taken care of the missing values. Now we can finally power up and build the DecisionTree model we have been inching towards.

# In[ ]:


"""
X = df_pr.drop(['estimated_cost'], axis = 1).values
y = df_pr['estimated_cost'].values


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size = .3, random_state=0)
"""


# In[ ]:


"""
y_df = df_pr['estimated_cost']
X_df = df_pr.drop(['estimated_cost'], axis = 1)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.25, random_state=5)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter = 500000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(accuracy)
"""


# In[ ]:


"""
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ridge=Ridge()
parameters= {'alpha':[x for x in [0.1,0.2,0.4,0.5,0.7,0.8,1]]}

ridge_reg=GridSearchCV(ridge, param_grid=parameters)
ridge_reg.fit(X_train,Y_train)
print("The best value of Alpha is: ",ridge_reg.best_params_)
"""


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
m.fit(df_pr, y)
m.score(df_pr,y)


# The n_jobs is set to -1 to use all the available cores on the machine. This gives us a score (r^2) of 0.99, which is excellent. The caveat here is that we have trained the model on the training set, and checked the result on the same. There’s a high chance that this model might not perform as well on unseen data (test set, in our case).
# 
# The only way to find out is to create a validation set and check the performance of the model on it. So let’s create a validation set and the train set will contain the rest.

# In[ ]:


y_df = df_pr['estimated_cost']
X_df = df_pr.drop(['estimated_cost'], axis = 1)

def split_vals(a,n): 
    return a[:n].copy(), a[n:].copy()

n_valid = int(len(df_pr)*0.25)  # same as Kaggle's test set size
n_trn = len(df_pr)-n_valid
#raw_train, raw_valid = split_vals(X_df, n_trn)
X_train, X_valid = split_vals(X_df, n_trn)
y_train, y_valid = split_vals(y_df, n_trn)

X_train.shape, y_train.shape, X_valid.shape


# Here, we will train the model on our new set (which is a sample of the original set) and check the performance across both – train and validation sets.

# In[ ]:


import math 
#define a function to check rmse value
def rmse(x,y): 
    return math.sqrt(((x-y)**2).mean())


# In order to compare the score against the train and test sets, the below function returns the RMSE value and score for both datasets.

# In[ ]:


def print_score(m):
    res = [rmse(m.predict(X_train), y_train),
           rmse(m.predict(X_valid), y_valid),
           m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# The result of the above code is shown below. The train set has a score of 0.99, while the validation set has a score of 0.99.

# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# # Creating a Validation set
# Creating a good validation set that closely resembles the test set is one of the most important tasks in machine learning. The validation score is representative of how our model performs on real-world data, or on the test data.
# 
# Keep in mind that if there’s a time component involved, then the most recent rows should be included in the validation set. So, our validation set will be of the same size as the test set (last 25% rows from the training data).

# In[ ]:


def split_vals(a,n):
   return a[:n].copy(), a[n:].copy()

n_valid = int(len(df_pr)*0.25)  
n_trn = len(df_pr)-n_valid

raw_train, raw_valid = split_vals(df_pr, n_trn)
X_train, X_valid = split_vals(X_df, n_trn)
y_train, y_valid = split_vals(y_df, n_trn)


# The data points from 0 to (length – 25%) are stored as the train set (x_train, y_train). A model is built using the train set and its performance is measured on both the train and validation sets as before.

# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# From the above code, we get the results:
# 
# * RMSE on the validation set
# * R-square on validation set

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor(random_state = 42) 
dt_model.fit(train_X, train_Y)


# In[ ]:


m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


draw_tree(m.estimators_[0], df_trn, precision=3)


# In[ ]:


y_pred = dt_model.predict([[116,2.0,1.0,5.0,10.0,81.0,9.781119,0,1,0,0,1,0,0]])
print(y_pred)


# In[ ]:


Y_pred = dt_model.predict(validation_X)


# In[ ]:


s = pd.Series(Y_pred)
validation_Y.reindex()
df = pd.concat([s.reset_index(drop=True), validation_Y.reset_index(drop=True)], axis=1, ignore_index=True)
df.tail(15)


# In[ ]:


Y = df_pr.estimated_cost
X = df_pr.drop(['estimated_cost'], axis = 1)

from sklearn.model_selection import train_test_split
train_X, validation_X, train_Y, validation_Y = train_test_split(X, Y, random_state = 42)

print(("Training set: Xt:{} Yt:{}".format(train_X.shape, train_Y.shape))) 
print(("Validation set: Xv:{} Yv:{}".format(validation_X.shape, validation_Y.shape))) 
print("-") 
print(("Full dataset: X:{} Y:{}".format(X.shape, Y.shape)))


# In[ ]:


from sklearn.metrics import accuracy_score
score = accuracy_score(validation_Y, Y_pred)
print(score)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state = 42) 
model.fit(train_X, train_Y)


# In[ ]:


from sklearn.metrics import mean_absolute_error

# instruct our model to make predictions for the prices on the validation set 
validation_predictions = model.predict(validation_X)

# calculate the MAE between the actual prices (in validation_Y) and the predictions made 
validation_prediction_errors = mean_absolute_error(validation_Y, validation_predictions)

validation_prediction_errors


# In[ ]:


from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=10, random_state=17, shuffle=True)


# In[ ]:


features_for_trees=['street_number', 'number_of_existing_stories', 'existing_units',
       'existing_construction_type', 'zipcode', 'sf_find_neighborhoods',
       'distance', 'year', 'existing_use_1 family dwelling',
       'existing_use_2 family dwelling', 'dt_pr_<0.5H Downtown',
       'dt_pr_<1H Downtown', 'dt_pr_Downtown', 'dt_pr_Outside SF']


# In[ ]:


TOTAL = df_pr.count()[0] 
N_VALID = 0.25 # Three months 
TRAIN = int(TOTAL*N_VALID)
df_small = df_pr
features = ['street_number', 'number_of_existing_stories', 'existing_units',
       'existing_construction_type', 'zipcode', 'sf_find_neighborhoods',
       'distance', 'year', 'existing_use_1 family dwelling',
       'existing_use_2 family dwelling', 'dt_pr_<0.5H Downtown',
       'dt_pr_<1H Downtown', 'dt_pr_Downtown', 'dt_pr_Outside SF']
df_pr
y_df = df_small['estimated_cost']
X_train, X_val = X_df[:TRAIN], X_df[TRAIN:]
y_train, y_val = y_df[:TRAIN], y_df[TRAIN:]
#define a function to check rmse value
import  math 
def rmse(x,y): 
    return math.sqrt(((x-y)**2).mean())
def print_score(m):
    res = [rmse(m.predict(X_train), y_train),
           rmse(m.predict(X_val), y_val),
           m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[ ]:


model = RandomForestRegressor(n_estimators=40, bootstrap=True, min_samples_leaf=25)
model.fit(X_train, y_train)
#draw_tree(model.estimators_[0], X_train, precision=2)
print_score(model)


# In[ ]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

#training_scores_encoded = lab_enc.fit_transform(df_pr.estimated_cost)
#print(training_scores_encoded)


y = df_pr.estimated_cost
X = df_pr.drop(['estimated_cost'], axis = 1)
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.25, random_state = 17)
X.shape, y.shape


# In[ ]:


"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score




ml_arr =[SVC(), GaussianNB(), Perceptron(), SGDClassifier(), DecisionTreeClassifier(), RandomForestClassifier()]


for el in ml_arr:
    el.fit(X_train, y_train)
    Y_pred = el.predict(X_valid)
    Y_pred.reshape(-1, 1)
    #Y_pred = lab_enc.fit_transform(Y_pred)
    #acc = round(el.score(y_valid, Y_pred) * 100, 2)
    score = accuracy_score(y_valid, Y_pred)
    print(score)
"""


# In[ ]:


# using scaled data
X=pd.concat([train_df[dummies_names], X_train_scaled[numerical_features]], axis=1, ignore_index = True)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
print((np.sqrt(-cv_scores.mean())))


# **# Cross-validation and adjustment of model hyperparameters
# 
# Let's prepare cross validation samples. As far as there are not a lot of data we can easily divide it on 10 folds, that are taken from shuffled train data. Within every split we will train our model on 90% of train data and compute CV metric on the other 10%.
# 
# We fix the random state for the reproducibility.

# In[ ]:


from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=10, random_state=17, shuffle=True)


# In[ ]:


from sklearn.linear_model import Ridge

model=Ridge(alpha=1)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
print((np.sqrt(-cv_scores.mean())))


# In[ ]:




