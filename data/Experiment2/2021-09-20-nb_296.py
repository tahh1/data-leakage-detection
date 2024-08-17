#!/usr/bin/env python
# coding: utf-8

# In[101]:


import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import traceback
from tqdm import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')
from google.colab import files


# In[2]:


# from google.colab import drive
# drive.mount('/content/drive')
# path_csv = 'drive/MyDrive/02.TCCs/3.TCC Pós Puc/data/'

path_csv = './data/'


#  # Virus Tracker
#  Classe utilizada para realizar algumas computações no dataset de covid

# In[109]:


class VirusTracker:

  def __init__(self, last_days=30, path_csv=None):
    self.last_days = last_days
    if path_csv:
      self.df = pd.read_csv(path_csv, index_col='date', parse_dates=True)
    else:
      self.json_response = self.__get_json_data(self.last_days)
      self.df = self.__parse_json_response(self.json_response)

  def get_country(self, country):
    df = self.df[self.df["country"] == country]
    return df

  def plot_all_country_stats(self, country):
    self.df[self.df['country'] == country]["cases"].plot(figsize=(16,8))
    self.df[self.df['country'] == country]["recovered"].plot()
    self.df[self.df['country'] == country]["deaths"].plot()
    plt.legend()

  def make_daily(self, country='Brazil'):
    df_tmp = self.df[self.df['country'] == country]
    df_tmp.sort_index(inplace=True)
    cases_daily = []
    deaths_daily = []
    last = 0
    for i in df_tmp['cases']:
      cases_daily.append(abs(i-last))
      last = i

    last = 0
    for i in df_tmp['deaths']:
      deaths_daily.append(abs(i-last))
      last = i

    df_tmp["cases_daily"] = cases_daily
    df_tmp["deaths_daily"] = deaths_daily
    
    return df_tmp
    
  
  def mort_per_100(self, df):
    df = df[["deaths_daily", "cases_daily"]]
    df = pd.DataFrame(pd.to_numeric(df.sum()),dtype=np.float64).transpose()
    return np.round(100*df["deaths_daily"]/df["cases_daily"],2)
    

  def __get_json_data(self, last_days):
    print('Collecting data...')
    response = requests.get(
        "https://corona.lmao.ninja/v2/historical?lastdays={}".format(last_days)
    ) 
    json_response = response.json()
    print(('Data of last {} days collected...'.format(last_days)))
    return json_response            

  def __parse_json_response(self, json_response):
    print('Parsing data...')
    pandas_data = []
    for data in json_response:
      for i, kv in enumerate(data['timeline']['cases'].items()):
        d = {}
        d['country'] = data['country']
        d['date'] = kv[0]
        d['cases'] = kv[1]
        d['deaths'] = list(data['timeline']['deaths'].values())[i]
        d['recovered'] = list(data['timeline']['recovered'].values())[i]
        pandas_data.append(d)
      
    print('COVID-19 data parsed')
    print(pandas_data)
    df = pd.DataFrame(pandas_data)
    df['date'] =  pd.to_datetime(df['date'],
                              format='%m/%d/%Y', infer_datetime_format=True )
    df = df.set_index('date')   
    return df

def get_mean_df(df, start, end, by_hour=False):
  date_range = pd.date_range(start=start, end=end)
  mean = []
  for date in date_range:
    try:
      d = str(date).split()[0]
      if by_hour: 
        serie =  df.loc['{} 00:00:00'.format(d):'{} 23:59:00'.format(d)].mean().to_dict()
      else:
        serie = df.loc['{}'.format(d)].mean().to_dict()
      serie['date'] = date
    except Exception as e:
      # print(traceback.print_exc())
      continue
    mean.append(serie)
  df_mean = pd.DataFrame(mean)
  df_mean = df_mean.set_index('date')
  df_mean.index = pd.to_datetime(df_mean.index)
  return df_mean  


# In[4]:


tracker = VirusTracker(path_csv=path_csv+'covid_all.csv')


# In[122]:


brazil = tracker.make_daily()
usa = tracker.make_daily('USA')
india = tracker.make_daily('India')
print((tracker.mort_per_100(brazil)[0]))
print((tracker.mort_per_100(usa)[0]))
print((tracker.mort_per_100(india)[0]))


# In[6]:


tracker.df.loc['2021-04-07'].nlargest(3, 'cases')


# ## 4.1  Qual é a taxa de contágio do COVID no Brasil?

# In[123]:


brazil = tracker.make_daily()
usa = tracker.make_daily('USA')
india = tracker.make_daily('India')
france = tracker.df[tracker.df['country'] == 'France']
russia = tracker.make_daily('Russia')


# In[8]:


brazil['cases'].plot(label='casos', title='Casos de COVID no Brasil', xlabel='mês', ylabel='Nº de casos (x 10 milhões)', grid=True)


# In[12]:


brazil['logInfection'].plot(label='casos', title='Casos de COVID no Brasil', xlabel='mês', ylabel='log(Nº de casos)', grid=True)


# In[13]:


from sklearn.linear_model import LinearRegression

def get_exponential_growth(df, column='cases', new_column='logInfection'):
  df[new_column] = np.log(df[column])
  df[new_column].replace([np.inf, -np.inf], np.nan, inplace=True)
  df.dropna(inplace=True)
  df = df.loc['2020-02-26':]
  X = np.array([i for i in range(len(df[new_column]))]).reshape((-1, 1))
  y = np.array(df[new_column])
  lr = LinearRegression()
  lr.fit(X, y)
  return np.exp(lr.intercept_)



# # 4.2 Existe semelhança nas taxas de contágio entre os 3 paises com maior número de casos?

# In[15]:


print(usa_growth)
print(brazil_growth)
print(india_growth)


# In[16]:


usa['logInfection'].plot() 
india['logInfection'].plot() 
brazil['logInfection'].plot(xlabel='mês', ylabel='log(Nº de infectados)')
plt.legend(['EUA','Índia', 'Brasil'])

# france.logInfection.plot()
# tracker.df[tracker.df['country'] == 'France']


# # 4.3 Qual é a taxa de mortalidade do COVID no Brasil?

# In[18]:


brazil['deaths'].plot(label='mortes', title='Mortes por COVID no Brasil', xlabel='mês', ylabel='Nº de mortes', grid=True)


# In[19]:


brazil['logDeaths'].plot(label='mortes', title='Mortes por COVID no Brasil', xlabel='mês', ylabel='log(Nº de mortes)', grid=True)


# # 4.4 Existe semelhança nas taxas de mortalidade entre os 3 países com maior número de casos?

# In[20]:


print(usa_death_growth)
print(brazil_death_growth)
print(india_death_growth)


# In[21]:


usa['logDeaths'].plot() 
india['logDeaths'].plot() 
brazil['logDeaths'].plot(xlabel='mês', ylabel='log(Nº de infectados)')
plt.legend(['EUA','Índia', 'Brasil'])


# # Dataset Tempo

# ## Download data

# In[58]:


def get_waether_data(country):
  r = requests.get('https://api.oikolab.com/weather',
                  params={'param': [
                                    'temperature',
                                    'urban_temperature',
                                    'wind_speed',
                                    'humidex',
                                    'soil_temperature_level_1',
                                    'windspeed',
                                    'wind_direction',
                                    'surface_solar_radiation',
                                    'surface_thermal_radiation',
                                    'surface_direct_solar_radiation',
                                    'direct_normal_solar_radiation',
                                    'surface_diffuse_solar_radiation',
                                    'relative_humidity',
                                    'surface_pressure',
                                    'total_cloud_cover',
                                    'total_precipitation',
                                    'snowfall'
                                      ],
                          'start': '2020-01-01',
                          'end': '2021-04-08',
                          'location':country,
                          'api-key': '871088fe53894bc9b1e9cba9663f9fde'}
                  )
  jst = r.text
  print(jst)
  import json
  weather_data = json.loads(r.json()['data'])
  df = pd.DataFrame(index=pd.to_datetime(weather_data['index'], 
                                        unit='s'),
                    data=weather_data['data'],
                    columns=weather_data['columns'])
  return df


# In[59]:


df_brazil_weather = get_waether_data('Brazil')
df_brazil_weather.head()


# In[60]:


df_usa_weather = get_waether_data('USA')
df_usa_weather.head()


# In[61]:


df_india_weather = get_waether_data('India')
df_india_weather.head()


# In[62]:


df_brazil_weather.to_csv(path_csv+'wheater_20200101_20210401_Brazil_D.csv')
df_usa_weather.to_csv(path_csv+'wheater_20200101_20210401_USA_D.csv')
df_india_weather.to_csv(path_csv+'wheater_20200101_20210401_India_D.csv')


# ## Load data

# In[63]:


df_brazil_weather = pd.read_csv(path_csv+'wheater_20200101_20210401_Brazil_D.csv', index_col='Unnamed: 0', parse_dates=True)
df_usa_weather = pd.read_csv(path_csv+'wheater_20200101_20210401_Brazil_D.csv', index_col='Unnamed: 0', parse_dates=True)
df_india_weather = pd.read_csv(path_csv+'wheater_20200101_20210401_Brazil_D.csv', index_col='Unnamed: 0', parse_dates=True)

df_brazil_weather = get_mean_df(df_brazil_weather, start='2020-02-07', end='2021-04-07')
df_usa_weather = get_mean_df(df_usa_weather, start='2020-02-07', end='2021-04-07')
df_india_weather = get_mean_df(df_india_weather, start='2020-02-07', end='2021-04-07')


# In[68]:


df_brazil_weather.info()


# In[70]:


brazil_weather = brazil.join(df_brazil_weather)
usa_weather = usa.join(df_usa_weather)
india_weather = india.join(df_india_weather)

# brazil.head(50)
# df_weather.loc['2020-02-26':'2021-04-08']['urban_temperature (degC)'].plot()
# brazil['cases'].plot()


# In[71]:


brazil_weather.to_csv(path_csv+'brazil_cov+wheater.csv')
usa_weather.to_csv(path_csv+'usa_cov+wheater.csv')
india_weather.to_csv(path_csv+'india_cov+wheater.csv')


# # Dataset Mobilidade

# ## download

# In[ ]:


get_ipython().system('wget https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv')
get_ipython().system('wget https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip')


# In[ ]:


get_ipython().system('mv Global_Mobility_Report.csv drive/MyDrive/02.TCCs/3.TCC\\ Pós\\ Puc/data/Global_Mobility_Report.csv')
get_ipython().system('mv Region_Mobility_Report_CSVs.zip drive/MyDrive/02.TCCs/3.TCC\\ Pós\\ Puc/data/Region_Mobility_Report_CSVs.zip')
#!unzip -x Region_Mobility_Report_CSVs.zip


# ## read csv region

# In[90]:


mob_brazil = pd.read_csv(path_csv+'Region_Report_Google/2021_BR_Region_Mobility_Report.csv',
                     index_col='date', parse_dates=True)

mob_usa = pd.read_csv(path_csv+'Region_Report_Google/2021_US_Region_Mobility_Report.csv',
                     index_col='date', parse_dates=True)

mob_india = pd.read_csv(path_csv+'Region_Report_Google/2021_IN_Region_Mobility_Report.csv',
                     index_col='date', parse_dates=True)
# mob_br.head()


# ## read csv global

# In[91]:


mob = pd.read_csv(path_csv+'Global_Mobility_Report.csv',
                  index_col='date', parse_dates=True)


# In[113]:


br_mob = mob[mob['country_region'] == 'Brazil']
br_mob.sort_index(inplace=True)

usa_mob = mob[mob['country_region'] == 'United States']
usa_mob.sort_index(inplace=True)

india_mob = mob[mob['country_region'] == 'India']
india_mob.sort_index(inplace=True)


# In[105]:


mob['country_region'].unique()


# In[93]:


def sanitize_mobility(df):
  df = df[['retail_and_recreation_percent_change_from_baseline',
        'grocery_and_pharmacy_percent_change_from_baseline',
        'parks_percent_change_from_baseline',
        'transit_stations_percent_change_from_baseline',
        'workplaces_percent_change_from_baseline',
        'residential_percent_change_from_baseline']]
      
  df = df.fillna(df.mean())

  df_mean = get_mean_df(df, start='2020-02-15', end='2021-04-08')
  return df_mean


# In[107]:


usa.head()


# In[124]:


br_mean = sanitize_mobility(br_mob)
usa_mean = sanitize_mobility(usa_mob)
india_mean = sanitize_mobility(india_mob)


# In[128]:


brazil_mob = brazil.join(br_mean)
usa_mob = usa.join(usa_mean)
india_mob = india.join(india_mean)


# ## 4.7 Quais são os indicadores de mobilidade que mais se correlacionam com o número de infectados?

# In[127]:


from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor

def build_decision_tree(df, title):
  df=df.drop('country', axis=1)
  normalized_df=(df-df.min())/(df.max()-df.min())
  droped_nan = normalized_df.fillna(0)
  
  y = droped_nan["cases"]
  to_drop = ['cases', 'deaths', 'recovered', 'cases_daily', 'deaths_daily']
  X = droped_nan.drop(to_drop, axis=1)

  train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, random_state = 42)
  
  rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
  # Train the model on training data
  rf.fit(train_features, train_labels);

  # Use the forest's predict method on the test data
  predictions = rf.predict(test_features)
  # Calculate the absolute errors
  errors = abs(predictions - test_labels)
  # Print out the mean absolute error (mae)
  print(('MAE:', round(np.mean(errors), 2)))

  features = df.columns
  features = features.drop(to_drop)
  importances = rf.feature_importances_
  indices = np.argsort(importances)

  plt.title(title)
  plt.barh(list(range(len(indices))), importances[indices], color='#8f63f4', align='center')
  plt.yticks(list(range(len(indices))), features[indices])
  plt.xlabel('Importância relativa')
  plt.show()


# In[130]:


build_decision_tree(brazil_mob, 'Importâncias do dataset do Brasil ')


# ## 4.8 O indicador de mobilidade mais relevante no Brasil também se repete nos outros dois países? 

# In[131]:


build_decision_tree(usa_mob, 'Importâncias do dataset do EUA')


# In[132]:


build_decision_tree(india_mob, 'Importâncias do dataset do Índia ')


# ## 4.5 Qual é o impacto do clima no contágio brasileiro, quais são as variáveis climáticas que mais se relacionam com o número de contagiados e qual a força desta correlação?

# ### Corr weather

# In[86]:


import seaborn as sn
import matplotlib.pyplot as plt

def get_correlation_matrix(df, title):
  fields = [
            'cases', 'temperature (degC)', 'urban_temperature (degC)',
            'wind_speed (m/s)', 'humidex (degC)',
            'soil_temperature_level_1 (degC)',
            'wind_speed (m/s).1', 'wind_direction (deg)',
            'surface_solar_radiation (W/m^2)', 'surface_thermal_radiation (W/m^2)',
            'surface_direct_solar_radiation (W/m^2)',
            'direct_normal_solar_radiation (W/m^2)',
            'surface_diffuse_solar_radiation (W/m^2)', 'relative_humidity (0-1)',
            'surface_pressure (Pa)', 'total_cloud_cover (0-1)',
            'total_precipitation (mm of water equivalent)',
            'snowfall (mm of water equivalent)']

  df_weather = df[fields]
  corrMatrix = df_weather.corr()
  corrMatrix.drop('wind_speed (m/s).1', inplace=True)
  t = corrMatrix[(corrMatrix['cases'] >= 0.20) | (corrMatrix['cases'] <= -0.20)]
  t = t.loc[:, t.index]
  sn.heatmap(t, annot=True, vmin =-1, vmax=1)
  plt.title(title, fontsize =20)
  plt.show()

# print(brazil.columns)


# In[87]:


get_correlation_matrix(brazil, title='Brasil')


# ## 4.6 As mesmas correlações observadas na questão 4.5 também são observadas nos outros 2 países com mais casos de infecção? 

# In[88]:


get_correlation_matrix(usa, title="Estados Unidos")


# In[89]:


get_correlation_matrix(india, title='Índia')

