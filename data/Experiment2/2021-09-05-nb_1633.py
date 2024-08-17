#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install covid
#!pip install iso3166
#!pip install covid19dh
#!pip install streamlit


# In[2]:


#https://ahmednafies.github.io/covid/
#!pip install covid

# Initial imports
import warnings
warnings.filterwarnings('ignore')
import os
import plotly
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
# Initialize the Panel Extensions (for Plotly)
import panel as pn
import param
pn.extension('plotly')
pn.extension()
import hvplot.pandas
import matplotlib.pyplot as plt
import os
from pathlib import Path
from dotenv import load_dotenv
import ipywidgets as widgets
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime as dt
import streamlit as st


import nltk as nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
from newsapi import NewsApiClient
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model


# ### Global Parameters

# In[3]:


# Set the random seed for reproducibility
# Note: This is used for model prototyping, but it is good practice to comment this out and run multiple experiments to evaluate your model.
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(2)

# Establish API 
api_key = "f49e8065af064801ba16d9322ddcca43"
newsapi = NewsApiClient(api_key=api_key)
vaccine_array = ["Pfizer", "Moderna", "Janssen", "Johnson&Johnson", "Sinopharm", "Vero Cell", "Sputnik V", "Sinovac"]


# In[4]:


def download_fully_vacinated_data(country):
    url="https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv"
    df=pd.read_csv(url)
    # Set the Date column to datetime format
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "location", "people_fully_vaccinated_per_hundred"]]
    df = df.set_index('date')
    df = df.dropna()
    data = df.groupby('location')
    return data.get_group(country)

def download_vacination_data(country):
    url="https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv"
    df=pd.read_csv(url)
    # Set the Date column to datetime format
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "location", "people_vaccinated_per_hundred"]]
    df = df.set_index('date')
    df = df.dropna()
    data = df.groupby('location')
    return data.get_group(country)

def download_covid_new_cases(country):
    url="https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
    df=pd.read_csv(url)
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "location", "new_cases"]]
    df = df.set_index("date")
    df = df.dropna()
    df = df.loc[:, (df != 0).any(axis=0)]
    data = df.groupby('location')
    return data.get_group(country)


# In[ ]:





# In[ ]:





# #### Create the Features `X` and Target `y` Data
# 
# Use the `window_data()` function bellow, to create the features set `X` and the target vector `y`. Define a window size of `30` days and use the column of "people_vaccinated_per_hundred" and "people_fully_vaccinated_per_hundred" as feature and target column; this will allow the model to predict vaccination date.

# In[5]:


# Scale Data
def scale_train_data(df):
    # Use the MinMaxScaler to scale data between 0 and 1.
    scaler = MinMaxScaler()   
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    for col in df.columns:
        if col == "new_cases":
            df[["new_cases"]] = scaler.fit_transform(df[["new_cases"]])
        if col == "people_fully_vaccinated_per_hundred":
            df[["people_fully_vaccinated_per_hundred"]] = scaler.fit_transform(df[["people_fully_vaccinated_per_hundred"]])
        if col == "people_vaccinated_per_hundred":
            df[["people_vaccinated_per_hundred"]] = scaler.fit_transform(df[["people_vaccinated_per_hundred"]])
        
    return df, scaler

def scale_test_data(df, scaler):
    for col in df.columns:
        if col == "new_cases":
            df[["new_cases"]] = scaler.transform(df[["new_cases"]])
        if col == "people_fully_vaccinated_per_hundred":
            df[["people_fully_vaccinated_per_hundred"]] = scaler.fit_transform(df[["people_fully_vaccinated_per_hundred"]])
        if col == "people_vaccinated_per_hundred":
            df[["people_vaccinated_per_hundred"]] = scaler.fit_transform(df[["people_vaccinated_per_hundred"]])
    
    return df


def split_train_test(df):    
    train_size  = int(0.7 * len(df))
    test_size = len(df) - train_size
    
    train = df[: train_size]
    test = df[train_size:]
    
    return train, test

# This function accepts the column number for the features (X) and the target (y)
# It chunks the data up with a rolling window of Xt-n to predict Xt
# It returns a numpy array of X any y
def window_data(df, window, feature_col_number, target_col_number):
    
    X = []
    y = []

    for i in range(len(df) - window - 1):
        features = df.iloc[i:(i + window), feature_col_number]
        target = df.iloc[(i + window), target_col_number]
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y)

def reshape_data(X_train, X_test):
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    return X_train, X_test

# Build the LSTM model. 
def lstm_model_ini(X_train, y_train, window, epochs, batch_size):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    model = Sequential()
    number_units = 10
    
    model.add(LSTM(number_units, input_shape=(1, window)))
    
    # Output layer
    model.add(Dense(units=1))
    # Compile the model
    model.compile(optimizer="adam", loss="mean_squared_error")
    
    model.fit(X_train, y_train, epochs=epochs, shuffle=False, batch_size=batch_size, verbose=0)
    model.reset_states()
    return model

def test_prediction(df, model, X_test, y_test, scaler):
    predicted = model.predict(X_test)
    predicted_value = scaler.inverse_transform(predicted)
    real_value = scaler.inverse_transform(y_test.reshape(-1, 1))
    compare_df = pd.DataFrame({
                "Real": real_value.ravel(),
                "Predicted": predicted_value.ravel()}, index = df.index[-len(real_value): ]) 
    return compare_df

def create_model(df):
    # Split train-test
    train, test = split_train_test(df)
    
    # Scale the data:
    train, scaler = scale_train_data(train)
    test = scale_test_data(test, scaler)
    
    X_train, y_train = window_data(train, window, 1,1)
    X_test, y_test = window_data(test, window, 1,1)
        
    X_train, X_test = reshape_data(X_train, X_test)

    # Create LTSM Model
    model = lstm_model_ini(X_train, y_train, window, epochs, batch_size)
    model.evaluate(X_test, y_test, verbose=0)
    # Test the model
    compare_df = test_prediction(df, model, X_test, y_test, scaler)
    plot = compare_df.hvplot(title="Model Performance")
    model.reset_states()
    return model, scaler, compare_df, plot


def scale_main_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    for col in df.columns:
        if col == "new_cases":
            df[["new_cases"]] = scaler.fit_transform(df[["new_cases"]])
        if col == "people_fully_vaccinated_per_hundred":
            df[["people_fully_vaccinated_per_hundred"]] = scaler.fit_transform(df[["people_fully_vaccinated_per_hundred"]])
        if col == "people_vaccinated_per_hundred":
            df[["people_vaccinated_per_hundred"]] = scaler.fit_transform(df[["people_vaccinated_per_hundred"]])    
    return df, scaler
def reverse_scale_data(df, scaler):
    for col in df.columns:
        if col == "new_cases":
            df[["new_cases"]] = scaler.inverse_transform(df[["new_cases"]])
        if col == "people_fully_vaccinated_per_hundred":
            df[["people_fully_vaccinated_per_hundred"]] = scaler.inverse_transform(df[["people_fully_vaccinated_per_hundred"]])
        if col == "people_vaccinated_per_hundred":
            df[["people_vaccinated_per_hundred"]] = scaler.inverse_transform(df[["people_vaccinated_per_hundred"]])
    
    return df

def predict_main_data(df, model, prediction_days):
    
    for i in range(prediction_days+1):
        df, scaler = scale_main_data(df)

        # Shape data of the last window period 
        latest = [df.iloc[len(df) - window: len(df),1]]
        latest = np.array(latest)
        latest = np.reshape(latest, (latest.shape[0], 1, latest.shape[1]))

        # Make Prediction
        pred = model.predict(latest)
        forecast = scaler.inverse_transform(pred)[0][0]
        #forecast = pred[0][0]

        # Add to main dataframe
        df = reverse_scale_data(df, scaler)
        df.loc[df.index[-1] + dt.timedelta(days=1)] = [country, forecast]
        
    return df.hvplot()

# Fetch the Covid news articles

def get_news(q, language):
    news_articles = newsapi.get_everything(q=q, language=language)

    return news_articles

# Sentiment DataFrames
def create_sentiment_df(news):
    articles = []
    for article in news["articles"]:
        try:
            title = article["title"]
            description = article["description"]
            text = article["content"]
            date = article["publishedAt"][:10]
            sentiment = analyzer.polarity_scores(str(title)+" "+str(description) +" "+ str(title))
            compound = sentiment["compound"]
            pos = sentiment["pos"]
            neu = sentiment["neu"]
            neg = sentiment["neg"]
            articles.append({
                "title": title,
                "description": description,
                "text": text,
                "date": date,
                "compound": compound,
                "positive": pos,
                "negative": neg,
                "neutral": neu
            })
        except AttributeError:
            pass
    return pd.DataFrame(articles) 

def mean_sentiment_scores(df):
    positive_score = df['positive'].mean()
    negative_score = df['negative'].mean()
    return positive_score, negative_score

def create_sentiment_plot(vaccine_array, language):
    sen_score_df = pd.DataFrame(columns=['positive', 'negative'])
    for vac in vaccine_array:
        news = get_news(vac, language)
        df = create_sentiment_df(news)
        pos, neg = mean_sentiment_scores(df)
        sen_score_df.loc[vac] = [pos, neg]
        plot = sen_score_df.hvplot.bar(rot=45, stacked=True, legend='right').opts(yformatter="%.0f", width=800, xlabel="Vaccine", ylabel="Sentiment Score", title="Sentiment toward Vaccine types")
    return plot


# In[6]:


window = 5
epochs = 990
batch_size = 15
number_units = 150
prediction_days = 21
country= "England"

days_forecast = 10

df = download_fully_vacinated_data(country)
model, scaler, compare_df, plot = create_model(df)
plot


# In[8]:


# Make Prediction
vac_df = download_fully_vacinated_data(country)
pre_plot = predict_main_data(vac_df, model, prediction_days)
pre_plot


# In[9]:


plot = create_sentiment_plot(vaccine_array, 'en')
plot


# In[ ]:





# ### Input for dashboard

# In[10]:


# Create Widget Elements
countries = ['Afghanistan', 'Africa', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Asia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo', 'Costa Rica', "Cote d'Ivoire", 'Croatia', 'Cuba', 'Cyprus', 'Czechia', 'Democratic Republic of Congo', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia', 'Europe', 'European Union', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hong Kong', 'Hungary', 'Iceland', 'India', 'Indonesia', 'International', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico', 'Micronesia (country)', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North America', 'North Macedonia', 'Norway', 'Oceania', 'Oman', 'Pakistan', 'Palau', 'Palestine', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'Rwanda', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South America', 'South Korea', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor', 'Togo', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Vatican', 'Venezuela', 'Vietnam', 'World', 'Yemen', 'Zambia', 'Zimbabwe']
country_list = pn.widgets.Select(name='Country:', options=countries)
window_input = pn.widgets.IntInput(name='How many days window', start=7, end=30, step=1, value=7)
epochs_input = pn.widgets.IntInput(name='Epochs', value=10, step=100, start=100, end=1000)
batch_size_input = pn.widgets.IntInput(name='Batch Size', start=1, end=15, step=1, value=1)
number_units_input = pn.widgets.IntInput(name='Units', value=50, step=10, start=0, end=1000)
day_prediction_input = pn.widgets.IntInput(name='How many days of prediction?', start=0, end=30, step=3, value=7)
model_functions_list = pn.widgets.Select(name='Prediction of...:', options=["---", "Cases", "One Dose", "Fully Vacinated"], value="---")
#predict_functions_list = pn.widgets.Select(name='Prediction of...:', options=["---", "Cases", "One Dose", "Fully Vacinated"], value="---")


# In[17]:


@pn.depends(country_list, window_input, epochs_input, batch_size_input, number_units_input, model_functions_list)
def disp_create_model_data(country_list, window_input, epochs_input, batch_size_input, number_units_input, model_functions_list):
    if model_functions_list == "Cases":
        df = download_covid_new_cases(country_list)
        model, scaler, compare_df, compare_plot = create_model(df)
        # save model to single file
        model.save('new_case_model.h5')
        return compare_plot
    elif model_functions_list == "One Dose":
        df = download_vacination_data(country_list)
        model, scaler, compare_df, compare_plot = create_model(df)
        # save model to single file
        model.save('one_dose_model.h5')
        return compare_plot
    elif model_functions_list == "Fully Vacinated":
        df = download_fully_vacinated_data(country_list)
        model, scaler, compare_df, compare_plot = create_model(df)
        # save model to single file
        model.save('fully_vaccinated_model.h5')
        return compare_plot
@pn.depends(model_functions_list, country_list, day_prediction_input)    
def saved_model_prediction(model_functions_list, country_list, day_prediction_input):
    if model_functions_list == "Cases":
        saved_model = load_model('new_case_model.h5')
        df = download_covid_new_cases(country_list)
        pred_plot = predict_main_data(df, saved_model, prediction_days)
        return pred_plot
    elif model_functions_list == "One Dose":
        saved_model = load_model('one_dose_model.h5')
        df = download_vacination_data(country_list)
        pred_plot = predict_main_data(df, saved_model, prediction_days)
        return pred_plot
    elif model_functions_list == "Fully Vacinated":
        saved_model = load_model("fully_vaccinated_model.h5")
        df = download_fully_vacinated_data(country_list)
        pred_plot = predict_main_data(df, saved_model, prediction_days)
        return pred_plot


# In[15]:


# Input Parameters
options_input = pn.Row(country_list, model_functions_list, align="center")
model_input1 = pn.Row(window_input, epochs_input, align="center")
model_input2 = pn.Row(batch_size_input, number_units_input, align="center")
option_output = pn.Row(disp_create_model_data, align="center")
prediction_output = pn.Row(day_prediction_input, align="center")
model_configuration_input = pn.Column(model_input1, model_input2, options_input, option_output, align="center")
saved_model_configuration = pn.Row(day_prediction_input)
saved_model_output = pn.Row(saved_model_prediction)
prediction = pn.Column(saved_model_configuration, saved_model_prediction)


# In[16]:


title = "#COVID Forecast Dasboard"
welcome = "This Dashboard provide the function of building and training model for Covid cases/Vacination Projection"

# Create a layout for the dashboard

sentiment_data_row = pn.Row(create_sentiment_plot(vaccine_array, 'en'))


dashboard = pn.WidgetBox(pn.Column(title,
                        pn.WidgetBox(pn.Tabs(    
                            ("Sentimment toward Vacine types", sentiment_data_row),
                            ("COVID-19 Forecast", model_configuration_input),
                            ("Prediction", prediction)
                            )), align="center"))

dashboard.servable()


# In[ ]:





# In[ ]:





# In[ ]:





# ##### 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




