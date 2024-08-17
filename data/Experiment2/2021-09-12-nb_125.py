#!/usr/bin/env python
# coding: utf-8

# Добавляем файлы

# In[ ]:





# In[ ]:


import pandas as pd
df = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')


# In[ ]:


df


# In[ ]:


df2


# In[ ]:


df.info()


# In[ ]:


df2.info()


# In[ ]:


y = df['critical_temperature'].astype(float)
X = df.drop('critical_temperature', axis=1)
X


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = scaler.fit_transform(X)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[ ]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
answers_pred = model.predict(X_train)

print((model.score(X_test, y_test)))


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
print((model.score(X_test, y_test)))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
print((model.score(X_test, y_test)))


# In[ ]:


predictions = model.predict(df2)
file_pred = pd.DataFrame({'Index': [i for i in range(len(predictions))],'critical_temp': predictions})


# In[ ]:


from google.colab import files
file_pred.to_csv('Dynnik.csv', index=False)
files.download("Dynnik.csv")

