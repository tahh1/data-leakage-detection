#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[11]:


df = pd.read_csv('./Datasets/canada_per_capita_income.csv')
df.head()


# In[20]:


plt.scatter(df["year"], df["per capita income (US$)"], color="red", marker=".")
plt.xlabel("Year")
plt.ylabel("Income")
plt.show()


# In[23]:


model = linear_model.LinearRegression()
model.fit(df[["year"]], df["per capita income (US$)"])


# In[27]:


model.predict([[2021], [2020]])


# In[33]:


print(model.coef_)
print(model.intercept_)


# In[42]:


plt.xlabel("Year", fontsize=16)
plt.ylabel("Income", fontsize=16)
plt.plot(df[["year"]], model.predict(df[["year"]]), color="red")
plt.scatter(df["year"], df["per capita income (US$)"], color="blue", marker=".")

