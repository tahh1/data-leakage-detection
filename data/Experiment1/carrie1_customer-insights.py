#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the data, specify data types
import pandas as pd
df = pd.read_csv('../input/data.csv',encoding="ISO-8859-1",dtype={'CustomerID': str,'InvoiceID': str})
df.InvoiceDate = pd.to_datetime(df.InvoiceDate, format="%m/%d/%Y %H:%M")
df.info() 


# In[2]:


df.head()


# In[3]:


df.describe()


# In[4]:


#remove the negative values and replace with nan
import numpy as np
df[df['Quantity'] < 0] = np.nan
df[df['UnitPrice'] < 0] = np.nan
df.describe()


# In[5]:


#get the total spent for each line item
df['total_dollars'] = df['Quantity']*df['UnitPrice']
df.describe()


# ## Building a customer table

# Let's aggregrate transaction data to learn more about our customers.

# In[6]:


#how many orders have they made
invoice_ct = df.groupby(by='CustomerID', as_index=False)['InvoiceNo'].count()
invoice_ct.columns = ['CustomerID', 'NumberOrders']
invoice_ct.describe()


# In[7]:


#how much money have they spent
total_spend = df.groupby(by='CustomerID', as_index=False)['total_dollars'].sum()
total_spend.columns = ['CustomerID', 'total_spent']
total_spend.describe()


# In[8]:


#how many items they bought
total_items = df.groupby(by='CustomerID', as_index=False)['Quantity'].sum()
total_items.columns = ['CustomerID', 'NumberItems']
total_items.describe()


# In[9]:


#when was their first order and how long ago was that from the last date in file (presumably
#when the data were pulled)
earliest_order = df.groupby(by='CustomerID', as_index=False)['InvoiceDate'].min()
earliest_order.columns = ['CustomerID', 'EarliestInvoice']
earliest_order['now'] = pd.to_datetime((df['InvoiceDate']).max())
earliest_order['days_as_customer'] = 1 + (earliest_order.now-earliest_order.EarliestInvoice).astype('timedelta64[D]')
earliest_order.drop('now', axis=1, inplace=True)
earliest_order


# In[10]:


#when was their last order and how long ago was that from the last date in file (presumably
#when the data were pulled)
last_order = df.groupby(by='CustomerID', as_index=False)['InvoiceDate'].max()
last_order.columns = ['CustomerID', 'last_purchase']
last_order['now'] = pd.to_datetime((df['InvoiceDate']).max())
last_order['days_since_purchase'] = 1 + (last_order.now-last_order.last_purchase).astype('timedelta64[D]')
last_order.drop('now', axis=1, inplace=True)
last_order.head


# In[11]:


#combine all the dataframes into one
import functools
dfs = [total_spend,invoice_ct,earliest_order,last_order,total_items]
CustomerTable = functools.reduce(lambda left,right: pd.merge(left,right,on='CustomerID', how='outer'), dfs)
CustomerTable.head()


# In[12]:


#how many customers?
len(CustomerTable)


# In[13]:


CustomerTable.describe()


# In[14]:


#identify and separate big spenders, lots of orders, long-time customers, dormant customers for
#sales and marketing campaign use; need to be separate flags because they aren't all mutually
#exclusive

def big_spender(row):
    if row['total_spent'] >= 1661.64:
        return 'Yes'
    else:
        return 'No'

def many_orders(row):
    if row['NumberOrders'] >= 100:
        return 'Yes'
    else:
        return 'No'

def loyal_customer(row):
    if row['days_as_customer'] >= 326:
        return 'Yes' 
    else:
        return 'No'

def dormant_customer(row):
    if row['days_since_purchase'] >= 141:
        return 'Yes' 
    else:
        return 'No'

CustomerTable['BigSpender'] = CustomerTable.apply(big_spender, axis=1)
CustomerTable['ManyOrders'] = CustomerTable.apply(many_orders, axis=1)
CustomerTable['LoyalCustomer'] = CustomerTable.apply(loyal_customer, axis=1)
CustomerTable['DormantCustomer'] = CustomerTable.apply(dormant_customer, axis=1)

CustomerTable['OrderFrequency'] = CustomerTable['NumberOrders']/CustomerTable['days_as_customer']

CustomerTable.head(10)


# In[15]:


#look at the distributions and relationships with other continuous variables
import seaborn as sns
sns.pairplot(CustomerTable, vars=["total_spent", "NumberOrders",'days_as_customer',
                                  'days_since_purchase','NumberItems','OrderFrequency'])


# In[16]:


RF = CustomerTable[["NumberOrders",'days_as_customer','NumberItems','BigSpender','CustomerID']]
features = RF.columns[:3]
features


# In[17]:


RF['is_train'] = np.random.uniform(0, 1, len(RF)) <= .8
train, test = RF[RF['is_train']==True], RF[RF['is_train']==False]
print(('Number of observations in the training data:', len(train)))
print(('Number of observations in the test data:',len(test)))


# In[18]:


y = pd.factorize(train['BigSpender'])[0]
y[0:10] #show the first ten; 'No' = 0


# In[19]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_jobs=2)
clf.fit(train[features], y)
list(zip(train[features], clf.feature_importances_))


# In[20]:


clf.predict_proba(test[features])[0:10]


# In[21]:


test['Prediction'] = clf.predict(test[features])
test.head()

