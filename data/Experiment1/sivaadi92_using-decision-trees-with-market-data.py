#!/usr/bin/env python
# coding: utf-8

# # This notebook seeks to train the best decision tree using only the market data

# ## Methodology:
# 1.  The y-label given to us is the 10 day forward looking market returns.
# 2.  We transform this into a classification problem by establishing thresholds.
# 3.  If the market returns are greater than that threshold we assign a +1 and if it is lesser than a threshold we assign -1. Otherwise it is assigned a label of 1.
# 4. It is trained on the first 80 % and tested to identify the best hyper-parameters on the remaining 20%.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Ctrl + Shift + P
import os
print((os.listdir("../input")))

toy = True
# Any results you write to the current directory are saved as output.


# In[ ]:


from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()

# train_my_model(market_train_df, news_train_df)


# In[ ]:


print((market_train_df.shape, news_train_df.shape))
print((market_train_df.columns))
print((news_train_df.columns))


# In[ ]:


# We will reduce the number of samples for memory reasons
toy = False

if toy:
    market_train_df = market_train_df.tail(100)
    news_train_df = news_train_df.tail(300)
else:
    market_train_df = market_train_df.tail(3_000_000)
    news_train_df = news_train_df.tail(6_000_000)



# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier



# In[ ]:


market_train_df['time'] = market_train_df['time'].dt.floor('1D')
market_train_df = market_train_df.reset_index(drop=True)


# * X_pruned only contains the labels 'returnsOpenPrevMktres10' and 'returnsOpenPrevMktres1'.
# * Y_pruned is the lable containing the forward looking return.

# In[ ]:


# market_train_df.head(5)

X_pruned = market_train_df[['returnsOpenPrevMktres10', 'returnsOpenPrevMktres1']]
Y_pruned = market_train_df[['returnsOpenNextMktres10']]

start_test_data = int(0.8 * market_train_df.shape[0])
X_train = X_pruned.iloc[:start_test_data]
Y_train = Y_pruned.iloc[:start_test_data]

X_test = X_pruned.iloc[start_test_data:].reset_index(drop=True)
Y_test = Y_pruned.iloc[start_test_data:].reset_index(drop=True)
Y_test_metadata = market_train_df.iloc[-start_test_data:][['time', 'universe']].reset_index(drop=True)



# This marks the end of the part where we get the data and divide into the train and test samples.

# In the below cell, we assign a confidence value +1 for 'returnsOpenNextMktres10' > threshold and -1 for 'returnsOpenNextMktres10' < threshold

# In[ ]:


Y_train.hist(bins=10)


# In[ ]:


#Hyperparameter : threshold
def assignConfidence(pthreshold, nthreshold, Y):
    Y[(Y['returnsOpenNextMktres10'] > pthreshold)]  = 1.0
    Y[(Y['returnsOpenNextMktres10'] < -nthreshold)] = -1.0
    Y[(Y != 1.0) & (Y != -1.0)] = 0.0
    return Y

def train_model(X, Y, **kwargs):
    decTree = DecisionTreeClassifier(**kwargs)
    X = X.ffill()
    decTree.fit(X, Y)
    return decTree

def overridden_predict(trained_model, X_test):
    X_test = X_test.ffill()
    y_pred = trained_model.predict(X_test)
    y_pred = pd.DataFrame({'confidenceValue':y_pred})
    return y_pred

def sigma_score(Y_pred, Y_test):
    score = Y_pred['confidenceValue'] * Y_test['returnsOpenNextMktres10'] * Y_test_metadata['universe']
    score = score.to_frame('score')
    score['time'] = Y_test_metadata['time'].values
    score_per_day = score.groupby('time').sum()
    try:
        sigma_score = score_per_day['score'].mean() / score_per_day['score'].std()
    except:
        sigma_score = 0.0
        
    return sigma_score


# The below cell;
# Input: X test data 
# Output : A dataframe with "confidence values''

# In[ ]:


# positive_threshold = [0.05, 0.075, 0.1]
# negative_threshold = [0.05, 0.075, 0.1]
# depth_levels = [3, 5, 9, 15, 19, None]
# max_sig_score = -100
# for pthresh in positive_threshold:
#     for nthresh in negative_threshold:
#         for max_depth in depth_levels:
#             Y_train_modified = Y_train.copy()
#             kwargs = {'max_depth':max_depth}
#             Y_train_modified = assignConfidence(pthresh, nthresh, Y_train_modified)
#             trained_model = train_model(X_train, Y_train_modified, **kwargs)
#             Y_pred = overridden_predict(trained_model, X_test)
#             sig_score = sigma_score(Y_pred=Y_pred, Y_test=Y_test)
#             print("Sigma score for phresh, nthresh and max depth is " + '{} {} {} '.format(pthresh, nthresh, max_depth)+
#                  "{}".format(sig_score))
#             print(trained_model.get_params()['max_depth'])
#             if sig_score > max_sig_score:
#                 best_pthresh = pthresh
#                 best_nthresh = nthresh
#                 if max_depth:
#                     best_max_depth = min(max_depth, trained_model.get_params()['max_depth'])
#                 else:
#                     best_max_depth = trained_model.get_params()['max_depth']
#                 max_sig_score = sig_score

# print("The best parameters are pthresh, nthresh and max depth are " + '{} {} {} '.format(best_pthresh, best_nthresh, best_max_depth))



# We take the best parameters and build our model. We have to join the training and testing set together as we don't want to loose the training on the testing set.

# In[ ]:


Y = pd.concat([Y_train, Y_test])
X = pd.concat([X_train, X_test])


# In[ ]:





# In[ ]:


best_pthresh = 0.05
best_nthresh = 0.05
best_max_depth = 9

Y = pd.concat([Y_train, Y_test])
Y_modified = Y.copy()
Y_modified = assignConfidence(best_pthresh, best_nthresh, Y_modified)
kwargs = {'max_depth':best_max_depth}
trained_model = train_model(X, Y_modified, **kwargs)
Y_pred = overridden_predict(trained_model, X)
# sig_score = sigma_score(Y_pred=Y, Y_test=Y_test)


# This serves as a boundary between the training and the actual submission.

# In[ ]:


def make_predictions(trained_model,predictions_template_df, market_obs_df, news_obs_df):
    sample = market_obs_df[['returnsOpenPrevMktres10', 'returnsOpenPrevMktres1']]
    sample = sample.ffill()
#     y_pred = lm.predict(sample)
    y_pred = trained_model.predict(sample)
    predictions_template_df.confidenceValue = y_pred.clip(-1, 1)


# In[ ]:


days = env.get_prediction_days()


# In[ ]:


for (market_obs_df, news_obs_df, predictions_template_df) in days:
    make_predictions(trained_model, predictions_template_df, market_obs_df, news_obs_df)
    env.predict(predictions_template_df)

print('Done!')



# In[ ]:


env.write_submission_file()
print("Fourth run")

