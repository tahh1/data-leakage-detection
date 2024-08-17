#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering of PUBG data with Model Developement
# 
# I previously created a notebook going through some [exploratory anaylsys] of the PUBG data. I went through many of the different features avalailable and displayed an interesting plot describing the data and potential correlation with the target variable.
# 
# * I found that there was one missing value for the target variable and decided that this row of data should be removed, as there was only one player for the match identified by the missing value.
# 
# * I also made a few decisions about creating new features and one important way of breaking the data up to gain higher correllations with our features for seperate match types.
# 
# [exploratory anaylsys]: https://www.kaggle.com/beaubellamy/pubg-eda#

# ## Import libraries
# We import the required libraries and import the data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')


# Lets check out the data again.

# In[ ]:


train = train[0:1000000]
train.head()


# In[ ]:


test.head()


# ## Missing Data
# Based on our EDA, we found a row that had a NULL value for the target variable. We will remove the irrelevant row of data.

# In[ ]:


# Remove the row with the missing target value
train = train[train['winPlacePerc'].isna() != True]


# ## Lets Engineer some features
# We'll process the testing data the same way we do for the training data so the testing data has the same features and scaling as our training data.
# 
# ### PlayersJoined
# We can determine the number of players that joined each match by grouping the data by matchID and counting the players.

# In[ ]:


# Add a feature containing the number of players that joined each match.
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
test['playersJoined'] = test.groupby('matchId')['matchId'].transform('count')


# In[ ]:


# Lets look at only those matches with more than 50 players.
#data = train[train['playersJoined'] > 50]

#plt.figure(figsize=(15,15))
#sns.countplot(data['playersJoined'].sort_values())
#plt.title('Number of players joined',fontsize=15)
#plt.show()


# You can see that there isn't always 100 players in each match, in fact its more likely to have between 90 and 100 players. It may be benficial to normalise those features that are affected by the number of players.
# 
# ### Normalised Features
# Here, I am making the assumption that it is easier to find an enemy when there are 100 players, than it is when there are 90 players.
# 

# In[ ]:


def normaliseFeatures(train):
    train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)
    train['headshotKillsNorm'] = train['headshotKills']*((100-train['playersJoined'])/100 + 1)
    train['killPlaceNorm'] = train['killPlace']*((100-train['playersJoined'])/100 + 1)
    train['killPointsNorm'] = train['killPoints']*((100-train['playersJoined'])/100 + 1)
    train['killStreaksNorm'] = train['killStreaks']*((100-train['playersJoined'])/100 + 1)
    train['longestKillNorm'] = train['longestKill']*((100-train['playersJoined'])/100 + 1)
    train['roadKillsNorm'] = train['roadKills']*((100-train['playersJoined'])/100 + 1)
    train['teamKillsNorm'] = train['teamKills']*((100-train['playersJoined'])/100 + 1)
    train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)
    train['DBNOsNorm'] = train['DBNOs']*((100-train['playersJoined'])/100 + 1)
    train['revivesNorm'] = train['revives']*((100-train['playersJoined'])/100 + 1)

    # Remove the original features we normalised
    train = train.drop(['kills', 'headshotKills', 'killPlace', 'killPoints', 'killStreaks', 
                        'longestKill', 'roadKills', 'teamKills', 'damageDealt', 'DBNOs', 'revives'],axis=1)

    return train

train = normaliseFeatures(train)
test = normaliseFeatures(test)


# In[ ]:


train.head()


# ### TotalDistance
# An additional feature we can create is the total distance the player travels. This is a combination of all the distance features in the original data set.

# In[ ]:


# Total distance travelled
train['totalDistance'] = train['walkDistance'] + train['rideDistance'] + train['swimDistance']
test['totalDistance'] = test['walkDistance'] + test['rideDistance'] + test['swimDistance']


# # Standardize the matchType feature
# Here I decided that many of the existing 16 seperate modes of game play were just different versions of four types of game.
# 
# 1. Solo: Hunger Games style, last man/women standing.
# 2. Duo: Teams of two against all other players.
# 3. Squad: Teams of up to 4 players against All other players
# 4. Other: These modes consist of custom and special events modes

# In[ ]:


# Normalise the matchTypes to standard fromat
def standardize_matchType(data):
    data['matchType'][data['matchType'] == 'solo'] = 'Solo'
    data['matchType'][data['matchType'] == 'normal-solo'] = 'Solo'
    data['matchType'][data['matchType'] == 'solo-fpp'] = 'Solo'
    data['matchType'][data['matchType'] == 'normal-solo-fpp'] = 'Solo'
    data['matchType'][data['matchType'] == 'normal-duo-fpp'] = 'Duo'
    data['matchType'][data['matchType'] == 'normal-duo'] = 'Duo'
    data['matchType'][data['matchType'] == 'duo'] = 'Duo'
    data['matchType'][data['matchType'] == 'duo-fpp'] = 'Duo'
    data['matchType'][data['matchType'] == 'squad'] = 'Squad'
    data['matchType'][data['matchType'] == 'squad-fpp'] = 'Squad'
    data['matchType'][data['matchType'] == 'normal-squad'] = 'Squad'
    data['matchType'][data['matchType'] == 'normal-squad-fpp'] = 'Squad'
    data['matchType'][data['matchType'] == 'flaretpp'] = 'Other'
    data['matchType'][data['matchType'] == 'flarefpp'] = 'Other'
    data['matchType'][data['matchType'] == 'crashtpp'] = 'Other'
    data['matchType'][data['matchType'] == 'crashfpp'] = 'Other'

    return data


train = standardize_matchType(train)
test = standardize_matchType(test)


# In[ ]:


# We need a copy of the test data with the player id later on
test_submission = test.copy()

train = train.drop(['Id','groupId','matchId'], axis=1)
test = test.drop(['Id','groupId','matchId'], axis=1)


# In[ ]:


# We need to keep a copy of the test data for the test id's later 
train_copy = train.copy()
test_copy = test.copy()


# Now we can transform the matchTypes into dummy values so we can use them in the model.

# In[ ]:


# Transform the matchType into scalar values
le = LabelEncoder()
train['matchType']=le.fit_transform(train['matchType'])
test['matchType']=le.fit_transform(test['matchType'])


# In[ ]:


# We can do a sanity check of the data, making sure we have the new 
# features created and the matchType feature is standardised.
train.head()


# # Scale the features
# Some features have very large values and a vary large variance. For any regresion analysis, it is good practice to scale all features to similar variance. This would not be neccessary if the variance of all features was between 6 and 10.

# In[ ]:


train.describe()


# You can see most features range 0 to 100 or 1000's, but there are two features that doesn't really need scaling, VehicleDestroys and matchType, as they only range between 0 to 5, 6. Its not neccassary to scale these features, but we will any way, because it makes the code easier.

# In[ ]:


scaler = MinMaxScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)

test_scaled = pd.DataFrame(scaler.fit_transform(test), columns=test.columns)

train_scaled.head()


# In[ ]:


train_scaled.describe()


# As you can see, all the features have the same minimum, maximum and standard deviation. All our features are now normalised and scaled for modelling.
# 
# # Model Developement
# 
# The great thing about programming is that you can automate so much. That's what we are going to do here, we'll try to fit the data using a group of models with some basic settings. We could try to use a gridsearch method to search for the best hyper parameters for each model to truly define the "best" model, but this is a Kernel only competition and I dont have a vast amount of memory on my local machine (only 8GB). So my strategy here will be to;
# 
# 1. Fit a group of models using the all features available.
# 2. Fit a group of models using feature selection. We'll select our features using multicollinearity and the variance inflation factor.
# 3. Seperate the matchTypes into four seperate data sets, as described in my [exploratory analysis], and fit the group of models to each matchType data set.
#   * We'll combine the predictions of these into one submission at the end.
# 
# Each time we will pick the best model and fit the full data set to obtain the best predictions.
# 
# I would expect a linear regression model to perform reasonably well due to what we found in the [exploratory anaylsys].
# 
# 
# [exploratory anaylsys]: https://www.kaggle.com/beaubellamy/pubg-eda#

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor


# In[ ]:


# Create a master copy of the data, so we can restore the default features.
#train_master = train_scaled.copy()

# Extract the target variable.
y = train_scaled['winPlacePerc']
X = train_scaled.drop(['winPlacePerc'],axis=1)

# Split the data in to training and validation set
size = 0.3
seed = 42
   
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=size, random_state=seed)


# Lets create a function to run all the models we want. I'm typically only running this on a Dell laptop (i7 core @ 2.4GHz, 8GB RAM), so it may take an hour or two to complete. 
# 
# One way to reduce this is to just subsample the data from the beginning with `train = train[0:1000000]` or even better using random sampling without replacement to select the indecies`train[random.sample(range(0, len(train)), 1000000)]`

# In[ ]:


# The function takes the training and validation data to fit and score a group of models
def runAllModels(X_train, X_validation, Y_train, Y_validation):
        
    linear = LinearRegression(copy_X=True)
    linear.fit(X_train,Y_train)
    print(("Linear Model score: {0:.3f}%".format(linear.score(X_validation,Y_validation)*100)))

    ridge = Ridge(copy_X=True)
    ridge.fit(X_train,Y_train)
    print(("Ridge Model score: {0:.3f}%".format(ridge.score(X_validation,Y_validation)*100)))

    lasso = Lasso(copy_X=True)
    lasso.fit(X_train,Y_train)
    print(("Lasso Model score: {0:.3f}%".format(lasso.score(X_validation,Y_validation)*100)))

    elastic = ElasticNet(copy_X=True)
    elastic.fit(X_train,Y_train)
    print(("ElasticNet Model score: {0:.3f}%".format(elastic.score(X_validation,Y_validation)*100)))

    ada = AdaBoostRegressor(learning_rate=0.8)
    ada.fit(X_train,Y_train)
    print(("AdaBoostRegressor Model score: {0:.3f}%".format(ada.score(X_validation,Y_validation)*100)))

    GBR = GradientBoostingRegressor(learning_rate=0.8)
    GBR.fit(X_train,Y_train)
    print(("GradientBoostingRegressor Model score: {0:.3f}%".format(GBR.score(X_validation,Y_validation)*100)))

    forest = RandomForestRegressor(n_estimators=10)
    forest.fit(X_train,Y_train)
    print(("RandomForestRegressor Model score: {0:.3f}%".format(forest.score(X_validation,Y_validation)*100)))

    tree = DecisionTreeRegressor()
    tree.fit(X_train,Y_train)
    print(("DecisionTreeRegressor Model score: {0:.3f}%".format(tree.score(X_validation,Y_validation)*100)))


# In[ ]:


train_scaled.head()


# In[ ]:


test_scaled.head()


# In[ ]:





# In[ ]:





# In[ ]:


runAllModels(X_train, X_validation, Y_train, Y_validation)


# The linear regression models performed reasonably well, but it seems there are some better models available, Gradient Boost or Random Forest would be a good choice.
# 
# We'll Choose the **Gradient Boost Regressor** to make our predictions.

# In[ ]:


GBR = GradientBoostingRegressor(learning_rate=0.8)
GBR.fit(X,y)

predictions_all = GBR.predict(test_scaled)


# # Model Development with feature selection
# We can use Multicolinearity to see what features are collinear with each other to help decide which features to remove.
# 
# ## Multicollinearity
# [Multicollinearity] exists whenever two or more of the predictors in a regression model are moderately or highly correlated. There are two types of multicollinearity:
# 1. **Structural multicollinearity** is a mathematical artifact caused by creating new predictors from other predictors — such as, creating the predictor *x*<sup>2</sup> from the predictor *x*.
# 2. **Data-based multicollinearity** is a result of a poorly designed experiment, reliance on purely observational data, or the inability to manipulate the system on which the data are collected.
# 
# What this really means is that when predictor variables are correlated, the estimated regression coefficient of any one variable will depend on which predictor variables, and in which order, are included in the model.
# 
# ## Variance Inflation Factor
# As the name suggests, the [variance inflation factor] is a measure of how much the variance of a coefficient is inflated by when multicollinearity exists. We will use this to determine which features to keep and which to discard.
# 
# The general rul of thumb is that a VIF of 4 - 10 should be investigated further, and anything above 10 indicates a serious multicollinearity that needs to be corrected. We will correct it by removeing the features with a VIF >= 10
# 
# [Ordinal regresion]: https://en.wikipedia.org/wiki/Ordinal_regression
# [Multicollinearity]: https://onlinecourses.science.psu.edu/stat501/node/344/
# [variance inflation factor]: https://en.wikipedia.org/wiki/Variance_inflation_factor

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def display_VIF(data):
    x_features=list(data)
    data_mat = data[x_features].as_matrix()                                                                                                              
    vif = [ variance_inflation_factor( data_mat,i) for i in range(data_mat.shape[1]) ]
    vif_factors = pd.DataFrame()
    vif_factors['Feature'] = list(x_features)
    vif_factors['VIF'] = vif
    
    return vif_factors

vif = display_VIF(train_scaled)
vif.sort_values(by=['VIF'],ascending=False)


# You can see that there are a number of features that are extremely multicollinear with other features in the data set. We will drop the features with the largest VIF and have another look at the variance inflation factor of the remaining features.

# In[ ]:


# Drop the features with the largest VIF and check for multicollinearity
train_scaled = train_scaled.drop(['totalDistance','rideDistance','swimDistance','walkDistance','numGroups','maxPlace',
                                 'playersJoined','winPoints','rankPoints'], axis=1)
test_scaled = test_scaled.drop(['totalDistance','rideDistance','swimDistance','walkDistance','numGroups','maxPlace',
                                 'playersJoined','winPoints','rankPoints'], axis=1)

vif = display_VIF(train_scaled)
vif.sort_values(by=['VIF'],ascending=False)


# You can see that removing some the the features with the highest VIF reduces the VIF for most of the remaining features, some more than others. The matchDuration and KillPlaceNorm have almost halved. We'll drop the remaining features with a VIF > 10 and keep the rest to build our model. We may decide to remove more later.

# In[ ]:


# Drop the the remaining features that have a VIF greater than 10.
train_scaled = train_scaled.drop(['matchDuration','killsNorm','killPlaceNorm'], axis=1)
test_scaled = test_scaled.drop(['matchDuration','killsNorm','killPlaceNorm'], axis=1)

vif = display_VIF(train_scaled)
vif.sort_values(by=['VIF'],ascending=False)


# Lets have a look at the pearson correlation between all the remaining features.

# In[ ]:


f,ax = plt.subplots(figsize=(20, 20))
sns.heatmap(train_scaled.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()


# In[ ]:


train_scaled.head()


# Because we are removing some features we need to perform the training testing split again on the new features.

# In[ ]:


# Train Test Split
y = train_scaled['winPlacePerc']
X = train_scaled.drop(['winPlacePerc'],axis=1)

# We will maintain the existing size and random state parameters for repeatability.
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=size, random_state=seed)


# In[ ]:


# Run all models with the reduced set of features.
runAllModels(X_train, X_validation, Y_train, Y_validation)


# Dropping all those features we identified as having a high multicolinearity with other features has reduced the accuracey of all the models we tried. So we will reinstate all those features for the next iteration of models. We probably wouldn't use any of these models for predictions since we already have bettor models with all the features.
# 
# So we wont use the reduced set of features to fit to the "best" model because we already have a model that does better with all the features.

# # Model Development for seperate matchTypes
# Again, with the insight we learnt from our [EDA], we found that it is likely that we will improve our overall results by creating four seperate models, one for each matchType. This allows us to potentially use a seperate type of model for each matchType.
# 
# Here we'll split the data into the matchTypes and see how the models perform for each one.
# 
# [EDA]: https://www.kaggle.com/beaubellamy/pubg-eda#

# In[ ]:


train_copy.head()


# In[ ]:


# Create a data set for each matchType and drop that feature, as there will be no variance, and hence no predictive power.
solo = train_copy[train_copy['matchType'] == 'Solo']
solo = solo.drop(['matchType'], axis=1)
duo = train_copy[train_copy['matchType'] == 'Duo']
duo = duo.drop(['matchType'], axis=1)
squad = train_copy[train_copy['matchType'] == 'Squad']
squad = squad.drop(['matchType'], axis=1)
other = train_copy[train_copy['matchType'] == 'Other']
other = other.drop(['matchType'], axis=1)

# since we used a copy of the trained data that hasn't been scaled, we need to scale the features again.
scaler = MinMaxScaler()
solo_scaled = pd.DataFrame(scaler.fit_transform(solo), columns=solo.columns)
duo_scaled = pd.DataFrame(scaler.fit_transform(duo), columns=duo.columns)
squad_scaled = pd.DataFrame(scaler.fit_transform(squad), columns=squad.columns)
other_scaled = pd.DataFrame(scaler.fit_transform(other), columns=other.columns)

# Seperate the matchType data
test_solo = test_copy[test_copy['matchType'] == 'Solo']
test_solo = test_solo.drop(['matchType'], axis=1)
test_duo = test_copy[test_copy['matchType'] == 'Duo']
test_duo = test_duo.drop(['matchType'], axis=1)
test_squad = test_copy[test_copy['matchType'] == 'Squad']
test_squad = test_squad.drop(['matchType'], axis=1)
test_other = test_copy[test_copy['matchType'] == 'Other']
test_other = test_other.drop(['matchType'], axis=1)

solo_test_scaled = pd.DataFrame(scaler.fit_transform(test_solo), columns=test_solo.columns)
duo_test_scaled = pd.DataFrame(scaler.fit_transform(test_duo), columns=test_duo.columns)
squad_test_scaled = pd.DataFrame(scaler.fit_transform(test_squad), columns=test_squad.columns)
other_test_scaled = pd.DataFrame(scaler.fit_transform(test_other), columns=test_other.columns)


# In[ ]:


solo_y = solo_scaled['winPlacePerc']
solo_X = solo_scaled.drop(['winPlacePerc'],axis=1)

# We will maintain the existing size and random state parameters for repeatability.
X_train, X_validation, Y_train, Y_validation = train_test_split(solo_X, solo_y, test_size=size, random_state=seed)

runAllModels(X_train, X_validation, Y_train, Y_validation)


# The model that performed the best for the solo matches was a GradientBoostingRegressor model with a score of 95.0%

# In[ ]:


GBR = GradientBoostingRegressor(learning_rate=0.8)
GBR.fit(solo_X,solo_y)
print(("GradientBoost Model traininig: {0:.3f}%".format(GBR.score(solo_X,solo_y)*100)))

predictions_solo = GBR.predict(solo_test_scaled)


# In[ ]:


duo_y = duo_scaled['winPlacePerc']
duo_X = duo_scaled.drop(['winPlacePerc'],axis=1)

# We will maintain the existing size and random state parameters for repeatability.
X_train, X_validation, Y_train, Y_validation = train_test_split(duo_X, duo_y, test_size=size, random_state=seed)

runAllModels(X_train, X_validation, Y_train, Y_validation)


# The model that performed the best for the duo matches was a GradientBoostingRegressor model, with a score of 94.4%

# In[ ]:


GBR = GradientBoostingRegressor(learning_rate=0.8)
GBR.fit(duo_X,duo_y)
print(("GradientBoost Model traininig: {0:.3f}%".format(GBR.score(duo_X,duo_y)*100)))

predictions_duo = GBR.predict(duo_test_scaled) 


# In[ ]:


squad_y = squad_scaled['winPlacePerc']
squad_X = squad_scaled.drop(['winPlacePerc'],axis=1)

# We will maintain the existing size and random state parameters for repeatability.
X_train, X_validation, Y_train, Y_validation = train_test_split(squad_X, squad_y, test_size=size, random_state=seed)

runAllModels(X_train, X_validation, Y_train, Y_validation)


# The model that performed the best for the squad matches was GradientBoostRegressor, with a score of 90.7%

# In[ ]:


GBR = GradientBoostingRegressor(learning_rate=0.8)
GBR.fit(squad_X,squad_y)
print(("GradientBoost Model traininig: {0:.3f}%".format(GBR.score(squad_X,squad_y)*100)))

predictions_squad = GBR.predict(squad_test_scaled)


# In[ ]:


other_y = other_scaled['winPlacePerc']
other_X = other_scaled.drop(['winPlacePerc'],axis=1)

# We will maintain the existing size and random state parameters for repeatability.
X_train, X_validation, Y_train, Y_validation = train_test_split(other_X, other_y, test_size=size, random_state=seed)

runAllModels(X_train, X_validation, Y_train, Y_validation)


# The model that performed the best for the other matches was the simple GradientBoostingRegressor, with a score of 83.1%
# 
# The Linear Regression model probably won here, because there isn't that much data available on these types of matches compared to the other match types.

# In[ ]:


GBR = GradientBoostingRegressor(learning_rate=0.8)
GBR.fit(other_X,other_y)
print(("GradientBoost Model traininig: {0:.3f}%".format(GBR.score(other_X,other_y)*100)))

predictions_other = GBR.predict(other_test_scaled) 


# In[ ]:


def create_submission(submission_Id, predictions, filename):
    submission = pd.DataFrame({'Id': submission_Id, 'winPlacePerc': predictions})
    
    submission.to_csv(filename+'.csv',index=False)


# In[ ]:


test_submission_solo = test_submission[test_submission['matchType'] == 'Solo']
test_submission_duo = test_submission[test_submission['matchType'] == 'Duo']
test_submission_squad = test_submission[test_submission['matchType'] == 'Squad']
test_submission_other = test_submission[test_submission['matchType'] == 'Other']

matchTypeId = test_submission_solo['Id'].append(test_submission_duo['Id']).append(test_submission_squad['Id']).append(test_submission_other['Id'])

predictions_solo[predictions_solo > 1] = 1
predictions_solo[predictions_solo < 0] = 0

predictions_duo[predictions_duo > 1] = 1
predictions_duo[predictions_duo < 0] = 0

predictions_squad[predictions_squad > 1] = 1
predictions_squad[predictions_squad < 0] = 0

predictions_other[predictions_other > 1] = 1
predictions_other[predictions_other < 0] = 0


predications_matchtype = np.append(np.append(predictions_solo,predictions_duo),np.append(predictions_squad,predictions_other))

create_submission(matchTypeId, predications_matchtype, 'submission_matchType')


# If you liked this post, please upvote.

# In[ ]:




