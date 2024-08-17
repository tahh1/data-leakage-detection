#!/usr/bin/env python
# coding: utf-8

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


# In[2]:


#Imports
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)


# Link to Official Starter Notebook:
# https://www.kaggle.com/dster/nfl-big-data-bowl-official-starter-notebook

# In[3]:


# Training data is in the competition dataset as usual
train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

train_df.head()


# In[4]:


train_df.shape # (509762, 49) (rows, columns)


# In[5]:


train_df.columns


# **Data Dictionary:**
# 
# * GameId - a unique game identifier
# * PlayId - a unique play identifier
# * Team - home or away
# * X - player position along the long axis of the field. See figure below.
# * Y - player position along the short axis of the field. See figure below.
# * S - speed in yards/second
# * A - acceleration in yards/second^2
# * Dis - distance traveled from prior time point, in yards
# * Orientation - orientation of player (deg)
# * Dir - angle of player motion (deg)
# * NflId - a unique identifier of the player
# * DisplayName - player's name
# * JerseyNumber - jersey number
# * Season - year of the season
# * YardLine - the yard line of the line of scrimmage
# * Quarter - game quarter (1-5, 5 == overtime)
# * GameClock - time on the game clock
# * PossessionTeam - team with possession
# * Down - the down (1-4)
# * Distance - yards needed for a first down
# * FieldPosition - which side of the field the play is happening on
# * HomeScoreBeforePlay - home team score before play started
# * VisitorScoreBeforePlay - visitor team score before play started
# * NflIdRusher - the NflId of the rushing player
# * OffenseFormation - offense formation
# * OffensePersonnel - offensive team positional grouping
# * DefendersInTheBox - number of defenders lined up near the line of scrimmage, spanning the width of the ---offensive line
# * DefensePersonnel - defensive team positional grouping
# * PlayDirection - direction the play is headed
# * TimeHandoff - UTC time of the handoff
# * TimeSnap - UTC time of the snap
# * Yards - the yardage gained on the play (you are predicting this)
# * PlayerHeight - player height (ft-in)
# * PlayerWeight - player weight (lbs)
# * PlayerBirthDate - birth date (mm/dd/yyyy)
# * PlayerCollegeName - where the player attended college
# * HomeTeamAbbr - home team abbreviation
# * VisitorTeamAbbr - visitor team abbreviation
# * Week - week into the season
# * Stadium - stadium where the game is being played
# * Location - city where the game is being player
# * StadiumType - description of the stadium environment
# * Turf - description of the field surface
# * GameWeather - description of the game weather
# * Temperature - temperature (deg F)
# * Humidity - humidity p- WindSpeed - wind speed in miles/hour WindDirection - wind direction

# In[6]:


train_df.info()


# In[7]:


train_df.isna().sum().sort_values(ascending=False)[0:12] #11 columns are missing data


# Deal with NA later

# # Plays per Team. Plays per Game

# In[8]:


print(f"Total games: {train_df.GameId.nunique()}")
print(f"Total plays: {train_df.PlayId.nunique()}")
print(f"Total players: {train_df.NflId.nunique()}")
print(f"Total rushers: {train_df.NflIdRusher.nunique()}")


# In[9]:


print(f"Total Teams: {train_df.PossessionTeam.nunique()}")
playPoss = train_df.groupby(["PlayId", 'PossessionTeam']).GameId.count().reset_index()
playPoss.columns = ["PlayId", 'PossessionTeam', 'PlayersOnField']
playPoss.PlayersOnField.describe() #22 players on the field for each play, as it should be


# In[10]:


teamPoss = playPoss.groupby("PossessionTeam").PlayId.count().sort_values(ascending=True)
print((sum(teamPoss.values))) #2371, boom


# In[11]:


#All Unique Team Plays
plt.figure(figsize=(20,10))
plt.barh(teamPoss.index, teamPoss.values, color="firebrick")
plt.title("All Team Plays", weight="bold", fontsize=20)
plt.xlabel("Number of Plays", fontsize=16)
plt.ylabel("")

plt.grid(axis="x")
plt.show()


# In[12]:


playsPerGame = train_df.groupby("GameId").PlayId.nunique()
playsPerGame.sum() #23171, boom


# In[13]:


playsPerGame.describe() #wow which game had 85 plays that is bonkers


# In[14]:


#All Unique Plays per game
plt.figure(figsize=(20,10))
plt.hist(playsPerGame.values, color="firebrick")
plt.title("Plays per Game", weight="bold", fontsize=20)
plt.xlabel("Number of Plays", fontsize=16)
plt.ylabel("")

plt.show()


# In[15]:


playDF = pd.DataFrame(playsPerGame)
playDF.loc[playDF.PlayId > 80].index[0] #2017121000


# In[16]:


train_df.loc[train_df.GameId == 2017121000].head() #It's the SNOW BOWL


# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Ralph_Wilson_Stadium_%28NFL_Buffalo_Bills%29_-_Orchard_Park%2C_NY.jpg/1200px-Ralph_Wilson_Stadium_%28NFL_Buffalo_Bills%29_-_Orchard_Park%2C_NY.jpg)
# 
# Buffalo vs Indianpolis in the snow. No wonder there were 85 plays...
# 
# https://en.wikipedia.org/wiki/Snow_Bowl_(2017)

# # Games by Stadium

# In[17]:


stads = pd.DataFrame(train_df.groupby("GameId").Stadium.max().reset_index())
stads.columns = ["GameId", "Stadium"]
stadGames = stads.groupby("Stadium").count().sort_values(by="GameId", ascending=True)
print((stadGames.GameId.sum())) #512 boom


# In[18]:


#All Unique Stadiums by games
plt.figure(figsize=(20,10))
plt.barh(stadGames.index, stadGames.GameId, color="seagreen")
plt.title("Games per Stadium", weight="bold", fontsize=20)
plt.xlabel("Number of Games", fontsize=16)
plt.ylabel("")

plt.grid(axis="x")
plt.show() #ugh more pats


# Should probably look at distance and downs now

# In[19]:


yardsToGo = train_df.groupby("PlayId").Distance.max()
yardsToGo.describe() #ha what play had 40 yards to go, that is also bonkers


# In[20]:


#All Unique Plays per game
plt.figure(figsize=(20,10))
plt.hist(yardsToGo.values, color="purple")
plt.title("Yards to Go per Play", weight="bold", fontsize=20)
plt.xlabel("Number of Yards", fontsize=16)
plt.ylabel("Count")

plt.show()


# In[21]:


downOfPlay = train_df.groupby("PlayId").Down.max()
downOfPlay.describe()


# In[22]:


#All Unique downs per play
plt.figure(figsize=(20,10))
plt.hist(downOfPlay.values, color="purple")
plt.title("Down of Play", weight="bold", fontsize=20)
plt.xlabel("Down Number", fontsize=16)
plt.ylabel("Count")

plt.show() #real ground-breaking stuff here


# # Where on the field do most plays start?

# In[23]:


yardlineOfPlay = train_df.groupby(["PlayId", "PlayDirection"]).YardLine.max().reset_index()
yardlineOfPlayDF = yardlineOfPlay.groupby(["YardLine", "PlayDirection"]).count().reset_index()
yardlineOfPlayDF.columns = ["YardLine", "PlayDirection", "Count"]
left = yardlineOfPlayDF.loc[yardlineOfPlayDF.PlayDirection == "left"].sort_values("YardLine", ascending=False)
right = yardlineOfPlayDF.loc[yardlineOfPlayDF.PlayDirection == "right"].sort_values("YardLine", ascending=True)

sortedLeftRight = pd.concat([right, left]).reset_index(drop=True)
sortedLeftRight.head()


# In[24]:


plt.figure(figsize=(20,10))

plt.bar(sortedLeftRight.index, sortedLeftRight.Count, color="purple")

plt.title("Plays vs YardLine", weight="bold", fontsize=20)
plt.xlabel("YardLine", fontsize=16)
plt.ylabel("Count")

plt.show() #most plays start on the 25 yeard line, brilliant.


# In[25]:


#without direction
sortedLeftRight2 = sortedLeftRight.groupby("YardLine").Count.sum()

plt.figure(figsize=(20,10))

plt.bar(sortedLeftRight2.index, sortedLeftRight2.values, color="purple")

plt.title("Plays vs YardLine", weight="bold", fontsize=20)
plt.xlabel("YardLine", fontsize=16)
plt.ylabel("Count")

plt.show() #most plays start on the 25 yeard line, brilliant.


# # Plays per quarter

# In[26]:


quarts = pd.DataFrame(train_df.groupby("PlayId").Quarter.max()).reset_index()
quarts.columns = ["PlayId", "Quarter"]

quartsCount = quarts.groupby("Quarter").count() 
quartsCount#quarter 5?!?


# In[27]:


#All Unique Plays per quarter
plt.figure(figsize=(20,10))
plt.bar(quartsCount.index, quartsCount.PlayId, color="lightblue")
plt.title("Quarter of Play", weight="bold", fontsize=20)
plt.xlabel("Quarter", fontsize=16)
plt.ylabel("Count")

plt.show()


# What are we predicting again

# In[28]:


yrds = train_df.groupby("PlayId").Yards.max()
yrds.describe()


# In[29]:


sum(yrds.values < 0) #oh, can gain negative yards, right I knew that


# In[30]:


print(("Percent of plays ending in negative yards: " + str(round(100* 2561/23171,1)) + "%"))


# In[31]:


#All Unique Yards gained
plt.figure(figsize=(20,10))
plt.hist(yrds.values, color="maroon")
plt.title("Yards Gained Per Play", weight="bold", fontsize=20)
plt.xlabel("Yards Gained", fontsize=16)
plt.ylabel("Count")

plt.show() #quite right skewed


# # Basic Team Columns

# In[32]:


teamCols = ["PlayId", "PossessionTeam", "Down", "Quarter", "Distance", "PlayDirection", "DefensePersonnel", "OffensePersonnel",
           'FieldPosition', 'HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 
                 'OffenseFormation', 'DefendersInTheBox', 'Week', 'Turf']#, 'Temperature', "Humidity"]
numCols = ["Distance", 'DefendersInTheBox', 'HomeScoreBeforePlay', 'VisitorScoreBeforePlay']#, 'Temperature', "Humidity"]
yCols = ["PlayId", "Yards"]


# In[33]:


X = train_df[teamCols].drop_duplicates(subset="PlayId")
y = train_df[yCols].drop_duplicates(subset="PlayId")


# In[34]:


X.Turf.value_counts()


# In[35]:


X.Turf = X.Turf.map({'Grass': 'Grass', 'Natural Grass': 'Grass', 'Naturall Grass': 'Grass', 'Natural': 'Grass', 'Natural grass': 'Grass', 'grass': 'Grass',
           'natural grass': 'Grass'})
X.Turf = X.Turf.fillna("Turf")
X.Turf.value_counts()


# In[36]:


X.DefendersInTheBox = X.DefendersInTheBox.fillna(X.DefendersInTheBox.mean())
#X.Temperature = X.Temperature.fillna(X.Temperature.mean())
#X.Humidity = X.Humidity.fillna(X.Humidity.mean())
X.OffenseFormation = X.OffenseFormation.fillna(X.OffenseFormation.value_counts().index[0])
X.FieldPosition = X.FieldPosition.fillna(X.FieldPosition.value_counts().index[0])


# In[37]:


X.info()


# In[38]:


y.info()


# In[39]:


X = X.astype(str)
X[numCols] = X[numCols].astype(float)
playIds = X.PlayId
X.drop("PlayId", axis=1, inplace=True)
X.info()


# In[40]:


y = y.astype(int)
y.drop("PlayId", axis=1, inplace=True)
y.info()


# In[41]:


from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)


# In[42]:


from catboost import CatBoostRegressor


# In[43]:


cat_features = [0, 1, 2, 4, 5, 6, 7, 10, 12, 13]


# In[44]:


model = CatBoostRegressor(iterations = 100, learning_rate = 0.5, depth = 15)


# In[45]:


model.fit(X_train, Y_train, cat_features)


# In[46]:


model.feature_importances_


# In[47]:


from sklearn.metrics import mean_squared_error
val_preds = model.predict(X_val)
print((mean_squared_error(Y_val, val_preds)))
print((mean_squared_error(Y_val, np.repeat(np.mean(Y_val.Yards), len(Y_val)))))


accuracy = model.score(X_val,Y_val)
print(accuracy)


# In[48]:


plt.hist(Y_val.Yards - val_preds)


# In[49]:


pd.Series(val_preds).describe()


# In[50]:


from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[51]:


X2 = X.copy()
X2.drop("Distance", axis=1, inplace=True)
X2.drop("DefendersInTheBox", axis=1, inplace=True)
X2.drop("HomeScoreBeforePlay", axis=1, inplace=True)
X2.drop("VisitorScoreBeforePlay", axis=1, inplace=True)
#X2.drop("Temperature", axis=1, inplace=True)
#X2.drop("Humidity", axis=1, inplace=True)

X2 = pd.get_dummies(X2)
X2["Distance"] = X.Distance
X2["DefendersInTheBox"] = X.DefendersInTheBox
X2["HomeScoreBeforePlay"] = X.HomeScoreBeforePlay
X2["VisitorScoreBeforePlay"] = X.VisitorScoreBeforePlay
#X2["Temperature"] = X.Temperature
#X2["Humidity"] = X.Humidity

print(("X shape: : ", X2.shape))


# In[52]:


X_train, X_val, Y_train, Y_val = train_test_split(X2, y, shuffle=True, test_size=0.2, random_state=42)


# In[53]:


xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3)


# In[54]:


xgb.fit(X_train, Y_train)


# In[55]:


val_preds = xgb.predict(X_val)
print((mean_squared_error(Y_val, val_preds)))
print((mean_squared_error(Y_val, np.repeat(np.mean(Y_val.Yards), len(Y_val)))))


accuracy = xgb.score(X_val,Y_val)
print(accuracy)


# In[56]:


from sklearn.ensemble import RandomForestRegressor


# In[57]:


rfr = RandomForestRegressor(n_estimators=20)


# In[58]:


rfr.fit(X_train, Y_train)


# In[59]:


val_preds = rfr.predict(X_val)
print((mean_squared_error(Y_val, val_preds)))
print((mean_squared_error(Y_val, np.repeat(np.mean(Y_val.Yards), len(Y_val)))))


accuracy = rfr.score(X_val,Y_val)
print(accuracy)


# All of these are terrible but at least xgboost is better than the mean

# In[60]:


from kaggle.competitions import nflrush


# In[61]:


X_train.columns


# In[62]:


def make_my_predictions(test_df, sample_prediction_df):
    X_test = test_df[teamCols].drop_duplicates(subset="PlayId")
    
    temp = pd.DataFrame(np.zeros(shape = (1,len(X2.columns))))
    temp.columns = X2.columns

    temp["Distance"] = X_test.Distance
    temp["DefendersInTheBox"] = X_test.DefendersInTheBox
    temp["HomeScoreBeforePlay"] = X_test.HomeScoreBeforePlay
    temp["VisitorScoreBeforePlay"] = X_test.VisitorScoreBeforePlay

    temp["PossessionTeam_" + X_test.PossessionTeam.values[0]] = 1
    temp["Down_" + str(X_test.Down.values[0])] = 1
    temp["Quarter_" + str(X_test.Quarter.values[0])] = 1
    temp["PlayDirection_" + X_test.PlayDirection.values[0]] = 1
    temp["Week_" + str(X_test.Week.values[0])] = 1
    
    if (np.logical_not(pd.isnull(X_test.FieldPosition.values[0]))):
        temp["FieldPosition_" + X_test.FieldPosition.values[0]] = 1

    if (sum([X_test.OffensePersonnel.values[0] in x for x in X2.columns]) > 0):
        temp["OffensePersonnel_" + X_test.OffensePersonnel.values[0]] = 1
    if (sum([X_test.DefensePersonnel.values[0] in x for x in X2.columns]) > 0):
        temp["DefensePersonnel_" + X_test.DefensePersonnel.values[0]] = 1
    if (sum([X_test.OffenseFormation.values[0] in x for x in X2.columns]) > 0):   
        temp["OffenseFormation_" + X_test.OffenseFormation.values[0]] = 1
        
    X_test.Turf = X_test.Turf.map({'Grass': 'Grass', 'Natural Grass': 'Grass', 'Naturall Grass': 'Grass', 'Natural': 'Grass', 'Natural grass': 'Grass', 'grass': 'Grass',
           'natural grass': 'Grass'})
    X_test.Turf = X_test.Turf.fillna("Turf")
    temp["Turf_" + X_test.Turf.values[0]] = 1
    
    pred = xgb.predict(temp)
    sample_prediction_df.iloc[:, 0:int(round(pred[0]))+ 100] = 0
    sample_prediction_df.iloc[:, int(round(pred[0])+ 100):-1] = 1
    sample_prediction_df.iloc[:, -1] = 1
    sample_prediction_df.iloc[:, int(round(pred[0]) + 100)] = .95
    sample_prediction_df= sample_prediction_df.T
    sample_prediction_df = sample_prediction_df.interpolate(axis = 0, method = 'linear').T
    return sample_prediction_df


# In[63]:


env = nflrush.make_env()
for (test_df, sample_prediction_df) in env.iter_test():
    predictions_df = make_my_predictions(test_df, sample_prediction_df)
    env.predict(predictions_df)

env.write_submission_file()


# In[ ]:




