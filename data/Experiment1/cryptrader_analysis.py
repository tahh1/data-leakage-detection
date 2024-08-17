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


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')

data = pd.read_csv('../input/beat-the-bookie-worldwide-football-dataset/closing_odds.csv.gz', compression='gzip', sep=',', quotechar='"')
print(('Reading completed. Total rows {}'.format(len(data))))


# In[3]:


data.head()


# In[4]:


teams = data["home_team"].unique()


# In[5]:


all_games = data[["home_team","away_team","home_score","away_score","avg_odds_home_win","avg_odds_draw","avg_odds_away_win"]].values
all_games = sorted(all_games,key=lambda x:x[0])
games = dict()
for n in range(len(all_games)):
    home = all_games[n][0]
    away = all_games[n][1]
    if home not in games:
        games[home] = dict()
    if away not in games[home]:
        games[home][away] = dict()
        for result in ["win","draw","lose"]:
            games[home][away][result] = 0 
    if all_games[n][2] > all_games[n][3]:
        games[home][away]["win"] += 1
    if all_games[n][2] == all_games[n][3]:
        games[home][away]["draw"] += 1
    if all_games[n][2] < all_games[n][3]:
        games[home][away]["lose"] += 1


# In[6]:


odds = data[["avg_odds_home_win","avg_odds_draw","avg_odds_away_win"]].values
payoff = list()
for x in odds:
    p = 1/(1/x[0]+1/x[1]+1/x[2])
    payoff.append(p)


# In[7]:


np.average(payoff)


# In[8]:


results = list()
for x in games:
    w = 0
    c = 0
    for y in games[x]:
        for side in games[x][y]:
            c += games[x][y][side]
            if side == "win":
                w += games[x][y][side]
    results.append([x,w/c,c])


# In[9]:


results = [x for x in results if x[2] > 50]


# In[10]:


results = sorted(results, key=lambda x:x[1])[::-1]
results


# In[11]:


all_games


# In[12]:


payout_vs_winning_prob = dict()
for x in all_games:
    if x[4] not in payout_vs_winning_prob:
        payout_vs_winning_prob[x[4]] = list()
    if x[2] > x[3]:
        payout_vs_winning_prob[x[4]].append(1)
    else:
        payout_vs_winning_prob[x[4]].append(0)


# In[13]:


import numpy as np
from operator import mul
from functools import reduce

def cmb(n,r):
    r = min(n-r,r)
    if r == 0: return 1
    over = reduce(mul, list(range(n, n - r, -1)))
    under = reduce(mul, list(range(1,r + 1)))
    return over // under
bins = [1+0.01*n for n in range(100)]
payout_vs_prob = list()
for n in range(len(bins)-1):
    tmp = [0,0]
    for x in payout_vs_winning_prob:
        if bins[n] < x and x <= bins[n+1] :
            w_count = len([y for y in payout_vs_winning_prob[x] if y == 1])
            t_count = len(payout_vs_winning_prob[x])
            l_count = t_count - w_count
            tmp[0] += w_count
            tmp[1] += t_count
    if tmp[1] > 0:
        p_value = cmb(tmp[1], tmp[0])/(2**tmp[1])
        payout_vs_prob.append([bins[n], tmp[0]/tmp[1], p_value])


# In[14]:


payout_vs_prob


# In[15]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
payout_vs_prob = np.array(payout_vs_prob)
plt.scatter(payout_vs_prob[:,0],payout_vs_prob[:,1])
plt.grid(True)
plt.show()


# In[16]:


payout_vs_prob = np.array(payout_vs_prob)
plt.scatter(payout_vs_prob[:,0],payout_vs_prob[:,1]*((payout_vs_prob[:,0]+0.005)))
plt.grid(True)
plt.show()


# In[17]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# generate points and keep a subset of them
x = payout_vs_prob[:,0].reshape(-1,1)
y = payout_vs_prob[:,1]

colors = ['teal', 'yellowgreen', 'gold']
lw = 2
plt.scatter(x, y, color='navy', s=30, marker='o', label="training points")

for count, degree in enumerate([3, 4, 5]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(x, y)
    y_plot = model.predict(x)
    plt.plot(x, y_plot, color=colors[count], linewidth=lw,
             label="degree %d" % degree)

plt.legend(loc='lower left')
plt.grid(True)
plt.show()


# In[18]:


import pandas as pd
df = pd.DataFrame(sorted(payout_vs_prob, key=lambda x:x[0]), columns=["payoff","win_prob","p_value"])
df


# In[19]:


df.to_csv("payoff_vs_winprob.csv")


# In[20]:


grouping = list()
splits = 50
for x in range(splits):
    threshold = [1/splits*x, 1/splits*(x+1)]
    tmp = list()
    for y in results:
        if threshold[0] < y[1] and y[1] <= threshold[1]:
            tmp.append(y[0])
    grouping.append(tmp)
grouping

matrix = list()
draw_matrix = list()
sides = ["home_team","away_team"]
scores = {"home_team":"home_score","away_team":"away_score"}
for x in grouping:
    res = list()
    d_res = list()
    for y in grouping:
        total_count = 0
        winning_count = 0
        draw_count = 0
        lose_count = 0
        for h in x:
            for side in range(2):
                home = sides[side%2]
                away = sides[(side+1)%2]
                for a in y:
                    try:
                        games[h][a][result]
                    except Exception:
                        try:
                            games[a][h][result]
                            tmp = a
                            a = h
                            h = tmp
                        except Exception:
                            continue
                    for result in ["win","draw","lose"]:
                        total_count += games[h][a][result]
                        if result == "win":
                            winning_count += games[h][a][result]
                        if result == "draw":
                            draw_count += games[h][a][result]
                        if result == "lose":
                            lose_count += games[h][a][result]
                        
        if total_count > 0:
            res.append(winning_count/total_count)
            d_res.append(draw_count/total_count)
        else:
            d_res.append(0)
            res.append(0)
    print(res)
    matrix.append(res)
    draw_matrix.append(d_res)


# In[21]:


matrix


# In[22]:


matrix[12]


# In[23]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


plt.plot(matrix[35])
plt.grid(True)
plt.xlabel("opponents")
plt.ylabel("winning rate")
plt.show()


# In[25]:


belong = dict()
for t in teams:
    for n in range(len(grouping)):
        if t in grouping[n]:
            belong[t] = n/splits


# In[26]:


import pickle
with open("belong.pickle",mode="wb") as f:
    pickle.dump(belong,f)


# In[27]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# generate points and keep a subset of them
X = list()
Y = list()
for x in all_games:
    try:
        X.append([belong[x[0]], belong[x[1]], x[4]])
        if x[2] > x[3]:
            Y.append(1)
        else:
            Y.append(0)
    except Exception:
        continue

colors = ['teal', 'yellowgreen', 'gold']
lw = 2
for count, degree in enumerate([3, 4, 5]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, Y)
    y_plot = model.predict(X)
    plt.plot(Y, y_plot, color=colors[count], linewidth=lw,
             label="degree %d" % degree)

plt.legend(loc='lower left')
plt.grid(True)
plt.show()


# In[28]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

X_train = list()
y_train = list()
X_test = list()
y_test = list()
length = len(all_games)
for n in range(len(all_games)):
    x = all_games[n]
    try:
        if n < length*0.5:
            X_train.append([belong[x[0]], belong[x[1]], x[4]])
            if x[2] > x[3]:
                y_train.append(1)
            else:
                y_train.append(0)
        else:
            X_test.append([belong[x[0]], belong[x[1]], x[4]])
            if x[2] > x[3]:
                y_test.append(1)
            else:
                y_test.append(0)
    except Exception:
        continue


# In[29]:


clf = QuadraticDiscriminantAnalysis()
clf.fit(X_train, y_train)


# In[30]:


with open("prediction_model.pickle",mode="wb") as f:
    pickle.dump(clf, f)


# In[31]:


from sklearn.metrics import confusion_matrix
new_X = list()
new_Y = list()
for n in range(len(X_test)):
    if X_test[n][2] > 2:
        new_X.append(X_test[n])
        new_Y.append(y_test[n])
confusion_matrix(clf.predict(new_X), new_Y)


# In[32]:


clf.score(X_test, y_test)


# In[33]:


y_plot


# In[34]:


belong


# In[35]:


all_games[:10]


# In[36]:


for x in teams:
    if "Liverpool" in x:
        print(x)


# In[37]:


for x in teams:
    if "Watford" in x:
        print(x)


# In[38]:


belong["Watford"]


# In[39]:


belong["Liverpool"]


# In[40]:


clf.predict([[belong["Liverpool"],belong["Watford"],1.22]])[0]


# In[41]:


matrix[10][7]


# In[42]:


games["Liverpool"]


# In[43]:


d_count = 0
t_count = 0
for x in games:
    for y in games[x]:
        for side in games[x][y]:
            t_count += games[x][y][side]
            if side == "draw":
                d_count += games[x][y][side]
print((t_count, d_count))


# In[44]:


all_games[:1]


# In[45]:


#backtest
import numpy as np
es = list()
draw_vs_performance = list()
for k in range(20):
    simple_result = 0
    assets = [1]
    c = 0
    w = 0
    for n in range(len(all_games)):
        home = all_games[n][0]
        away = all_games[n][1]
        odds = all_games[n][4]*1.08
        o_draw = all_games[n][5]*1.08
        o_away = all_games[n][6]*1.08
        #draw_matrix[belong[home]][belong[away]]
        try:
            E = 0.25*o_draw
            if o_draw > k and o_draw <= k+1:
                es.append(E)
                c += 1
                last_asset = assets[-1]
                betting_size = last_asset*E/o_draw
                b_size = 2
                simple_result -= b_size
                if all_games[n][2] > all_games[n][3]:
                    w += 1
                    simple_result += odds
                    assets.append(simple_result)
                if all_games[n][2] < all_games[n][3]:
                    w += 1
                    simple_result += o_away
                    assets.append(simple_result)
                else:
                    assets.append(simple_result)
        except Exception:
            continue
    draw_vs_performance.append([simple_result/len(assets),w,c])
    print(k)


# In[46]:


#backtest
import numpy as np
es = list()
draw_vs_performance = list()
simple_result = 0
assets = [1]
c = 0
w = 0
for n in range(len(all_games)):
    home = all_games[n][0]
    away = all_games[n][1]
    odds = all_games[n][4]*1.08
    o_draw = all_games[n][5]*1.08
    o_away = all_games[n][6]*1.08
    #draw_matrix[belong[home]][belong[away]]
    try:
        E = 0.25*o_draw
        if odds >= 1+0.1 and odds < 1.2:
            es.append(E)
            c += 1
            last_asset = assets[-1]
            b_size = last_asset*0.04/odds
            simple_result -= b_size
            if all_games[n][2] > all_games[n][3]:
                w += 1
                simple_result += odds
                assets.append(simple_result)
            else:
                assets.append(simple_result)
    except Exception:
        continue


# In[47]:


plt.plot([x[0] for x in draw_vs_performance])
plt.show()


# In[48]:


draw_vs_performance


# In[49]:


4371/4797


# In[50]:


413/419


# In[51]:


w,c


# In[52]:


simple_result/len(assets)


# In[53]:


len(assets)


# In[54]:


np.average(es)


# In[55]:


w,c


# In[56]:


performance = [x[0] for x in draw_vs_performance]


# In[57]:


assets


# In[58]:


plt.plot(assets)
plt.grid(True)
plt.show()


# In[59]:


len(assets)


# In[ ]:




