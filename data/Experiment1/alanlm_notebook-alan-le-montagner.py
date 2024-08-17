#!/usr/bin/env python
# coding: utf-8

# L'objectif de ce challenge Kaggle est de prédire, avec le plus de succès possible, l'issue de matchs de tennis afin de réaliser des paris sportifs.
# 
# On passera tout d'abord par une exploration et une visualisation des données, et l'on poursuivra par des modèles prédictifs.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
import os
from random import *
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dropout



get_ipython().run_line_magic('matplotlib', 'inline')


# La première étape consiste simplement en l'importation des données du fichier ATP.csv.

# In[2]:


df = pd.read_csv("/kaggle/input/atpdata/ATP.csv",dtype=str)


# Par ailleurs, il est nécessaire pour cette base de données, où de nombreuses variables sont manquantes, de la nettoyer. 
# 
# On passera ici, d'abord, par la sélection des données intéressantes pour notre cas, à savoir les données à partir de 1991, qui contiennent davantage de statistiques sur les matchs. Au vu de la quantité de matchs disponibles, le rejet des données avant 1991 ne devrait pas poser de problème, et nous simplifiera la tâche.
# 
# Par la suite, on constate que plusieurs variables contiennent de trop nombreuses valeurs manquantes. On les retire donc de notre étude.
# 
# Enfin, on retire de notre base de données les matchs dont au moins une variable est manquante, encore une fois pour simplifier la tâche par la suite. On aurait également pu chercher à remplir certaines de ces valeurs manquantes, par exemple en affectant "0" ou "-1" à toutes les valeurs quantitatives manquantes, ou en créant une catégorie "inconnu" pour les valeurs qualitatives telles que la nationalité ou la main dominante.

# In[3]:


df['tourney_month']=df.tourney_date.astype(str).str[:6]
df = df[df['tourney_month'].between('199101','201909')]

df.info()

df = df.drop(columns=['draw_size','winner_entry','winner_seed','loser_entry','loser_seed'])
df = df.dropna()
df.shape

#nb_data=df.shape[0]
#nb_param=df.shape[1]
#count=np.zeros(nb_param)
#for i in range(1,nb_data):
#    for j in range(1,nb_param):
#        if np.isnan(df[df.columns[j]].iloc[i]):
#            count[j]+=1

    

#df = df.dropna().astype(float)


# La commande ci-dessous nous permet d'explorer rapidement le panel de valeurs prises par les variables utilisées, et notamment les valeurs les plus fréquentes.

# In[4]:


df.describe().transpose()


# On passe ensuite par la construction d'histogrammes pour visualiser la répartition des valeurs prises par quelques statistiques propres aux joueurs, et les éventuelles différences notables entre gagnants et perdants.

# In[5]:


#Visualisation des variables propres aux joueurs

df['winner_age']=pd.to_numeric(df['winner_age'])
df['winner_ht']=pd.to_numeric(df['winner_ht'])
df['winner_id']=pd.to_numeric(df['winner_id'])
df['winner_rank']=pd.to_numeric(df['winner_rank'])
df['winner_rank_points']=pd.to_numeric(df['winner_rank_points'])

df['loser_age']=pd.to_numeric(df['loser_age'])
df['loser_ht']=pd.to_numeric(df['loser_ht'])
df['loser_id']=pd.to_numeric(df['loser_id'])
df['loser_rank']=pd.to_numeric(df['loser_rank'])
df['loser_rank_points']=pd.to_numeric(df['loser_rank_points'])


plt.figure(figsize=(20,10))
plt.subplot(2,4,1)
df['winner_age'].plot(kind='hist',bins=26, xlim=(15,40), ylim=(0,10000), title='Age du gagnant')

plt.subplot(2,4,2)
df['loser_age'].plot(kind='hist',bins=26, xlim=(15,40), ylim=(0,10000), title='Age du perdant')

plt.subplot(2,4,3)
df['winner_ht'].plot(kind='hist',bins=15, xlim=(160,210), ylim=(0,25000), title='Taille du gagnant')

plt.subplot(2,4,4)
df['loser_ht'].plot(kind='hist',bins=15, xlim=(160,210), ylim=(0,25000), title='Taille du perdant')

plt.subplot(2,4,5)
df['winner_rank'].plot(kind='hist',bins=100, xlim=(0,800), ylim=(0,25000), title='Rang du gagnant')

plt.subplot(2,4,6)
df['loser_rank'].plot(kind='hist',bins=100, xlim=(0,800), ylim=(0,25000), title='Rang du perdant')

plt.subplot(2,4,7)
df['winner_rank_points'].plot(kind='hist',bins=100, xlim=(0,14000), ylim=(0,15000), title='Points de classement du gagnant')

plt.subplot(2,4,8)
df['loser_rank_points'].plot(kind='hist',bins=100, xlim=(0,14000), ylim=(0,15000), title='Points de classement du perdant')


# Par la suite, on effectue le même type de visualisation pour les variables propres à l'issue du match, telles que les statistiques de services, de breakpoints ou encore d'aces.

# In[6]:


df['w_1stWon']=pd.to_numeric(df['w_1stWon'])
df['w_2ndWon']=pd.to_numeric(df['w_2ndWon'])
df['w_SvGms']=pd.to_numeric(df['w_SvGms'])
df['w_ace']=pd.to_numeric(df['w_ace'])
df['w_bpFaced']=pd.to_numeric(df['w_bpFaced'])
df['w_bpSaved']=pd.to_numeric(df['w_bpSaved'])
df['w_df']=pd.to_numeric(df['w_df'])
df['w_svpt']=pd.to_numeric(df['w_svpt'])

df['l_1stWon']=pd.to_numeric(df['l_1stWon'])
df['l_2ndWon']=pd.to_numeric(df['l_2ndWon'])
df['l_SvGms']=pd.to_numeric(df['l_SvGms'])
df['l_ace']=pd.to_numeric(df['l_ace'])
df['l_bpFaced']=pd.to_numeric(df['l_bpFaced'])
df['l_bpSaved']=pd.to_numeric(df['l_bpSaved'])
df['l_df']=pd.to_numeric(df['l_df'])
df['l_svpt']=pd.to_numeric(df['l_svpt'])


plt.figure(figsize=(20,15))

plt.subplot(3,4,1)
df['w_1stWon'].plot(kind='hist', title='Premiers services remportés par le gagnant (%)', bins=100, xlim=(0,100), ylim=(0,10000))

plt.subplot(3,4,2)
df['l_1stWon'].plot(kind='hist', title='Premiers services remportés par le perdant (%)', bins=100, xlim=(0,100), ylim=(0,10000))

plt.subplot(3,4,3)
df['w_2ndWon'].plot(kind='hist', title='Seconds services remportés par le gagnant (%)', bins=50, xlim=(0,50), ylim=(0,10000))

plt.subplot(3,4,4)
df['l_2ndWon'].plot(kind='hist', title='Seconds services remportés par le perdant (%)', bins=50, xlim=(0,50), ylim=(0,10000))

plt.subplot(3,4,5)
df['w_bpFaced'].plot(kind='hist', title='Breakpoints endurés par le gagnant', bins=30, xlim=(0,30), ylim=(0,20000))

plt.subplot(3,4,6)
df['l_bpFaced'].plot(kind='hist', title='Breakpoints endurés par le perdant', bins=30, xlim=(0,30), ylim=(0,20000))

plt.subplot(3,4,7)
df['w_bpSaved'].plot(kind='hist', title='Breakpoints sauvés par le gagnant', bins=25, xlim=(0,25), ylim=(0,20000))

plt.subplot(3,4,8)
df['l_bpSaved'].plot(kind='hist', title='Breakpoints sauvés par le perdant', bins=25, xlim=(0,25), ylim=(0,20000))

plt.subplot(3,4,9)
df['w_ace'].plot(kind='hist', title='Aces du gagnant', bins=40, xlim=(0,40), ylim=(0,30000))

plt.subplot(3,4,10)
df['l_ace'].plot(kind='hist', title='Aces du perdant', bins=40, xlim=(0,40), ylim=(0,30000))

plt.subplot(3,4,11)
df['w_df'].plot(kind='hist', title='Double fautes du gagnant', bins=20, xlim=(0,20), ylim=(0,30000))

plt.subplot(3,4,12)
df['l_df'].plot(kind='hist', title='Double fautes du perdant', bins=20, xlim=(0,20), ylim=(0,30000))


# On peut également regarder quelques statistiques qualitatives et non quantitatives, telles que la nationalité des joueurs.

# In[7]:


plt.figure(1, figsize=(20,12))

country_names_loser=np.array(Counter(df['loser_ioc']).most_common())[:,0]
country_appearances_loser=list(map(int,np.array(Counter(df['loser_ioc']).most_common())[:,1]))
plt.subplot(2,4,1)
P1=plt.pie(country_appearances_loser,labels=country_names_loser)

country_names_winner=np.array(Counter(df['winner_ioc']).most_common())[:,0]
country_appearances_winner=list(map(int,np.array(Counter(df['winner_ioc']).most_common())[:,1]))
plt.subplot(2,4,2)
P2=plt.pie(country_appearances_winner,labels=country_names_winner)


# On choisit ensuite les variables que l'on souhaite prendre en compte pour les modèles prédictifs.
# 
# Au départ, j'ai pensé que l'objectif ne se limitait pas uniquement à la prédiction du résultat (gagné/perdu) du match, mais aussi à la prédiction de quelques statistiques sur lesquelles on peut également parier sur les sites de paris en ligne. J'ai donc ajouté l'ensemble des statistiques du match à la liste des variables souhaitées en sortie de nos modèles.
# 
# Par ailleurs, utiliser ces statistiques du match, indisponibles pour les parieurs avant le début du match, pour tenter de prédire l'issue du match, ne me semblait pas être une bonne idée, je ne les ai donc pas inclues dans les variables d'entrée de notre modèle, même lorsque l'on m'a confirmé que l'objectif était uniquement d'obtenir l'issue du match (Joueur gagnant).

# In[8]:


#On définit les variables d'entrée (= paramètres connus avant le match)
input_param=['p1_hand','p1_ht','p1_age','p1_rank','p1_rank_points','p2_hand','p2_ht','p2_age','p2_rank','p2_rank_points','surface']

#On définit les variables de sortie (= statistiques du match utilisées sur les sites de paris sportifs)
output_param=['p1_1stWon', 'p1_2ndWon', 'p1_SvGms', 'p1_ace', 'p1_bpFaced', 'p1_bpSaved', 'p1_df', 'p1_svpt', 'p2_1stWon', 'p2_2ndWon', 'p2_SvGms', 'p2_ace', 'p2_bpFaced', 'p2_bpSaved', 'p2_df', 'p2_svpt', 'winner_number']


# Afin d'anonymiser l'identité du gagnant et du perdant, on affecte aléatoirement un numéro "1" ou "2" aux joueurs gagnants et perdants, et donc à l'ensemble des variables utilisées dans cette analyse.
# 
# En pratique, on passe ici par un échange de valeurs entre le joueur gagnant et le joueur perdant dans une copie de notre jeu de données.
# 
# Cependant, cet algorithme est très long à exécuter (30 minutes pour l'ensemble des données), et mériterait d'être optimisé.

# In[9]:


df_shuffle=df.copy()
winner_number=np.zeros(df.shape[0])

pd.options.mode.chained_assignment = None  # default='warn'

#for i in range(0,1000):
for i in range(0,df.shape[0]):
    n=randint(1,2)
    winner_number[i]=n-1
    if n==1:
        
        #Inputs
        
        tmp=df_shuffle.iloc[i]['loser_id']
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('loser_id')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_id')]
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_id')]=tmp
  
        tmp=df_shuffle.iloc[i]['loser_hand']
        if (df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_hand')]=='R'):
            df_shuffle.iloc[i,df_shuffle.columns.get_loc('loser_hand')]=0
        else:
            df_shuffle.iloc[i,df_shuffle.columns.get_loc('loser_hand')]=1
            
        if (tmp=='R'):
            df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_hand')]=0
        else:
            df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_hand')]=1
        
        tmp=df_shuffle.iloc[i]['loser_ht']
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('loser_ht')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_ht')]
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_ht')]=tmp
        
        tmp=df_shuffle.iloc[i]['loser_age']
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('loser_age')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_age')]
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_age')]=tmp
        
        tmp=df_shuffle.iloc[i]['loser_rank']
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('loser_rank')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_rank')]
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_rank')]=tmp
        
        tmp=df_shuffle.iloc[i]['loser_id']
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('loser_rank_points')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_rank_points')]
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_rank_points')]=tmp
        
        #Outputs
        
        tmp=df_shuffle.iloc[i]['l_1stWon']
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('l_1stWon')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_1stWon')]
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_1stWon')]=tmp
        
        tmp=df_shuffle.iloc[i]['l_2ndWon']
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('l_2ndWon')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_2ndWon')]
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_2ndWon')]=tmp
        
        tmp=df_shuffle.iloc[i]['l_SvGms']
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('l_SvGms')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_SvGms')]
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_SvGms')]=tmp
        
        tmp=df_shuffle.iloc[i]['l_ace']
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('l_ace')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_ace')]
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_ace')]=tmp
        
        tmp=df_shuffle.iloc[i]['l_bpFaced']
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('l_bpFaced')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_bpFaced')]
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_bpFaced')]=tmp
        
        tmp=df_shuffle.iloc[i]['l_bpSaved']
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('l_bpSaved')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_bpSaved')]
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_bpSaved')]=tmp
        
        tmp=df_shuffle.iloc[i]['l_df']
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('l_df')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_df')]
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_df')]=tmp
        
        tmp=df_shuffle.iloc[i]['l_svpt']
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('l_svpt')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_svpt')]
        df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_svpt')]=tmp
        
    else:
        if (df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_hand')]=='R'):
            df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_hand')]=0
        else:
            df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_hand')]=1
            
        if (df_shuffle.iloc[i,df_shuffle.columns.get_loc('loser_hand')]=='R'):
            df_shuffle.iloc[i,df_shuffle.columns.get_loc('loser_hand')]=0
        else:
            df_shuffle.iloc[i,df_shuffle.columns.get_loc('loser_hand')]=1   


# Une fois l'anonymisation effectuée, on définit réellement notre jeu de données d'entrée

# In[10]:


#input_df=pd.DataFrame(columns=['p1_id','p1_hand','p1_ht','p1_age','p1_rank','p1_rank_points','p2_id','p2_hand','p2_ht','p2_age','p2_rank','p2_rank_points'])#,'surface'])
input_df=pd.DataFrame(columns=['p1_hand','p1_ht','p1_age','p1_rank','p1_rank_points','p2_hand','p2_ht','p2_age','p2_rank','p2_rank_points'])#,'surface'])

#input_df.iloc[:,input_df.columns.get_loc('p1_id')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('loser_id')].copy()
#input_df.iloc[:,input_df.columns.get_loc('p2_id')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('winner_id')].copy()
input_df.iloc[:,input_df.columns.get_loc('p1_hand')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('loser_hand')].copy()
input_df.iloc[:,input_df.columns.get_loc('p2_hand')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('winner_hand')].copy()
input_df.iloc[:,input_df.columns.get_loc('p1_ht')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('loser_ht')].copy()
input_df.iloc[:,input_df.columns.get_loc('p2_ht')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('winner_ht')].copy()
input_df.iloc[:,input_df.columns.get_loc('p1_age')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('loser_age')].copy()
input_df.iloc[:,input_df.columns.get_loc('p2_age')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('winner_age')].copy()
input_df.iloc[:,input_df.columns.get_loc('p1_rank')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('loser_rank')].copy()
input_df.iloc[:,input_df.columns.get_loc('p2_rank')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('winner_rank')].copy()
input_df.iloc[:,input_df.columns.get_loc('p1_rank_points')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('loser_rank_points')].copy()
input_df.iloc[:,input_df.columns.get_loc('p2_rank_points')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('winner_rank_points')].copy()
#input_df.iloc[:,input_df.columns.get_loc('surface')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('surface')].copy()


# On définit également notre jeu de données de sortie

# In[11]:


output_df=pd.DataFrame(columns=['p1_1stWon', 'p1_2ndWon', 'p1_SvGms', 'p1_ace', 'p1_bpFaced', 'p1_bpSaved', 'p1_df', 'p1_svpt', 'p2_1stWon', 'p2_2ndWon', 'p2_SvGms', 'p2_ace', 'p2_bpFaced', 'p2_bpSaved', 'p2_df', 'p2_svpt', 'winner_number'])

output_df.iloc[:,output_df.columns.get_loc('p1_1stWon')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_1stWon')].copy()
output_df.iloc[:,output_df.columns.get_loc('p2_1stWon')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_1stWon')].copy()
output_df.iloc[:,output_df.columns.get_loc('p1_2ndWon')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_2ndWon')].copy()
output_df.iloc[:,output_df.columns.get_loc('p2_2ndWon')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_2ndWon')].copy()
output_df.iloc[:,output_df.columns.get_loc('p1_SvGms')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_SvGms')].copy()
output_df.iloc[:,output_df.columns.get_loc('p2_SvGms')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_SvGms')].copy()
output_df.iloc[:,output_df.columns.get_loc('p1_ace')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_ace')].copy()
output_df.iloc[:,output_df.columns.get_loc('p2_ace')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_ace')].copy()
output_df.iloc[:,output_df.columns.get_loc('p1_bpFaced')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_bpFaced')].copy()
output_df.iloc[:,output_df.columns.get_loc('p2_bpFaced')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_bpFaced')].copy()
output_df.iloc[:,output_df.columns.get_loc('p1_bpSaved')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_bpSaved')].copy()
output_df.iloc[:,output_df.columns.get_loc('p2_bpSaved')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_bpSaved')].copy()
output_df.iloc[:,output_df.columns.get_loc('p1_df')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_df')].copy()
output_df.iloc[:,output_df.columns.get_loc('p2_df')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_df')].copy()
output_df.iloc[:,output_df.columns.get_loc('p1_svpt')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_svpt')].copy()
output_df.iloc[:,output_df.columns.get_loc('p2_svpt')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_svpt')].copy()
#output_df.iloc[:,output_df.columns.get_loc('winner_number')]=df_shuffle.iloc[:,df.columns.get_loc('winner_id')].copy()
output_df.iloc[:,output_df.columns.get_loc('winner_number')]=winner_number


# On sépare notre jeu de données en une base d'entraînement pour nos modèles, et une base de test, pour évaluer l'efficacité de nos modèles.
# 
# J'ai ici choisi de séparer les données sans les mélanger, pour prédire les 20% de matchs les plus récents en s'appuyant sur les 80% de matchs les plus anciens, comme si les matchs de test étaient situés dans le futur du point où notre modèle est entraîné.

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(input_df[:],output_df[:],test_size=0.2) #20%
print((X_train.shape))
print((y_train.shape))
print((X_test.shape))
print((y_test.shape))


# In[13]:


print(X_train)
print(y_train)


# Le premier modèle est un modèle de régression linéaire basique, pour obtenir une sorte de "probabilité" pour faire un choix du gagnant entre le joueur 1 et le joueur 2.
# 
# Ce modèle présente l'avantage de pouvoir, si nécessaire, établir des intervalles de rejet du résultat, par exemple avec une affectation à 0 si les résultats sont inférieurs à 0.3, une affectation à 1 si les résultats sont supérieurs à 0.7, et un rejet (qui pourrait correspondre à l'absence de pari, jugé trop risqué) pour un résultat compris entre 0.3 et 0.7.

# In[14]:


linear_model = LinearRegression()

m = linear_model.fit(X_train,y_train['winner_number'])


# On calcule le RMSE des prédictions de l'échantillon d'apprentissage et de l'échantillon de test. 
# 
# Cependant, il apparaît visiblement que les résultats sont "trop" parfaits et qu'un problème est donc rencontré.
# 
# Malgré une longue période passée à chercher une explication à ces résultats, je n'ai pu comprendre la raison de ces résultats.

# In[15]:


#calcul du RMSE
RMSE_train = np.sqrt(((y_train['winner_number'] - linear_model.predict(X_train))**2).sum()/len(y_train['winner_number']))
RMSE_test = np.sqrt(((y_test['winner_number'] - linear_model.predict(X_test))**2).sum()/len(y_test['winner_number']))

print(("RMSE en apprentissage : ", RMSE_train))
print(("RMSE en test : ", RMSE_test))


# J'ai également essayé un modèle de régression logistique, qui est largement utilisé dans le cas d'une prédiction d'un certain résultat précis parmi un nombre fini de résultats possibles, ce qui est ici notre cas.

# In[16]:


from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(solver='lbfgs', max_iter=5000)
logistic_model.fit(X_train, y_train['winner_number'])


# Cependant, comme pour la régression linéaire classique, les résultats obtenus sont "trop" bons, avec un taux d'erreur égal à 0 en apprentissage comme en test. 

# In[17]:


RMSE_train = np.sqrt(((y_train['winner_number'] - logistic_model.predict(X_train))**2).sum()/len(y_train['winner_number']))
RMSE_test = np.sqrt(((y_test['winner_number'] - logistic_model.predict(X_test))**2).sum()/len(y_test['winner_number']))
print(("RMSE en apprentissage : ", RMSE_train))
print(("RMSE en test : ", RMSE_test))


# Enfin, suite au temps perdu à chercher l'explication des résultats des modèles de régression, j'ai expérimenté différents modèles de deep learning pour tenter d'obtenir des résultats plus fiables.

# In[18]:


model = Sequential()
X_param_nb = X_train.shape[1]
model.add(Dropout(0.1, input_shape = (X_param_nb, )))
model.add(Dense(X_param_nb, activation = 'relu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, y_train['winner_number'], epochs = 20, validation_split = 0.2, batch_size = 256, shuffle=True)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training_set', 'Validation_set'])
plt.show()

#Affichage du taux de prédiction sur la base de test
print((model.evaluate(X_test, y_test['winner_number'])))


# Ce modèle ne réussit cependant pas à prédire avec fiabilité l'issue du match.
# 
# Bien que je ne m'attendais pas à un extrêmement bon résultat au vu des variables prises en compte dans la méthode, ce résultat est cependant très décevant. La variable de score et celle de classement notamment auraient dû fournir suffisamment d'informations pour qu'une légère tendance à la victoire pour les joueurs les mieux classés se dégage, et soit obtenue par ce modèle.

# Par la suite, j'ai expérimenté les résultats obtenus en modifiant les statistiques utilisées pour prédire le résultat du match. 
# 
# Cette fois-ci, on utilise donc les statistiques directement issues du match, ce qui est cependant un abus et qui ne pourrait être réellement appliqué pour de vrais paris sportifs.

# In[19]:


input_df_2=pd.DataFrame(columns=['p1_1stWon', 'p1_2ndWon', 'p1_SvGms', 'p1_ace', 'p1_bpFaced', 'p1_bpSaved', 'p1_df', 'p1_svpt', 'p2_1stWon', 'p2_2ndWon', 'p2_SvGms', 'p2_ace', 'p2_bpFaced', 'p2_bpSaved', 'p2_df', 'p2_svpt'])

input_df_2.iloc[:,input_df_2.columns.get_loc('p1_1stWon')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_1stWon')].copy()
input_df_2.iloc[:,input_df_2.columns.get_loc('p2_1stWon')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_1stWon')].copy()
input_df_2.iloc[:,input_df_2.columns.get_loc('p1_2ndWon')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_2ndWon')].copy()
input_df_2.iloc[:,input_df_2.columns.get_loc('p2_2ndWon')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_2ndWon')].copy()
input_df_2.iloc[:,input_df_2.columns.get_loc('p1_SvGms')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_SvGms')].copy()
input_df_2.iloc[:,input_df_2.columns.get_loc('p2_SvGms')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_SvGms')].copy()
input_df_2.iloc[:,input_df_2.columns.get_loc('p1_ace')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_ace')].copy()
input_df_2.iloc[:,input_df_2.columns.get_loc('p2_ace')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_ace')].copy()
input_df_2.iloc[:,input_df_2.columns.get_loc('p1_bpFaced')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_bpFaced')].copy()
input_df_2.iloc[:,input_df_2.columns.get_loc('p2_bpFaced')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_bpFaced')].copy()
input_df_2.iloc[:,input_df_2.columns.get_loc('p1_bpSaved')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_bpSaved')].copy()
input_df_2.iloc[:,input_df_2.columns.get_loc('p2_bpSaved')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_bpSaved')].copy()
input_df_2.iloc[:,input_df_2.columns.get_loc('p1_df')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_df')].copy()
input_df_2.iloc[:,input_df_2.columns.get_loc('p2_df')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_df')].copy()
input_df_2.iloc[:,input_df_2.columns.get_loc('p1_svpt')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_svpt')].copy()
input_df_2.iloc[:,input_df_2.columns.get_loc('p2_svpt')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_svpt')].copy()

X_train, X_test, y_train, y_test = train_test_split(input_df_2[:],output_df[:],test_size=0.2) #20%


# On réessaie donc les modèles testés plus tôt, en obtenant cette fois-ci de bien meilleurs résultats

# In[20]:


linear_model = LinearRegression()

m = linear_model.fit(X_train,y_train['winner_number'])

RMSE_train = np.sqrt(((y_train['winner_number'] - linear_model.predict(X_train))**2).sum()/len(y_train['winner_number']))
RMSE_test = np.sqrt(((y_test['winner_number'] - linear_model.predict(X_test))**2).sum()/len(y_test['winner_number']))

print(("RMSE en apprentissage : ", RMSE_train))
print(("RMSE en test : ", RMSE_test))


# In[21]:


from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(solver='lbfgs', max_iter=5000)
logistic_model.fit(X_train, y_train['winner_number'])

RMSE_train = np.sqrt(((y_train['winner_number'] - logistic_model.predict(X_train))**2).sum()/len(y_train['winner_number']))
RMSE_test = np.sqrt(((y_test['winner_number'] - logistic_model.predict(X_test))**2).sum()/len(y_test['winner_number']))
print(("RMSE en apprentissage : ", RMSE_train))
print(("RMSE en test : ", RMSE_test))


# In[22]:


from keras.callbacks import EarlyStopping
model = Sequential()
X_param_nb = X_train.shape[1]
es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20)
model.add(Dropout(0.1, input_shape = (X_param_nb, )))
model.add(Dense(X_param_nb, activation = 'relu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, y_train['winner_number'], epochs = 1000, validation_split = 0.2, batch_size = 256, shuffle=True, callbacks=[es])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training_set', 'Validation_set'])
plt.show()

#Affichage du taux de prédiction sur la base de test
print(("Loss value for test data : ", model.evaluate(X_test, y_test['winner_number'])[0]))
print(("Accuracy value for test data : ", model.evaluate(X_test, y_test['winner_number'])[1]))


# Si j'avais eu davantage de temps, plusieurs pistes auraient pu être intéressantes pour améliorer les résultats ou pousser davantage le sujet, à savoir :
# * Une simple normalisation (ou centrage et réduction) des données aurait pu être intéressante pour éviter les biais qui ont pu être apportés par les différentes échelles de valeurs prises par les variables
# * Étudier les précédents résultats des joueurs (par exemple par le biais d'un modèle LSTM) pourrait apporter de solides modèles davantage personnalisés
# * Définir des caractéristiques supplémentaires, notamment la différence entre les statistiques du joueur 1 et du joueur 2, pour les donner en entrée des modèles, pourrait aider les modèles à apprendre et prédire plus efficacement
# * J'aurais souhaité comprendre la cause des résultats obtenus dans les modèles de régression sur les premières données d'entrée.
# * Un calcul des covariances ou l'évaluation des corrélations entre les joueurs "gagnants" et "perdants" pourrait permettre de détecter les variables permettant de discriminer les joueurs selon leur classe "gagnant" ou "perdant", pour laisser de côté les variables moins utiles.
# * Certaines variables qualitatives, notamment la surface, le Round et le Tourney_level, pourraient être transformées en plusieurs variables quantitatives binaires pour les fournir en entrée aux modèles devrait également apporter de meilleurs résultats.
# * En outre de l'issue du match, tenter de prédire le score des différents sets pourrait également être intéressant, dans une optique de paris en ligne sur les scores précis;
# * Enfin, la méthode actuellement utilisée pour "anonymiser" les joueurs "gagnants" et "perdants" est très longue à exécuter et est également coûteuse en ressources (CPU à 100% durant 30 minutes). Trouver une meilleure méthode ou au moins l'optimiser devrait permettre de réduire fortement ce coût en temps et en ressources.
