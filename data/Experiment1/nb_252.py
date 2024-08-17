#!/usr/bin/env python
# coding: utf-8

# In[105]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix, classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt


import plotly.express as px

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV



from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier


from sklearn.pipeline import Pipeline
import plotly.figure_factory as ff


#!pip install pyod

from pyod.models.copod import COPOD
from sklearn.manifold import TSNE


# In[2]:


pd.set_option('display.max_columns', None)


# In[3]:


player = pd.read_csv(r'C:\Users\tab\Desktop\Modul 3\ujian\Ujian_Modul3_JCDS_Bekasi-master\Ujian_Modul3_JCDS_Bekasi-master\nba_players.csv')
player.head()


# In[4]:


player.drop(['Unnamed: 0.1', 'Unnamed: 0'],axis=1, inplace=True)


# In[5]:


len(player)


# In[6]:


player.describe().T


# In[7]:


list_item = []
for col in player.columns:
    list_item.append([col, player[col].dtype, player[col].isna().sum(), round((player[col].isna().sum()/len(player[col]))*100,2),
                      player[col].nunique()])

desc = pd.DataFrame(columns=['column', 'dtype', 'NumbOfNull', 'NullPercent', 'NUnique'],data=list_item)


# In[8]:


desc


# In[9]:


player.info()


# In[10]:


player.shape


# # Data cleansing & preprocessing

# In[11]:


player.head()


# Removing unused and non-numeric features :

# In[12]:


player2 =player.drop(['player_name', 'team_abbreviation', 'college' , 'country', 'team_abbreviation', 'draft_year', 'draft_round', 'draft_number', 'season'], axis=1 )


# In[13]:


player2.head()


# In[14]:


player2.info()


# Check if there are any duplicate values

# In[15]:


for col in player2.columns:
    player2[col] =player2[col].astype(int)
    


# In[16]:


player2.duplicated().values.any()


# In[17]:


player2.corr()


# # Data Insights and visualization

# In[18]:


plt.figure(figsize=(16,8))
sns.heatmap(player2.corr())
plt.show()


# Relationship between potential player and depended Var

# In[36]:


#better vizualisation

sns.pairplot(player2)


# Correlation with better Visualization

# In[20]:


player2.corr()['potential_player'].sort_values(ascending=False).to_frame()


# In[27]:


sns.countplot(x='ast', hue='potential_player', data=player2)
plt.show()


# from the graph we can conclude that lowest ast have so much number of unpotential player

# In[32]:


plt.figure(figsize=(16,10))
sns.countplot(x='age', hue='potential_player', data=player2)
plt.show()


# from the graph we can conclude that age distributin around 26 have so much number of potential player

# In[31]:


plt.figure(figsize=(16,10))
sns.countplot(x='pts', hue='potential_player', data=player2)
plt.show()


# from the graph we can conclude that pts distributin around 21 have so much number of potential player

# # Building training set and testing set

# In[38]:


from sklearn.model_selection import train_test_split
x = player2.drop('potential_player', axis=1)
y = player2['potential_player']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# # Using 3 Machine Learning Model for Classification

# In[39]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[43]:


model1 = KNeighborsClassifier()
model2 = LogisticRegression(max_iter=400)
model3 = RandomForestClassifier()


# We use and compare 3 classification models which are KNN, LR, RFC, the work of these models are  :
# 
# 1. Logistic Regression make a linear regression  to plot a line and split the trained data to be false and true according value of potential player. Logistic Regression make a model by giving an respective intercept for each feature and sum them with value of slope. In this case since this model have many features, this model is called multinomial logistic regression, which use sigmoid function to predict the probability of coming results.
# 2. K-Nearest Neightborhood works based on minimum distance from the query instance to the training samples to determine the K-nearest neighbors. After we gather K nearest neighbors, we take simple majority of these K-nearest neighbors to be the prediction of probability of potential player
# 3. Random Forest performing regression and classification task by using technique called bagging. First RFC select k features, and calculate the node "d", next RFC split node into daughter and repeat until create many trees which ensemble forest. The final predictions of the random forest are made by averaging the predictions of each individual tree.

# In[45]:


model2.fit(x_train, y_train)
model1.fit(x_train, y_train)
model3.fit(x_train, y_train)


# In[46]:


plot_confusion_matrix(model2, x_train, y_train)


# In[47]:


plot_confusion_matrix(model1, x_train, y_train)


# In[48]:


plot_confusion_matrix(model3, x_train, y_train)


# In[53]:


logreg_prob = model2.predict_proba(x_test)
problogreg = logreg_prob[:,1]
fpr_logreg, tpr_logreg, threshold_logreg = roc_curve(y_test, problogreg)
roc_auc_logreg = auc(fpr_logreg, tpr_logreg)


# In[55]:


plt.title('ROC Logistic Regression')
plt.plot(fpr_logreg, tpr_logreg, 'blue', label='AUC Logistic Regression = {}'.format(round(roc_auc_logreg,3)))
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.fill_between(fpr_logreg, tpr_logreg, 0, facecolor='azure', alpha=1)
plt.legend(loc='lower right')


# In[ ]:


logreg_grid_pred = model2.predict(x_test)
rfc_grid_pred = model1.predict(x_test)
knn_random_pred = model3.predict(x_test)

knn_random_pred


# In[93]:


tnlogreg, fplogreg, fnlogreg, tplogreg = confusion_matrix(y_test, logreg_grid_pred).ravel()
print((confusion_matrix(y_test, logreg_grid_pred)))


# In[95]:


tnrfc, fprfc, fnrfc, tprfc = confusion_matrix(y_test, rfc_grid_pred).ravel()
print((confusion_matrix(y_test, rfc_grid_pred)))


# In[97]:


tnknn, fpknn, fnknn, tpknn = confusion_matrix(y_test, knn_random_pred).ravel()
print((confusion_matrix(y_test, knn_random_pred)))


# In[61]:


knn_prob = model1.predict_proba(x_test)
probknn = knn_prob[:,1]
fpr_knn, tpr_knn, threshold_knn = roc_curve(y_test, probknn)
roc_auc_knn = auc(fpr_knn, tpr_knn)


# In[62]:


plt.title('ROC K-Nearest Neighbor')
plt.plot(fpr_knn, tpr_knn, 'blue', label='AUC K-Nearest Neighbor = {}'.format(round(roc_auc_knn,3)))
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.fill_between(fpr_knn, tpr_knn, 0, facecolor='azure', alpha=1)
plt.legend(loc='lower right')


# In[63]:


rfc_prob = model3.predict_proba(x_test)
probrfc = rfc_prob[:,1]
fpr_rfc, tpr_rfc, threshold_rfc = roc_curve(y_test, probrfc)
roc_auc_rfc = auc(fpr_rfc, tpr_rfc)


# In[64]:


plt.title('ROC Random Forest')
plt.plot(fpr_rfc, tpr_rfc, 'blue', label='AUC Random Forest = {}'.format(round(roc_auc_rfc,3)))
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.fill_between(fpr_rfc, tpr_rfc, 0, facecolor='azure', alpha=1)
plt.legend(loc='lower right')


# In[89]:


pred1 = model1.predict(x_test)
pred2 = model2.predict(x_test)
pred3 = model3.predict(x_test)


# In[87]:


accuracy_score(y_test, pred)


# In[123]:


roc_auc_logreg


# In[140]:


modelComparation = pd.DataFrame({
    "Logistic Regression": [accuracy_score(y_test, pred2),precision_score(y_test, logreg_grid_pred),recall_score(y_test, logreg_grid_pred),f1_score(y_test, logreg_grid_pred), tplogreg, tnlogreg, fplogreg, fnlogreg, roc_auc_logreg],
    
    "Random Forest": [accuracy_score(y_test, pred1),precision_score(y_test, rfc_grid_pred),recall_score(y_test, rfc_grid_pred),f1_score(y_test, rfc_grid_pred), tprfc, tnrfc, fprfc, fnrfc, roc_auc_rfc],
    
    "K-Nearest Neighbor": [accuracy_score(y_test, pred3),precision_score(y_test, knn_random_pred),recall_score(y_test, knn_random_pred),f1_score(y_test, knn_random_pred), tpknn, tnknn, fpknn, fnknn, roc_auc_knn],
}, index=['Accuracy Score', 'Precision Score', 'Recall Score', 'F1 Score', 'True Positive', 'True Negative', 'False Positive', 'False Negative', 'ROC'])
modelComparation


#  From this table, it can be concluded that the KNN can provide the best results because it has the highest accuracy, precision, and F1 score. Also, this model has a low False Positive. This is important because the more False Positive the model has, the greater the loss for the institution.KNN also have the highest ROC score. To validate the model, we'll be using the ROC and AUC scores to determine the sensitivity of the three models to the threshold.

# In[138]:


plt.figure(figsize=(16,10))
plt.subplot(221)
plt.title('ROC Logistic Regression')
plt.plot(fpr_logreg, tpr_logreg, 'red', label='AUC Logistic Regression = {}'.format(round(roc_auc_logreg,3)))
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.fill_between(fpr_logreg, tpr_logreg, 0, facecolor='pink', alpha=1)
plt.legend(loc='lower right')

plt.subplot(222)
plt.title('ROC Random Forest')
plt.plot(fpr_rfc, tpr_rfc, 'red', label='AUC Random Forest = {}'.format(round(roc_auc_rfc,4)))
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.fill_between(fpr_rfc, tpr_rfc, 0, facecolor='pink', alpha=1)
plt.legend(loc='lower right')


plt.subplot(223)
plt.title('ROC K-Nearest Neighbor')
plt.plot(fpr_knn, tpr_knn, 'blue', label='AUC K-Nearest Neighbor = {}'.format(round(roc_auc_knn,3)))
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.fill_between(fpr_knn, tpr_knn, 0, facecolor='azure', alpha=1)
plt.legend(loc='lower right')


# ## Model Optimization

# In[108]:


param_random_knn = {
    'algorithm' : ['auto','ball_tree', 'kd_tree','brute'],
    'n_neighbors' : [3,5,7],
    'leaf_size' : [10, 30] 
}
random_search_knn = RandomizedSearchCV(estimator=model1, param_distributions=param_random_knn,cv=5, scoring='roc_auc', n_jobs=-1)
random_search_knn.fit(x_train, y_train)


# In[109]:


knn_random = KNeighborsClassifier(leaf_size=10, n_neighbors=7)
knn_random.fit(x_train, y_train)


# In[110]:


knn_random = random_search_knn.best_estimator_


# In[119]:


knn_random_pred = knn_random.predict(x_test)
knn_random_proba = knn_random.predict_proba(x_test)

tnknn, fpknn, fnknn, tpknn = confusion_matrix(y_test, knn_random_pred).ravel()


# In[120]:


knn_prob = knn_random.predict_proba(x_test)
probknn = knn_prob[:,1]
fpr_knn, tpr_knn, threshold_knn = roc_curve(y_test, probknn)
roc_auc_knn = auc(fpr_knn, tpr_knn)


# In[118]:


print((classification_report(y_test, knn_random_pred)))


# # Using model to predict new players dataset

# In[142]:


new_player = pd.read_csv(r'C:\Users\tab\Desktop\Modul 3\ujian\Ujian_Modul3_JCDS_Bekasi-master\Ujian_Modul3_JCDS_Bekasi-master\new_players.csv')
new_player.head()


# In[143]:


new_player.drop([ 'Unnamed: 0', 'player_id', 'college', 'country'],axis=1, inplace=True)


# In[144]:


for col in new_player.columns:
    player2[col] =player2[col].astype(int)
    


# In[145]:


pred1 = model1.predict(new_player)


# In[149]:


new_player['pred'] = pred1
new_player

