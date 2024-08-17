#!/usr/bin/env python
# coding: utf-8

# ## linear regression

# In[ ]:


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import xgboost as xgb

import matplotlib.pyplot as plt


# In[ ]:


pca_df = pd.read_csv("pca_train.csv", low_memory = False)

X_train, X_test, y_train, y_test = train_test_split(pca_df.iloc[:,1:], pca_df.iloc[:,0], 
                                                    test_size=0.4, random_state=42, stratify = pca_df.iloc[:,0])
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, 
                                                    test_size=0.5, random_state=42, stratify = y_test)

print(f'Shape of train {X_train.shape} \n Shape of validate {X_val.shape} \n Shape of test {X_test.shape}')


# In[ ]:


clf = LogisticRegression(max_iter=10000, tol=0.1, solver = "saga")
clf.fit(X_train, y_train)


# In[ ]:


y_val_pred = clf.predict(X_val)
y_test_pred = clf.predict(X_test)


# In[ ]:


print((roc_auc_score(y_val, y_val_pred)))
print((roc_auc_score(y_test, y_test_pred)))


# ## random forest

# In[ ]:


# feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, 
                               random_state=42, 
                               max_features = 'auto',
                               n_jobs=-1, verbose = 1)

# Fit on training data
model.fit(X_train, y_train.values.ravel())


# In[ ]:


X_train_rf_predictions = model.predict(X_train)
X_train_rf_probs = model.predict_proba(X_train)[:, 1]

rf_predictions = model.predict(X_test)
rf_probs = model.predict_proba(X_test)[:, 1]
rf_probs_val = model.predict_proba(X_val)[:, 1]


# In[ ]:


print((roc_auc_score(y_test, rf_probs)))
print((roc_auc_score(y_val.values.ravel(), rf_probs_val)))


# In[ ]:


# AUC-ROC curve
rf_roc_auc = roc_auc_score(y_test, rf_predictions)
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % roc_auc_score(y_test.values.ravel(), rf_probs))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('rf_ROC')
plt.show()


# ## xgboost

# In[ ]:


train_dmatrix = xgb.DMatrix(data=X_train,label=y_train)
val_dmatrix = xgb.DMatrix(data=X_val,label=y_val)
test_dmatrix = xgb.DMatrix(data=X_test,label=y_test)


# In[ ]:


param = {'eval_metric': 'auc', 'alpha' : 69.22788372146633, 'eta' : 0.2094786309587594,
        'min_child_weight' : 3.2793154567014025, 'subsample' : 0.8724888256198183,
        'objective': 'binary:logistic'}


# In[ ]:


evallist = [(val_dmatrix, 'eval'), (train_dmatrix, 'train')]
num_round = 50

bst = xgb.train(param, train_dmatrix, num_round, evallist)

bst.save_model('xgb1.model')


# In[ ]:


preds = bst.predict(test_dmatrix)
roc_auc_score(y_test, preds)


# In[ ]:


# plot importance features
xgb.plot_importance(bst, max_num_features = 25)

# plot tree importance
fig, ax = plt.subplots(figsize=(80, 70))
xgb.plot_tree(bst, num_trees=3, ax=ax)
# plt.show()
plt.savefig("tree.jpeg", quality = 100)


# ----------

# ## predicting for the holdout file

# In[ ]:


pca_holdout_df = pd.read_csv("pca_holdout.csv", low_memory = False)

holdout_dmatrix = xgb.DMatrix(data=pca_holdout_df)

holdout_pred = bst.predict(holdout_dmatrix)


# In[ ]:


ID = pd.read_csv("../data/2020_Competition_Holdout.csv", usecols = [0])

holdout_pred_df = pd.concat([ID, pd.DataFrame(holdout_pred)], axis = 1)

holdout_pred_df.columns = ['ID', 'SCORE']

holdout_pred_df["RANK"] = holdout_pred_df["SCORE"].rank(method='max', ascending = False)

holdout_pred_df.to_csv("CaseCompetition_Jennie_Sun.csv", index = False)

