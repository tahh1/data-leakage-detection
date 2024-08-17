#!/usr/bin/env python
# coding: utf-8

# # Titanic survival
# 
# ### Who will make it?
# 

# This notebook is prepared for training purpose.
# 
# We will explore the [Titanic survival data](https://www.kaggle.com/c/titanic) , and model the survival with decision trees.
# (see [Decision Tree course](https://www.linkedin.com/learning/machine-learning-ai-foundations-decision-trees))

# ## 1. GOALS
# 
#  - predict survival rate of titanic passengers
#  - practice decision trees
#  - build a small data science project
# 

# ## 2. Data understanding
# 
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("../input/train.csv")
print((data.info()))
print(("\n Embarked options: ", data["Embarked"].value_counts()))

data.describe()


# ## Apply less memory-consuming data types to the data (see [Dataquest blog](https://www.dataquest.io/blog/pandas-big-data/))

# In[ ]:


data.head()


# In[4]:


from sklearn.model_selection import train_test_split


# setting up new data types
dtypes_col       = data.columns
dtypes_type_old  = data.dtypes
dtypes_type      = ['int16', 'bool','category','object','category','float32','int8','int8','object','float32','object','category']
optimized_dtypes = dict(list(zip(dtypes_col, dtypes_type)))

#read data once again with optimized columns
data_optimized = pd.read_csv("../input/train.csv",dtype=optimized_dtypes)
test_optimized = pd.read_csv("../input/test.csv",dtype=optimized_dtypes)

#splitting data to train and validation
train, valid = train_test_split(data_optimized, test_size=0.2)


combined = {"train":train,
            "valid":valid,
            "test":test_optimized}

print((data_optimized.info()))


# <span style="color:green"> Hooray! We saved more than half of the memory and the table is read properly!</span>

# ## Do we have any missing data?

# In[5]:


data_optimized.isnull().sum()


# "Cabin" column will not be interesting for us, because there is a lot of missing data.
# 
# "Age" column may be important for the model, and but some of decision tree models have a mechanism for estimating missing value based on correlation with other values if possible.
# 
# 
# I will use [Kaggle, Anisotropic profile](https://www.kaggle.com/arthurtok/interactive-porto-insights-a-plot-ly-tutorial) to get the data analysis step by step.

# In[6]:


combined_cleaned = {}
for i,data in list(combined.items()):
    combined_cleaned[i] = data.drop('Cabin', 1).copy()


# In[7]:


#numerical features

train_numeric = combined_cleaned["train"].select_dtypes(include=['float32','int16','int8','bool'])

colormap = plt.cm.cubehelix_r
plt.figure(figsize=(16,12))

plt.title('Pearson correlation of numeric features', y=1.05, size=15)
sns.heatmap(train_numeric.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# Two interesting correlations are:
#  - positive (0.41) for "SibSp" and "Parch"
#  - negative  (-0.31) for "SibSp" and "Age"

# In[8]:


# category features# category features

#we do not count NaN categories
def survived_percent(categories,column):
    survived_list = []
    for c in categories.dropna():
        count = combined_cleaned["train"][combined_cleaned["train"][column] == c][column].count()
        survived = combined_cleaned["train"][combined_cleaned["train"][column] == c]["Survived"].sum()/count
        survived_list.append(survived)
    return survived_list    
   
category_features_list = ["Sex", "Embarked","Pclass"]
category_features = {}

for x in category_features_list:
    unique_values = combined_cleaned["train"][x].unique().dropna()
    survived = survived_percent(unique_values,x)
    category_features[x] = [unique_values, survived]


fig, axs = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
cb_dark_blue = (0/255,107/255,164/255)
cb_orange = (255/255, 128/255, 14/255)
cb_grey = (89/255, 89/255, 89/255)
color=[cb_dark_blue,cb_orange,cb_grey]

font_dict = {'fontsize':20, 
             'fontweight':'bold',
             'color':"white"}

for i,cat in enumerate(category_features.keys()):
    number_categories = len(category_features[cat][0])
    axs[i].bar(list(range(number_categories)), category_features[cat][1], color=color[:number_categories])
    axs[i].set_title("Survival rate " + cat ,fontsize=20, fontweight='bold' )
    for j,indx in enumerate(category_features[cat][1]):
        label_text = category_features[cat][0][j]
        x = j
        y = indx
        axs[i].annotate(label_text, xy = (x-0.15 ,y/2), **font_dict )

for i in range(3):
    axs[i].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    axs[i].patch.set_visible(False)



# Gender is a key factor insurvival rate.

# ## 3. Filling missing data

# For "Embarked" colum we will use most popular category which is "S", because there are only 2 records missing.
# 
# For "Age" column as a first attempt we will use age distribution $F_{Age}$ known from the rest of the data and generate random number from it. 

# In[13]:


# filling NaN in "Embarked" and "Fare"

for i,data in list(combined_cleaned.items()):
    data["Embarked"].fillna(value="S",inplace=True) # S is most popular value 
    mean_Fare = data["Fare"].mean()
    data["Fare"].fillna(value=mean_Fare,inplace=True)


# In[14]:


# filling NaN in "Age" 
fig, ax = plt.subplots( figsize=(6,4))
x = combined_cleaned["train"]["Age"].dropna()
hist, bins = np.histogram( x,bins=15)

#plot of histogram
ax.hist(x, normed=True, color='grey')
ax.set_title('Age histogram')
plt.show()


# In[15]:


from random import choices

bin_centers = 0.5*(bins[:len(bins)-1]+bins[1:])
probabilities = hist/hist.sum()

#dictionary with random numbers from existing age distribution
for i,data in list(combined_cleaned.items()):
    data["Age_rand"] = data["Age"].apply(lambda v: np.random.choice(bin_centers, p=probabilities))
    Age_null_list   = data[data["Age"].isnull()].index
    
    data.loc[Age_null_list,"Age"] = data.loc[Age_null_list,"Age_rand"]
    


# 

# ## 4. Modelling

# ### a) C&RT decision tree
# 

# In[16]:


from sklearn import preprocessing,tree
from sklearn.model_selection import GridSearchCV

tree_data = {}
tree_data_category = {}

for i,data in list(combined_cleaned.items()):
    tree_data[i] = data.select_dtypes(include=['float32','int16','int8']).copy()
    tree_data_category[i] = data.select_dtypes(include=['category'])

    #categorical variables handling
    for column in tree_data_category[i].columns:
        le = preprocessing.LabelEncoder()
        le.fit(data[column])
        tree_data[i][column+"_encoded"] = le.transform(data[column])


# In[ ]:


#finding best fit with gridsearch
param_grid = {'min_samples_leaf':np.arange(20,50,5),
              'min_samples_split':np.arange(20,50,5),
              'max_depth':np.arange(3,6),
              'min_weight_fraction_leaf':np.arange(0,0.4,0.1),
              'criterion':['gini','entropy']}
clf = tree.DecisionTreeClassifier()
tree_search = GridSearchCV(clf, param_grid, scoring='average_precision')

X =  tree_data["train"].drop("PassengerId",axis=1)
Y = combined_cleaned["train"]["Survived"]
tree_search.fit(X,Y)

print(("Tree best parameters :",tree_search.best_params_))
print(("Tree best estimator :",tree_search.best_estimator_ ))
print(("Tree best score :",tree_search.best_score_ ))


# In[17]:


tree_best_parameters = tree_search.best_params_
tree_optimized = tree.DecisionTreeClassifier(**tree_best_parameters)
tree_optimized.fit(X,Y)

train_columns = list(tree_data["train"].columns)
train_columns.remove("PassengerId")
fig, ax = plt.subplots( figsize=(6,4))
ax.bar(list(range(len(X.columns))),tree_optimized.feature_importances_ )
plt.xticks(list(range(len(X.columns))),X.columns,rotation=90)
ax.set_title("Feature importance")
plt.show()


# In[18]:


import graphviz 

dot_data = tree.export_graphviz(tree_optimized, 
                                out_file=None,
                                filled=True, 
                                rounded=True,  
                                special_characters=True,
                               feature_names = train_columns) 
graph = graphviz.Source(dot_data)
graph


# ## Prediction on the test set

# In[21]:


test_without_PId = tree_data["test"].drop("PassengerId",axis=1)
prediction_values = tree_optimized.predict(test_without_PId).astype(int)
prediction = pd.DataFrame({"PassengerId":tree_data["test"]["PassengerId"],
                           "Survived":prediction_values})

prediction.head()
prediction.to_csv("Titanic_tree_prediction.csv",index=False)


# ## Performance evaluation

# In[ ]:


from sklearn.metrics import confusion_matrix

evaluation = {}
cm = {}


valid_without_PId = tree_data["valid"].drop("PassengerId",axis=1)
evaluation["tree"] = tree_optimized.predict(valid_without_PId).astype(int)
survival_from_data = combined_cleaned["valid"]["Survived"].astype(int)

print((survival_from_data.value_counts()))

cm["tree"] = confusion_matrix(survival_from_data, evaluation["tree"])
cm["tree"] = cm["tree"].astype('float') / cm["tree"].sum(axis=1)[:, np.newaxis]

cm["tree"]


# In[ ]:


import itertools

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(list(range(cm.shape[0])), list(range(cm.shape[1]))):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ## Into the woods - random forest
# 

# In[22]:


from sklearn.ensemble import RandomForestClassifier

clf_forest = RandomForestClassifier(n_estimators=10,min_samples_leaf=20, max_depth=4,min_weight_fraction_leaf=0.1)
clf_forest.fit(X,Y)


# In[25]:


prediction_values_forest = clf_forest.predict(test_without_PId).astype(int)
prediction_forest = pd.DataFrame({"PassengerId":tree_data["test"]["PassengerId"],
                           "Survived":prediction_values_forest})

#prediction_forest.to_csv("Titanic_tree_prediction_forest.csv",index=False)


# ## Random forest with parameters optimization

# In[ ]:


param_grid = {'n_estimators':np.arange(3,11,2),
              'max_depth':np.arange(3,6),
              'min_weight_fraction_leaf':np.arange(0,0.4,0.1),
              'criterion':['gini','entropy']}
clf = RandomForestClassifier()
forest_search = GridSearchCV(clf, param_grid, scoring='precision')

forest_search.fit(X,Y)

print(("Forest best parameters :",forest_search.best_params_))
print(("Forest best estimator :",forest_search.best_estimator_ ))
print(("Forest best score :",forest_search.best_score_ ))



# In[ ]:


clf_forest = RandomForestClassifier(**forest_search.best_params_)
clf_forest.fit(X,Y)


# In[ ]:


prediction_values_forest = clf_forest.predict(test_without_PId).astype(int)
prediction_forest = pd.DataFrame({"PassengerId":tree_data["test"]["PassengerId"],
                           "Survived":prediction_values_forest})

prediction_forest.to_csv("Titanic_tree_prediction_forest.csv",index=False)


# ## Performance evaluation

# In[ ]:


evaluation["forest"] = clf_forest.predict(valid_without_PId).astype(int)

cm["forest"] = confusion_matrix(survival_from_data, evaluation["forest"])
cm["forest"] = cm["forest"].astype('float') / cm["forest"].sum(axis=1)[:, np.newaxis]

cm["forest"]

plot_confusion_matrix(cm["forest"], classes=["No","Yes"], 
                      title='Normalized confusion matrix')


# ## Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X, Y)


# In[28]:


prediction_values_NaiveBayes = gnb.predict(test_without_PId).astype(int)
prediction_NaiveBayes = pd.DataFrame({"PassengerId":tree_data["test"]["PassengerId"],
                           "Survived":prediction_values_NaiveBayes})

#prediction_NaiveBayes.to_csv("Titanic_tree_prediction_NaiveBayes.csv",index=False)


# ## Performance evaluation

# In[ ]:


evaluation["NB"] = gnb.predict(valid_without_PId).astype(int)

cm["NB"] = confusion_matrix(survival_from_data, evaluation["NB"])
cm["NB"] = cm["NB"].astype('float') / cm["NB"].sum(axis=1)[:, np.newaxis]

cm["NB"]

plot_confusion_matrix(cm["NB"], classes=["No","Yes"], 
                      title='Normalized confusion matrix')


# ## Support Vector Machines

# In[ ]:


from sklearn import svm
clf_svm = svm.SVC()
clf_svm.fit(X,Y)


# In[ ]:





# In[ ]:


prediction_values_svm = clf_svm.predict(test_without_PId).astype(int)
prediction_svm = pd.DataFrame({"PassengerId":tree_data["test"]["PassengerId"],
                           "Survived":prediction_values_svm})

#prediction_svm.to_csv("Titanic_tree_prediction_svm.csv",index=False)

