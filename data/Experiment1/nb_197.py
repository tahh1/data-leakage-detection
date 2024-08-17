#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.bigdatauniversity.com"><img src="https://ibm.box.com/shared/static/cw2c7r3o20w9zn8gkecaeyjhgw3xdgbj.png" width="400" align="center"></a>
# 
# <h1 align="center"><font size="5">Classification with Python</font></h1>

# In this notebook we try to practice all the classification algorithms that we learned in this course.
# 
# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Lets first load required libraries:

# In[1]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### About dataset

# This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# |----------------|---------------------------------------------------------------------------------------|
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |

# Lets download the dataset

# In[2]:


get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# ### Load Data From CSV File  

# In[3]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[4]:


df.shape


# ### Convert to date time object 

# In[5]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 
# 

# Let’s see how many of each class is in our data set 

# In[6]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection 
# 

# Lets plot some columns to underestand data better:

# In[7]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[8]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[9]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction

# ### Lets look at the day of the week people get the loan 

# In[10]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4 

# In[11]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values

# Lets look at gender:

# In[12]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay their loans while only 73 % of males pay theirs.
# 

# Lets convert male to 0 and female to 1:
# 

# In[13]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# Convert loan_status PAIDOFF to 1 and COLLECTION to zero to make evaluation later on easier

# In[14]:


df['loan_status'].replace(to_replace = ['PAIDOFF', 'COLLECTION'], value = [0, 1], inplace = True)
df['loan_status'].head()


# ## One Hot Encoding  
# #### How about education?

# In[15]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Feature before One Hot Encoding

# In[16]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame 

# In[17]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.head()


# ### Feature selection

# Lets define feature sets, X:

# In[18]:


X = Feature
X[0:5]


# What are our labels?

# In[19]:


y = df['loan_status'].values
y[0:5]


# ## Normalize Data 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split)

# In[20]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Classification 

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression
# 
# 
# 
# __ Notice:__ 
# - You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# - You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# - You should include the code of the algorithm in the following cells.

# # K Nearest Neighbor(KNN)
# Notice: You should find the best k to build the model with the best accuracy.  
# **warning:** You should not use the __loan_test.csv__ for finding the best k, however, you can split your train_loan.csv into train and test to find the best __k__.

# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2, random_state = 4)
# test accuracy of various values of K
Ks = 10
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))
for n in range(1, Ks):
    neighbour = KNeighborsClassifier(n_neighbors = n).fit(X_train, y_train)
    yhat = neighbour.predict(X_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n - 1] = np.std(yhat == y_test)/np.sqrt(yhat.shape[0])

# Show accuracy of various K values
import matplotlib.pyplot as plt
k_values = list(range(1, Ks))
plt.plot(list(range(1,Ks)),mean_acc,'g')
plt.fill_between(list(range(1,Ks)),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(list(range(1,Ks)),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print(("The best accuracy was", mean_acc.max(), "with k =", mean_acc.argmax() + 1))


# In[22]:


# Train model using entire dataset (i.e. X and y, not X_train, y_train) with k = 7
k = 7
final_neighbour_model = KNeighborsClassifier(n_neighbors = k).fit(X, y)


# # Decision Tree

# In[23]:


# Train decision tree model using X_train, y_train at various depths
from sklearn.tree import DecisionTreeClassifier
depths = 10
mean_acc = np.zeros((depths - 1))
std_acc = np.zeros((depths - 1))
for depth in range(1, depths):
    loanstatustree = DecisionTreeClassifier(criterion = "entropy", max_depth = depth)
    loanstatustree.fit(X_train, y_train)
    predTree = loanstatustree.predict(X_test)
    mean_acc[depth - 1] = metrics.accuracy_score(y_test, predTree)
    std_acc[depth - 1] = np.std(predTree == y_test) / np.sqrt(predTree.shape[0])


# In[24]:


# Show accuracy of various K values
import matplotlib.pyplot as plt
k_values = list(range(1, Ks))
plt.plot(list(range(1,Ks)),mean_acc,'g')
plt.fill_between(list(range(1,Ks)),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(list(range(1,Ks)),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Maximum depth')
plt.tight_layout()
plt.show()


# In[25]:


# Choose a max_depth of 2 to prevent over-fitting and increase test accuracy without restricting model too much
final_decisionTree = DecisionTreeClassifier(criterion = "entropy", max_depth = 2).fit(X, y)


# # Support Vector Machine

# In[26]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools


# In[27]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(list(range(cm.shape[0])), list(range(cm.shape[1]))):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[28]:


from sklearn import svm
# Train model with training set
loanSVM = svm.SVC(kernel = 'rbf')
loanSVM.fit(X_train, y_train)

# Predict labels using test set
yhat = loanSVM.predict(X_test)

# Evaluate effectiveness of model using confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1, 0])
np.set_printoptions(precision=2)
print((classification_report(y_test, yhat)))
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['PAIDOFF','COLLECTION'], normalize= False,  title='Confusion matrix')


# In[29]:


# Re-train model using unsplit training set
loanSVM.fit(X, y)


# # Logistic Regression

# In[30]:


from sklearn.linear_model import LogisticRegression

# Train model
loanRegression = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

# Produce prediction and probability using test set
y_pred = loanRegression.predict(X_test)
y_pred_prob = loanRegression.predict_proba(X_test)
# print(y_pred, '\n', y_pred_prob)


# In[31]:


# Evaluate model using metrics library
from sklearn.metrics import jaccard_score
print((jaccard_score(y_test, y_pred)))

# Evaluate effectiveness of model using confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
np.set_printoptions(precision=2)
print((classification_report(y_test, y_pred)))
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['PAIDOFF','COLLECTION'], normalize= False,  title='Confusion matrix')


# In[32]:


# Re-train model on entire training set
loanRegression.fit(X, y)


# # Model Evaluation using Test set

# In[33]:


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# First, download and load the test set:

# In[34]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# ### Load Test set for evaluation 

# In[35]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[36]:


# Clean data as for training set
test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek

test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)

test_df['Gender'].replace(to_replace = ['male','female'], value = [0, 1], inplace = True)
test_df['loan_status'].replace(to_replace = ['PAIDOFF', 'COLLECTION'], value = [0, 1], inplace = True)
Feature = test_df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(test_df['education'])], axis=1)

X_test = Feature
y_test = test_df['loan_status'].values

X = preprocessing.StandardScaler().fit(X).transform(X)
X_test.head()


# In[37]:


# Define list to store accuracy measurements
Evaluation = []

# K-nearest neigbours
y_pred1 = final_neighbour_model.predict(X_test)
KNN_Jaccard = jaccard_score(y_test, y_pred1, average = "weighted")
KNN_f1 = f1_score(y_test, y_pred1, average = "weighted")
Evaluation.append(['KNN', KNN_Jaccard, KNN_f1, None])

# Decision Tree
y_pred2 = final_decisionTree.predict(X_test)
tree_Jaccard = jaccard_score(y_test, y_pred2, average = "weighted")
tree_f1 = f1_score(y_test, y_pred2, average = "weighted")
Evaluation.append(['Decision Tree', tree_Jaccard, tree_f1, None])

# Support Vector Machine
y_pred3 = loanSVM.predict(X_test)
SVM_Jaccard = jaccard_score(y_test, y_pred3, average = "weighted")
SVM_f1 = f1_score(y_test, y_pred3, average = "weighted")
Evaluation.append(['SVM', SVM_Jaccard, SVM_f1, None])

# Logistic Regression
y_pred4 = loanRegression.predict(X_test)
y_prob = loanRegression.predict_proba(X_test)
logRegr_Jaccard = jaccard_score(y_test, y_pred4, average = "weighted")
logRegr_f1 = f1_score(y_test, y_pred4, average = "weighted")
logRegr_logloss = log_loss(y_test, y_prob)
Evaluation.append(['Logistic Regression', logRegr_Jaccard, logRegr_f1, logRegr_logloss])


# In[38]:


report = pd.DataFrame(Evaluation, columns = ['Algorithm', 'Jaccard', 'F1-score', 'LogLoss'])
report


# # Report
# You should be able to report the accuracy of the built model using different evaluation metrics:

# | Algorithm          | Jaccard | F1-score | LogLoss |
# |--------------------|---------|----------|---------|
# | KNN                | ?       | ?        | NA      |
# | Decision Tree      | ?       | ?        | NA      |
# | SVM                | ?       | ?        | NA      |
# | LogisticRegression | ?       | ?        | ?       |

# <h2>Want to learn more?</h2>
# 
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href="http://cocl.us/ML0101EN-SPSSModeler">SPSS Modeler</a>
# 
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href="https://cocl.us/ML0101EN_DSX">Watson Studio</a>
# 
# <h3>Thanks for completing this lesson!</h3>
# 
# <h4>Author:  <a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a></h4>
# <p><a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>
# 
# <hr>
# 
# <p>Copyright &copy; 2018 <a href="https://cocl.us/DX0108EN_CC">Cognitive Class</a>. This notebook and its source code are released under the terms of the <a href="https://bigdatauniversity.com/mit-license/">MIT License</a>.</p>
