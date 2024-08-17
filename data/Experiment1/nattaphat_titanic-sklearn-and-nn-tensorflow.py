#!/usr/bin/env python
# coding: utf-8

# ** Importing Python libraries**
# * Python libraries will be used for this project

# In[1]:


import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#To encode data
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

# tensorflow for NN machine learning
import tensorflow as tf
from tensorflow.python.framework import ops

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier


# **Importing training and test data sets**
# * Imported Train data and test data, checkd header name for both files, only "Survived" header that is not exist in the test data. 

# In[2]:


#Import Training and Test Data
train_ori = pd.read_csv('../input/train.csv')
test_ori  = pd.read_csv('../input/test.csv')

print(("Train header name = ", train_ori.columns.values))
print(("Test header name = ", test_ori.columns.values))
print(("Train shape = ", train_ori.shape))
print(("test shape = ", test_ori.shape))


# In[3]:


train_Y = train_ori.Survived
train_X = train_ori
train_X.drop('Survived', 1, inplace = True)

Merged = train_X.append(test_ori)
print(("train_Y shape = ", train_Y.shape))
print(("Train shape = ", train_X.shape))
print(("Merged shape = ", Merged.shape))


# **Feature type**
# * 4  features are integer and 2 features are float
# * 5 features are string objects

# In[4]:


train_X.info()
print(('_'*40))
test_ori.info()


# **Created new feature "Title"**
# * 18 title categories were found
# * Reduced to 5 categories

# In[5]:


Merged['Title']  = Merged['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
Merged.head(5)


# In[6]:


pd.crosstab(Merged['Title'], Merged['Sex'])


# In[7]:


def Group_Title():
    Title_Dictionary = []
    Title_Dictionary = {
                        "Capt"             : "Rare",
                        "Col"              : "Rare",
                        "Don"              : "Rare", 
                        "Dona"             : "Rare",
                        "Dr"               : "Rare",
                        "Jonkheer"         : "Rare", 
                        "Lady"             : "Rare",
                        "Major"            : "Rare",
                        "Master"           : "Master",
                        "Miss"             : "Miss", 
                        "Mlle"             : "Rare",    
                        "Mme"              : "Rare", 
                        "Mr"               : "Mr",
                        "Mrs"              : "Mrs",
                        "Ms"               : "Rare",
                        "Rev"              : "Rare", 
                        "Sir"              : "Rare",    
                        "the Countess"     : "Rare", 
    }
    Merged['Title']     =  Merged.Title.map(Title_Dictionary)
    
Group_Title()
#CountPerTitle = Merged_data['Title'][:].value_counts() 
pd.crosstab(Merged['Title'], Merged['Sex'])


# **Create familiy size feature**
# * Family size = "SibSp" + "Parch" +1

# In[8]:


Merged['FamilySize'] = Merged['SibSp'] + Merged['Parch'] + 1
Merged.head(5)


# **Fill in Null data of Embarked and Fare**
# * Only 2 examples show null
# * Both are from the same 1st class, family size = 1 and fare = 80
# I think Embarked, Pclass and family size should be related and  we can use  to predict the null.

# In[9]:


Merged[Merged['Embarked'].isnull()]


# Based on the summary table below, we can guest the Embarked of passenger 62 and 830, it should be C.

# In[10]:


Merged[[ "Embarked", "FamilySize", "Pclass", "Fare"]].groupby(['Embarked','Pclass', 'FamilySize'], as_index=False).mean()


# In[11]:


Merged['Embarked'].fillna('C', inplace=True)
Merged.info()


# Ony one null observed on Fare. Using the information from above table, we now can guest the value on null.
# * Pclass = 3, Family size = 1 and Embarked = S, Fare should be 9.6$

# In[12]:


Merged[Merged['Fare'].isnull()]


# In[13]:


Merged['Fare'].fillna(9.6, inplace=True)
Merged.info()


# In[14]:


Merged.head(20)


# **Fill in Age data**
# * Number of null = 263 ==>Master =8, Miss =50, Mr = 176, Mrs = 27 and Rare =2
# 

# In[15]:


Merged['Age'].isnull().sum()


# In[16]:


Merged[Merged['Age'].isnull()][['Title','Fare']].groupby(['Title']).count()


# In[17]:


Merged[['Title','Age']].groupby(['Title']).median()


# In[18]:


grid = sns.FacetGrid(Merged, col='Title', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)


# In[19]:


MedMaster = np.nanmedian(Merged[(Merged['Title'] == "Master")]['Age'])
MedMiss = np.nanmedian(Merged[(Merged['Title'] == "Miss")]['Age'])
MedMr = np.nanmedian(Merged[(Merged['Title'] == "Mr")]['Age'])
MedMrs = np.nanmedian(Merged[(Merged['Title'] == "Mrs")]['Age'])
MedRare = np.nanmedian(Merged[(Merged['Title'] == "Rare")]['Age'])
print(("median Master = ",MedMaster))
print(("median Miss = ",MedMiss))
print(("median Mr = ",MedMr))
print(("median Mrs = ",MedMrs))
print(("median Rare = ",MedRare))


# In[20]:


Merged.head(20)


# In[21]:


Merged.loc[(Merged['Age'].isnull()) & (Merged['Title']=="Master"),'Age'] = MedMaster
Merged.loc[(Merged['Age'].isnull()) & (Merged['Title']=="Miss"),'Age'] = MedMiss
Merged.loc[(Merged['Age'].isnull()) & (Merged['Title']=="Mr"),'Age'] = MedMr
Merged.loc[(Merged['Age'].isnull()) & (Merged['Title']=="Mrs"),'Age'] = MedMrs
Merged.loc[(Merged['Age'].isnull()) & (Merged['Title']=="Rare"),'Age'] = MedRare


# In[22]:


Merged.info()


# In[23]:


print(("Train header name = ", Merged.columns.values))


# In[24]:


Merged.head(5)


# **Encode label data**
# * "Sex", "Title" and "Embarked" need to be transformed to numeric  feature.

# In[25]:


label_encoder = LabelEncoder()
Merged.loc[:,'Sex'] = label_encoder.fit_transform(Merged.loc[:,'Sex'])
Merged.loc[:,'Title'] = label_encoder.fit_transform(Merged.loc[:,'Title'])
Merged.loc[:,'Embarked'] = label_encoder.fit_transform(Merged.loc[:,'Embarked'])
Merged.head(10)


# Lets drop large missing data "cabin" and other nominal features "Name" " Ticket".  Since we created the new Family size feature, "SibSp" "Parch" are also dropped. 

# In[26]:


Merged = Merged.drop(['Parch', 'SibSp', 'Ticket','Name','Cabin'], axis=1)


# In[27]:


Merged.head(5)


# **Add two new features**
# * SP1 = Age * Pclass
# * SP2 = (3/Pclass)(2-Sex)^2

# In[28]:


Merged['SP1'] = Merged['Age'] * Merged['Pclass']
Merged['SP2'] = (3/Merged['Pclass'])*(2-Merged['Sex'])**2 
Merged.head()


# In[29]:


Merged.info()
Merged = preprocessing.scale(Merged)


# In[30]:


Merged.shape


# Split Train and Test data

# In[31]:


Train_new = Merged[0:891]
Test_new = Merged[891:Merged.shape[0]]
X_train, X_Dev, Y_train, Y_Dev = train_test_split(Train_new, train_Y, test_size=0.2, random_state = 0)

print((X_train.shape,Y_train.shape,X_Dev.shape,Y_Dev.shape,Test_new.shape))


# **Machine learning**

# In[32]:


# Logistic Regression
Model = LogisticRegression()
Model.fit(X_train, Y_train)
Y_pred = Model.predict(X_Dev)
print((round(Model.score(X_train, Y_train) * 100,2),round(Model.score(X_Dev, Y_Dev)*100,2)))


# In[33]:


#Suport Vector Machine
Model = SVC()
Model.fit(X_train, Y_train)
Y_pred = Model.predict(X_Dev)
print((round(Model.score(X_train, Y_train) * 100,2),round(Model.score(X_Dev, Y_Dev)*100,2)))


# In[34]:


#Linear SVC
Model = LinearSVC()
Model.fit(X_train, Y_train)
Y_pred = Model.predict(X_Dev)
print((round(Model.score(X_train, Y_train) * 100,2),round(Model.score(X_Dev, Y_Dev)*100,2)))


# In[35]:


# Stochastic Gradient Descent
Model = SGDClassifier()
Model.fit(X_train, Y_train)
Y_pred = Model.predict(X_Dev)
print((round(Model.score(X_train, Y_train) * 100,2),round(Model.score(X_Dev, Y_Dev)*100,2)))


# In[36]:


# Random Forest
Model = RandomForestClassifier(n_estimators=100)
Model.fit(X_train, Y_train)
Y_pred = Model.predict(X_Dev)
print((round(Model.score(X_train, Y_train) * 100,2),round(Model.score(X_Dev, Y_Dev)*100,2)))
Y_pred_Test = Model.predict(Test_new)


# In[37]:


len(Y_pred_Test)


# **Implementing NN with Tensorflow**

# In[38]:


seed = 1
np.random.seed(seed)


# Initializing parameters function

# In[39]:


def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        W = tf.get_variable("W" + str(l),[layer_dims[l],layer_dims[l-1]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b = tf.get_variable("b" + str(l),[layer_dims[l],1], initializer =  tf.constant_initializer(0.0) )

        parameters['W' + str(l)] = W
        parameters['b' + str(l)] = b
        print(('W' + str(l), W))
        print(('b' + str(l), b))

    return parameters


# Create place holder for input X and Y

# In[40]:


def create_placeholders(n_x, n_y):
    ### START CODE HERE ### (approx. 2 lines)
    X = tf.placeholder(dtype = "float", shape = [n_x, None])
    Y = tf.placeholder(dtype = "float", shape = [n_y, None])
    ### END CODE HERE ###
    print(("X = " + str(X)))
    print(("Y = " + str(Y)))
    return X, Y


# Forward prop function

# In[41]:


def forward_propagation(X, parameters,layer_dims):
    L = len(layer_dims)                  # number of layers in the neural network
    A = X

    for l in range(1, L):
        A_prev = A
        Z = tf.add(tf.matmul(parameters['W' + str(l)],A_prev),parameters['b' + str(l)])
        A = tf.nn.tanh(Z)               #activation function is tanh
    return Z


# Cost function

# In[42]:


def compute_cost(Z_l, Y):

    logits = tf.transpose(Z_l)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    return cost


# In[43]:


def compute_cost_regularization(Z_l, Y, parameters, lambd = 0.01):
    L = len(parameters) // 2                  # number of layers in the neural network
    regularizer = 0
    for l in range(1, L):
        regularizer = regularizer + tf.nn.l2_loss(parameters['W' + str(l)])

    loss = compute_cost(Z_l, Y)
    cost = tf.reduce_mean(loss + lambd * regularizer)
    ### END CODE HERE ###

    return cost


# In[44]:


def indices_to_one_hot(data, nb_classes):
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


# In[45]:


def model(X_train, Y_train, layers_dims, learning_rate = 0.0001, iteration = 100000, print_cost = True):
    L = len(layers_dims)                                # number of layers in the neural networks                              # to keep consistent results
    (n_x, m) = X_train.shape                            # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                              # n_y : output size
    Train_costs = []                                          # To keep track of the cost
    Dev_costs = []
    iterations = []
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters(layers_dims)

    Z_l = forward_propagation(X, parameters,layers_dims)
    print(('Z_l', Z_l))
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost_regularization(Z_l, Y,parameters,lambd=0.003)
    # Optimizer is Adadelta
    optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()
    
     # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)
        for iter in range(iteration):

            _ , train_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
            cost_Dev = sess.run(cost,feed_dict={X: X_Dev, Y: Y_Dev})
            
        # Print the cost every epoch
            if print_cost == True and iter % 10000 == 0:
                print(("Cost after epoch %i: %f" % (iter, train_cost)))
                print(("CostDev after epoch %i: %f" % (iter, cost_Dev)))
                
                
            if print_cost == True and iter % 1000 == 0:
                Train_costs.append(train_cost)
                Dev_costs.append(cost_Dev)
                iterations.append(iter)

        # plot the cost
        plt.plot(iterations,np.squeeze(Train_costs),'b',label="Train")
        plt.plot(iterations, np.squeeze(Dev_costs),'r',label="Dev")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        Parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z_l,0), tf.argmax(Y,0))

        prediction=tf.argmax(Z_l,0)
        predTrain = sess.run([[prediction]],feed_dict={X: X_train})
        predTDev = sess.run([[prediction]],feed_dict={X: X_Dev})
        predTest = sess.run([[prediction]],feed_dict={X: X_test})
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(("correct_prediction:", correct_prediction))
        print(("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train})))
        print(("Test Accuracy:", accuracy.eval({X: X_Dev, Y: Y_Dev})))
        return  Parameters, predTrain, predTDev, predTest


# In[46]:


X_train = X_train.T
X_Dev = X_Dev.T
Y_train = Y_train.T
Y_Dev = Y_Dev.T

X_test = Test_new.T


# In[47]:


Y_train = indices_to_one_hot(Y_train,2).T
Y_Dev = indices_to_one_hot(Y_Dev,2).T


# In[48]:


print(("X_train shape", X_train.shape))
print(("Y_train shape", Y_train.shape))
print(("X_Dev shape", X_Dev.shape))
print(("Y_Dev shape", Y_Dev.shape))


# In[49]:


layers_dims = [X_train.shape[0], 10, 5, 5, 3, 2]
tf.reset_default_graph() 
parameters, predTrain, predDev, predTest = model(X_train, Y_train, layers_dims, learning_rate = 0.01, iteration = 50000)


# In[50]:


layers_dims = [X_train.shape[0], 10, 5, 2]
tf.reset_default_graph() 
parameters, predTrain, predDev, predTest = model(X_train, Y_train, layers_dims, learning_rate = 0.01, iteration = 50000)


# In[51]:


Y_pred_Test=np.resize(predTest,(X_test.shape[1]))


# In[52]:


submission = pd.DataFrame({
        "PassengerId": test_ori["PassengerId"],
        "Survived": Y_pred_Test
    })
submission.to_csv('submission.csv', index=False)


# In[53]:




