#!/usr/bin/env python
# coding: utf-8

# ## Learn Machine Learning models with examples
# #### The motive behind working on this Kernal is to revise the basic steps or formal mathmatical calculation running behind the models and not treating any model as Black Box
# Please comment the suggestion or upvote if you like.

# *Note* : Only done for Logistics Regression- Few more things to be added.
#  Work In Progress. 
#  Will do for other algos also.

# ### Importing all the required Libs

# In[1]:


# Importing all the required libraries
import os
import re
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import WordNetLemmatizer

stopwords = set(STOPWORDS)


# ### Adding few mords in stopwords - got from wordcloud below

# In[2]:


stopwords.add('will')
stopwords.add('ve')
stopwords.add('now')
stopwords.add('gonna')
stopwords.add('wanna')
stopwords.add('lol')
stopwords.add('via')
            


# Changing the current directory using python os module to the data directory

# In[3]:


os.chdir('/kaggle/input/nlp-getting-started/')


# Importing data to memory using pandas

# In[4]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# Doing basic checks on the data

# In[5]:


# Data Overview
train.head()


# In[6]:


test.head()


# How many null values are there?

# In[7]:


print('Null values in Train data set%')
train.isnull().sum()/len(train)*100


# In[8]:


print('Null values in Test dataset %')
test.isnull().sum()/len(test)*100


# As we will only deal with teh text column, so we will not be focusing on the missing values in other columns

# ### Combining all the data train and test, for cleaning purpose

# In[9]:


# Combining all the text data 
tweets_data = pd.concat([train, test], axis=0, sort=False, ignore_index=True)
tweets_data.head()


# ### Below function contains the data cleaning steps using regex

# In[10]:


# Data cleaning steps
def clean_data(df, text_col, new_col='cleaned_text', stemming=False, lemmatization=True):
    
    '''
    It will remove the noise from the text data(@user, characters not able to encode/decode properly)    
    ----Arguments----
    df : Data Frame
    col : column name (string)
    steming : boolean
    lemmatization : boolean
    '''
    tweets_data = df.copy() # deep copying the data in order to avoid any change in the main data col  
    
    # Creating one more new column for new text transformation steps
    tweets_data[new_col] = tweets_data[text_col]
    
    # removing @<userid>, as it is very common in the twitter data
    tweets_data[new_col] = tweets_data[new_col].apply(lambda x : re.sub(
        '@[A-Za-z0-9_]+', '', x)) 
    
    # Removing &amp 
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub('&amp',' ', str(x)))
    
    # Removing URLs from the data
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub('https?:\/\/[a-zA-z0-9\.\/]+','',
                                                                     str(x)))
    tweets_data[new_col] = tweets_data[new_col].str.lower()
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\’", "\'", str(x)))
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\s\'", " ", str(x)))
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"won\'t", "will not", str(x)))
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"can\'t", "can not", str(x)))
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"don\'t", "do not", str(x)))
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"dont", "do not", str(x)))
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"n\’t", " not", str(x)))
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"n\'t", " not", str(x)))
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\'re", " are", str(x)))
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub("\'s", " is", str(x)))
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\’d", " would", str(x)))
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\'ll", " will", str(x)))
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\'t", " not", str(x)))
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\'ve", " have", str(x)))
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\'m", " am", str(x)))
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\n", "", str(x)))
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\r", "", str(x)))
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\'", "", str(x)))
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\"", "", str(x)))
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r'[?|!|\'|"|#]',r'', str(x)))
    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r'[.|,|)|(|\|/]',r' ', str(x)))
    
    # Trimming the sentences
    tweets_data[new_col] = tweets_data[new_col].str.strip() 
    tweets_data[new_col] = tweets_data[new_col].apply(lambda x : re.findall(
       "[A-Za-z0-9]+", x))
    tweets_data[new_col] = tweets_data[new_col].apply(lambda x : " ".join(x))
    
    # Remove stopwords
    tweets_data[new_col] = tweets_data[new_col].apply(
        lambda x : ['' if word in stopwords else word for word in x.split()])
    
    tweets_data[new_col] = tweets_data[new_col].apply(lambda x : " ".join(x))
        
    # Removing extra spaces
    tweets_data[new_col] = tweets_data[new_col].apply(lambda x : re.sub("\s+", " ", x))
    
    # lemmatization
    if lemmatization:
        
        lemma = WordNetLemmatizer()
        
        tweets_data[new_col] = tweets_data[new_col].apply(lambda sentence : 
                                         [lemma.lemmatize(word,'v') for word in sentence.split(" ")])
        
        tweets_data[new_col] = tweets_data[new_col].apply(lambda x : " ".join(x))
     
    # Stemming code
    if stemming:
        stemming = PorterStemmer()
        
        tweets_data[new_col] = tweets_data[new_col].apply(lambda sentence : 
                                         [stemming.stem(x) for x in sentence.split(" ")])
        
        tweets_data[new_col] = tweets_data[new_col].apply(lambda x : " ".join(x))

    return tweets_data


# In[11]:


tweets_data = clean_data(tweets_data, "text", 
                         'cleaned_text', 
                         lemmatization=True,
                         stemming=False)

pd.set_option('display.max_colwidth', -1)

print('----- Text Before And After Cleaning -----')

tweets_data[['text', 'cleaned_text']].head(10)


# ### Stemming of Text - Stemming tries to convert word into it's root form

# In[12]:


stemming = PorterStemmer()
print(f"Runs converted to {stemming.stem('runs')}")
print(f"Stemming converted to {stemming.stem('stemming')}")


# ### Plotting Word Cloud

# In[13]:


wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10)
wordcloud.generate(" ".join(tweets_data['cleaned_text']))
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None, dpi=80) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  


# Wildfire seems to be one of the major issue.

# ### %cent of classes

# In[14]:


print('% of classes')
print((tweets_data['target'].value_counts() / len(train) * 100))


# ### Preparing data using TF-IDF [Term Frequency Inverse Document Matrix]

# In[15]:


vec = TfidfVectorizer(ngram_range=(1,5),
                      #max_features=10000,
                      min_df=3,
                      stop_words='english')

tfidf_matrix = vec.fit_transform(tweets_data['cleaned_text'])

tfidf_matrix.shape


# In[16]:


tfidf_matrix = pd.DataFrame(tfidf_matrix.toarray(),
                            columns = vec.get_feature_names(),
                            dtype='float32')

print(("Shape of the dataframe ",tfidf_matrix.shape))
print('Data Frame Info')
tfidf_matrix.info()


# ### Splitting the data for testing 

# In[17]:


# Prepare the data set for model training

X = tfidf_matrix.iloc[list(range(0, train.shape[0])), :]

test_dataset = tfidf_matrix.iloc[train.shape[0]:, :] 
                           
y = tweets_data.loc[0:train.shape[0]-1, 'target']

x_train, x_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state=123, 
                                                    test_size = 0.3)


# # Learn And Train Different Types Models
# My main objective is not to treat any model as a black box, let also know what is going on behind when we fit any model. 
# When we we will discuss anything about models just keep in mind the below steps needed for a model
# > 1. Objective
# > 2. Model structure (e.g. variables, formula, equation)
# > 3. Model assumptions
# > 4. Parameter estimates and interpretation
# > 5. Model fit (e.g. goodness-of-fit tests and statistics)
# > 6. Model selection

# ## 1. Logistic Regression 
# 
# - It is a statistical technique that is used to map the the values of a depedent variable(Y) to it's indepedendent or predictor variables(X).
# - It is used for classification purpose either binary like yes/no or ordinal like good, better, best.
# - Also known as logit or log odds - odds means the probability of success divided by probability of failure the function used for parameter estimation other function used is the sigmoid function.
# 
#   Logistic function - Calculations
#  > log(p/1-p) = b + b1x1 + b2x2 + ... + bkxk | where __p__ is the probaility and __x__ is the feature 
# 
#  > log(p/1-p) is known as log odds
# 
#  > p/1-p = e^(b + b1x1 + b2x2 +...+ bkXk)
# 
#  > p = 1/(1 + e^(b + b1x1 + b2x2 +...+ bkXk))
# 

# ### Summary
# 
# 1. __Objective__ : Is to build a mode the expected value of Y as the function of X
# 2. __Model Structure__: p = e^(b0+b1x1+b2x2+...bkXk)/(1+e^(b0+b1x1+b2x2+...bkXk))
# - __Model Assumption__: 
#    1. Independent variables should be linearly dependent on log odds.
#    2. No Multicolinearity - Independent features(X) should not be correlated with each other.
#    3. Needed a larger sample size.
#    4. Dependent variable must be binary or ordinal.
# - __Parameter Estimation__:
#    1. b0(beta) or Intercept : It's basically a constant which means the predicted or avg value of y when the independent variables are 0.
#    2. b1, b2,b3. : It means a slope also defines the association between Y and X. In other terms - By changing 1 unit value of X1 the y will change by b1 times.
#  - __Model fit(goodness-of-fit tests)__:
#     1. Accuracy(If dataset is balanced)
#     2. F1 score, precision, recall, auc-roc score, etc
# - __Model Selection__ : Removing unwanted features using Lasso or l1 penality or using Ridge(l2) penality 

# ### Fitting or Training Linear Regression

# In[18]:


clf = LogisticRegression(max_iter=1500,
                        solver='lbfgs')

clf.fit(x_train, y_train)


# ### Testing on Test Data Set

# In[19]:


print(("F1 Score is ", f1_score(y_test, clf.predict(x_test))))
confusion_matrix(y_test, clf.predict(x_test))


# In[20]:


clf.fit(X, y)

act_pred = clf.predict(test_dataset)
act_pred = act_pred.astype('int')

submission_file = pd.DataFrame({'id' : test['id'],
                               'target' : act_pred})

submission_file.to_csv('/kaggle/working/sub_140120_v0.4lr.csv', index = False)


# ### 2. Naive Bayes Classifier

# In[21]:


nv = GaussianNB()
nv.fit(x_train, y_train)


# Testing on the test data set

# In[22]:


print(("F1 Score is ", f1_score(y_test, nv.predict(x_test))))
print('Confusion Matrix')
confusion_matrix(y_test, nv.predict(x_test))


# ### 3. Random Forest

# In[23]:


rf = RandomForestClassifier(n_estimators=1500,
                            max_depth=6,
                            oob_score=True)
rf.fit(x_train, y_train)


# Checking on test dataset

# In[24]:


print(("F1 Score is ", f1_score(y_test, rf.predict(x_test))))
print('--------Confusion Matrix---------')
confusion_matrix(y_test, rf.predict(x_test))


# Training on complete dataset for Prediction

# In[25]:


rf.fit(X, y)

act_pred = rf.predict(test_dataset)
act_pred = act_pred.astype('int')

submission_file = pd.DataFrame({'id' : test['id'],
                               'target' : act_pred})

submission_file.to_csv('/kaggle/working/sub_140120_v0.1rf.csv', index = False)


# ### 4. XGBOOST

# In[26]:


from xgboost import XGBClassifier

xgb = XGBClassifier(max_depth=6,
                    learning_rate=0.3,
                    n_estimators=1500,
                    objective='binary:logistic',
                    random_state=123,
                    n_jobs=4)

xgb.fit(x_train, y_train)


# Checking score on the test data set

# In[27]:


print(("F1 Score is ", f1_score(y_test, xgb.predict(x_test))))
print('--------Confusion Matrix---------')
confusion_matrix(y_test, xgb.predict(x_test))


# In[28]:


xgb.fit(X, y)

act_pred = xgb.predict(test_dataset)
act_pred = act_pred.astype('int')

submission_file = pd.DataFrame({'id' : test['id'],
                               'target' : act_pred})

submission_file.to_csv('/kaggle/working/sub_130120_v0.1xgb.csv', index = False)


# ### Using Word2vec now
# 

# In[29]:


import gensim.models.word2vec as w2v
import nltk
import multiprocessing
from nltk.tokenize import TweetTokenizer


# In[30]:


num_of_features = 300
min_word_count = 3
num_of_threads = multiprocessing.cpu_count()
context_size = 7
downsampling = 1e-3


# In[31]:


vecs = w2v.Word2Vec(sg=1,
                   seed=123,
                   workers=num_of_threads,
                   size=num_of_features,
                   min_count=min_word_count,
                   window=context_size,
                   sample=downsampling)


# In[32]:


tokens_vec = []
tokens = TweetTokenizer()
tokens_vec = tweets_data['cleaned_text'].apply(lambda x : tokens.tokenize(x))


# In[33]:


vecs.build_vocab(tokens_vec.values.tolist())


# In[ ]:





# In[ ]:




