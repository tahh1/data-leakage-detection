#!/usr/bin/env python
# coding: utf-8

# This kernel notebook provides the insight of Amazon Products reviews, analyze the sentiments of customer,counts the number of of positive and negative reviews,summary of statistical data ,data visualization, formation of wordclouds based on overall reviews, positive reviews and negative reviews,fitting the dataset into different models to find out the accuracy score of the dataset.
# Hope it will be useful for the beginner.

# About This Data:
# This is a list of over 34,000 customer reviews for Amazon products like the Kindle, Fire TV Stick, and more provided by Datafiniti's Product Database. The dataset includes basic product information, rating, review text, and more for each product.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print((check_output(["ls", "../input"]).decode("utf8")))


# In[2]:


#IMPORTING LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import nltk.classify.util
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.classify import NaiveBayesClassifier
import numpy as np
import re
import string
import nltk
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#importing dataset
df = pd.read_csv(r"../input/1429_1.csv")
df.head()


# In[4]:


dff = df[['reviews.rating' , 'reviews.text' , 'reviews.title' , 'reviews.username']]
print((dff.isnull().sum())) #Checking for null values
dff.head()


# In[5]:


#Filtering Null Values
check =  dff[dff["reviews.rating"].isnull()]
check.head()


# In[6]:


#Filtering not null values
senti=dff[dff["reviews.rating"].notnull()]
dff.head()


# In[7]:


#Classifying text based on sentiments(positive or negative)
senti["senti"] = senti["reviews.rating"]>=4
senti["senti"] = senti["senti"].replace([True , False] , ["pos" , "neg"])


# In[8]:


#Count of reviews based on sentiments
senti["senti"].value_counts().plot.bar(color=['cyan', 'blue'])


# In[9]:


#Cleaning Text
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import numpy as np
import re
import string
import nltk

cleanup_re = re.compile('[^a-z]+')
def cleanup(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = cleanup_re.sub(' ', sentence).strip()
    #sentence = " ".join(nltk.word_tokenize(sentence))
    return sentence

senti["Summary_Clean"] = senti["reviews.text"].apply(cleanup)
check["Summary_Clean"] = check["reviews.text"].apply(cleanup)


# In[10]:


#Splitting training and testing dataset
split = senti[["Summary_Clean" , "senti"]]
train=split.sample(frac=0.8,random_state=200)
test=split.drop(train.index)


# In[11]:


#Feature extracter for NLTK Naives Bayes
def word_feats(words):
    features = {}
    for word in words:
        features [word] = True
    return features


# In[12]:


train["words"] = train["Summary_Clean"].str.lower().str.split()
test["words"] = test["Summary_Clean"].str.lower().str.split()
check["words"] = check["Summary_Clean"].str.lower().str.split()

train.index = list(range(train.shape[0]))
test.index = list(range(test.shape[0]))
check.index = list(range(check.shape[0]))
prediction =  {} ## For storing results of different classifiers

train_naive = []
test_naive = []
check_naive = []

for i in range(train.shape[0]):
    train_naive = train_naive +[[word_feats(train["words"][i]) , train["senti"][i]]]
for i in range(test.shape[0]):
    test_naive = test_naive +[[word_feats(test["words"][i]) , test["senti"][i]]]
for i in range(check.shape[0]):
    check_naive = check_naive +[word_feats(check["words"][i])]


classifier = NaiveBayesClassifier.train(train_naive)
print(("NLTK Naive bayes Accuracy : {}".format(nltk.classify.util.accuracy(classifier , test_naive))))
classifier.show_most_informative_features(5)


# In[13]:


#Predicting result for NLTK Classifier
y =[]
only_words= [test_naive[i][0] for i in range(test.shape[0])]
for i in range(test.shape[0]):
    y = y + [classifier.classify(only_words[i] )]
prediction["Naive"]= np.asarray(y)

y1 = []
for i in range(check.shape[0]):
    y1 = y1 + [classifier.classify(check_naive[i] )]

check["Naive"] = y1


# In[14]:


#BUILDING COUNTER AND TFIDF VECTOR FOR TEST<TRAIN AND CHECK DATA
from wordcloud import STOPWORDS

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
stopwords = set(STOPWORDS)
stopwords.remove("not")

count_vect = CountVectorizer(min_df=2 ,stop_words=stopwords , ngram_range=(1,2))
tfidf_transformer = TfidfTransformer()

X_train_counts = count_vect.fit_transform(train["Summary_Clean"])        
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


X_new_counts = count_vect.transform(test["Summary_Clean"])
X_test_tfidf = tfidf_transformer.transform(X_new_counts)

checkcounts = count_vect.transform(check["Summary_Clean"])
checktfidf = tfidf_transformer.transform(checkcounts)


# In[15]:


#Fitting Multinomial NB
from sklearn.naive_bayes import MultinomialNB
model1 = MultinomialNB().fit(X_train_tfidf , train["senti"])
prediction['Multinomial'] = model1.predict_proba(X_test_tfidf)[:,1]
print(("Multinomial Accuracy : {}".format(model1.score(X_test_tfidf , test["senti"]))))

check["multi"] = model1.predict(checktfidf)## Predicting Sentiment for Check which was Null values for rating


# In[16]:


#Fitting Bernouli NB
from sklearn.naive_bayes import BernoulliNB
model2 = BernoulliNB().fit(X_train_tfidf,train["senti"])
prediction['Bernoulli'] = model2.predict_proba(X_test_tfidf)[:,1]
print(("Bernoulli Accuracy : {}".format(model2.score(X_test_tfidf , test["senti"]))))

check["Bill"] = model2.predict(checktfidf)## Predicting Sentiment for Check which was Null values for rating


# In[17]:


#Fitting Logistic Regression
from sklearn import linear_model
logreg = linear_model.LogisticRegression(solver='lbfgs' , C=1000)
logistic = logreg.fit(X_train_tfidf, train["senti"])
prediction['LogisticRegression'] = logreg.predict_proba(X_test_tfidf)[:,1]
print(("Logistic Regression Accuracy : {}".format(logreg.score(X_test_tfidf , test["senti"]))))

check["log"] = logreg.predict(checktfidf)## Predicting Sentiment for Check which was Null values for rating


# In[18]:


#Getting Most Occurence words in training test
words = count_vect.get_feature_names()
feature_coefs = pd.DataFrame(
    data = list(zip(words, logistic.coef_[0])),
    columns = ['feature', 'coef'])
feature_coefs.sort_values(by="coef")


# In[19]:


#To check out which classifier is doing what
def formatt(x):
    if x == 'neg':
        return 0
    if x == 0:
        return 0
    return 1
vfunc = np.vectorize(formatt)

cmp = 0
colors = ['k', 'b', 'g', 'm', 'y']
for model, predicted in list(prediction.items()):
    if model not in 'Naive':
        false_positive_rate, true_positive_rate, thresholds = roc_curve(test["senti"].map(vfunc), predicted)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))
        cmp += 1

plt.title('Classifiers comparaison with ROC')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Lets see precision and recall of different classifiers

# In[20]:


test.senti = test.senti.replace(["pos" , "neg"] , [True , False] )


# In[21]:


keys = list(prediction.keys())
for key in ['Multinomial', 'Bernoulli', 'LogisticRegression']:
    print((" {}:".format(key)))
    print((metrics.classification_report(test["senti"], prediction.get(key)>.5, target_names = ["positive", "negative"])))
    print("\n")


# In[22]:


#lets test our classifier with some samples
def test_sample(model, sample):
    sample_counts = count_vect.transform([sample])
    sample_tfidf = tfidf_transformer.transform(sample_counts)
    result = model.predict(sample_tfidf)[0]
    prob = model.predict_proba(sample_tfidf)[0]
    print(("Sample estimated as %s: negative prob %f, positive prob %f" % (result.upper(), prob[0], prob[1])))

test_sample(logreg, "The product was good and easy to  use")
test_sample(logreg, "the whole experience was just horrible and product is worst")
test_sample(logreg, "product is worst")


# In[23]:


check.head(5)


# In[24]:


#WORDCLOUD for overall reviews
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)


mpl.rcParams['font.size']=12                #10 
mpl.rcParams['savefig.dpi']=100             #72 
mpl.rcParams['figure.subplot.bottom']=.1 


def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=300,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
        
    ).generate(str(data))
    
    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
show_wordcloud(senti["Summary_Clean"])


# In[25]:


#WORDCLOUD based on positive reviews
show_wordcloud(senti["Summary_Clean"][senti.senti == "pos"] , title="Postive Words")


# In[26]:


#wordcloud based on negative reviews
show_wordcloud(senti["Summary_Clean"][senti.senti == "neg"] , title="Negative Words")


# In[27]:


#pie plot
senti["senti"].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',shadow=True,figsize=(10,8))
plt.show()


# In[28]:


#Statistical Summary of the data
senti.describe().plot(kind = "area",fontsize=27, figsize = (20,8), table = True,colormap="summer")
plt.xlabel('Statistics',)
plt.ylabel('Value')
plt.title("General Statistics of Products Dataset")


# Thanks for reading my kernel .If you found it helpful,then please upvote it
