#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re


# In[2]:


os.getcwd()


# In[3]:


os.chdir(r"D:\OneDrive - Manipal Global Education Services Pvt Ltd\Official\MGAIT\Datasets")


# In[5]:


#!pip install nltk


# In[6]:


#!pip install wordcloud


# In[4]:


import nltk


# In[8]:


#nltk.download()


# In[5]:


reviews = pd.read_csv("K8 Reviews v0.2.csv")


# In[10]:


reviews.head()


# In[11]:


#Dataset is scrapped from Amazon for Lenovo K8 mobile phones
# Review in the form of free text was scrapped and the user rating
# A user rating of 1,2,3 -> sentiment 0
# A user rating of 4 and 5 -> sentiment 1


# In[12]:


reviews.shape


# In[13]:


reviews.sentiment.value_counts()


# Getting insignts from the reviews:
# 
#         1. Use resular expressions
#         2. Word cloud
#         3. Bar graph

# ## Use resular expressions to get insignts about the reviews

# In[14]:


#1. Find out the Reviews Which have some numbers followed by the gb


# In[15]:


count=0
for review in reviews.review.values:
    review = review.strip()
    result = re.search("[0-9]+gb",review)
    if result:
        print(review)
        count+=1


# In[16]:


print(count)


# In[17]:


reviews.review.values[100]


# In[18]:


#1. Find out the Reviews Which have some numbers followed by the /-


# In[19]:


count=0
for review in reviews.review.values:
    review = review.strip()
    result = re.search("[0-9]+/-",review)
    if result:
        print(review)
        count+=1


# In[20]:


print(count)


# ### Get the word cloud:
# 
#     1. combine all the reviews into a single string
#     2. instantiate word cloud
#     3. generate the word cloud

# In[21]:


from wordcloud import WordCloud


# In[22]:


reviews_combined = " ".join(reviews.review.values)


# In[23]:


#Understanding join
lst = ["A","B","C"]
" ".join(lst)


# In[24]:


len(reviews_combined)


# In[25]:


reviews_combined[:200]


# In[26]:


word_cloud = WordCloud().generate(reviews_combined)


# In[27]:


word_cloud = WordCloud(width=800,height=800,
                       background_color='white',
                       max_words=150).\
generate(reviews_combined)


# In[28]:


plt.figure(figsize=[8,8])
plt.imshow(word_cloud)
plt.show()


# ## Bar graph of top 25 used words

# 1. get the words and the frequencies
# 2. sort based on frequencies and get the top 25 words
# 3. plot the bar graph

# In[29]:


from nltk.probability import FreqDist
all_terms = reviews_combined.split(" ")
fdist = FreqDist(all_terms)


# In[30]:


fdist


# In[31]:


#From the dict obtain a data frame of words and freq's
df_dist = pd.DataFrame(list(fdist.items()), columns = ["words","freq"])


# In[32]:


df_dist.head()


# In[33]:


#Top 5 words based on frequency
df_dist.sort_values(ascending=False, by="freq").head(5)


# In[34]:


get_ipython().run_line_magic('matplotlib', 'inline')
df_dist.sort_values(ascending=False, by="freq").head(25).\
plot.bar(x= "words", y= "freq",figsize=(20,5)) 


# Problems with these visuals:
# 
#     1.Too many distinct words in the corpus of reviews
#         1.Text is non unifrom case
#         2.Punctuations present in the text
#         3.There are language connectors in the text - stop words
#         4.Words with different forms/tenses - charge, charged, charging

# Basic Text processing tasks:
#     1. Text cleaning
#     2. Get the visuals - word cloud and bar graph or frequency graph
#     3. Converting text to numeric matrices
#     4. Advanced visuals using collocations(n-grams - unigrams, bigrams, trigrams)
#     5. Sentiment analysis and sentiment classification
#     6. text classification
#     7. document clustering

# In[35]:


from nltk.tokenize import word_tokenize


# In[36]:


all_terms = word_tokenize(reviews_combined.lower())


# In[37]:


print((all_terms[:200]))


# In[38]:


len(set(all_terms))


# In[39]:


from nltk.probability import FreqDist


# In[40]:


fdist = FreqDist(all_terms)
fdist


# In[41]:


fdist.plot(30)
plt.show()


# The head and the tail of the above graph consists of either punct or stop words which needs to be removed

# In[42]:


# Remove Every thing other than alphabets, numbers and space


# In[43]:


reviews_combined_clean = re.sub("[^\w\s]+","",reviews_combined)
all_terms = word_tokenize(reviews_combined_clean.lower())


# In[44]:


len(set(all_terms))


# In[45]:


from nltk.corpus import stopwords


# In[46]:


stop_nltk = stopwords.words("english")


# In[47]:


stop_updated = stop_nltk + ["mobile","phone","lenovo","k8","note"]


# In[48]:


reviews_updated1 = [term for term in all_terms if term not in stop_updated and len(term)>2]


# In[49]:


len(set(reviews_updated1))


# In[50]:


print((reviews_updated1[:200]))


# In[51]:


from nltk.stem import SnowballStemmer
stemmer_s = SnowballStemmer("english")


# In[52]:


reviews_updated_stem = [ stemmer_s.stem(word) for word in reviews_updated1]


# In[53]:


print((len(set(reviews_updated_stem))))


# stemmer reduces the number of distinct words in the corpus to a greater exten

# lets build a udf
# 
# - input : review 
# - Tasks : All the above cleaning steps
# - Return : string of cleaned reveiw
# 
# based on the outcome of this function, you should be able to add a new column in the data frame called as "cleaned_review"

# In[54]:


def clean_txt(sent):
    #Stripping white spaces before and after the text
    sent = sent.strip()
    #Replacing multiple spaces with a single space
    result = re.sub("\s+", " ", sent)
    #Replacing Non-Alpha-numeric and non space charecters with nothing
    result1 = re.sub(r"[^\w\s]","",result)
    tokens = word_tokenize(result1.lower())
    stemmed = [stemmer_s.stem(term) for term in tokens \
               if term not in stop_updated and \
               len(term) > 2] 
    res = " ".join(stemmed)
    return res


# In[55]:


reviews['clean_review'] = reviews.review.apply(clean_txt)


# In[56]:


reviews.head()


# # Word cloud on cleaned dataset

# In[57]:


reviews_combined_clean = " ".join(reviews.clean_review.values)


# In[58]:


reviews_combined_clean[:500]


# In[59]:


word_cloud = WordCloud(width=800,height=800,background_color='white',max_words=150).\
generate_from_text(reviews_combined_clean)


# In[60]:


plt.figure(figsize=[8,8])
plt.imshow(word_cloud)
plt.show()


# # Bar Graph on top 25 words

# In[61]:


from nltk.probability import FreqDist
all_terms = word_tokenize(reviews_combined_clean)
fdist = FreqDist(all_terms)


# In[62]:


fdist


# In[63]:


#From the dict obtain a data frame of words and freq's
df_dist = pd.DataFrame(list(fdist.items()), columns = ["words","freq"])


# In[64]:


get_ipython().run_line_magic('matplotlib', 'inline')
df_dist.sort_values(ascending=False, by="freq").head(25).\
plot.bar(x= "words", y= "freq",figsize=(20,5)) 


# # Plot the bar graph for top 25 frequenctly used bigrams

# In[65]:


from sklearn.feature_extraction.text import CountVectorizer


# In[66]:


# create a bigram count vectorizer object
bigram_count_vectorizer = CountVectorizer(ngram_range=(2,2),max_features=150)


# In[67]:


X_bigram = bigram_count_vectorizer.fit_transform(reviews['clean_review'])

# Creating a DTM
DTM_bigram = pd.DataFrame(X_bigram.toarray(), columns=bigram_count_vectorizer.get_feature_names())


# In[68]:


print((bigram_count_vectorizer.get_feature_names()[:20]))


# In[69]:


DTM_bigram.sum().sort_values(ascending=False).head(25).plot.bar(figsize=(20,5))  


# # Sentiment Classification

# ### Sentiment Prediction: Building our own model based on the Sentiment labels

# - Step1: get the X and y
# - Step2: converting text to numbers (countvectorizer or tfidfvectorizer)
# - Step3: Split into test and train
# - Step4: train the model
# - Step5: test and get the accuracy scores
# - Step6: Make predictions for an Input Review

# In[132]:


#Create a document term matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[164]:


count_vect = CountVectorizer()


# In[134]:


X_text = reviews.clean_review.values
y = reviews.sentiment.values


# In[165]:


#Extract the features on the reviews for train - fit
#Compute the count of every word extarcted in every document(review)
X = count_vect.fit_transform(X_text)


# In[136]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,
                                                 random_state=42)


# In[137]:


from sklearn.metrics import accuracy_score


# In[138]:


from sklearn.linear_model import LogisticRegression


# In[139]:


logreg = LogisticRegression()


# In[140]:


logreg.fit(X_train,y_train)


# In[141]:


y_test_pred = logreg.predict(X_test)


# In[192]:


accuracy_score(y_test, y_test_pred)


# In[211]:


#Make Predictions:
review1 = "This is a fantastic mobile really like it but the battery drains fast"
review2 = "Camera is good but when I speak over phone continuously over long time it gets heated up"


# In[230]:


review = [review1,review2]


# In[233]:


c_review = list(map(clean_txt, review))


# In[234]:


X_test_new = count_vect.transform(c_review)


# In[235]:


y_test_pred_new = logreg.predict(X_test_new)


# In[224]:


y_test_pred_new


# In[114]:


from sklearn.naive_bayes import MultinomialNB


# In[115]:


nb = MultinomialNB()


# In[116]:


nb.fit(X_train,y_train)


# In[117]:


y_test_pred = nb.predict(X_test)


# In[118]:


accuracy_score(y_test, y_test_pred)


# In[236]:


from sklearn.svm import SVC


# In[237]:


svc = SVC()


# In[238]:


svc.fit(X_train,y_train)


# In[239]:


y_test_pred = svc.predict(X_test)


# In[240]:


accuracy_score(y_test, y_test_pred)

