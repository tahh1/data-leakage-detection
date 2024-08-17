#!/usr/bin/env python
# coding: utf-8

# ### News Category Classification Assignnment 
# 
#                                                                                         -- Submitted By Rinki Chatterjee

# #### Importing Necessary Libraries

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[35]:


data = pd.read_json("/kaggle/input/news-category-dataset/News_Category_Dataset_v2.json",lines=True)


# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report,confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM,Bidirectional,SpatialDropout1D
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import Embedding,Flatten,Dense,Dropout
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import asarray,zeros
from wordcloud import WordCloud

import warnings
warnings.filterwarnings('ignore')


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.isna().apply(pd.value_counts) #missing value check


# In[7]:


data.category.nunique() # number of unique categories


# In[8]:


data.category.unique()


# * WORLDPOST and THE WORLDPOST were given as two separate categories in the dataset. Here I change the category THE WORLDPOST to WORLDPOST 

# In[9]:


data.category = data.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)


# In[10]:


data.category.nunique() # number of unique categories


# In[11]:


# Plotting top 10 category
group_count = data['category'].value_counts()
sns.barplot(group_count.index[:10], group_count.values[:10], alpha=0.8)
plt.title('Top 10 category ')
plt.ylabel('Counts', fontsize=12)
plt.xlabel('Category groups', fontsize=12,)
plt.xticks(rotation=45)
plt.show()


# In[12]:


# Plotting top 10 authors
group_count = data['authors'].value_counts()
sns.barplot(group_count.index[:10], group_count.values[:10], alpha=0.8)
plt.title('Top 10 authors')
plt.ylabel('Count', fontsize=12)
plt.xlabel('Author types', fontsize=12,)
plt.xticks(rotation='vertical')
plt.show()


# In[13]:


#Plotting word cloud
from wordcloud import WordCloud
desc = " ".join(str(des) for des in data['headline'])

wc_desc = WordCloud(background_color='white', max_words=200, width=400, height=400,random_state=10).generate(desc)
plt.figure(figsize=(10,10))
plt.imshow(wc_desc)
plt.title("Word cloud for Headline column")


# In[14]:


#Plotting word cloud for short description
from wordcloud import WordCloud
desc = " ".join(str(des) for des in data['short_description'])

wc_desc = WordCloud(background_color='white', max_words=200, width=400, height=400,random_state=10).generate(desc)
plt.figure(figsize=(10,10))
plt.imshow(wc_desc)
plt.title("Word cloud for short description column")


# ### Data cleaning 

# In[15]:


import unicodedata
import nltk
import spacy
from nltk.tokenize.toktok import ToktokTokenizer
stopword_list = nltk.corpus.stopwords.words('english')
tokenizer = ToktokTokenizer()
nlp = spacy.load('en_core_web_sm', parse = False, tag=False, entity=False)
#from contractions import CONTRACTION_MAP
import nltk
nltk.download('words')
words = set(nltk.corpus.words.words())
stopword_list = nltk.corpus.stopwords.words('english')


# #### Defining all functions for data cleaning

# In[17]:


# Remove any emails 
def remove_emails(text):
    text = re.sub(r'\b[^\s]+@[^\s]+[.][^\s]+\b', ' ', text)
    return text

def remove_hyperlink(text):
    text=re.sub(r'(http|https)://[^\s]*',' ',text)
    return text

# Removing Digits
def remove_digits(text):
    #text= re.sub(r"\b\d+\b", "", text)
    text= re.sub(r"(\s\d+)", " ", text)
    return text
    

# Removing Special Characters
def remove_special_characters(text):
    text = re.sub('[^a-zA-Z\s]', ' ', text)
    return text


# removing accented charactors
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

 # Removing Stopwords
def remove_stopwords(text,is_lower_case):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]

    filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)   
    return filtered_text

# Lemmetization
def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


# In[18]:


# Combine all the functions and creating a preprocessing pipeline
# # Text preprocessing
def text_preprocessing(corpus,isRemoveEmail,isRemoveDigits,isRemoveHyperLink, 
                     isRemoveSpecialCharac,isRemoveAccentChar,
                       text_lower_case,text_lemmatization, stopword_removal):
    
    normalized_corpus = []
    
    for doc in corpus:
        
        if text_lower_case:
            doc = doc.lower()
        
        if isRemoveEmail:
            doc = remove_emails(doc)
        
        if isRemoveHyperLink:
            doc=remove_hyperlink(doc)
             
        if isRemoveAccentChar:
            doc = remove_accented_chars(doc)
            
        if isRemoveDigits:
            doc = remove_digits(doc)
        
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
        # insert spaces between special characters to isolate them    
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)
        
        if text_lemmatization:
            doc = lemmatize_text(doc)
        
        if isRemoveSpecialCharac:
            doc = remove_special_characters(doc)
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        
        if stopword_removal:
            doc = remove_stopwords(doc,is_lower_case=text_lower_case)
                
        normalized_corpus.append(doc)
        
    return normalized_corpus


# In[36]:


EMAIL_FLAG=True
DIGIT_FLAG=True
HYPER_LINK_FLAG=True
ALL_SPEC_CHAR_FLAG=True
ACCENT_CHAR_FLAG=True
LOWER_CASE_FLAG=True
LEMMETIZE_FLAG=False
STOPWORD_FLAG=True

clean_headline= text_preprocessing(data['headline'],EMAIL_FLAG,DIGIT_FLAG,HYPER_LINK_FLAG,
                   ALL_SPEC_CHAR_FLAG,ACCENT_CHAR_FLAG,
                  LOWER_CASE_FLAG,LEMMETIZE_FLAG,STOPWORD_FLAG)
clean_short_Desc = text_preprocessing(data['short_description'],EMAIL_FLAG,DIGIT_FLAG,HYPER_LINK_FLAG,
                   ALL_SPEC_CHAR_FLAG,ACCENT_CHAR_FLAG,
                  LOWER_CASE_FLAG,LEMMETIZE_FLAG,STOPWORD_FLAG)


# In[37]:


data['clean_headline']=clean_headline
data['clean_short_Desc'] = clean_short_Desc


# ### Wordcloud after cleaning the data

# ### Category wise word cloud

# In[21]:


#Description
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.figure(figsize=(15,15))

for index, i in enumerate(data['category'].unique()):
  s = str(i)
  i = str(data[data['category']==s].clean_headline)
  i = WordCloud(background_color='white', max_words=200, width=400, height=400,random_state=10).generate(i)
  c = index+1
  plt.subplot(10,5,c)
  plt.imshow(i)
  plt.title(s)


# * Merging the columns into one

# In[38]:


data['MergedColumn'] = data[data.columns[6:8]].apply(
    lambda x: ' '.join(x.astype(str)),
    axis=1
)


# In[23]:


pd.set_option('display.max_colwidth', -1)


# In[24]:


data['MergedColumn'][0:10]


# #### Creating the final data

# In[39]:


final_df = data.copy()
del data
final_df.drop(columns=['headline', 'authors', 'link', 'short_description', 'date',
                   'clean_headline', 'clean_short_Desc'],axis=1,inplace=True)


# In[26]:


final_df.columns


# In[27]:


final_df.to_csv('final_data.csv',index=False)


# In[28]:


#Plotting word cloud
from wordcloud import WordCloud
desc = " ".join(str(des) for des in final_df['MergedColumn'])

wc_desc = WordCloud(background_color='white', max_words=200, width=400, height=400,random_state=10).generate(desc)
plt.figure(figsize=(10,10))
plt.imshow(wc_desc)
plt.title("Word cloud for final data after cleaning")


# #### ML Algorithms

# In[29]:


### Count Vectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(final_df['MergedColumn']).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X,final_df['category'], test_size=0.2, random_state=42)

print((X_train.shape))
print((y_train.shape))
print((X_test.shape))
print((y_test.shape))




# ### Naive Bayes
# 

# In[30]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)

predictions = nb.predict(X_test)

print(('Training accuracy',nb.score(X_train,y_train)))

from sklearn.metrics import accuracy_score
print(("Testing accuracy " ,accuracy_score(predictions,y_test)))


# In[31]:


confusion_matrix(y_test,predictions)


# In[32]:


classification_report(y_test,predictions)


# ### Logistic Regression

# In[33]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
print(("Training accuracy",lr.score(X_train,y_train)))

predictions = lr.predict(X_test)

from sklearn.metrics import accuracy_score
print(("Testing accuracy",accuracy_score(predictions,y_test)))


# ### K Nearest Neighbours
# 

# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=5,weights='distance')
# knn.fit(X_train,y_train)
# print("Training accuracy",knn.score(X_train,y_train))
# 
# predictions = knn.predict(X_test)
# 
# from sklearn.metrics import accuracy_score
# print("Testing accuracy",accuracy_score(predictions,y_test))

# ### SVM

# from sklearn.svm import SVC
# svc= SVC(C=1.0,kernel='linear',degree=3,gamma='auto')
# svc.fit(X_train,y_train)
# 
# print("Training accuracy",svc.score(X_train,y_train))
# 
# predictions = svc.predict(X_test)
# 
# from sklearn.metrics import accuracy_score
# print("Testing accuracy",accuracy_score(predictions,y_test))

# ### Deep Learning Models

# In[40]:


import matplotlib.pyplot as plt

text_length = []

for text in range(len(final_df['MergedColumn'])):
    try:
        text_length.append(len(final_df['MergedColumn'][text]))

    except Exception as e:
        pass

print(("Maximum length of  Data", np.max(text_length)))
print(("Minimum length of  Data", np.min(text_length)))
print(("Median length of  Data", np.median(text_length)))
print(("Average length of  Data",np.mean(text_length)))
print(("Standard Deviation of  Data",np.std(text_length)))
plt.boxplot(text_length)
plt.show()


# ### Using TFidf Vectorizer for Merged columns
# 
# 

# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer

# tokenize and build vocab
tfidf = TfidfVectorizer(max_features=5000,min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english',max_df=0.7)
X = tfidf.fit_transform(final_df.MergedColumn).toarray()


print((tfidf.vocabulary_))
print(("length of the vocabulary ",len(tfidf.vocabulary_)))

y = np.asarray(final_df['category'])
y = pd.get_dummies(final_df['category']).values
print((X.shape))
print((y.shape))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


model = Sequential() 
model.add(Dense(100, activation='softmax',input_shape=(x_train[0].shape)))
model.add(Dropout(0.2))
model.add(Dense(100, activation='softmax'))
model.add(Dense(40, activation='softmax'))
# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# summarize the model
print((model.summary())) 


# In[10]:


history= model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=5,batch_size=64)


# In[11]:


loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(('Accuracy: %f' % (accuracy*100)))


# In[17]:


plt.title('Accuracy')
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show();


# In[18]:


plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();


# In[ ]:




