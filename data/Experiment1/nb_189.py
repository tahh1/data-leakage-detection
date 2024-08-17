#!/usr/bin/env python
# coding: utf-8

# # Homework 4
# 
# Gabriel Idris Gilling (gig2106@columbia.edu) & Juan Lopez-Martin (jl5522@columbia.edu) 

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, RepeatedStratifiedKFold, GridSearchCV,train_test_split, cross_val_score
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re


# In[ ]:


import spacy
nlp = spacy.load("en_core_web_lg", disable = ["tagger", "parser", "ner"])


# In[ ]:


df = pd.read_csv("drive/My Drive/ML/winemag-data-130k-v2.csv")
df = df.sample(10000, random_state= 2020)
df.columns


# ## Ex 1

# We start building a model based on some of the features except for the description. Using LASSO, we get a $R^2$ of 0.34.

# In[ ]:


X = pd.get_dummies(df[['country', 'province', 'region_1','region_2', 'taster_name', 'variety']], dummy_na=True)
X['price'] = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, df['points'], random_state = 2020)


# In[ ]:


ctransformer = make_column_transformer((make_pipeline(StandardScaler(), SimpleImputer()), X.columns=='price'), 
                                       remainder='passthrough')

pipe_lasso = make_pipeline(ctransformer, Lasso())

gridlasso = GridSearchCV(estimator=pipe_lasso, param_grid={'lasso__alpha': [1, 0.1, 0.01, 0.0001, 0.00001]}, cv=3)
gridlasso.fit(X_train, y_train)
gridlasso.score(X_test, y_test)


# We get a similar score using Random Forest.

# In[ ]:


pipe_rf = make_pipeline(ctransformer, RandomForestRegressor())

pipe_rf.fit(X_train, y_train)
pipe_rf.score(X_test, y_test)


# ### 1.2 Simple Model using Bag of Words

# Using a very simple BoW model with CountVectorizer gets us a baseline of $R^2 = 0.6$. Note that we are using max_features = 2000 to reduce computing time. 

# In[ ]:


vect = CountVectorizer(max_features=2000)

X1_train, X1_test, y1_train, y1_test = train_test_split(df['description'], df['points'], random_state = 2020)

X1_train_bow = vect.fit_transform(X1_train)
X1_test_bow = vect.transform(X1_test)

lr = LassoCV().fit(X1_train_bow, y1_train)
lr.score(X1_test_bow, y1_test)


# ### 1.3 Simple Model with preprocessing, TF-IDF and bigrams
# 
# To start, we define a simple preprocessing function that lemmatizes our Wine descriptions after keeping only words that are constituted of characters and are not stopwords. Previous trials have shown that this basic preprocessing increases our models' accuracy by a small but significant margin.
# 
# We then deploy the TF-IDF vectorizer, setting the ngram_range argument to (1,2), instructing it to find bigrams and transform the data accordingly.

# In[ ]:


def preprocess2(series):
    series = series.apply(lambda x : [w.lemma_ for w in nlp(x) if w.is_alpha and not w.is_stop])

    return (series)


# In[ ]:


df['description_preprocess'] = preprocess2(df['description'])
df['description_preprocess'].head()


# In[ ]:


X = pd.get_dummies(df[['country', 'province', 'region_1','region_2', 'taster_name', 'variety']], dummy_na=True)
X['price'] = df['price']
X['description_preprocess'] = df['description_preprocess']
X_train, X_test, y_train, y_test = train_test_split(X, df['points'], random_state = 2020)

vect = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
X_train_tfidf = vect.fit_transform(X_train['description_preprocess'].apply(lambda x: ' '.join(x)))
X_test_tfidf = vect.transform(X_test['description_preprocess'].apply(lambda x: ' '.join(x)))


# In[ ]:


lr = LassoCV().fit(X_train_tfidf, y_train)
lr.score(X_test_tfidf, y_test)


# Surprisingly, we get a slightly worse score than the one using CountVectorizer. We also removed the use of bigrams to check if that could improve the score, but the increment was minimal.

# In[ ]:


vect = TfidfVectorizer(max_features=2000)
X_train_tfidf = vect.fit_transform(X_train['description_preprocess'].apply(lambda x: ' '.join(x)))
X_test_tfidf = vect.transform(X_test['description_preprocess'].apply(lambda x: ' '.join(x)))
lr = LassoCV().fit(X_train_tfidf, y_train)
lr.score(X_test_tfidf, y_test)


# ## 1.4 Merging BoW with non-text features

# Although there is a small improvement when including the non-textual features, we expected a greater increase. Potentially, a better model could increase this score.

# In[ ]:


X = pd.get_dummies(df[['country', 'province', 'region_1','region_2', 'taster_name', 'variety']], dummy_na=True)
X['price'] = df['price']
X['description'] = df['description']
X_train, X_test, y_train, y_test = train_test_split(X, df['points'], random_state = 2020)

vect = CountVectorizer(max_features=2000)
X_train_bow = vect.fit_transform(X_train['description'])
X_test_bow = vect.transform(X_test['description'])


# In[ ]:


X_train_all = np.concatenate((X_train_bow.todense(), X_train.loc[:, X_train.columns != 'description'].to_numpy()), axis = 1)
X_test_all = np.concatenate((X_test_bow.todense(), X_test.loc[:, X_train.columns != 'description'].to_numpy()), axis = 1)


# In[ ]:


condition = np.concatenate(([False]*(X_train_all.shape[1]-1), [True]), axis = 0)
ctransformer = make_column_transformer((make_pipeline(StandardScaler(), SimpleImputer()), condition), 
                                       remainder='passthrough')

pipe_lasso = make_pipeline(ctransformer, Lasso())

gridlasso = GridSearchCV(estimator=pipe_lasso, param_grid={'lasso__alpha': [0.01, 0.001, 0.0001, 0.00001]}, cv=3)
gridlasso.fit(X_train_all, y_train)
gridlasso.score(X_test_all, y_test)


# # Ex 2
# 
# In the section below, we use Spacy's implementation of Word2Vec to predict the wine qualities.
# 
# First, we use a standalone model using Word2vec only.
# 
# Afterwards, we merge out TF-IDF model with our Word2Vec models, and we get better results.

# ### a) Standalone Spacy Word2Vec

# In[ ]:


X3_train, X3_test, y3_train, y3_test = train_test_split(df['description_preprocess'].apply(lambda x: ' '.join(x)), df['points'], random_state = 2020)


# In[ ]:


X3_train


# In[ ]:


X_docs_train = [nlp(d).vector for d in X3_train]
X_docs_test = [nlp(d).vector for d in X3_test]


# In[ ]:


X_docs_train = np.vstack(X_docs_train)
X_docs_test = np.vstack(X_docs_test)


# In[ ]:


X_docs_train.shape


# In[ ]:


lr = LassoCV().fit(X_docs_train, y3_train)


# The standalone model using only text features that were converted to Word2Vec format using spacy achieves an $R^2$ of nearly 0.5, which is relatively low compared to our previous models, but understandable given the fact that we're only using a single column in our dataset.

# In[ ]:


lr.score(X_docs_test, y3_test)


# ## b) Let's stack Word2vec and BoW
# 
# We want to see whether we can merge our vectorized representations of text features in both TF-IDF format to achieve a better result.
# 
# First, we run a simple Lasso model using the TFIDF vectorizer on our descriptions to establish a baseline. We can see that the standalone TFIDF model, using 2000 features performs better than the Spacy word2vec embeddings. We are keeping 2000 features to speed up computation.
# 
# 

# In[ ]:


vect = TfidfVectorizer(stop_words=None, max_features= 2000)
X3_train_tfidf = vect.fit_transform(X3_train)
X3_test_tfidf = vect.transform(X3_test)

lr = LassoCV().fit(X3_train_tfidf, y3_train)
lr.score(X3_test_tfidf, y3_test)


# In[ ]:


X_docs_train = [nlp(d).vector for d in X3_train]
X_docs_test = [nlp(d).vector for d in X3_test]


# In order to concatenate our TF-IDF vectorized representations with the Spacy word embeddings, we first need to transform them into dense format, while also converting the word embeddings back to Numpy array format.

# In[ ]:


X3_train_comb = np.concatenate((X3_train_tfidf.todense(), np.array(X_docs_train)), axis = 1)


# In[ ]:


X3_test_comb = np.concatenate((X3_test_tfidf.todense(), np.array(X_docs_test)), axis = 1)


# We achieve an $R^2$ of 0.61 which is higher than our baseline for both the standalone word2vec embeddings and TFIDF models, effectively demonstrating that combining both types of features can increase model accuracy by a substantive margin.

# In[ ]:


lr = LassoCV().fit(X3_train_comb, y3_train)
lr.score(X3_test_comb, y3_test)


# # Part 3
# Our code is based on McCormick and Ryan's tutorial [BERT Fine-Tuning Tutorial with PyTorch](https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX#scrollTo=EKOTlwcmxmej) which is in turn based on the [run_glue.py](https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py) script in the transformers library.
# 
# However, there are important differences in our implementation. 
# * First and foremost, instead of a classification problem we are trying to solve a regression problem. Therefore, the last layer is of the model only has one neuron with a linear activation function that thus returns a continous numerical value. We use MSE as the loss and R^2 as the metric to easily compare with the previous models.
# * Second, we are using [ALBERT](https://arxiv.org/abs/1909.11942) instead of BERT. This should make training faster while keeping a similar performance.
# * We use a batch size of 32 and 4 epochs instead of 2. Some of the hyperparameters are slightly modified to follow the recommendations for ALBERT, still using AdamW.
# 
# For now we are using a random sample of 100,000 rows instead of the full dataset.
# 
# 
# 

# In[ ]:


seed_val = 2020

sample = 100000
batch_size = 32
epochs = 2


# In[ ]:


from google.colab import drive

import numpy as np
import pandas as pd
import random
import time
import datetime

import torch
from transformers import *

from sklearn.metrics import r2_score

from transformers import AlbertForSequenceClassification, AdamW, AlbertConfig
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print(('There are %d GPU(s) available.' % torch.cuda.device_count()))
    print(('We will use the GPU:', torch.cuda.get_device_name(0)))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

drive.mount('/content/drive')
df = pd.read_csv("drive/My Drive/ML/winemag-data-130k-v2.csv")
df = df.sample(sample)
df.columns

criterion = torch.nn.MSELoss()


# In[ ]:


X = df['description'].to_numpy()
y = torch.tensor((df['points'].to_numpy()-80)/20).float()
nlabels = 1

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

def tokenize(sentences):
  input_ids = []
  attention_masks = []
  for sent in sentences:
      encoded_dict = tokenizer.encode_plus(
                          sent,                      # Sentence to encode.
                          add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                          max_length = 256,           # Pad & truncate all sentences.
                          pad_to_max_length = True,
                          return_attention_mask = True,   # Construct attn. masks.
                          return_tensors = 'pt',     # Return pytorch tensors.
                    )  
      input_ids.append(encoded_dict['input_ids'])
      attention_masks.append(encoded_dict['attention_mask'])

  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)

  return input_ids, attention_masks

X_input, X_attention = tokenize(X)

model = AlbertForSequenceClassification.from_pretrained(
    "albert-base-v2",
    num_labels = nlabels,
    output_attentions = False,
    output_hidden_states = False,
)
model.cuda()

optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)

dataset = TensorDataset(X_input, X_attention, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(('{:>5,} training samples'.format(train_size)))
print(('{:>5,} validation samples'.format(val_size)))

train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size)
validation_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset),
            batch_size = batch_size)

total_steps = len(train_dataloader) * epochs
# Create the learning rate scheduler as in run_glue.py
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss from McCormick and Ryan's tutorial
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


training_stats = []
total_t0 = time.time()

for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    print("")
    print(('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs)))
    print('Training...')
    t0 = time.time()

    # Reset loss
    total_train_loss = 0

    model.train()
    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print(('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed)))

        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        
        output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        loss = criterion(output[0].squeeze(), b_labels)
        total_train_loss += loss.item()
        loss.backward()

        # Clip the norm of the gradients to 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    # Average loss
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    training_time = format_time(time.time() - t0)
    print("")
    print(("  Average training loss: {0:.2f}".format(avg_train_loss)))
    print(("  Training epoch took: {:}".format(training_time)))
        
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
      
        with torch.no_grad():
            output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            loss = criterion(output[0].squeeze(), b_labels)

        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        output = output[0].squeeze().detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_eval_accuracy += r2_score(label_ids, output)
        
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print(("  R2: {0:.2f}".format(avg_val_accuracy)))

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    validation_time = format_time(time.time() - t0)    
    print(("  Validation Loss: {0:.2f}".format(avg_val_loss)))
    print(("  Validation took: {:}".format(validation_time)))

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print(("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0))))


# We get a final R^2 of 0.74. This is  better than any previous models we have trained, although note that all the others were trained on a smaller subset of the data. Note this score only takes into account the description; including other of the relevant features (e.g. price, denomination, etc.) would probably increase this number much more.
# 
