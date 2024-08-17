#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/junghoum/Hello-world/blob/main/Untitled41.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


import requests
from urllib import parse
from bs4 import BeautifulSoup
base_url = 'http://movie.naver.com/movie/point/af/list.nhn?&page={}'
url = base_url.format(1)
res = requests.get(url)

if res.status_code == 200:
  soup = BeautifulSoup(res.text)
  tds = soup.select('table.list_netizen > tbody > tr > td.title')
  print(len(tds))
  for td in tds:
    movie_title = td.select_one('a.movie').text.strip()
    link = td.select_one('a.movie').get('href')
    link = parse.urljoin(base_url, link)
    score = td.select_one('div.list_netizen_score > em').text.strip()
    comment = td.select_one('br').next_sibling.strip()
    print(movie_title, link, score, comment, sep=' :: ')
    print('-------------------------------------------------')


# In[2]:


import random
random.uniform(0.2, 1.2)


# In[3]:


import requests
import time
import random
from bs4 import BeautifulSoup


base_url = 'https://movie.naver.com/movie/point/af/list.nhn?&page={}'
#결과 저장할 리스트
comment_list = []
for page in range(1, 101):
    url = base_url.format(page)
    res=requests.get(url)
    if res.status_code == 200:
        soup = BeautifulSoup(res.text, 'lxml')
        tds = soup.select('table.list_netizen > tbody > tr > td.title')
        for td in tds:
            movie_title = td.select_one('a.movie').text.strip()
            link = td.select_one('a.movie').get('href')
            link = parse.urljoin(base_url, link)
            score = td.select_one('div.list_netizen_score > em').text.strip()
            comment = td.select_one('br').next_sibling.strip()
            # 리스트에 저장
            comment_list.append((movie_title, link, score, comment))
        interval = round(random.uniform(0.2, 1.2), 2)
        time.sleep(interval)
print('종료') 


# In[4]:


import pandas as pd
df = pd.DataFrame(comment_list, 
                  columns=['영화제목','영화링크', '평점','댓글'])
df.to_csv('naver_comment.csv', encoding='utf-8', index=False)


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


get_ipython().system('pip install konlpy')


# In[8]:


import pandas as pd
import numpy as np

from konlpy.tag import Okt
import re

okt = Okt()
tagset_df = pd.Series(okt.tagset)
tagset_df


# In[9]:


def get_pos(x):
  tagger = Okt()
  pos = tagger.pos(x)

  result = []

  for al in pos:
    result.append(f'{al[0]}/{al[1]}')

  return result


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[11]:


def apply_regular_expression(text):
    hangul = re.compile('[^ ㄱ-ㅣ 가-힣]')  # 한글 추출 규칙: 띄어 쓰기(1 개)를 포함한 한글
    result = hangul.sub('', text)  # 위에 설정한 "hangul"규칙을 "text"에 적용(.sub)시킴
    return result


a = apply_regular_expression(df['댓글'][0])


print(get_pos(a))


# In[12]:


df['댓글'][0]


# In[13]:


nouns = okt.nouns(apply_regular_expression(df['댓글'][0]))
nouns


# In[14]:


#말뭉치 생성

corpus = "".join(df['댓글'].tolist())
corpus


# In[15]:


# 정규 표현식 적용

apply_regular_expression(corpus)


# In[16]:


# 전체 말뭉치(corpus)에서 명사 형태소 추출

nouns = okt.nouns(apply_regular_expression(corpus))
print(nouns)


# In[17]:


# 빈도 탐색
from collections import Counter

counter = Counter(nouns)


# In[18]:


# 가장 많은 단어 top 10

counter.most_common(10)


# In[19]:


# 한글자 명사 제거하고 top 10

available_counter = Counter({x: counter[x] for x in counter if len(x) > 1})
available_counter.most_common(10)


# In[20]:


# "우리", "매우"와 같은 실질적인 의미가 없고 꾸밈 역할을 하는 불용어를 불용어 사전을 정의하여 제거

stopwords = pd.read_csv("https://raw.githubusercontent.com/yoonkt200/FastCampusDataset/master/korean_stopwords.txt").values.tolist()
stopwords[:10]


# In[21]:


# 위 단어들 외에도 분석하고자 하는 데이터셋에 특화된 불용어 단어들도 불용어 사전에 추가하기

jeju_hotel_stopwords = ['영화', '리뷰']

for word in jeju_hotel_stopwords:
  stopwords.append(word)


# In[22]:


# BoW 벡터 생성

from sklearn.feature_extraction.text import CountVectorizer

def text_cleaning(text):
  hangul = re.compile('[^ ㄱ-ㅣ 가-힣]') # 정규 표현식 처리
  result = hangul.sub('', text)
  okt = Okt() #형태소 추출
  nouns = okt.nouns(result)
  nouns = [x for x in nouns if len(x) > 1] # 한글자 키워드 제거
  nouns = [x for x in nouns if x not in stopwords] # 불용어 제거
  return nouns

vect = CountVectorizer(tokenizer = lambda x: text_cleaning(x))
bow_vect = vect.fit_transform(df['댓글'].tolist())
word_list = vect.get_feature_names()
count_list = bow_vect.toarray().sum(axis=0)


# In[23]:


# 단어 리스트

word_list


# In[24]:


# 각 단어가 전체 리뷰중에 등장한 총 횟수

count_list


# In[25]:


# 각 단어의 리뷰별 등장 횟수

bow_vect.toarray()


# In[26]:


bow_vect.shape


# In[27]:


# "단어" - "총 등장 횟수" Matching

word_count_dict = dict(zip(word_list, count_list))

word_count_dict


# In[28]:


# TF-IDF 적용(변환)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_vectorizer = TfidfTransformer()
tf_idf_vect = tfidf_vectorizer.fit_transform(bow_vect)


# In[29]:


print(tf_idf_vect.shape)


# In[30]:


# 변화 후 1000*3599 matrix가 출력된다. 여기서 한 행은 한 리뷰를 의미라고, 한 열은 한 단어를 의미한다.

# 첫 번째 리뷰에서의 단어 중요도(TF-IDF 값) -- 0이 아닌 것만 출력

print(tf_idf_vect[0])


# In[31]:


# 첫 번째 리뷰에서 모든 단어의 중요도 -- 0인 값까지 포함

print(tf_idf_vect[0].toarray().shape)
print(tf_idf_vect[0].toarray())


# In[32]:


# "벡터", "단어" mapping

vect.vocabulary_


# In[33]:


invert_index_vectorizer = {v: k for k, v in vect.vocabulary_.items()}
print(str(invert_index_vectorizer)[:100]+'...')


# In[34]:


# 감성 분류 -- Logistic Regression


# In[35]:


# 데이터셋 생성, 우리는 이용자의 리뷰를 "긍정"/"부정" 두가지 부류로 나누고자 한다.하지만 이러한 이용자의 감성을 대표할 수 있는 "평가 점수" 변수는 1~5의  value를 가지고 있다. 따라서 "평가 점수"변수 (rating: 1~5)를 이진 변수(긍정:1 부정:0)으로 변환해야 한다.

df.sample(10)


# In[36]:


df['평점'] = df['평점'].astype(str).astype(int)


# In[37]:


# 리뷰 내용과 평점을 보면 4~5 는 긍정적인 리뷰, 1~3은 부정적인 리뷰가 많이 보였기에 4~5는 1, 1~3은 0 으로 분류

df['평점'].hist()

a = df['평점']


# In[38]:


def rating_to_label(a):
  if a > 7:
    return 1
  else:
    return 0

df['y'] = a.apply(lambda x: rating_to_label(x))


# In[39]:


df.head()


# In[40]:


# Training set/ Test set 나누기

from sklearn.model_selection import train_test_split

x = tf_idf_vect
y = df['y']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=2)


# In[41]:


# model 학습하기

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# fit in training set
lr = LogisticRegression(random_state = 0)
lr.fit(x_train, y_train)

# predict in test set
y_pred = lr.predict(x_test)


# In[42]:


# 분류 결고 평가

# classification result for test set

print('accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('precision: %.2f' % precision_score(y_test, y_pred))
print('recall: %.2f' % recall_score(y_test, y_pred))
print('F1: %.2f' % f1_score(y_test, y_pred))


# In[43]:


# confusion matrix

from sklearn.metrics import confusion_matrix

confu = confusion_matrix(y_true = y_test, y_pred = y_pred)

plt.figure(figsize=(4, 3))
sns.heatmap(confu, annot=True, annot_kws={'size':15}, cmap='OrRd', fmt='.10g')
plt.title('Confusion Matrix')
plt.show()

# 모델 평가결과를 살펴보면, 모델이 지나치게 긍정("1")으로만 예측하는 경향이 있습니다. 따라서 긍정 리뷰를 잘 예측하지만, 부정 리뷰에 대한 예측 정확도가 매우 낮다. 이는 샘플데이터의 클래스 불균형으로 인한 문제로 보인다.
# 따라서, 클래스 불균형 조정을 진행


# In[44]:


# 샘플링 재조정하기

# 1:1 Sampling

df['y'].value_counts()


# In[45]:


positive_random_idx = df[df['y']==1].sample(275, random_state=12).index.tolist()
negative_random_idx = df[df['y']==0].sample(275, random_state=12).index.tolist()



# In[46]:


random_idx = positive_random_idx + negative_random_idx
x = tf_idf_vect[random_idx]
y = df['y'][random_idx]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 2)


# In[47]:


# 모델 재학습

lr2 = LogisticRegression(random_state = 0)
lr2.fit(x_train, y_train)
y_pred = lr2.predict(x_test)


# In[48]:


#clasification result for test set

print('accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('precision: %.2f' % precision_score(y_test, y_pred))
print('recall: %.2f' % recall_score(y_test, y_pred))
print('F1: %.2f' % f1_score(y_test, y_pred))


# In[49]:


# confusion matrix

from sklearn.metrics import confusion_matrix

confu = confusion_matrix(y_true = y_test, y_pred = y_pred)

plt.figure(figsize=(4,3))
sns.heatmap(confu, annot=True, annot_kws={'size':15}, cmap='OrRd', fmt='.10g')
plt.show()

# 이제 모델이 "긍정적인" 케이스와 "부정적인"케이스를 모두 적당히 잘 맞춘 것을 확인할 수 있습니다.


# In[50]:


# 긍정/ 부정 키원드 분석

# 긍정/ 부정 키원드를 추출하기 위해 먼저 Logistic Regression 모델에 각 단어의
# coeficient를 시각화해보기

lr2.coef_


# In[51]:


# print logistic regression's coef

plt.figure(figsize=(10, 8))
plt.bar(range(len(lr2.coef_[0])), lr2.coef_[0])

# 여기서 계수가 양인 경우는 단어가 긍정적인 영향을 미쳤다고 볼 수 있고, 반면에, 음인 경우는 부정적인 영향을 미쳤다고 볼 수 있습니다.
# 이 계수들을 크기순으로 정렬하면, 긍정/ 부정 키워드를 출력하는 지표가 되겠습니다.


# In[52]:


# 긍정 키원드와 부정 키워드의 top 5를 각각 출력해보기

print(sorted(((value, index) for index, value in enumerate(lr2.coef_[0])), reverse = True)[:5])

print(sorted(((value, index) for index, value in enumerate(lr2.coef_[0])), reverse = True)[-5:])

# 단어의 coeficient와 index가 출력된다.


# In[53]:


# 전체 단어가 포함한 "긍정 키원드 리스트"와 "부정 키원드 리스트"를 정의하고 출력하기

coef_pos_index = sorted(((value, index) for index, value in enumerate(lr2.coef_[0])), reverse=True)

coef_neg_index = sorted(((value, index) for index, value in enumerate(lr2.coef_[0])), reverse=False)

coef_pos_index


# In[54]:


invert_index_vectorizer = {v: k for k, v in vect.vocabulary_.items()}
invert_index_vectorizer


# In[55]:


for coef in coef_pos_index[:20]:
  print(invert_index_vectorizer[coef[1]], coef[0])


# In[56]:


for coef in coef_neg_index[:20]:
  print(invert_index_vectorizer[coef[1]], coef[0])


# In[56]:




