#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# # توابع مورد استفاده برای چاپ نمودار ها

# In[5]:


def plot_loss(history, label, n):
  # Use a log scale to show the wide range of values.
  plt.semilogy(history.epoch,  history.history['loss'],
               color=colors[n], label='Train '+label)
  plt.semilogy(history.epoch,  history.history['val_loss'],
          color=colors[n], label='Val '+label,
          linestyle="--")
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  
  plt.legend()

def plot_metrics(history):
  metrics =  ['loss', 'auc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.subplots_adjust(right=1.5)
    plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.7,1])
    else:
      plt.ylim([0,1])

    plt.legend()
    
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]


# In[6]:


dataframe = pd.read_csv('/kaggle/input/web-club-recruitment-2018/train.csv')
dataframe.head()


# In[7]:


dataframe.describe()


# In[8]:


neg, pos = np.bincount(dataframe['Y'])
total = neg + pos
print(('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total)))


# <p lang="fa" dir="rtl" align="right">
#     با توجه به خروجی بالا داده ها بالانس نیستند وقتی قبل از بالانس کردن داده ها مدل رو ترین کردیم مدل برای اکثریت داده ها خروجی رو دسته 0 تشخیص می داد
# </p>
# 
# 
# 

# In[9]:


data_labels = np.array(dataframe['Y'])
bool_data_labels = data_labels != 0

data_features = np.array(dataframe)

pos_features = data_features[bool_data_labels]
neg_features = data_features[~bool_data_labels]

pos_labels = data_labels[bool_data_labels]
neg_labels = data_labels[~bool_data_labels]



# In[10]:


ids = np.arange(len(pos_features))
choices = np.random.choice(ids, len(neg_features))

res_pos_features = pos_features[choices]
res_pos_labels = pos_labels[choices]

res_pos_features.shape


# <p lang="fa" dir="rtl" align="right">
# برای بالانس کردن داده ها به وسیله کد زیر سعی کردیم تعداد داده های 1 را به وسیله resample افزایش بدیم
# </p>
# 
# 

# In[11]:


resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)

order = np.arange(len(resampled_labels))
np.random.shuffle(order)
resampled_features = resampled_features[order]
resampled_labels = resampled_labels[order]

resampled_features.shape


# In[12]:


neg1, pos1 = np.bincount(resampled_labels)
total1 = neg1 + pos1
print(('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total1, pos1, 100 * pos1 / total1)))


# In[13]:


res = np.insert(resampled_features, 1, resampled_labels, axis=1)
pres = pd.DataFrame(data=resampled_features,columns=dataframe.columns)
pres.head()


# In[14]:


train, test = train_test_split(pres, test_size=0.08)
train, val = train_test_split(train, test_size=0.08)
print((len(train), 'train examples'))
print((len(val), 'validation examples'))
print((len(test), 'test examples'))


# In[15]:


train_labels = np.array(train['Y'])
bool_train_labels = train_labels != 0
val_labels = np.array(val['Y'])
# test_labels = np.array(test['Y'])


# In[16]:


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Y')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


# In[17]:


def demo(feature_column):
    train_ds = df_to_dataset(train, batch_size=5)
    example_batch = next(iter(train_ds))[0]
    feature_layer = layers.DenseFeatures(feature_column)
    print((feature_layer(example_batch).numpy()))


# <p lang="fa" dir="rtl" align="right">
#    بعد از بالانس کردن داده ها مدل شروع به تشخیص دو دسته کرد ولی مشکلی که پیش اومد دقت مدل خیلی کم بود و همچنین مدل مشکل overfit نیز داشت.
#     برای رفع این مشکلات اومدیم و داده ها رو با توابع زیر نرمال کردیم و دقت مدل رسید به 98 درصد.
#     دقت خیلی بهبود پیدا کرد ولی مدل عمیقا overfit شده بود.
#     نمودار history بر روی train و val از  هم فاصله می گرفت
# </p>

# In[18]:


def zscore_x6(col):
  mean = dataframe.describe()['X6']['mean']
  std = dataframe.describe()['X6']['std']
  return (col - mean)/std

def zscore_x7(col):
  mean = dataframe.describe()['X7']['mean']
  std = dataframe.describe()['X7']['std']
  return (col - mean)/std

def zscore_x8(col):
  mean = dataframe.describe()['X8']['mean']
  std = dataframe.describe()['X8']['std']
  return (col - mean)/std

def zscore_x9(col):
  mean = dataframe.describe()['X9']['mean']
  std = dataframe.describe()['X9']['std']
  return (col - mean)/std

def zscore_x10(col):
  mean = dataframe.describe()['X10']['mean']
  std = dataframe.describe()['X10']['std']
  return (col - mean)/std

def zscore_x11(col):
  mean = dataframe.describe()['X11']['mean']
  std = dataframe.describe()['X11']['std']
  return (col - mean)/std

def zscore_x12(col):
  mean = dataframe.describe()['X12']['mean']
  std = dataframe.describe()['X12']['std']
  return (col - mean)/std

def zscore_x13(col):
  mean = dataframe.describe()['X13']['mean']
  std = dataframe.describe()['X13']['std']
  return (col - mean)/std

def zscore_x14(col):
  mean = dataframe.describe()['X14']['mean']
  std = dataframe.describe()['X14']['std']
  return (col - mean)/std

def zscore_x15(col):
  mean = dataframe.describe()['X15']['mean']
  std = dataframe.describe()['X15']['std']
  return (col - mean)/std

def zscore_x16(col):
  mean = dataframe.describe()['X16']['mean']
  std = dataframe.describe()['X16']['std']
  return (col - mean)/std
def zscore_x17(col):
  mean = dataframe.describe()['X17']['mean']
  std = dataframe.describe()['X17']['std']
  return (col - mean)/std

def zscore_x18(col):
  mean = dataframe.describe()['X18']['mean']
  std = dataframe.describe()['X18']['std']
  return (col - mean)/std

def zscore_x19(col):
  mean = dataframe.describe()['X19']['mean']
  std = dataframe.describe()['X19']['std']
  return (col - mean)/std

def zscore_x20(col):
  mean = dataframe.describe()['X20']['mean']
  std = dataframe.describe()['X20']['std']
  return (col - mean)/std

def zscore_x21(col):
  mean = dataframe.describe()['X21']['mean']
  std = dataframe.describe()['X21']['std']
  return (col - mean)/std

def zscore_x22(col):
  mean = dataframe.describe()['X22']['mean']
  std = dataframe.describe()['X22']['std']
  return (col - mean)/std

def zscore_x23(col):
  mean = dataframe.describe()['X23']['mean']
  std = dataframe.describe()['X23']['std']
  return (col - mean)/std


# In[19]:


feature_columns = []

X1_buckets = feature_column.bucketized_column(feature_column.numeric_column('X1'), boundaries=[i*10000 for i in range(1, 100)])
# demo(X1_buckets)
feature_columns.append(X1_buckets)

X2_categories = feature_column.indicator_column(feature_column.categorical_column_with_identity('X2', 3))
# demo(X2_categories)
feature_columns.append(X2_categories)

X3_categories = feature_column.indicator_column(feature_column.categorical_column_with_identity('X3', 7))
# demo(X3_categories)
feature_columns.append(X3_categories)

X4_categories = feature_column.indicator_column(feature_column.categorical_column_with_identity('X4', 4))
# demo(X4_categories)
feature_columns.append(X4_categories)


X5_buckets = feature_column.bucketized_column(feature_column.numeric_column('X5'), boundaries=[25, 30, 35, 40, 45, 50, 55, 60, 65, 70 , 75])
# demo(X5_buckets)
feature_columns.append(X5_buckets)



feature_columns.append(feature_column.numeric_column('X6', normalizer_fn=zscore_x6))
feature_columns.append(feature_column.numeric_column('X7', normalizer_fn=zscore_x7))
feature_columns.append(feature_column.numeric_column('X8', normalizer_fn=zscore_x8))
feature_columns.append(feature_column.numeric_column('X9', normalizer_fn=zscore_x9))
feature_columns.append(feature_column.numeric_column('X10', normalizer_fn=zscore_x10))
feature_columns.append(feature_column.numeric_column('X11', normalizer_fn=zscore_x11))
feature_columns.append(feature_column.numeric_column('X12', normalizer_fn=zscore_x12))
feature_columns.append(feature_column.numeric_column('X13', normalizer_fn=zscore_x13))
feature_columns.append(feature_column.numeric_column('X14', normalizer_fn=zscore_x14))
feature_columns.append(feature_column.numeric_column('X15', normalizer_fn=zscore_x15))
feature_columns.append(feature_column.numeric_column('X16', normalizer_fn=zscore_x16))
feature_columns.append(feature_column.numeric_column('X17', normalizer_fn=zscore_x17))
feature_columns.append(feature_column.numeric_column('X18', normalizer_fn=zscore_x18))
feature_columns.append(feature_column.numeric_column('X19', normalizer_fn=zscore_x19))
feature_columns.append(feature_column.numeric_column('X20', normalizer_fn=zscore_x20))
feature_columns.append(feature_column.numeric_column('X21', normalizer_fn=zscore_x21))
feature_columns.append(feature_column.numeric_column('X22', normalizer_fn=zscore_x22))
feature_columns.append(feature_column.numeric_column('X23', normalizer_fn=zscore_x23))


# In[20]:


feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


# In[21]:


batch_size = 200
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


# <p lang="fa" dir="rtl" align="right">
#    برای رفع مشکل overfit از لایه های dropout استفاده کردیم و مشکل حل شد ولی دقت مدل به ۸۳ کاهش پیدا کرد.
#     همچنین رگرسیون خطی L2 رو هم تست کردیم ولی نتیجه dropput خیلی بهتر بود
# </p>
# 

# In[22]:


model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(200, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(100, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(80, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(40, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=METRICS)


history = model.fit(train_ds,
          validation_data=val_ds,
          epochs=150)
model.summary()


# In[23]:


model.evaluate(test_ds)


# In[24]:


plot_loss(history, "history", 1)


# In[25]:


plot_metrics(history)


# <p lang="fa" dir="rtl" align="right">
#   در آخر در نمودار خطا می بینیم که مدل مشکل high variance نداره
#     همچنین دقت و recal و precision مدل به حدود ۸۰ رسیده
#     بنظر می رسه که برای بهبود بیشتر دقت و سایر معیار های مدل نیاز به داده های بیشتر هست چونکه با کاهش سایز ولیدیشن و تست ست دقت مدل بیشتر می شه ولی چون که دوست نداریم مدل احیانا آورفیت بشه بیشتر از این اندازه مجموعه داده های تست و ولیدیت رو کاهش نمی دیم
#     نکته مهم بعدی معیار ROC هست که با مقدار 0.9 نشون میده مدل با احتمال خوبی دسته بندی رو انجام میده
# </p>

# In[43]:


test_dataframe = pd.read_csv('/kaggle/input/web-club-recruitment-2018/test.csv')
kaggle_test_ds = tf.data.Dataset.from_tensor_slices((dict(test_dataframe))).batch(32)
print(kaggle_test_ds)
print(train_ds)


# In[53]:


pred= model.predict(kaggle_test_ds).round().astype(int)
df = pd.DataFrame(pred)
df.index.name = 'id'
df.columns = ['predicted_val']
df.head()


# In[55]:


df.to_csv('output.csv', index=True)


# In[ ]:




