#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import imdb
#num_words表示加载影评时，确保影评里面的单词使用频率保持在前1万位，于是有些很少见的生僻词在数据加载时会舍弃掉
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# In[2]:


print((train_data[0]))
print((train_labels[0]))


# In[4]:


#频率与单词的对应关系存储在哈希表word_index中,它的key对应的是单词，value对应的是单词的频率
word_index = imdb.get_word_index()
#我们要把表中的对应关系反转一下，变成key是频率，value是单词
reverse_word_index = dict([(value, key) for (key, value) in list(word_index.items())])
'''
在train_data所包含的数值中，数值1，2，3对应的不是单词，而用来表示特殊含义，1表示“填充”，2表示”文本起始“，
3表示”未知“，因此当我们从train_data中读到的数值是1，2，3时，我们要忽略它，从4开始才对应单词，如果数值是4，
那么它表示频率出现最高的单词
'''
text = ""
for wordCount in train_data[0]:
    if wordCount > 3:
        text += reverse_word_index.get(wordCount - 3)
        text += " "
    else:
        text += "?"

print(text)


# In[5]:


import numpy as np
def oneHotVectorizeText(allText, dimension=10000):
    '''
    allText是所有文本集合，每条文本对应一个含有10000个元素的一维向量，假设文本总共有X条，那么
    该函数会产生X条维度为一万的向量，于是形成一个含有X行10000列的二维矩阵
    '''
    oneHotMatrix = np.zeros((len(allText), dimension))
    for i, wordFrequence in enumerate(allText):
        oneHotMatrix[i, wordFrequence] = 1.0
    return oneHotMatrix

x_train = oneHotVectorizeText(train_data)
x_test =  oneHotVectorizeText(test_data)

print((x_train[0]))

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# In[6]:


from keras import models
from keras import layers

model = models.Sequential()
#构建第一层和第二层网络，第一层有10000个节点，第二层有16个节点
#Dense的意思是，第一层每个节点都与第二层的所有节点相连接
#relu 对应的函数是relu(x) = max(0, x)
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
#第三层有16个神经元，第二层每个节点与第三层每个节点都相互连接
model.add(layers.Dense(16, activation='relu'))
#第四层只有一个节点，输出一个0-1之间的概率值
model.add(layers.Dense(1, activation='sigmoid'))


# In[8]:


import matplotlib.pyplot as plt
x = np.linspace(-10, 10)
y_relu = np.array([0*item if item < 0 else item for item in x])
plt.figure()
plt.plot(x, y_relu, label='ReLu')
plt.legend()


# In[9]:


from keras import losses
from keras import metrics
from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])


# In[10]:


x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[: 10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, 
                    validation_data = (x_val, y_val))


# In[12]:


train_result = history.history
print((list(train_result.keys())))


# In[15]:


import matplotlib.pyplot as plt

acc = train_result['acc']
val_acc = train_result['val_acc']
loss = train_result['loss']
val_loss = train_result['val_loss']

epochs = list(range(1, len(acc) + 1))
#绘制训练数据识别准确度曲线
plt.plot(epochs, loss, 'bo', label='Trainning loss')
#绘制校验数据识别的准确度曲线
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Trainning and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[25]:


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
print(results)


# In[27]:


model.predict(x_test)


# In[1]:


import pandas as pd
df = pd.read_json('/Users/chenyi/Documents/人工智能出书/数据集/第6章/News_Category_Dataset.json', lines=True)
df.head()


# In[2]:


categories = df.groupby('category')
print(("total categories: ", categories.ngroups))
print((categories.size()))


# In[3]:


df.category = df.category.map(lambda x:"WORLDPOST" if x == "THE WORLDPOST" else x)


# In[4]:


categories = df.groupby('category')
print(("total categories: ", categories.ngroups))
print((categories.size()))


# In[5]:


from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence, one_hot
df['text'] = df.headline + " " + df.short_description

# 将单词进行标号
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.text)
X = tokenizer.texts_to_sequences(df.text)
df['words'] = X


# In[27]:


#记录每条数据的单词数
df['word_length'] = df.words.apply(lambda i: len(i))
#清除单词数不足5个的数据条目
df = df[df.word_length >= 5]
df.word_length.describe()


# In[70]:


def word2Frequent(sequences):
    word_index = {}
    for sequence in sequences:
        for word in sequence:
            word_index[word] = word_index.get(word, 0) + 1
    return word_index
word_index = word2Frequent(df.words)


count = 10000
#将单词按照频率按照升序排序，然后取出排在第一万位的单词频率
s = [(k, word_index[k]) for k in sorted(word_index, key=word_index.get, reverse=True)]
print((s[0]))
frequent_to_index = {}
for i in range(count):
    frequent_to_index[s[i][0]] = 9999 - i


# In[30]:


# 将分类进行编号
categories = df.groupby('category').size().index.tolist()
category_int = {}
int_category = {}
for i, k in enumerate(categories):
    category_int.update({k:i})
    int_category.update({i:k})

df['c2id'] = df['category'].apply(lambda x: category_int[x])


# In[74]:


import numpy as np
import keras.utils as utils
from sklearn.model_selection import train_test_split
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i in range(len(sequences)):
        for word in sequences[i]:
            if frequent_to_index.get(word, None) is not None:
                pos = frequent_to_index[word]
                results[i, pos] = 1.0   
    return results

X = np.array(df.words)
X = vectorize_sequences(X)
print((X[0]))
Y = utils.to_categorical(list(df.c2id))


# 将数据分成两部分,80%用于训练，20%用于测试

seed = 29
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=seed)


# In[79]:


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
#当结果是输出多个分类的概率时，用softmax激活函数,它将为30个分类提供不同的可能性概率值
model.add(layers.Dense(len(int_category), activation='softmax'))

#对于输出多个分类结果，最好的损失函数是categorical_crossentropy
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[80]:


history = model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val), batch_size=512)


# In[82]:


import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = list(range(1, len(loss) + 1))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()
plt.show()


# In[89]:


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
#当结果是输出多个分类的概率时，用softmax激活函数,它将为30个分类提供不同的可能性概率值
model.add(layers.Dense(len(int_category), activation='softmax'))

#对于输出多个分类结果，最好的损失函数是categorical_crossentropy
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=20, batch_size=512)


# In[86]:


results = model.evaluate(x_val, y_val)
print(results)


# In[139]:


import pandas as pd
data_path = '/Users/chenyi/Documents/人工智能/housing.csv'
housing = pd.read_csv(data_path)
housing.info()


# In[93]:


housing.head()


# In[94]:


housing.describe()


# In[95]:


housing.hist(bins=50, figsize=(15,15))


# In[96]:


housing['ocean_proximity'].value_counts()


# In[101]:


import seaborn as sns
total_count = housing['ocean_proximity'].value_counts()
plt.figure(figsize=(10,5))
sns.barplot(total_count.index, total_count.values, alpha=0.7)
plt.title("Ocean Proximity Summary")
plt.ylabel("Number of Occurences", fontsize=12)
plt.xlabel("Ocean of Proximity", fontsize=12)
plt.show()


# In[102]:


print((housing.shape))


# In[173]:


#将ocean_proximity转换为数值
housing['ocean_proximity'] = housing['ocean_proximity'].astype('category')
housing['ocean_proximity'] = housing['ocean_proximity'].cat.codes
#将median_house_value分离出来最为被预测数据
data = housing.values
train_data = data[:, [0,1,2,3,4,5,6,7,9]]
train_value = data[:,[8]]
print((train_data[0]))
print((train_value[0]))


# In[178]:


print((np.isnan(train_data).any()))
print((np.argwhere(np.isnan(train_data))))
train_data[np.isnan(train_data)] = 0
print((np.isnan(train_data).any()))


# In[181]:


mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis = 0)
train_data /= std


# In[187]:


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])


# In[197]:


history = model.fit(train_data, train_value, epochs=300, 
                    validation_split=0.2, 
                    batch_size=32)


# In[198]:


val_mae_history = history.history['val_mean_absolute_error']
plt.plot(list(range(1, len(val_mae_history) + 1)), val_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# In[199]:


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(val_mae_history)

plt.plot(list(range(1, len(smooth_mae_history)+1)), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# In[1]:


import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
try:
    from PIL import Image
except ImportError:
    import Image

# Open image file
image = Image.open('doggy.jpeg')
my_dpi=300.

# Set up figure
fig=plt.figure(figsize=(float(image.size[0])/my_dpi,float(image.size[1])/my_dpi),dpi=my_dpi)
ax=fig.add_subplot(111)

# Remove whitespace from around the image
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

# Set the gridding interval: here we use the major tick interval
myInterval=100.
loc = plticker.MultipleLocator(base=myInterval)
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)

# Add the grid
ax.grid(which='major', axis='both', linestyle='-')

# Add the image
ax.imshow(image)

# Find number of gridsquares in x and y direction
nx=abs(int(float(ax.get_xlim()[1]-ax.get_xlim()[0])/float(myInterval)))
ny=abs(int(float(ax.get_ylim()[1]-ax.get_ylim()[0])/float(myInterval)))

# Add some labels to the gridsquares
for j in range(ny):
    y=myInterval/2+j*myInterval
    for i in range(nx):
        x=myInterval/2.+float(i)*myInterval
        ax.text(x,y,'{:d}'.format(i+j*nx),color='w',ha='center',va='center')

# Save the figure
fig.savefig('doggy.tiff',dpi=my_dpi)

