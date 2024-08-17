#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os


# In[2]:


datapath='dataset'


# In[3]:


os.listdir(datapath)


# In[4]:


category=os.listdir(datapath)


# In[5]:


category


# In[6]:


labels=[i for i in range(len(category))]


# In[7]:


labels


# In[8]:


dict_label=dict(list(zip(category,labels)))


# In[9]:


dict_label


# In[10]:


size_im=100


# In[11]:


data=[]


# In[12]:


target=[]


# In[13]:


for cat in category:
    folder_path=os.path.join(datapath,cat)
    img_names=os.listdir(folder_path)
    
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)
        
        try:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            resized_img=cv2.resize(gray,(size_im,size_im))
            data.append(resized_img)
            target.append(dict_label[cat])
        
        except Exception as e:
            print(e)


# In[14]:


data


# In[15]:


target


# In[16]:


len(data)


# In[17]:


len(target)


# In[18]:


import matplotlib.pyplot as plt


# In[19]:


plt.imshow(data[300])


# In[20]:


target[300]


# In[21]:


data[0].shape


# In[22]:


data[0].max()


# In[23]:


import numpy as np


# In[24]:


data=np.array(data)/255.0


# In[25]:


data[0].max()


# In[26]:


data.shape[0]


# In[27]:


data=np.reshape(data,(data.shape[0],size_im,size_im,1))


# In[28]:


target=np.array(target)


# In[29]:


from keras.utils import np_utils


# In[30]:


new_target=np_utils.to_categorical(target)


# In[31]:


new_target


# In[32]:


np.save('data',data)
np.save('target',new_target)


# In[33]:


from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout,Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint


# In[34]:


model=Sequential()


# In[35]:


model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


# In[36]:


model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


# In[37]:


model.add(Flatten())


# In[38]:


model.add(Dropout(0.5))


# In[39]:


model.add(Dense(50,activation='relu'))


# In[40]:


model.add(Dense(2,activation='softmax'))


# In[41]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


x_train,x_test,y_train,y_test=train_test_split(data,new_target,test_size=0.1)


# In[44]:


checkpoint=ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history=model.fit(x_train,y_train,epochs=10,callbacks=[checkpoint],validation_split=0.2)


# In[45]:


plt.plot(history.history['loss'],'r',label='Training Loss')
plt.plot(history.history['val_loss'],'g',label='Validation Loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[46]:


plt.plot(history.history['accuracy'],'r',label='Training Accuracy')
plt.plot(history.history['val_accuracy'],'g',label='Validation Accuracy')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[47]:


model.evaluate(x_test,y_test)


# In[48]:


model.save('model.model')


# In[60]:


from keras.models import load_model
model=load_model('pre-trained.model')


# In[68]:


classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[65]:


color_dict={1:(0,255,0),0:(0,0,255)}


# In[66]:


labels_dict={1:'Mask',0:'No Mask'}


# In[69]:


source=cv2.VideoCapture(0)
while True:
    
    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=classifier.detectMultiScale(gray)
    
    for x,y,w,h in faces:
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normal=resized/255.0
        reshaped=np.reshape(normal,(1,100,100,1))
        prediction=model.predict(reshaped)
        result=np.argmax(prediction,axis=1)[0]
        
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[result],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[result],-1)
        cv2.putText(img,labels_dict[result],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    
    
    cv2.imshow("Detector",img)
    key=cv2.waitKey(1)
    
    if key==27:
        break

cv2.destroyAllWindows()
source.release()


# In[ ]:




