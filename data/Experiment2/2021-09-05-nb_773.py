#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn  
import torch.optim as optim  
import torchvision.transforms as transforms 
import torchvision
import os
from skimage import io
from torch.utils.data import (Dataset,DataLoader) 
import cv2


# In[2]:


torch.cuda.get_device_name(0)


# In[3]:


from google.colab import drive
drive.mount('/content/drive')


# In[4]:


df_train = pd.read_csv('/content/drive/MyDrive/SoftComputing Assignment/Verification code/verification_train.csv')
df_test  = pd.read_csv('/content/drive/MyDrive/SoftComputing Assignment/Verification code/verification_test.csv')


# In[5]:


df_train


# In[6]:


df_test


# In[7]:


y_train = df_train['label']
y_test  = df_test['label']
print(y_train)
print(y_test)


# In[8]:


x_train = df_train.drop(['label'], axis = 1)
x_test = df_test.drop(['label'], axis = 1)


# In[9]:


x_train


# In[10]:


x_test


# In[11]:


y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# In[12]:


x_train = x_train.to_numpy()
x_test = x_test.to_numpy()


# In[14]:


train_dataset = []
for i in range(len(x_train)):
  arr = x_train[i].reshape(28,28).astype('float32')
  width = 180
  height = 180
  dim = (width,height)
  resized = cv2.resize(arr, dim, interpolation = cv2.INTER_AREA)
  resized = cv2.normalize(resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  z = y_train[i]
  resized = torch.tensor(resized,dtype=torch.float32)
  train_dataset.append((resized,z))


# In[15]:


len(train_dataset)


# In[16]:


train_dataset


# In[18]:


test_dataset = []
for i in range(len(x_test)):
  arr = x_test[i].reshape(28,28).astype('float32')
  width = 180
  height = 180
  dim = (width,height)
  resized = cv2.resize(arr, dim, interpolation = cv2.INTER_AREA)
  resized = cv2.normalize(resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  z = y_test[i]
  resized = torch.tensor(resized,dtype=torch.float32)
  test_dataset.append((resized,z))


# In[19]:


len(test_dataset)


# In[20]:


batch_size = 20
num_iters = 20000
input_dim = 180*180 
num_hidden = 200
output_dim = 10
learning_rate = 0.01 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[21]:


train_loader = torch.utils.data.DataLoader(dataset=train_dataset , 
                                           batch_size=batch_size, 
                                           shuffle=True) 

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)  


# In[22]:


class DeepNeuralNetworkModel(nn.Module):
    def __init__(self, input_size, num_classes, num_hidden):
        super().__init__()
      
        self.linear_1 = nn.Linear(input_size, num_hidden)
        self.relu_1 = nn.ReLU()

        self.linear_2 = nn.Linear(num_hidden, num_hidden)
        self.relu_2 = nn.ReLU()

        self.linear_3 = nn.Linear(num_hidden, num_hidden)
        self.relu_3 = nn.ReLU()

        self.linear_4 = nn.Linear(num_hidden, num_hidden)
        self.relu_4 = nn.ReLU()

        self.linear_5 = nn.Linear(num_hidden, num_hidden)
        self.relu_5 = nn.ReLU()

        self.linear_6 = nn.Linear(num_hidden, num_hidden)
        self.relu_6 = nn.ReLU()

        self.linear_out = nn.Linear(num_hidden, num_classes)

    def forward(self, x):
       
        out  = self.linear_1(x)
        out = self.relu_1(out)
        
        out  = self.linear_2(out)
        out = self.relu_2(out)

        out  = self.linear_3(out)
        out = self.relu_3(out)

        out  = self.linear_4(out)
        out = self.relu_4(out)

        out  = self.linear_5(out)
        out = self.relu_5(out)

        out  = self.linear_6(out)
        out = self.relu_6(out)
        
        probas  = self.linear_out(out)
        return probas



model = DeepNeuralNetworkModel(input_size = input_dim,
                               num_classes = output_dim,
                               num_hidden = num_hidden)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# In[23]:


num_epochs = num_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)
print(num_epochs)


# In[24]:


iter = 0
total_loss = []
iteration = []
correctly_classified_PerEpoch = []
accuracy_perEpoch = []
num_of_epoch = []
# for epoch in range(num_epochs):
for epoch in range(num_epochs):
    print(('Epoch: ',epoch+1))
    num_of_epoch.append(epoch+1)
    for i, (images, labels) in enumerate(train_loader):

        images = images.view(-1, 180*180).to(device)

        labels = labels.to(device)


        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images) 
        

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1
        print(('Iteration no: ',iter))
        if iter % 750 == 0:
            print(('loss in iteration ',iter,'= ',loss.item()))
            total_loss.append(loss.item())
            iteration.append(iter)

    correct = 0
    total = 0
    for images, labels in test_loader:
          
          images = images.view(-1,180*180).to(device)
          outputs = model(images)
 
          _, predicted = torch.max(outputs, 1)

          total += labels.size(0)
 
          if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.cpu()).sum() 
          else:
                correct += (predicted == labels).sum()

    accuracy = 100 * correct.item() / total
    accuracy_perEpoch.append(accuracy)
    correctly_classified_PerEpoch.append(correct.item())
    print(('Total Data: {}.  CorrectlyPredicted: {}.'.format(total,correct)))
    print(('Accuracy in epoch ',epoch+1,'=',accuracy))


# In[26]:


print(('loss count after every 750 iteration: ',total_loss))
print(('iteration count: ',iteration))
print(('Correctly classfied per epoch: ',correctly_classified_PerEpoch))
print(('accuracy per epoch: ',accuracy_perEpoch))
print(('Epoch list: ',num_of_epoch))


# In[28]:


plt.plot(iteration,total_loss)
plt.title('Iteration vs Loss(verification of Experiment 1)')
plt.show()


# In[29]:


plt.plot(num_of_epoch,correctly_classified_PerEpoch)
plt.title('Correctly classified per epoch(verification of Experiment 1)')
plt.show()


# In[31]:


plt.plot(num_of_epoch,accuracy_perEpoch)
plt.title('Accuracy per epoch(verification of Experiment 1)')
plt.show()


# In[32]:


root_path = '/content/drive/MyDrive/SoftComputing Assignment/Verification code/Experiment1'
save_model = True

if save_model is True:
    torch.save(model.state_dict(), root_path + 'VerificationFullDone.pkl') 

