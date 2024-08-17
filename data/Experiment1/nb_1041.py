#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from imutils import face_utils
font = cv2.FONT_HERSHEY_SIMPLEX


# In[2]:


import os

def get_files(path):
    return os.listdir(path)

cascPath = "/Users/abdulrehman/opt/anaconda3/envs/Face-Detection/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml"

def return_bbx(image):
    faceCascade = cv2.CascadeClassifier(cascPath)
    faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
    return faces


# In[3]:


get_files('/Users/abdulrehman/Desktop/SML Project/FacesInTheWild/lfw-deepfunneled')


# In[4]:


Dataset_path = '/Users/abdulrehman/Desktop/SML Project/FacesInTheWild/'

Celebs = pd.read_csv(Dataset_path+'lfw_allnames.csv')
Celebs = Celebs[Celebs['images']>50]
Celebs


# In[5]:


for _,[name,images] in Celebs.iterrows():
    print(name)
    print((get_files(Dataset_path+'lfw-deepfunneled/'+name)))
    print('\n\n')


# In[6]:


image = cv2.imread(Dataset_path+'lfw-deepfunneled/Colin_Powell/Colin_Powell_0007.jpg')
faces = return_bbx(image)
(x,y,w,h) = faces[0]
cropped = image[x:x+w, y:y+h]
plt.imshow(cropped)
print((cropped.shape))


# In[7]:


resized = cv2.resize(cropped, (64,64), interpolation = cv2.INTER_AREA)
plt.imshow(resized)
print((resized.shape))


# In[8]:


X = []
Y = []

for _, [name,__] in Celebs.iterrows():
    celeb_path = Dataset_path+'lfw-deepfunneled/'+name+'/'
    
    images_paths = get_files(celeb_path)
    for image_path in images_paths:
        image = cv2.imread(celeb_path+image_path,1)
        faces = return_bbx(image)
        if len(faces) == 1:
            (x,y,w,h) = faces[0]
            cropped = image[x:x+w, y:y+h]
            dim = (64, 64)
            resized = cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA)
            image = np.array(resized).astype("float32")
            X.append(image)
            Y.append(name)

X_data = np.array(X)
Y_data = np.array(Y)


# In[9]:


X_data = np.array(X)
Y_data = np.array(Y)

print((X_data.shape))
print((Y_data.shape))


# In[ ]:





# In[10]:


import mahotas
bins = 20

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(int)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

def fd_lbp(image):    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(int)
    # compute the haralick texture feature vector
    haralick = mahotas.features.lbp(gray, 5, 5).mean(axis=0)
    return haralick
 
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    return hist.flatten()

def get_global_features(image):
    global_feature = np.hstack([fd_histogram(image), fd_haralick(image), fd_lbp(image), fd_hu_moments(image)])
    return global_feature


# In[11]:


X_temp = []
for i in range(len(X_data)):
    X_temp.append(get_global_features(X_data[i]))


# In[12]:


X_data = np.array(X_temp)
print((X_data.shape))
print((Y_data.shape))


# In[13]:


from collections import Counter

counter = Counter(Y_data)
print(counter)


# In[14]:


from imblearn.under_sampling import NearMiss

undersample = NearMiss(version=1, n_neighbors=3)
X_resampled, Y_resampled = undersample.fit_resample(X_data,Y_data)
X_data = X_resampled
Y_data = Y_resampled
counter = Counter(Y_data)
print(counter)

del undersample
del X_resampled
del Y_resampled
del counter


# In[15]:


print((X_data.shape))
print((Y_data.shape))


# In[16]:


from sklearn.preprocessing import MinMaxScaler, LabelEncoder

scaler = MinMaxScaler(feature_range=(0, 1))
X_data = scaler.fit_transform(X_data)
labelencoder = LabelEncoder()
Y_data = labelencoder.fit_transform(Y_data)


# In[17]:


from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import svm

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, train_size=0.8, random_state = 0)


# In[18]:


tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['poly'], 'degree': [0, 1, 2, 3, 4, 5, 6],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['sigmoid'], 'C': [1, 10, 100, 1000]}]


# In[19]:


scores = ['accuracy']


for score in scores:
    print(("# Tuning hyper-parameters for %s" % score))
    print()

    clf = GridSearchCV(svm.SVC(), tuned_parameters, scoring='%s' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print((clf.best_params_))
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print(("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params)))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print((classification_report(y_true, y_pred)))
    print()


# In[ ]:





# In[20]:


rbf = svm.SVC(kernel='rbf', gamma=0.1, C=100).fit(X_train, y_train)
accuracy_rbf = rbf.score(X_train, y_train)
print(("Training Accuracy Radial Basis Kernel:", accuracy_rbf*100))
accuracy_rbf = rbf.score(X_test, y_test)
print(("Testing Accuracy Radial Basis Kernel:", accuracy_rbf*100))


# In[ ]:




