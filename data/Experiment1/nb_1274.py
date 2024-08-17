#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 1. Import data from HW5.xlsx into data frame, and print the data.
# Age: customer's age in completed years
# Experience: years of professional experience
# Education: 1=undergrad; 2=grad; 3=advanced/professional
# Family: family size
# Zip: home address zip code
# Income: annual income in $,000
# Mortgage: value of house mortgage if any in $,000
# Credit: average monthly spending on credit cards in $,000
# Loan: Does the customer have a personal loan?
# Securities: Does the customer have a securities account?
# Deposit: Does the customer have a certificate of deposit account?
# Online: Does the customer use internet banking?
# Card: Does the customer use a credit card?
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.cluster.vq import kmeans, vq 
df = pd.read_excel("HW5.xlsx")
df


# In[ ]:


# 2. Apply pandas.DataFrame.corr to analyze pairwise correlations of columns.
# 2.1 Print the pairwise correlations.
# 2.2 Show pairwise correlations in colormap with colorbar.
# 2.3 Label axes and figure title.
cor = df.corr()
print(cor)
plt.imshow(df.corr(), cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(list(range(len(df.columns))), df.columns, rotation=90)
plt.yticks(list(range(len(df.columns))), df.columns)
plt.xlabel("Pairwise correlation columns")
plt.ylabel("Pairwise correlation columns")
plt.title("Correlation Colormap")


# In[ ]:


# 3. Apply scipy.cluster.vq to perform 2-means clustering between Income and Credit.
# 3.1 Show two clusters in red or blue dots of size 1.
# 3.2 Show two centroids in green squares of size 8.
# 3.3 Label axes and figure title.

data = pd.concat((df['Income'],df['Credit']),axis=1)
centroids,_ = kmeans(data,2) 
# assign each sample to a cluster 
index,_ = vq(data,centroids) 

# plot different color for each cluster by its index 
plt.plot(data['Income'][index==0],data['Credit'][index==0],'or',markersize=1) 
plt.plot(data['Income'][index==1],data['Credit'][index==1],'ob',markersize =1) 
plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8) 
plt.xlabel("Income")
plt.ylabel("Credit")
plt.title("2 Means Clustering between Income and Credit")
plt.show() 


# In[ ]:


# 4. Apply scipy.cluster.vq to perform 3-means clustering between Income and Credit.
# 4.1 Show three clusters in red, blue or magenta dots of size 1.
# 4.2 Show three centroids in green squares of size 8.
# 4.3 Label axes and figure title.


centroids,_ = kmeans(data,3) 
# assign each sample to a cluster 
index,_ = vq(data,centroids) 
# plot different color for each cluster by its index 
plt.plot(data['Income'][index==0],data['Credit'][index==0],'or',markersize=1) 
plt.plot(data['Income'][index==1],data['Credit'][index==1],'ob',markersize=1) 
plt.plot(data['Income'][index==2],data['Credit'][index==2],'om',markersize=1) 
plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
plt.xlabel("Income")
plt.ylabel("Credit")
plt.title("3 Means Clustering between Income and Credit") 
plt.show() 


# In[ ]:


# 5. Apply scipy.stats.linregress to perform linear regression on Experience given Age.
# 5.1 Print the estimation equation.
# 5.2 Show the data in blue dots of size 1.
# 5.3 Show the estimation equation in red line of width 4.
# 5.4 Label axes and figure title.

from scipy.stats import linregress 
import numpy as np
linSlope, linIntercept, linR, linP, linSE = linregress(df["Age"], df["Experience"]) 
print(("Experience = {}*Age + {}".format(linSlope,linIntercept)))
dem = np.linspace(0, 100, 5) 
plt.plot(dem, linIntercept + linSlope * dem, '-r', 
linewidth=4) 
plt.plot(df["Age"], df["Experience"], '.b',markersize=1) 
plt.xlabel("Age")
plt.ylabel("Experience")
plt.title("Linear Regression ")


# In[ ]:


# 6. Apply scipy.optimize.curve_fit to perform curve fitting on Experience given Age.
# 6.1 Print the estimation equation.
# 6.2 Show the data in blue dots of size 1.
# 6.3 Show the estimation equation in red line of width 4.
# 6.4 Label axes and figure title.
from scipy.optimize import curve_fit 
att, var = curve_fit(lambda dem, itc, slp: itc + slp * dem, df["Age"], df["Experience"]) 
print(("Experience = {}*Age + {}".format(att[1],att[0])))
plt.plot(dem, att[0] + att[1] * dem, '-r', 
linewidth=4) 
plt.plot(df["Age"], df["Experience"], '.b',markersize=1) 
plt.xlabel("Age")
plt.ylabel("Experience")
plt.title("Curve Fitting")


# In[ ]:


# 7. Apply sklearn.linear_model.LogisticRegression to predict Loan using eight columns.
# 7.1 Print the estimated intercept and coefficients.
# 7.2 Estimate Loan prediction and probabilities on all eight columns.
# 7.3 Print the estimated Loan and probabilities.
# 7.4 Print the prediction accuracy score.
# 7.5 Print the confusion matrix.
# 7.6 Apply roc_curve function to create the true positive rate and false positive rate.
# 7.7 Create a square plot to report the Receiver Operating Characteristics (ROC) curve.
# 7.8 Show the ROC curve with a text label in blue color.
# 7.9 Show the baseline in broken line with a text label in black color.
# 7.10 Label axes and figure title.
import numpy as np
from sklearn.metrics import confusion_matrix,roc_curve,auc
from sklearn.linear_model import LogisticRegression
res = LogisticRegression().fit(df.iloc[:, :8] ,df["Loan"])
print(("Estimated intercept and coefficients:{},{}".format(res.intercept_, res.coef_)))
print(("Predictions : {}".format(res.predict(df.iloc[:, :8]))))
print(("Estimated Loan and probabilities:{}".format(res.predict_proba(df.iloc[:, :8]))))
print(("Prediction Accuracy score:{}".format(res.score(df.iloc[:,:8],df['Loan']))))
print(("Confusion Matrix:{}".format(confusion_matrix(res.predict(df.iloc[:,:8]),df['Loan']))))

fpr,tpr,threshold = roc_curve(res.predict(df.iloc[:,:8]),df['Loan'])
print(("False Positive rate:{}".format(fpr[1])))
print(("True positive rate:{}".format(tpr[1])))
plt.axis([0,1,0,1])
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'ROC')
plt.plot([0, 1], [0, 1],'k--',label='Base Line')
plt.legend(loc = 'lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

