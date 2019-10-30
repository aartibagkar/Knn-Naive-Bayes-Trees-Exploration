#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as s


# # (3 A) Generate 600 independent numbers normally distributed with mean 0 and standard deviation

# ####  A Declaring sigma and covariance

# In[2]:


mu1 = [680,700]
mu2 = [10,550]

cov1 = [[149,-47],[-47,209]]
cov2 = [[200,15],[15,250]]


mean1 = [680,700]
mean2 = [10,550]

cov1 = [[149,-47],[-47,209]]
cov2 = [[200,15],[15,250]]


# In[3]:


array1 = np.random.normal(0, 1, 600)
print("First 5 elements for red which are normally distributed ",array1[1:5])


array2 = np.random.normal(0, 1, 600)
print("First 5 elements for blue which are normally distributed ",array2[1:5])


# #### B Organize these numbers into a 2 × 300 matrix, called data

# In[4]:


data = np.array([array1[1:300], array1[301:600]])
# print(data)


# #### C & D Multiply the matrix data by covariance

# In[5]:


train1 = np.matmul(cov1,data)
   
train1[0] = train1[0]+680   
train1[1] = train1[1]+700 


# #### A similar approach generates the 300 blue points. Just use µB and ΣB instead

# In[6]:


data2 = np.array([array1[1:300], array1[301:600]])
train2 = np.matmul(cov2,data2)
    
train2[0] = train2[0]+10
train2[1] = train2[1]+550 


# # 3b) Next, put these 600 point, along with their color value Y in a data frame called simData

# In[7]:


train2 = train2.transpose()
train1 = train1.transpose()


# In[8]:


df = pd.DataFrame(data= train1) 
df.columns = ['x1','x2']
df['Y'] = "red" 
df.head()

df2 = pd.DataFrame(data= train2) 
df2.columns = ['x1','x2']
df2['Y'] = "blue" 
df2.head()

simData = df.append(df2, ignore_index=True)
 
print("simdata along with the color assigned")    
print(simData.head())
print(simData.tail())
# train.tail()


# # 3c) Plot the red and blue points in a scatter plot, with the right color.

# In[9]:


# import seaborn as sns
# sns.set(color_codes=True)
# p = sns.lmplot('x1', 'x2', data= train, hue ='Y', palette="Set1", fit_reg = False, scatter_kws = {"s": 10})


plt.scatter(simData["x1"], simData["x2"], c=simData["Y"],s=50, cmap='RdBu')
lim = plt.axis()


# # 3d) Generate another 200 red and 200 blue points just as above, for testing purpose. Collect these 400 points in a data frame called testDF

# In[10]:


array3 = np.random.normal(0, 1, 400)


array4 = np.random.normal(0, 1, 400)

data3 = np.array([array3[1:200], array3[201:400]])
train3 = np.matmul(cov1,data3)
train3[0] = train3[0]+680   
train3[1] = train3[1]+700 



data4 = np.array([array4[1:200], array4[201:400]])
train4 = np.matmul(cov2,data4)
train4[0] = train4[0]+10
train4[1] = train4[1]+550 

df3 = pd.DataFrame(data= train3) 
df3 = df3.transpose()
df3.columns = ['x1','x2']
df3['Y'] = "red" 
df3.head()

df4 = pd.DataFrame(data= train4) 
df4 = df4.transpose()
df4.columns = ['x1','x2']
df4['Y'] = "blue" 
df4.head()

test = df3.append(df4, ignore_index=True)

print("test data generated as follows")
print(test.head())
print(test.tail())


# In[11]:


plt.scatter(test["x1"], test["x2"], c=test["Y"],s=50, cmap='RdBu')
lim = plt.axis()


# In[12]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
train_pred=le.fit_transform(simData["Y"])

test_pred=le.fit_transform(test["Y"])


# # 3e) Generate a 200 × 200 grid of in the range of X1 and X2. Using Bayes decision rule color these points red or blue according to the Bayes rule, and add them to the scatter plot of the simData (use a small dot for the grid points.)

# #### Naive Bayes model fitting and prediction

# In[13]:


from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics


gnb = GaussianNB()
gnb.fit(simData.loc[:, simData.columns != 'Y'], train_pred)


y_pred = gnb.predict(test.loc[:, test.columns != 'Y'])



# Model Accuracy, how often is the classifier correct?


print("Accuracy:",metrics.accuracy_score(test_pred, y_pred))


# In[14]:


predict =  test[['x1', 'x2']].copy()
predict['Y'] = y_pred

predict.head()

convert = lambda x: "red" if x==1 else("blue")
predict['Y'] = predict['Y'].map(convert)

predict.head()


# In[15]:



plt.scatter(test["x1"], test["x2"], c=predict["Y"],s=50, cmap='RdBu')
lim = plt.axis()


# #### 3f) Using Bayes decision rule predict the Y value of this test data and compare it to their actual value. Print the confusion table for testDF and also print the test error rate.

# In[16]:


confusion = metrics.confusion_matrix(predict["Y"], test["Y"])

print(metrics.confusion_matrix(predict["Y"], test["Y"]))
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]


# In[17]:


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

results = confusion_matrix(test["Y"], predict["Y"]) 
print ('Confusion Matrix :')
print(results) 
print ('Accuracy Score :',accuracy_score(test["Y"], predict["Y"]) )
print ('Report : ')
print (classification_report(test["Y"], predict["Y"]))


# # 3g) Load the kknn library for R or the KNeighborsClassifier for Python. Use the simData data frame you generated earlier. Using 1-folding cross validation find the best k for the kNN model for this data. Using this k build a kNN model called bestKnnModel on the training data simData. Predict the Y values of the test data testDF. Print the confusion table and the test error rate of the best kNN model.
# 

# In[18]:


from sklearn.neighbors import KNeighborsClassifier



kList=[]
minusList=[]
plusList=[]
totalList=[]
lowKey=1
highKey=501
Step=5 #Number of different k's to test
n=20 #Number of times for each k to generate train/test sets
p=0.2 #fraction of data set aside for test set

for k in range(lowKey, highKey,Step):
    kList.append(k)
    knnMinusErrorRate=0
    knnPlusErrorRate=0
    knnErrorRate=0
    for i in range(1,n):
        classifier = KNeighborsClassifier(n_neighbors=k)
        X=simData.iloc[:,[0,1]]
        classifier.fit(X, simData["Y"])
        yPred=classifier.predict(test[['x1','x2']])
        yPredP=classifier.predict_proba(test[['x1','x2']])
        C=confusion_matrix(test["Y"], yPred)
        knnMinusErrorRate+=C[0,1]/(C[0,0]+C[0,1])
        knnPlusErrorRate+=C[1,0]/(C[1,0]+C[1,1])
        knnErrorRate+=(C[1,0]+C[0,1])/sum(sum(C))
    #end inner for
    knnMinusErrorRate/=n #find the avg error over n tests
    knnPlusErrorRate/=n
    knnErrorRate/=n
    minusList.append(knnMinusErrorRate) #store for graphing
    plusList.append(knnPlusErrorRate)
    totalList.append(knnErrorRate)
#end for


# In[19]:



bestErrRate=min(totalList)
bestK=kList[np.argmin(totalList)]

print("Best K value :: ",bestK)
classifier=KNeighborsClassifier(n_neighbors=bestK)
X=simData.iloc[:,[0,1]]
classifier.fit(X, simData["Y"])

yTestPred=classifier.predict(test.iloc[:,[0,1]])
yTestPredP=classifier.predict_proba(test.iloc[:,[0,1]])
C=confusion_matrix(test["Y"],yTestPred)

   
s='Error rate for k={0:d} is {1:.3f}'
print(s.format(bestK, bestErrRate))
print('confusion matrix:')
print(C)


# In[312]:


print(" KNN Prediction plot")
plt.scatter(test["x1"], test["x2"], c=predict["Y"],s=50, cmap='RdBu')
lim = plt.axis()

