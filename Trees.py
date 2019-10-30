#!/usr/bin/env python
# coding: utf-8

# # Read the data adultTrain 

# In[249]:


import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Read data from adult train

# In[232]:


data = pd.read_csv("C:/Semester 2/Algorithmic machine learning/Homework 2/adult.data", na_values =" ?",  header = None)
data.dropna(axis= 0, inplace = True)


# #### Insert column names

# In[239]:


adultTrain = pd.DataFrame(data)

header = ["age", "workclass", "fnlwgt", "education", "educationNum", "maritalStatus", "occupation",
"relationship", "race", "sex", "capitalGain", "capitalLoss", "hoursPerWeek", "nativeCountry",
"income"]

adultTrain.columns = header


# In[240]:


adultTrain.info()


# #### print the list of data that are categorical, along with the levels in each of them

# In[75]:


num_cols = adultTrain._get_numeric_data().columns


cat_adult  = adultTrain.drop(num_cols, axis=1)


for (columnName,columnData) in cat_adult.iteritems():
    print(columnName)
    print(cat_adult[columnName].value_counts())


# In[91]:


for col in cat_adult.columns: 
    print(col) 


# In[77]:


adultTrain.head()


# In[79]:


adultTrain.tail()


# #### Initially I have build model with 500 sample and then assigned original data to sample data for final processing

# In[178]:


sample_adult = adultTrain
sample_adult.head()


# In[179]:


sample_adult["income"].value_counts()


# # 4b) Load the naivebayes package or the analogeous package in Python. Build a model called naiveModelA with income as the response variable, and other features as independent variables. The model should use no Laplace smoothing, and treat numerical data non-parametrically using a kernel based method. Print a summary of the model.

# #### preprocessing of categorical data

# In[241]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
sample_adult.loc[:,"income"] =le.fit_transform(sample_adult["income"])
sample_adult.loc[:,"workclass"] =le.fit_transform(sample_adult["workclass"])
sample_adult.loc[:,"education"] =le.fit_transform(sample_adult["education"])
sample_adult.loc[:,"maritalStatus"] =le.fit_transform(sample_adult["maritalStatus"])
sample_adult.loc[:,"occupation"] =le.fit_transform(sample_adult["occupation"])
sample_adult.loc[:,"relationship"] =le.fit_transform(sample_adult["relationship"])
sample_adult.loc[:,"race"] =le.fit_transform(sample_adult["race"])
sample_adult.loc[:,"sex"] =le.fit_transform(sample_adult["sex"])
sample_adult.loc[:,"nativeCountry"] =le.fit_transform(sample_adult["nativeCountry"])



# #### Building the naive baye model

# In[181]:


from sklearn.naive_bayes import MultinomialNB

nb_classifier = MultinomialNB(alpha=0.0,)
nb_classifier.fit(sample_adult.loc[:, sample_adult.columns != 'income'], sample_adult.loc[:,"income"])


print(nb_classifier.get_params(True))
nb_classifier
# y_pred = gnb.predict(test.loc[:, test.columns != 'Y'])


# # 4c) From the model extract the estimated class conditional probabilities of males and females under the sex feature for people with income over 50K and people with income under 50K per year.

# In[242]:


less_income = sample_adult["income"] == 0
high_income = sample_adult["income"] == 1

female = sample_adult["sex"] == 2
male = sample_adult["sex"] == 1

f_less_income = len(sample_adult[less_income & female])
f_highs_income = len(sample_adult[high_income & female])
m_highs_income = len(sample_adult[high_income & male])
m_less_income = len(sample_adult[less_income & male])
less = len(less_income)
high = len(high_income)
print("conditional probabilities of males under 50K ", m_less_income/less)
print("conditional probabilities of males over 50K ", m_less_income/high)
print("conditional probabilities of females over 50K ", f_less_income/high)
print("conditional probabilities of females under 50K ", f_less_income/less)


# # 4d) From the model extract the estimated class conditional mean and standard deviation of hoursPerWeek feature for people with income over 50K and people with income under 50K per year. Also draw boxplots for this feature for each income class. (You may wish to examine this information for all numerical and categorical features to get a feel of each one’s relation with income level.)

# In[247]:


means = sample_adult.groupby('income')['hoursPerWeek'].mean()
means

print("conditional mean for hoursPerWeek under 50K ", means[0])
print("conditional mean for hoursPerWeek over 50K ", means[1])
# o2.groupby(['YEAR', 'daytype', 'hourtype'])['option_value'].mean()


# In[248]:


sds = sample_adult.groupby('income')['hoursPerWeek'].std()
sds

print("conditional standard deviation for hoursPerWeek under 50K ", sds[0])
print("conditional standard deviation for hoursPerWeek under 50K ", sds[1])


# #### Box plot 

# In[253]:


sns.boxplot(x='hoursPerWeek', y='income', data=adultTrain)


# # 4e) Form the UCI archive read the test data from this link and save it in the data frame adultTest.If you examine the original file, you will see that the first few lines are comments, and start with the character “|”. Make sure to set the comments character while reading the data, so these lines are skipped.

# In[255]:


adultTest = pd.read_csv("C:/Semester 2/Algorithmic machine learning/Homework 2/adult.test",na_values= ' ?', comment='|', names=["age", "workclass", "fnlwgt", "education", "educationNum", "maritalStatus", "occupation",
"relationship", "race", "sex", "capitalGain", "capitalLoss", "hoursPerWeek", "nativeCountry",
"income"])
adultTest.dropna(axis=0,inplace=True)
adultTest['income'] = adultTest['income'].map( lambda x: str(x)[:-1])
adultTest.head()


# #### preprocessing of categorical data

# In[256]:


adultTest.loc[:,"income"] =le.fit_transform(adultTest["income"])
adultTest.loc[:,"workclass"] =le.fit_transform(adultTest["workclass"])
adultTest.loc[:,"education"] =le.fit_transform(adultTest["education"])
adultTest.loc[:,"maritalStatus"] =le.fit_transform(adultTest["maritalStatus"])
adultTest.loc[:,"occupation"] =le.fit_transform(adultTest["occupation"])
adultTest.loc[:,"relationship"] =le.fit_transform(adultTest["relationship"])
adultTest.loc[:,"race"] =le.fit_transform(adultTest["race"])
adultTest.loc[:,"sex"] =le.fit_transform(adultTest["sex"])
adultTest.loc[:,"nativeCountry"] =le.fit_transform(adultTest["nativeCountry"])


# In[257]:


naivePredA = nb_classifier.predict(adultTest.loc[:, adultTest.columns != 'income'])
naivePredA


# #### confusion matrix and the error rate for the test data

# In[270]:


from sklearn import metrics
confusion = metrics.confusion_matrix (adultTest['income'], naivePredA)
print(confusion)

error_naive = (confusion[0,1]+confusion[1,0])/sum(sum(confusion))

print("Error rate", error_naive)


# # 4f) Repeat parts 4b)-4e) but this time assume all numerical variables follow the normal distribution for numerical features. How does this change affect the test error rate?

# In[269]:


from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics


naiveNormal = GaussianNB()
naiveNormal.fit(sample_adult.loc[:, sample_adult.columns != 'income'], sample_adult.loc[:,"income"])


naiveNormal_pred = naiveNormal.predict(adultTest.loc[:, adultTest.columns != 'income'])

confusion = metrics.confusion_matrix (adultTest['income'], naiveNormal_pred)
print(confusion)

error = (confusion[0,1]+confusion[1,0])/sum(sum(confusion))

print("Error rate", error)


# In[261]:


less_income = sample_adult["income"] == 0
high_income = sample_adult["income"] == 1

female = sample_adult["sex"] == 2
male = sample_adult["sex"] == 1

f_less_income = len(sample_adult[less_income & female])
f_highs_income = len(sample_adult[high_income & female])
m_highs_income = len(sample_adult[high_income & male])
m_less_income = len(sample_adult[less_income & male])
less = len(less_income)
high = len(high_income)
print("conditional probabilities of males under 50K ", m_less_income/less)
print("conditional probabilities of males over 50K ", m_less_income/high)
print("conditional probabilities of females over 50K ", f_less_income/high)
print("conditional probabilities of females under 50K ", f_less_income/less)

means = sample_adult.groupby('income')['hoursPerWeek'].mean()
means

print("conditional mean for hoursPerWeek under 50K ", means[0])
print("conditional mean for hoursPerWeek over 50K ", means[1])

sds = sample_adult.groupby('income')['hoursPerWeek'].std()
sds

print("conditional standard deviation for hoursPerWeek under 50K ", sds[0])
print("conditional standard deviation for hoursPerWeek under 50K ", sds[1])

sns.boxplot(x='hoursPerWeek', y='income', data=adultTrain)


# # 4g) Now load the tree library in R or the analogeous library in Python and repeat parts 4b)-4e) for the default tree model. Also, print the resulting tree. Compare the error rate with the naive Bayes models.

# In[225]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=100, max_depth=3)
treeModel = tree.fit(sample_adult.loc[:, sample_adult.columns != 'income'], sample_adult.loc[:,"income"])
treeModel


# In[271]:


treePredA= treeModel.predict(adultTest.loc[:, adultTest.columns != 'income'])

confusion = metrics.confusion_matrix (adultTest['income'], treePredA)
print(confusion)

error_tree = (confusion[0,1]+confusion[1,0])/sum(sum(confusion))

print("Error rate", error_tree)


# In[274]:


print("Naive bayes error rate is ",error_naive," Decision Tree error rate is ",error_tree )


# #### Printing decision tree

# In[275]:


# Printing the Decision Tree 
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 
features = list(adultTrain.columns[:-1])
print(features)

dot_data = StringIO()  
export_graphviz(tree, out_file=dot_data,feature_names= features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())


# # 4h) Next apply the cost complexity pruning and find the tree it produces and name it bestTree. Plot the best tree, and also print its confusion matrix and error rate on the test data.
# 

# In[277]:


# Pruning
from sklearn.tree._tree import TREE_LEAF

def is_leaf(inner_tree, index):
    # Check whether node is leaf node
    return (inner_tree.children_left[index] == TREE_LEAF and 
            inner_tree.children_right[index] == TREE_LEAF)

def prune_index(inner_tree, decisions, index=0):
    # Start pruning from the bottom - if we start from the top, we might miss nodes that become leaves during pruning.
    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_index(inner_tree, decisions, inner_tree.children_left[index])
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_index(inner_tree, decisions, inner_tree.children_right[index])

    # Prune children if both children are leaves now and make the same decision:     
    if (is_leaf(inner_tree, inner_tree.children_left[index]) and
        is_leaf(inner_tree, inner_tree.children_right[index]) and
        (decisions[index] == decisions[inner_tree.children_left[index]]) and 
        (decisions[index] == decisions[inner_tree.children_right[index]])):
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF

def prune_duplicate_leaves(mdl):
    # Remove leaves if both 
    decisions = mdl.tree_.value.argmax(axis=2).flatten().tolist() # Decision for each node
    prune_index(mdl.tree_, decisions)


# In[279]:


#Using this on the DecisionTreeClassifier 'tree':
prune_duplicate_leaves(tree)


# In[281]:


tree


# In[283]:


# Tree After Pruning
dot_data = StringIO()  
export_graphviz(tree, out_file=dot_data,feature_names= features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())


# In[289]:


# Predicting the values of testdata again
treePred_prun = tree.predict(adultTest.loc[:, adultTest.columns != 'income'])
treePred_prun


# In[291]:



confusion = metrics.confusion_matrix (adultTest['income'], treePred_prun)
print(confusion)

error_naive = (confusion[0,1]+confusion[1,0])/sum(sum(confusion))

print("Error rate", error_naive)


# # 4i) Finally, develop a random forest model where at each partition only a random subset of features of size m = √d is used, where d is the number of features. Print the confusion matrix and the error rate for the test data.

# In[295]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_features='auto') # If “auto”, then max_features = sqrt(n_features)
rfc.fit(sample_adult.loc[:, sample_adult.columns != 'income'], sample_adult.loc[:,"income"])



rfcPred= rfc.predict(adultTest.loc[:, adultTest.columns != 'income'])
rfcPred


# #### Print the confusion matrix and the error rate for the test data. 

# In[297]:


confusion = metrics.confusion_matrix (adultTest['income'], rfcPred)
print(confusion)

error_naive = (confusion[0,1]+confusion[1,0])/sum(sum(confusion))

print("Error rate", error_naive)


# #### Importance of Features

# In[299]:


print(rfc.feature_importances_)


# In[302]:


import matplotlib.pyplot as plt
importances = rfc.feature_importances_
indices = np.argsort(importances)

# Plotting bar graph in order of importance 
plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')


# In[310]:


col = list(adultTest.columns)
col = col[:-1]


# In[311]:


feat_importances = pd.Series(importances, index=col)
feat_importances.plot()
plt.show()

