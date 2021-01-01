#!/usr/bin/env python
# coding: utf-8

# In[95]:


from random import seed
from random import randrange
from csv import reader
from math import sqrt
from math import pow
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np


# In[ ]:


1. For the IRIS dataset, prepare a training dataset and a testing dataset for classification model training and testing. For each class, take the first 40 samples into the training dataset, and the remaining 10 samples into the testing dataset.
2. Make a KNN classifier using the Minkowski distance function you made in HW2. The KNN function performs classification based on the majority voting of K-nearest neighbors. Implement the KNN classifiers to the IRIS dataset using K = 3, 5, 7 for K-nearest neighbors, and r = 1, 2, 4 for the distance order of Minkowski Distance. For each parameter setting of K and r, perform classification for the testing data samples you prepared in problem 1. 1) For each KNN parameter setting, report classification accuracy and the confusion matrix. 2) Calculate and report the classification accuracy for each class at each parameter setting. 3) Assume we use the average accuracy of each class as the overall model performance measure, find the best parameter setting that generates the highest average accuracy for the 3 classes.


# In[96]:


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# In[97]:


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# In[98]:


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# In[99]:


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# In[100]:


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# In[101]:


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# In[102]:


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# In[103]:


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    confusionMatrix(actual,predicted)
    return scores


# In[104]:


# Calculate the minkowski distance between two vectors


def minkowski(row1, row2, r):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += pow((row1[i] - row2[i]),r)
    distance = pow(distance,1/r)   
    return round(distance,3)



# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = minkowski(test_row, train_row,4) #### R --> value <-- 1, 2 or 4
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# In[105]:


# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return(predictions)


# In[106]:


# Making of consfusion matrix
def confusionMatrix(Y_test,predictions):
    arr = []
    print("\nConfusion Matrix:\n\n",confusion_matrix(Y_test,predictions))
    #arr.append(confusion_matrix(Y_test,predictions))


# #### We are calling the string to float & string to integer methods to convert pandas dataframe into a numpy array 
# #### n_fold --> 10 <-- will help you segregrate the data with a propotion of 10.
# #### hence, after a segretation of 1st 50 dataset:
# #### the training set will be 40 & the testing set will be 10

# In[107]:


seed(1)
filename = 'hw2_iris.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
# convert class column to integers
print("We are building a linear preprocessing from scratch:")
print("Which will convert the class from string to numericals as:")
str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 10
num_neighbors = 5
import numpy as np
np.random.shuffle(dataset)


# #### First 50 datasets:

# 

# ### R = 1

# In[108]:


hey = 0
count = 0
hello = 50
buru = 0
kadamba = 0
num_neighbors = 3
n_folds = 10
lenusama  = len(dataset)
while lenusama > 0:
    pulima = 1
    temp = dataset[hey:hello]
    hey += 50
    hello += 50
    scores = evaluate_algorithm(temp, k_nearest_neighbors, n_folds, num_neighbors)
    kadamba = sum(scores)/float(len(scores))
    buru += kadamba
    lenusama -=50
    count+=1
    pulima+=1
print(" Accuracy for k = 3: ",buru/count)    


# In[109]:


hey = 0
count = 0
hello = 50
buru = 0
kadamba = 0
num_neighbors = 5
n_folds = 10
lenusama  = len(dataset)
while lenusama > 0:
    pulima = 1
    temp = dataset[hey:hello]
    hey += 50
    hello += 50
    scores = evaluate_algorithm(temp, k_nearest_neighbors, n_folds, num_neighbors)
    kadamba = sum(scores)/float(len(scores))
    buru += kadamba
    lenusama -=50
    count+=1
    pulima+=1
print(" Accuracy for k = 5: ",buru/count)    


# In[110]:


hey = 0
count = 0
hello = 50
buru = 0
kadamba = 0
num_neighbors = 7
n_folds = 10
lenusama  = len(dataset)
while lenusama > 0:
    pulima = 1
    temp = dataset[hey:hello]
    hey += 50
    hello += 50
    scores = evaluate_algorithm(temp, k_nearest_neighbors, n_folds, num_neighbors)
    kadamba = sum(scores)/float(len(scores))
    buru += kadamba
    lenusama -=50
    count+=1
    pulima+=1
print(" Accuracy for k = 7: ",buru/count)    


# ### for R = 2:

# In[127]:


hey = 0
count = 0
hello = 50
buru = 0
kadamba = 0
num_neighbors = 3
n_folds = 10
lenusama  = len(dataset)
while lenusama > 0:
    pulima = 1
    temp = dataset[hey:hello]
    hey += 50
    hello += 50
    scores = evaluate_algorithm(temp, k_nearest_neighbors, n_folds, num_neighbors)
    kadamba = sum(scores)/float(len(scores))
    buru += kadamba
    lenusama -=50
    count+=1
    pulima+=1
print(" Accuracy for k = 3: ",buru/count)    


# In[128]:


hey = 0
count = 0
hello = 50
buru = 0
kadamba = 0
num_neighbors = 5
n_folds = 10
lenusama  = len(dataset)
while lenusama > 0:
    pulima = 1
    temp = dataset[hey:hello]
    hey += 50
    hello += 50
    scores = evaluate_algorithm(temp, k_nearest_neighbors, n_folds, num_neighbors)
    kadamba = sum(scores)/float(len(scores))
    buru += kadamba
    lenusama -=50
    count+=1
    pulima+=1
print(" Accuracy for k = 5: ",buru/count)    


# In[129]:


hey = 0
count = 0
hello = 50
buru = 0
kadamba = 0
num_neighbors = 7
n_folds = 10
lenusama  = len(dataset)
while lenusama > 0:
    pulima = 1
    temp = dataset[hey:hello]
    hey += 50
    hello += 50
    scores = evaluate_algorithm(temp, k_nearest_neighbors, n_folds, num_neighbors)
    kadamba = sum(scores)/float(len(scores))
    buru += kadamba
    lenusama -=50
    count+=1
    pulima+=1
print(" Accuracy for k = 7: ",buru/count)    


# ### R = 4

# In[114]:


hey = 0
count = 0
hello = 50
buru = 0
kadamba = 0
num_neighbors = 3
n_folds = 10
lenusama  = len(dataset)
while lenusama > 0:
    pulima = 1
    temp = dataset[hey:hello]
    hey += 50
    hello += 50
    scores = evaluate_algorithm(temp, k_nearest_neighbors, n_folds, num_neighbors)
    kadamba = sum(scores)/float(len(scores))
    buru += kadamba
    lenusama -=50
    count+=1
    pulima+=1
print(" Accuracy for k = 3: ",buru/count)    


# In[115]:


hey = 0
count = 0
hello = 50
buru = 0
kadamba = 0
num_neighbors = 5
n_folds = 10
lenusama  = len(dataset)
while lenusama > 0:
    pulima = 1
    temp = dataset[hey:hello]
    hey += 50
    hello += 50
    scores = evaluate_algorithm(temp, k_nearest_neighbors, n_folds, num_neighbors)
    kadamba = sum(scores)/float(len(scores))
    buru += kadamba
    lenusama -=50
    count+=1
    pulima+=1
print(" Accuracy for k = 5: ",buru/count)    


# In[116]:


hey = 0
count = 0
hello = 50
buru = 0
kadamba = 0
num_neighbors = 7
n_folds = 10
lenusama  = len(dataset)
while lenusama > 0:
    pulima = 1
    temp = dataset[hey:hello]
    hey += 50
    hello += 50
    scores = evaluate_algorithm(temp, k_nearest_neighbors, n_folds, num_neighbors)
    kadamba = sum(scores)/float(len(scores))
    buru += kadamba
    lenusama -=50
    count+=1
    pulima+=1
print(" Accuracy for k = 7: ",buru/count)    


# 3) As shown in the plot below, a simple decision tree is constructed to classify two iris flowers: Versicolor and Virginica using two features of petal width and petal length. Assume the binary decision boundary on Petal Length is 4.8, and the decision boundary on Petal Width is 1.7. Make a function to implement this simple decision tree and use your function to classify the 100 iris samples of Versicolor and Virginica. Report the classification accuracy, sensitivity, and specificity. Here we define sensitivity = accuracy for class Versicolor, and specificity = accuracy of class Virginica.

# In[130]:



data = pd.read_csv("iris_org.csv", header = None)
testdata=data.iloc[50:150]
vdata=testdata.to_numpy()

def whatflower(plength,pwidth):
    if plength>4.8:
        return 2
    else:
        if pwidth>1.7:
            return 2
        else:
            return 1
dec=np.zeros((100), dtype=int)
for i in range(100):
    dec[i]=whatflower(vdata[i,2],vdata[i,3])
count=0
for i in range(100):
    if dec[i]==vdata[i,4]:
        count+=1
acscore=(count/100)*100
count=0
for i in range(100):
    if dec[i]==vdata[i,4]:
        count+=1
sscore=(count/110)*100
count=0
for i in range(50):
    j=i+50
    if dec[j]==vdata[j,4]:
        count+=1
spscore=(count/50)*100
print("Accuracy : ",acscore,"%" )
print("Sensitivity : ",sscore,"%")
print("Specificity : ",spscore,"%")


# In[ ]:




