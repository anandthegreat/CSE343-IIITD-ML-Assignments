#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(fname= "Dataset.data")

def normalizer(data):
    sdeviations = np.std(data,axis = 0)
    means = np.mean(data, axis=0)
    return (data - means)/sdeviations

# data = normalizer(data)


# In[2]:


def optimalParameter(data):
    X = data[:,:-1]
    Y = data[:,np.size(data,1)-1]
    X = np.insert(X,0,1,1)                    #Insert a column of ones (X0 feature) 
    thetaVector = np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T, Y) )
    return thetaVector


# In[3]:


def CostCalculator(data, testData):           #fold-wise CostCalculator
    numFeatures     = np.size(data,1)-1       #n, same for both test and train data
    numTrainingRows = np.size(data,0)         #m
    numTestRows     = np.size(testData,0)     #m for test data
    thetaVector     = optimalParameter(data)  #theta vector is calculated for train data
    print("ThetaVector: " ,thetaVector)
    
    costsumTrain    = 0.0
    costsumTest     = 0.0 
    #calculate RMSE for training data 
    for i in range(numTrainingRows):      
        featureVector = np.array(data[i,:-1])
        featureVector = np.insert(featureVector,0,1)
        costsumTrain += np.square(np.dot(thetaVector.T, featureVector) - data[i][numFeatures])
    costsumTrain = costsumTrain/(2*numTrainingRows)
    costsumTrain = costsumTrain**0.5
    print("RMSE Training: ",costsumTrain)
    
    #computer RMSE for test data
    for i in range(numTestRows):      
        featureVector2 = np.array(testData[i,:-1])
        featureVector2 = np.insert(featureVector2,0,1)
        costsumTest   += np.square(np.dot(thetaVector.T, featureVector2) - testData[i][numFeatures])
    costsumTest = costsumTest/(2*numTestRows)
    costsumTest = costsumTest**0.5
    print("RMSE Testing: ",costsumTest, "\n")
    


# In[4]:


def folds(data):                                  #split the data and call CostCalculator for each split/fold
    numFeatures     = np.size(data,1)-1           #n
    numTrainingRows = np.size(data,0)             #m
    vsetSize        = int(0.2 * numTrainingRows)  #validation set size (=1/5th size of data)
    concMatrix      = np.zeros(vsetSize)
    for x in range(5):
        newTestData      = data[vsetSize*(x):vsetSize*(x+1),:]
        if(x==0):
            concMatrix   = newTestData
        if(x!=4):
            newTrainData = data[vsetSize*(x+1):,:]
        else:
            newTrainData = data[:vsetSize*x,:] 
        if(x!=0 and x!=4):
            newTrainData = np.concatenate((concMatrix, newTrainData),axis = 0)
        if(x!=0 and x!=4):
            concMatrix   = np.concatenate((concMatrix, newTestData) , axis=0)
        
        print("Fold [", x+1, "]")
        CostCalculator(newTrainData, newTestData)


# In[5]:


folds(data)


# In[ ]:




