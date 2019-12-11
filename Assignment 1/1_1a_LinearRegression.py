#!/usr/bin/env python
# coding: utf-8

# In[100]:


import time
import numpy as np
import matplotlib.pyplot as plt
start_time = time.time()
data = np.loadtxt(fname= "dataset.data")

def normalizer(data):
    sdeviations = np.std(data,axis = 0)
    means       = np.mean(data, axis=0)
    return (data - means)/sdeviations

# data = normalizer(data)


# In[101]:


def GradientDescent(data, testData, noOfIterations):           
    numFeatures     = np.size(data,1)-1           #n
    numTrainingRows = np.size(data,0)             #m
    numTestRows     = np.size(testData,0)         #m for test data
    thetavector     = np.ones(numFeatures+1)      #thetavector initialized with ones
    alpha           = 0.1                         #learning rate
    temp            = np.zeros(numFeatures+1)     #for saving temp values of theta(s) and simultaneous updation
    RMSEtrainItr    = np.zeros(noOfIterations)    #for saving error value after each iteration for training set
    RMSEtestItr     = np.zeros(noOfIterations)    #for saving error value after each iteration for test set
    
    for x in range(noOfIterations):               #number of iterations for gradient descent
        X0 = np.ones((numTrainingRows,1))
        X  = np.insert(data, 0, 1, axis = 1)
        X  = X[:,:-1]
        Y  = data[:, numFeatures]
        hypVector  = np.dot(X,thetavector)      #hypvector is hypothesis vector from i=1 to m
        difference = hypVector - Y
        gradient   = np.dot(X.T,difference)     #d/d(theta)(J(theta))
        gradient   = gradient/numTrainingRows
        thetavector = (thetavector - (alpha*gradient))
        
        #calculate cost function for training set
        costsum = 0.0
        for i in range(numTrainingRows):      
            costsum+= (difference[i]**2)
        costsum = costsum/(2*numTrainingRows)
        RMSEtrainItr[x] = np.sqrt(costsum)
#        print("Training ",costsum)
        
        Xtest = np.insert(testData, 0, 1, axis = 1)
        Xtest = Xtest[:,:-1]
        Ytest = testData[:,numFeatures]
        hypVectorTest = np.dot(Xtest, thetavector)
        differenceTest = hypVectorTest - Ytest
        
        #calculate cost function for test set
        costsum = 0.0
        for i in range(numTestRows):      
            costsum+= (differenceTest[i]**2)
        costsum = costsum/(2*numTestRows)
        RMSEtestItr[x] = np.sqrt(costsum)

    return RMSEtrainItr, RMSEtestItr 


# In[102]:


def plot(meanRMSEtrain, meanRMSEtest, numberOfIterations):
    xaxis = np.arange(1,numberOfIterations+1)    
    plt.plot(xaxis,meanRMSEtrain)
    plt.plot(xaxis,meanRMSEtest)
    plt.xlabel('#iterations')
    plt.ylabel('mean RMSE over 5 folds)')
    plt.legend(['mean Training Set RMSE', 'mean Testing Set RMSE'], loc='upper right')
    plt.show()
    


# In[103]:


def folds(data, noOfIterations):                  #split the data and call CostCalculator for each split/fold
    numFeatures     = np.size(data,1)-1           #n
    numTrainingRows = np.size(data,0)             #m
    vsetSize        = int(0.2 * numTrainingRows)  #validation set size (=1/5th size of data)
    concMatrix      = np.zeros(vsetSize)
    RMSEtrainItr    = np.zeros(noOfIterations)    #for saving error value after each iteration for training set
    RMSEtestItr     = np.zeros(noOfIterations)    #for saving error value after each iteration for test set
    
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
        RMSEtrainCatch, RMSEtestCatch = GradientDescent(newTrainData, newTestData, noOfIterations)
        RMSEtrainItr += 0.2*RMSEtrainCatch
        RMSEtestItr  += 0.2*RMSEtestCatch
        print("Training Error: ", RMSEtrainCatch[noOfIterations-1])
        print("Testing Error : ", RMSEtestCatch[noOfIterations-1])
        
    print("Mean Training Error: ", RMSEtrainItr[noOfIterations-1])
    print("Mean Testing Error : ", RMSEtestItr[noOfIterations-1])
    plot(RMSEtrainItr,RMSEtestItr, noOfIterations)


# In[104]:


folds(data, 1000)
print("--- Total time: %s seconds ---" % (time.time() - start_time))


# In[ ]:





# In[ ]:





# In[ ]:




