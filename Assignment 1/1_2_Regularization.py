#!/usr/bin/env python
# coding: utf-8

# In[6]:


import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings(action="ignore",category=DeprecationWarning)
start_time = time.time()

data = np.loadtxt(fname= "Dataset.data")
numFeatures     = np.size(data,1)-1              #n
numTrainingRows = np.size(data,0)                #m
vsetSize        = int(0.2 * numTrainingRows)     #validation set size (=1/5th size of data)

concMatrix      = data[:vsetSize,:]
testData        = data[vsetSize: 2*vsetSize, :]  #holded testset
newTrainValData = data[2*vsetSize:, :]           #remaining 80% train+val data
newTrainValData = np.concatenate((concMatrix, newTrainValData), axis = 0)

X = newTrainValData[:,:-1]
Y = newTrainValData[:,np.size(newTrainValData,1)-1]
X = np.insert(X,0,1,1)                    #Insert a column of ones (X0 feature)

# X = data[:,:-1]
# Y = data[:,np.size(data,1)-1]
# X = np.insert(X,0,1,1)                    #Insert a column of ones (X0 feature) 


# In[7]:


L1 = Lasso(max_iter = 10000)
L2 = Ridge(max_iter = 10000)

parameters     = {'alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1 ,1, 5, 10, 20 ]}
ridgeRegressor = GridSearchCV(L2, parameters, scoring = 'neg_mean_squared_error', cv=5)
ridgeRegressor.fit(X,Y)

lambdaL2 =(ridgeRegressor.best_params_.get('alpha'))      #optimal regularization parameter for L2

lassoRegressor = GridSearchCV(L1, parameters, scoring = 'neg_mean_squared_error', cv=5)
lassoRegressor.fit(X,Y)
lambdaL1 = (lassoRegressor.best_params_.get('alpha'))     #optimal regularization parameter for L1

print("Lambda1, Lambda2 ", lambdaL1, lambdaL2)


# In[8]:


def GradientDescent(data, noOfIterations, regType, regParam):  #regType = L1 or L2, regParam = lambda           
    numFeatures     = np.size(data,1)-1           #n
    numTrainingRows = np.size(data,0)             #m
    thetavector     = np.ones(numFeatures+1)      #thetavector initialized with ones
    alpha           = 0.1                         #learning rate
    temp            = np.zeros(numFeatures+1)     #for saving temp values of theta(s) and simultaneous updation
    RMSEtrainItr    = np.zeros(noOfIterations)    #for saving error value after each iteration for training set
    
    for x in range(noOfIterations):               #number of iterations for gradient descent
        X0 = np.ones((numTrainingRows,1))
        X  = np.insert(data, 0, 1, axis = 1)
        X  = X[:,:-1]
        Y  = data[:, numFeatures]
        hypVector  = np.dot(X,thetavector)      #hypvector is hypothesis vector from i=1 to m
        difference = hypVector - Y
        gradient   = np.dot(X.T,difference)     #d/d(theta)(J(theta))
        gradient   = gradient/numTrainingRows
       
        if(regType == 1):      #L1 regression
            thetavector = (thetavector - (alpha* (gradient + (regParam*np.sign(thetavector))/numTrainingRows)))
        elif(regType == 2):    #L2 regression  #(1-alpha*(lambda/m))
            thetavector = thetavector*(1-alpha*(regParam/numTrainingRows)) - alpha*gradient
            
        #calculate cost function for training set
        costsum = 0.0
        for i in range(numTrainingRows):      
            costsum+= (difference[i]**2)
        costsum = costsum/(2*numTrainingRows)
        RMSEtrainItr[x] = costsum
        
    RMSEtrainItr = np.sqrt(RMSEtrainItr)
    return RMSEtrainItr 


# In[9]:


def plot(meanRMSEtrain, title, numberOfIterations):
    xaxis = np.arange(1,numberOfIterations+1)    
    plt.plot(xaxis,meanRMSEtrain)
    plt.title(title)
    plt.xlabel('#iterations')
    plt.ylabel('mean RMSE')
    plt.show()


# In[10]:


print("L1 Regularization running ...")
RMSETrainItrL1 = GradientDescent(newTrainValData, 1000, 1, lambdaL1)      #for L1 on train+val set
print("L2 Regularization running ...")
RMSETrainItrL2 = GradientDescent(newTrainValData, 1000, 2, lambdaL2)      #for L2 on train+val set
print("RMSE L1 ", RMSETrainItrL1[999])
print("RMSE L2 ", RMSETrainItrL2[999])
plot(RMSETrainItrL1, "L1 regularization RMSE vs Iterations", 1000)
plot(RMSETrainItrL2, "L2 regularization RMSE vs Iterations", 1000)

print("L1 Regularization running on test data ...")
RMSEtestItrL1 = GradientDescent(testData,1000, 1, lambdaL1)
print("L2 Regularization running on test data ...")
RMSEtestItrL2 = GradientDescent(testData,1000, 2, lambdaL2)
print("\nRMSE Test using L1: ", RMSEtestItrL1[999])          #RMSE on test data using L1
print("RMSE Test using L2: ", RMSEtestItrL2[999])          #RMSE on test data using L2

print("--- Total time: %s seconds ---" % (time.time() - start_time))


# In[ ]:




