#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings(action="ignore",category=DeprecationWarning)

data = np.genfromtxt('data_brainXweight.csv', delimiter=',')
data = data[1:,:]                   #remove title row
brainwts = data[:,0]
bodywts  = data[:,1]


# In[2]:


# def optimalParameter(data):
#     X = data[:,:-1]
#     Y = data[:,np.size(data,1)-1]
#     X = np.insert(X,0,1,1)                    #Insert a column of ones (X0 feature) 
#     thetaVector = np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T, Y) )
#     return thetaVector
X = data[:,0]
Y = data[:,1]
X0= np.ones(len(brainwts))
X = np.c_[X0,X]                    #Insert a column of ones (X0 feature)
# print(Y)


# In[3]:


L1 = Lasso(max_iter = 10000)
L2 = Ridge(max_iter = 10000)

parameters     = {'alpha': [100 , 200, 500, 1000, 2000, 5000 ]}
ridgeRegressor = GridSearchCV(L2, parameters, scoring = 'neg_mean_squared_error', cv=5)
ridgeRegressor.fit(X,Y)

lambdaL2 =(ridgeRegressor.best_params_.get('alpha'))      #optimal regularization parameter for L2

lassoRegressor = GridSearchCV(L1, parameters, scoring = 'neg_mean_squared_error', cv=5)
lassoRegressor.fit(X,Y)
lambdaL1 = (lassoRegressor.best_params_.get('alpha'))     #optimal regularization parameter for L1

print(lambdaL1, lambdaL2)
# print("L1 Lambda %f L2 Lambda %f" % (lambdaL1, lambdaL2))


# In[4]:


def GradientDescent(data,noOfIterations,regType,regParam): #regType = L1/L2 or 0 for none,regParam=lambda           
    numFeatures     = np.size(data,1)-1           #n
    numTrainingRows = np.size(data,0)             #m
    thetavector     = np.ones(numFeatures+1)      #thetavector initialized with ones
    alpha           = 0.0001                         #learning rate
    temp            = np.zeros(numFeatures+1)     #for saving temp values of theta(s) and simultaneous updation
    RMSEtrainItr    = np.zeros(noOfIterations)
    for x in range(noOfIterations):               #number of iterations for gradient descent
        for i in range(numFeatures+1):            #for every parameter theta, or jth theta
            sum = 0.0
            for j in range(numTrainingRows):
                featureVector = np.array(data[j,:-1])
                featureVector = np.insert(featureVector,0,1)     #X0=1 (dummy feature)
                sum += ((np.dot(thetavector.T,featureVector) - data[j][numFeatures])*featureVector[i]) 
            if(regType == 0):                     #regression not used
                temp[i] = thetavector[i] - (alpha*sum)/numTrainingRows
            elif(regType == 1):                   #L1 Regression
                temp[i] = thetavector[i] - (alpha*(sum + regParam*np.sign(thetavector[i])))/numTrainingRows
            elif(regType == 2):                   #L2 Regression
                temp[i] = thetavector[i]*(1-alpha*(regParam/numTrainingRows)) - (alpha*sum)/numTrainingRows

        #simultaneously update thetas
        for i in range(numFeatures+1):        
            thetavector[i]= temp[i]
            
        #calculate cost function for training set
        costsum = 0.0
        for i in range(numTrainingRows):      
            featureVector = np.array(data[i,:-1])
            featureVector = np.insert(featureVector,0,1)
            costsum+= np.square(np.dot(thetavector.T, featureVector) - data[i][numFeatures])
        costsum = costsum/(2*numTrainingRows)
        RMSEtrainItr[x] = costsum
#        print("Training ",costsum)

    RMSEtrainItr = np.sqrt(RMSEtrainItr) 
    print("RMSE is: ",RMSEtrainItr[noOfIterations-1])
    return thetavector


# In[5]:


bestFits   = []
bestFitsL1 = []
bestFitsL2 = []

print("Without Regularization")
thetaV   = GradientDescent(data,100, 0, -1)
print("With L1 Regularization")
thetaVL1 = GradientDescent(data,100, 1, lambdaL1)
print("With L2 Regularization")
thetaVL2 = GradientDescent(data,100, 2, lambdaL2)
print("Without Reg Paramters: ",thetaV)
print("L1 Regularization Parameters: ",thetaVL1)
print("L2 Regularization Parameters: ",thetaVL2)

for x in brainwts:
    bestFits.append(thetaV[1]*x + thetaV[0])
    
for x in brainwts:
    bestFitsL1.append(thetaVL1[1]*x + thetaVL1[0])
    
for x in brainwts:
    bestFitsL2.append(thetaVL2[1]*x + thetaVL2[0])
    
plt.scatter(brainwts, bodywts, color='g', marker='.')
plt.plot(brainwts, bestFits, color='r')
plt.title("Best Fit Without Regularization")
plt.xlabel('Brain Weight')
plt.ylabel('Body Weight')
plt.show()

plt.scatter(brainwts, bodywts, color='g', marker='.')
plt.plot(brainwts, bestFitsL1, color='b')
plt.title("Best Fit With L1 Regularization")
plt.xlabel('Brain Weight')
plt.ylabel('Body Weight')
plt.show()

plt.scatter(brainwts, bodywts, color='g', marker='.')
plt.plot(brainwts, bestFitsL2, color='k')
plt.title("Best Fit With L2 Regularization")
plt.xlabel('Brain Weight')
plt.ylabel('Body Weight')
plt.show()


# In[ ]:





# In[ ]:




