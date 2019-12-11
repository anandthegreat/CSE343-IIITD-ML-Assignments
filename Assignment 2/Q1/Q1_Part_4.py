#!/usr/bin/env python
# coding: utf-8

# In[2]:


import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm
from numpy import mean
from numpy import std
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings(action="ignore",category=DeprecationWarning)
warnings.filterwarnings(action="ignore",category=FutureWarning)


# In[3]:


def h5opener(filename):
    with h5py.File(filename,'r') as f:
#         print("Keys: %s" % f.keys())
        X = f['x'][:]
        Y = f['y'][:]
    return X,Y

def plotUtil(filename, outlier_remove = 'no'):
    if(outlier_remove == 'yes'):
        X,Y = outlier_removal(filename)
    else:
        X,Y = h5opener(filename)

    numClasses = len(list(set(Y)))
    colors = ['red', 'green', 'blue']
    for i in range(numClasses):
        plt.scatter(X[Y==i,0],X[Y==i,1],c = colors[i], label = ('Class '+ str(i)), edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc = 'upper right' )
    plt.title(filename)
    plt.show()
    print('\n')
    
def outlier_removal(filename):
    X,Y = h5opener(filename)
    X1 = X[:,0]   #feature 1
    X2 = X[:,1]   #feature 2
#     print('Original No of Examples: ', len(X))
    
    meanX1 = mean(X1)
    sdX1   = std(X1)
    meanX2 = mean(X2)
    sdX2   = std(X2)
    
    X1temp = []
    X2temp = []
    Ytemp  = []

    cur_idx = 0
    for x in X1:
        if( (x > (meanX1 - 2*sdX1)) and (x < (meanX1 + 2*sdX1))):
            X1temp.append(x)
            X2temp.append(X2[cur_idx])
            Ytemp.append(Y[cur_idx])
        cur_idx += 1
    X1 = []
    X2 = []
    Y  = []
    cur_idx = 0
    for x in X2temp:
        if (x > (meanX2 - 3*sdX2)) and (x < (meanX2 + 3*sdX2)):
            X2.append(x)
            X1.append(X1temp[cur_idx])
            Y.append(Ytemp[cur_idx])
        cur_idx += 1
    
    X = np.vstack((X1,X2)).T
    Y = np.asarray(Y)
#     print('No of Examples after removing outliers: ', len(X))
    
    return X,Y


# In[4]:


def kernel_predict(X,Y, kernel_type, c):  
    
    xmin,xmax = X[:,0].min()-1, X[:,0].max()+1
    ymin,ymax = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(xmin,xmax,0.02), np.arange(ymin,ymax,0.02))
    if(kernel_type=='linear'):
        svm_model = svm.SVC(kernel= kernel_type,C = c).fit(X,Y)
    elif(kernel_type=='rbf'):
        svm_model = svm.SVC(kernel = kernel_type, gamma = 'auto',C = c).fit(X,Y)
    else: print('Kernel type not supported')
        
#     print(svm_model.support_vectors_)
    
    Z = svm_model.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z, cmap = plt.cm.coolwarm, alpha = 0.8)
    
    plt.scatter(X[:,0],X[:,1], c=Y, cmap=plt.cm.coolwarm, s=25,edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
#     plt.title('Dataset: '+ filename + '   ' + 'Kernel: '+kernel_type)
    plt.show()
    print('\n')
    


# In[22]:


def rbf_own(X,Y):
    rbf = np.zeros((X.shape[0],Y.shape[0]))
    for i,x in enumerate(X):
        for j,y in enumerate(Y):
            rbf[i,j] = np.exp((-1*np.linalg.norm(x-y)**2))   #for data4, use gamma = 9, 1 o/w
    return rbf


# In[23]:


def predict_own(kernel,test_data, dual_coef_,support_vectors_):
    w = np.zeros(len(test_data))
    rbf_kernel = rbf_own(support_vectors_, test_data)
    for j in range(len(test_data)):
        for i in range(0,len(support_vectors_)):
            if(kernel =='linear'):
                w[j]+= ((np.dot(support_vectors_[i],test_data[j]))*dual_coef_[0,i])
            elif(kernel == 'rbf'):
                w[j]+= (rbf_kernel[i,j]*dual_coef_[0,i])
            else:
                print("Invalid Kernel Type: " + kernel)
                
    for j in range(len(test_data)):
        if(w[j] < 0):
            w[j] = 0
        else:
            w[j] = 1
            
    return w


# In[24]:


if __name__ == '__main__':
    
    #DATASET 4
    print("DATASET 4")
    print("-------------------------")
    
    X,Y = outlier_removal('data_4.h5')      #for data4, use gamma = 9, 1 o/w
    X_train = X[0: int(0.8*len(Y)),:]
    Y_train = Y[0: int(0.8*len(Y))]
    X_test  = X[int(0.8*len(Y)):len(Y),:]
    Y_test  = Y[int(0.8*len(Y)):len(Y)]

    clf = svm.SVC(kernel = 'linear', C=1).fit(X_train, Y_train)
    y_pred_inbuilt = clf.predict(X_test)
    acc1 = accuracy_score(Y_test, y_pred_inbuilt) * 100
    
    y_pred_own = predict_own('linear',X_test,clf.dual_coef_,clf.support_vectors_)
    acc2 = accuracy_score(Y_test, y_pred_own) * 100
    
    print("Accuracy using Linear Kernel:")
    print("Accuracy using SVM prediction function = ", acc1)
    print("Accuracy using own prediction function = ", acc2)
    
    clf = svm.SVC(kernel = 'rbf', C=1).fit(X_train, Y_train)
    y_pred_inbuilt = clf.predict(X_test)
    acc1 = accuracy_score(Y_test, y_pred_inbuilt) * 100
    
    y_pred_own = predict_own('rbf',X_test,clf.dual_coef_,clf.support_vectors_)
    acc2 = accuracy_score(Y_test, y_pred_own) * 100

    print("Accuracy using RBF Kernel:")
    print("Accuracy using SVM prediction function = ", acc1)
    print("Accuracy using own prediction function = ", acc2)
    
    
    #DATASET 5
    print("DATASET 5")
    print("-------------------------")
    X,Y = outlier_removal('data_5.h5')      #for data4, use gamma = 9, 1 o/w
    X_train = X[0: int(0.8*len(Y)),:]
    Y_train = Y[0: int(0.8*len(Y))]
    X_test  = X[int(0.8*len(Y)):len(Y),:]
    Y_test  = Y[int(0.8*len(Y)):len(Y)]

    clf = svm.SVC(kernel = 'linear', C=1).fit(X_train, Y_train)
    y_pred_inbuilt = clf.predict(X_test)
    acc1 = accuracy_score(Y_test, y_pred_inbuilt) * 100
    
    y_pred_own = predict_own('linear',X_test,clf.dual_coef_,clf.support_vectors_)
    acc2 = accuracy_score(Y_test, y_pred_own) * 100
    
    print("Accuracy using Linear Kernel:")
    print("Accuracy using SVM prediction function = ", acc1)
    print("Accuracy using own prediction function = ", acc2)
    
    clf = svm.SVC(kernel = 'rbf', C=1).fit(X_train, Y_train)
    y_pred_inbuilt = clf.predict(X_test)
    acc1 = accuracy_score(Y_test, y_pred_inbuilt) * 100
    
    y_pred_own = predict_own('rbf',X_test,clf.dual_coef_,clf.support_vectors_)
    acc2 = accuracy_score(Y_test, y_pred_own) * 100

    print("Accuracy using RBF Kernel:")
    print("Accuracy using SVM prediction function = ", acc1)
    print("Accuracy using own prediction function = ", acc2)
  


# In[ ]:





# In[ ]:




