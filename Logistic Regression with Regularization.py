#!/usr/bin/python3
# Homework 2 Code
from typing_extensions import runtime
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from IPython.display import display
from sklearn.preprocessing import StandardScaler

def find_binary_error(w, X, y):
    # find_binary_error: compute the binary error of a linear classifier w on data set (X, y)
    # Inputs:
    y_pred = np.sign(np.dot(X, w))
    error = len(y_pred[y_pred != y])
    binary_error = error/ len(y)

    #        w: weight vector
    #        X: data matrix (without an initial column of 1s)
    #        y: data labels (plus or minus 1)
    # Outputs:
    #        binary_error: binary classification error of w on the data set (X, y)
    #           this should be between 0 and 1.

    # Your code here, assign the proper value to binary_error:

    return binary_error


def logistic_reg(X, y, w_init, max_its, eta, grad_threshold,lamda):
    # logistic_reg learn logistic regression model using gradient descent
    # Inputs:
    #        X : data matrix (without an initial column of 1s)
    #        y : data labels (plus or minus 1)
    #        w_init: initial value of the w vector (d+1 dimensional)
    #        max_its: maximum number of iterations to run for
    #        eta: learning rate
    #        grad_threshold: one of the terminate conditions;
    #               terminate if the magnitude of every element of gradient is smaller than grad_threshold
    # Outputs:
    #        t : number of iterations gradient descent ran for
    #        w : weight vector
    #        e_in : in-sample error (the cross-entropy error as defined in LFD)

    # Your code here, assign the proper values to t, w, and e_in:
    t = 0
    w = w_init
    N = X.shape[0]
    gradient = np.ones(X.shape[1])
    while not np.all(np.abs(gradient) < grad_threshold) and t < max_its:
        gradient=np.zeros(X.shape[1])
        for (i, x) in enumerate(X):
            gradient += (-1 / N) * (y[i] * x) / (1 + np.exp(y[i] * np.dot(w, x)))
        #gradient=np.sum(X*y.reshape(-1,1)/(1+np.exp(y*np.dot(X,w))).reshape(-1,1),axis=0)/N
        w =(1-2*eta*lamda)*w - eta * (gradient)
        t += 1
    return w


def main():
    #L1
    X_train, X_test, y_train, y_test = np.load("digits_preprocess.npy", allow_pickle=True)
    #mean=X_train.mean(axis=0)
    #std=X_train.std(axis=0)
    for i in range(len(y_train)):
        if y_train[i]==0:
            y_train[i]=-1
    for i in range(len(y_test)):
        if y_test[i]==0:
            y_test[i]=-1
    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    #X_train=(X_train-mean)/(std+10**-7)
    #X_test=(X_test-mean)/(std+10**-7)
    intercept_train=np.ones((X_train.shape[0],1))
    X_train=np.concatenate((intercept_train,X_train),axis=1)
    intercept_test=np.ones((X_test.shape[0],1))
    X_test=np.concatenate((intercept_test,X_test),axis=1)
    w_init=np.zeros(X_train.shape[1])
    grad_threshold=10**-6
    max_its=10**4
    eta=0.01
    lamda=0.1
    w=logistic_reg(X_train,y_train,w_init,max_its,eta,grad_threshold,lamda)
    print("binary test error: %3f"%find_binary_error(w,X_test,y_test))
    print("Number of zeros in weight vector: %d"%len(w[w==0]))
    
   




    


if __name__ == "__main__":
    main()
