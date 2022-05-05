#!/usr/bin/python3
# Homework 2 Code
from typing_extensions import runtime
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from IPython.display import display

def find_binary_error(w, X, y):
    # find_binary_error: compute the binary error of a linear classifier w on data set (X, y)
    # Inputs:
    y_pred = np.sign(np.dot(X, w))
    error = len(y_pred[y_pred != y])
    binary_error = error / len(y)

    #        w: weight vector
    #        X: data matrix (without an initial column of 1s)
    #        y: data labels (plus or minus 1)
    # Outputs:
    #        binary_error: binary classification error of w on the data set (X, y)
    #           this should be between 0 and 1.

    # Your code here, assign the proper value to binary_error:

    return binary_error


def logistic_reg(X, y, w_init, max_its, eta, grad_threshold):
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
        #gradient=np.sum(X*y.reshape(-1,1)/(1+np.exp(y*np.dot(X,w))).reshape(-1,1),axis=0)/X.shape[0]
        w = w - eta * gradient
        t += 1
    e_in=np.sum(np.log(1+np.exp(-y*np.dot(X,w))))/N
    return t, w, e_in


def main():
    #PART1
    train_data = pd.read_csv('clevelandtrain.csv')
    test_data = pd.read_csv('clevelandtest.csv')
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    for data in train_data:
        if data[-1] == 0:
            data[-1] = -1
    for data in test_data:
        if data[-1] == 0:
            data[-1] = -1
    X_train = train_data[:, :-1]
    #print(X_train.mean(axis=0))
    #print(X_train.std(axis=0))
    mean=X_train.mean(axis=0)
    std=X_train.std(axis=0)
    X_train=(X_train-mean)/std
    y_train = train_data[:, -1]
    X_test = test_data[:, :-1]
    X_test=(X_test-mean)/std
    y_test = test_data[:, -1]
   # print(1-LogisticRegression(max_iter=10**6).fit(X_train,y_train).score(X_test,y_test))
    intercept_test = np.ones((X_test.shape[0], 1))
    intercept_train = np.ones((X_train.shape[0], 1))
    X_train = np.concatenate((intercept_train, X_train), axis=1)
    X_test = np.concatenate((intercept_test, X_test), axis=1)
    w_init = np.zeros(X_train.shape[1])
    eta = 7.7
    grad_threshold = 10 ** -6
    max_its=10**6
    start_time=datetime.now()
    t, w, e_in = logistic_reg(X_train, y_train, w_init, max_its, eta, grad_threshold)
    runtime=datetime.now()-start_time
    # t_arr.append(t)
    # e_in_arr.append(e_in)
    # btr_err.append(find_binary_error(w,X_train,y_train))
    # bte_err.append(find_binary_error(w,X_test,y_test))
    # 
    # time_arr.append(runtime)
    # data={"Runtime":time_arr,"Number of Iterations":t_arr,"E_in":e_in_arr,"Binary Training Error":btr_err,"Binary Testing Error":bte_err}
    # df=pd.DataFrame(data=data)
    # display(df)
    print ("Runtime: %s"%runtime)
    print("num of iterations: %d"%t)
    print("E_in: %f"%e_in)
    print("binary training error: %f"%find_binary_error(w,X_train,y_train))
    print("binary testing error: %f"%find_binary_error(w,X_test,y_test))



    


if __name__ == "__main__":
    main()
