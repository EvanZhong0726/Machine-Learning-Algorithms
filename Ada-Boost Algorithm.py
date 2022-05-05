#!/usr/bin/python3
# Homework 5 Code
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt
from tables import Col

def adaboost_trees(X_train, y_train, X_test, y_test, n_trees):
    # %AdaBoost: Implement AdaBoost using decision trees
    # %   using decision stumps as the weak learners.
    # %   X_train: Training set
    # %   y_train: Training set labels
    # %   X_test: Testing set
    # %   y_test: Testing set labels
    # %   n_trees: The number of trees to use
    N=X_train.shape[0]
    weight=np.ones(N)/N
    y_train_pred=np.zeros((n_trees,X_train.shape[0]))
    y_test_pred=np.zeros((n_trees,X_test.shape[0]))
    alphas=np.zeros(n_trees)
    for i in range (n_trees):
        decision_stump=DecisionTreeClassifier(criterion="entropy",max_depth=1)
        decision_stump.fit(X_train,y_train,sample_weight=weight)
        y_pred=np.sign(decision_stump.predict(X_train))
        correct=np.where(np.equal(y_pred,y_train))
        wrong=np.where(np.not_equal(y_pred,y_train))
        error=1-decision_stump.score(X_train,y_train, sample_weight=weight)
        gamma=np.sqrt((1-error)/error)
        Z=gamma*error+(1-error)/gamma
        alpha=np.log((1-error)/error)/2
        alphas[i]=alpha
        y_train_pred[i]=y_pred
        y_test_pred[i]=np.sign(decision_stump.predict(X_test))
        weight[correct]=weight[correct]/(gamma*Z)
        weight[wrong]=weight[wrong]*gamma/Z
    train_pred=np.sign(np.multiply(y_train_pred,alphas.reshape(-1,1)).sum(axis=0))
    test_pred=np.sign(np.multiply(y_test_pred,alphas.reshape(-1,1)).sum(axis=0))
    train_error=len(y_train[y_train!=train_pred])/len(y_train)
    test_error=len(y_test[y_test!=test_pred])/len(y_test)

    return train_error, test_error


def main_hw5():
    # Load data
     # Load data
    # og_train_data = np.genfromtxt('zip.train')
    # og_test_data = np.genfromtxt('zip.test')

    # #filter data, split data, and update labels
    # og_train_data=og_train_data[np.logical_or(og_train_data[:,0]==5, og_train_data[:,0]==3)]
    # og_test_data=og_test_data[np.logical_or(og_test_data[:,0]==5, og_test_data[:,0]==3)]
    # X_train=og_train_data[:,1:]
    # X_test=og_test_data[:,1:]
    # y_train=og_train_data[:,0]
    # y_test=og_test_data[:,0]
    # y_train[y_train==5]=-1
    # y_train[y_train==3]=1
    # y_test[y_test==5]=-1
    # y_test[y_test==3]=1


    # train_errors=np.zeros(200)
    # test_errors=np.zeros(200)
    # x_axis=np.arange(1,201)
    # for i in range(200):
    #    train_error, test_error = adaboost_trees(X_train, y_train, X_test, y_test, i+1)
    #    train_errors[i]=train_error
    #    test_errors[i]=test_error
    # plt.plot(x_axis,train_errors, c='red', label='train') 
    # plt.plot(x_axis,test_errors, c='blue', label='test')
    # plt.xlabel("num_trees")  
    # plt.ylabel("error")
    # plt.title("Ada boost train test errors")
    # plt.legend()
    # plt.show()
    X=[[3],[5],[7],[2],[3]]
    y=[5,6,9,11,8]
    neigh = KNeighborsRegressor(n_neighbors=3)
    neigh.fit(X,y)
    print(neigh.predict([[3.2]]))

    

    
   

if __name__ == "__main__":
    main_hw5()
