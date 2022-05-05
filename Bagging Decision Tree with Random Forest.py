#!/usr/bin/python3
# Homework 4 Code
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt


def bagged_trees(X_train, y_train, X_test, y_test, num_bags):
    # The `bagged_tree` function learns an ensemble of numBags decision trees 
    # and also plots the  out-of-bag error as a function of the number of bags
    #
    # % Inputs:
    # % * `X_train` is the training data
    # % * `y_train` are the training labels
    # % * `X_test` is the testing data
    # % * `y_test` are the testing labels
    # % * `num_bags` is the number of trees to learn in the ensemble
    #
    # % Outputs:
    # % * `out_of_bag_error` is the out-of-bag classification error of the final learned ensemble
    # % * `test_error` is the classification error of the final learned ensemble on test data
    #
    # % Note: You may use sklearns 'DecisonTreeClassifier'
    # but **not** 'RandomForestClassifier' or any other bagging function
    
    #generate bootstrap datasets
    N=X_train.shape[0]
    predictions=np.zeros((num_bags,X_test.shape[0]))
    model=[]
    out_of_bag_count=np.zeros(X_train.shape[0])
    out_of_bag_pred_sum=np.zeros(X_train.shape[0])
    for i in range(num_bags):
        index=np.random.choice(N,N,replace=True)
        bootstrap_X_train=X_train[index,:]
        bootstrap_y_train=y_train[index]
        not_index=np.isin(np.arange(X_train.shape[0]),index,invert=True)
        out_of_bag_count[not_index]+=1
        not_in_bootstrap_X_train=X_train[not_index,:]
        not_in_bootstrap_y_train=y_train[not_index]
        decision_tree=DecisionTreeClassifier(criterion='entropy')
        decision_tree.fit(bootstrap_X_train,bootstrap_y_train)
        model.append(decision_tree)
        y_pred=np.sign(decision_tree.predict(X_test))
        out_of_bag_pred_sum[not_index]+=np.sign(decision_tree.predict(not_in_bootstrap_X_train))
        predictions[i]=y_pred
    non_zero_index=np.where(out_of_bag_count!=0)
    out_of_bag_pred=np.sign(np.divide(out_of_bag_pred_sum[non_zero_index],out_of_bag_count[non_zero_index]))
    out_of_bag_error=len(out_of_bag_pred[out_of_bag_pred!=y_train[non_zero_index]])/len(out_of_bag_pred)
    avg_predictions=np.sign(predictions.mean(axis=0))
    test_error=len(y_test[y_test!=avg_predictions])/len(y_test)

    return out_of_bag_error, test_error

def main_hw4():
    # Load data
    # og_train_data = np.genfromtxt('zip.train')
    # og_test_data = np.genfromtxt('zip.test')

    # #filter data, split data, and update labels
    # og_train_data=og_train_data[np.logical_or(og_train_data[:,0]==1, og_train_data[:,0]==3)]
    # og_test_data=og_test_data[np.logical_or(og_test_data[:,0]==1, og_test_data[:,0]==3)]
    # X_train=og_train_data[:,1:]
    # X_test=og_test_data[:,1:]
    # y_train=og_train_data[:,0]
    # y_test=og_test_data[:,0]
    # y_train[y_train==1]=-1
    # y_train[y_train==3]=1
    # y_test[y_test==1]=-1
    # y_test[y_test==3]=1

    # out_of_bag_error_arr=np.zeros(200)
    # test_error_arr=np.zeros(200)
    # x_axis=np.arange(1,201)
    # # Run bagged trees
    # for i in range(200):
    #     num_bags=i+1
    #     out_of_bag_error, test_error = bagged_trees(X_train, y_train, X_test, y_test, num_bags)
    #     out_of_bag_error_arr[i]=out_of_bag_error
    #     test_error_arr[i]=test_error
    # plt.plot(x_axis,out_of_bag_error_arr)
    # plt.title("Out of bag error")
    # plt.xlabel("num_bags")
    # plt.ylabel("out_of_bag_error")
    # plt.show()


    # out_of_bag_error, test_error = bagged_trees(X_train, y_train, X_test, y_test, 200)
    # single_decision_tree=DecisionTreeClassifier(criterion="entropy")
    # single_decision_tree.fit(X_train,y_train)
    # y_pred_test=np.sign(single_decision_tree.predict(X_test))
    # single_test_error=len(y_pred_test[y_pred_test!=y_test])/len(y_test)
    # print(out_of_bag_error)
    # print(test_error)
    # print(single_test_error)

if __name__ == "__main__":
    main_hw4()

