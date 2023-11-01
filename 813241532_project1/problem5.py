# -------------------------------------------------------------------------
'''
    Problem 5: Gradient Descent and Newton method Training of Logistic Regression
    20/100 points
'''

import problem3
import problem4
from problem2 import *
import numpy as np # linear algebra
import pickle

def batch_gradient_descent(X, Y, X_test, Y_test, num_iters = 50, lr = 0.01, log=True):
    '''
    Train Logistic Regression using Gradient Descent
    X: d x m training sample vectors
    Y: 1 x m labels
    X_test: test sample vectors
    Y_test: test labels
    num_iters: number of gradient descent iterations
    lr: learning rate
    log: True if you want to track the training process, by default True
    :return: (theta, training_log)
    training_log: contains training_loss, test_loss, and norm of theta
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    ## Initialize the theta
    
    Y = Y.reshape(X.shape[1],1).T
    Y_test = Y_test.reshape(X_test.shape[1],1).T
    theta = np.random.normal(scale=0.001,size=(X.shape[0],1))
    training_log = []
    for train_idx in range(num_iters):
        iter_log = []
        Z = problem4.linear(theta,X)
        A = problem4.sigmoid(Z)
        iter_log.append(problem4.loss(A,Y).reshape(-1))
        
        dtheta = problem4.dtheta(Z,X,Y)
        theta = theta - lr*dtheta
        
        Z_test = problem4.linear(theta,X_test)
        A_test = problem4.sigmoid(Z_test)
        iter_log.append(problem4.loss(A_test,Y_test).reshape(-1))
        
        iter_log.append(np.linalg.norm(theta))
        training_log.append(iter_log)
    return theta, training_log
    #########################################

def stochastic_gradient_descent(X, Y, X_test, Y_test, num_iters = 50, lr = 0.01, log=True):
    '''
    Train Logistic Regression using Gradient Descent
    X: d x m training sample vectors
    Y: 1 x m labels
    X_test: test sample vectors
    Y_test: test labels
    num_iters: number of gradient descent iterations
    lr: learning rate
    log: True if you want to track the training process, by default True
    :return: (theta, training_log)
    training_log: contains training_loss, test_loss, and norm of theta
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    Y = Y.reshape(X.shape[1],1).T
    Y_test = Y_test.reshape(X_test.shape[1],1).T
    theta = np.random.normal(scale=0.001,size=(X.shape[0],1))
    training_log = []
    for train_idx in range(num_iters):
        iter_log = []
        random_idx = np.random.randint(X.shape[1])
        X_train = X[:,random_idx]
        Y_train = Y[:,random_idx]
        Z = problem3.linear(theta.reshape(-1),X_train)
        A = problem3.sigmoid(Z)
        iter_log.append(problem3.loss(A,Y_train).reshape(-1))
        
        dtheta = problem3.dtheta(Z,X_train.reshape(-1,1),Y_train)
        theta = theta - lr*dtheta
        
        Z_test = problem4.linear(theta,X_test)
        A_test = problem4.sigmoid(Z_test)
        iter_log.append(problem4.loss(A_test,Y_test).reshape(-1))
        
        iter_log.append(np.linalg.norm(theta))
        training_log.append(iter_log)
    return theta, training_log
    #########################################


def Newton_method(X, Y, X_test, Y_test, num_iters = 50, log=True):
    '''
    Train Logistic Regression using Gradient Descent
    X: d x m training sample vectors
    Y: 1 x m labels
    X_test: test sample vectors
    Y_test: test labels
    num_iters: number of gradient descent iterations
    log: True if you want to track the training process, by default True
    :return: (theta, training_log)
    training_log: contains training_loss, test_loss, and norm of theta
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    Y = Y.reshape(X.shape[1],1).T
    Y_test = Y_test.reshape(X_test.shape[1],1).T
    theta = np.random.normal(scale=0.1,size=(X.shape[0],1))
    training_log = []
    for train_idx in range(num_iters):
        iter_log = []
        Z = problem4.linear(theta,X)
        A = problem4.sigmoid(Z)
        hessian = problem4.Hessian(Z,X)
        iter_log.append(problem4.loss(A,Y).reshape(-1))
        
        dtheta = problem4.dtheta(Z,X,Y)
        theta = theta - np.dot(np.linalg.inv(hessian),dtheta)
        
        Z_test = problem4.linear(theta,X_test)
        A_test = problem4.sigmoid(Z_test)
        iter_log.append(problem4.loss(A_test,Y_test).reshape(-1))
        
        iter_log.append(np.linalg.norm(theta))
        training_log.append(iter_log)
    return theta, training_log
    #########################################


# --------------------------
def train_SGD(**kwargs):
    # use functions defined in problem3.py to perform stochastic gradient descent

    tr_X = kwargs['Training X']
    tr_y = kwargs['Training y']
    te_X = kwargs['Test X']
    te_y = kwargs['Test y']
    num_iters = kwargs['num_iters']
    lr = kwargs['lr']
    log = kwargs['log']
    return stochastic_gradient_descent(tr_X, tr_y, te_X, te_y, num_iters, lr, log)


# --------------------------
def train_GD(**kwargs):
    # use functions defined in problem4.py to perform batch gradient descent

    tr_X = kwargs['Training X']
    tr_y = kwargs['Training y']
    te_X = kwargs['Test X']
    te_y = kwargs['Test y']
    num_iters = kwargs['num_iters']
    lr = kwargs['lr']
    log = kwargs['log']
    return batch_gradient_descent(tr_X, tr_y, te_X, te_y, num_iters, lr, log)

# --------------------------
def train_Newton(**kwargs):
    tr_X = kwargs['Training X']
    tr_y = kwargs['Training y']
    te_X = kwargs['Test X']
    te_y = kwargs['Test y']
    num_iters = kwargs['num_iters']
    log = kwargs['log']
    return Newton_method(tr_X, tr_y, te_X, te_y, num_iters, log)


if __name__ == "__main__":
    '''
    Load and split data, and use the three training methods to train the logistic regression model.
    The training log will be recorded in three files.
    The problem5.py will be graded based on the plots in plot_training_log.ipynb (a jupyter notebook).
    You can plot the logs using the "jupyter notebook plot_training_log.ipynb" on commandline on MacOS/Linux.
    Windows should have similar functionality if you use Anaconda to manage python environments.
    '''
    X, y = loadData()
    X = appendConstant(X)
    (tr_X, tr_y), (te_X, te_y) = splitData(X, y)

    kwargs = {'Training X': tr_X,
              'Training y': tr_y,
              'Test X': te_X,
              'Test y': te_y,
              'num_iters': 1000,
              'lr': 0.01,
              'log': True}

    theta, training_log = train_SGD(**kwargs)
    with open('./data/SGD_outcome.pkl', 'wb') as f:
        pickle.dump((theta, training_log), f)


    theta, training_log = train_GD(**kwargs)
    with open('./data/batch_outcome.pkl', 'wb') as f:
        pickle.dump((theta, training_log), f)
#
#
    theta, training_log = train_Newton(**kwargs)
    with open('./data/newton_outcome.pkl', 'wb') as f:
        pickle.dump((theta, training_log), f)


