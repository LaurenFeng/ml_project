# -------------------------------------------------------------------------
'''
    Problem 3: compute sigmoid(<theta, x>), the loss function, and the gradient.
    This is the single training example version.

    20/100 points
'''

import numpy as np # linear algebra

def linear(theta, x):
    '''
    theta: (n+1) x 1 column vector of model parameters
    x: (n+1) x 1 column vector of an example features. Must be a sparse csc_matrix
    :return: inner product between theta and x
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    z = np.dot(theta,x)
    return z
    #########################################

def sigmoid(z):
    '''
    z: scalar. <theta, x>
    :return: sigmoid(z)
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    a= 1/(1 + np.exp(-z))
    return a
    #########################################

def loss(a, y):
    '''
    a: 1 x 1, sigmoid of an example x
    y: {0,1}, the label of the corresponding example x
    :return: negative log-likelihood loss on (x, y).
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    loss = -1* (y * np.log(a) + (1-y)* np.log(1-a))
    return loss.reshape(-1)
    #########################################

def dz(z, y):
    '''
    z: scalar. <theta, x>
    y: {0,1}, label of x
    :return: the gradient of the negative log-likelihood loss on (x, y) wrt z.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    dz = sigmoid(z)-y
    return dz
    #########################################

def dtheta(z, x, y):
    '''
    z: scalar. <theta, x>
    x: (n+1) x 1 vector, an example feature vector
    y: {0,1}, label of x
    :return: the (n+1) x 1 gradient vector of the negative log-likelihood loss on (x, y) wrt theta.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    dtheta = (sigmoid(z)-y)* x
    return dtheta
    #########################################

def Hessian(z, x):
    '''
    C;ompute the Hessian matrix on a single training example.
    z: scalar. <theta, x>
    x: (n+1) x 1 vector, an example feature vector
    :return: the Hessian matrix of the negative log-likelihood loss wrt theta
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    hessian = sigmoid(z)*(1-sigmoid(z))*np.multiply(x,x.T)
    return hessian
    #########################################    
