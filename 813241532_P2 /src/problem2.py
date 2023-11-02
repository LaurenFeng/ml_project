# -------------------------------------------------------------------------
'''
    Problem 2: Compute the objective function and decision function of dual SVM.

'''
from problem1 import *

import numpy as np

# -------------------------------------------------------------------------
def dual_objective_function(alpha, train_y, train_X, kernel_function, sigma):
    """
    Compute the dual objective function value.

    alpha: 1 x m learned Lagrangian multipliers (the dual variables).
    train_y: 1 x m labels (-1 or 1) of training data.
    train_X: n x m training feature matrix. n: number of features; m: number training examples.
    kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
    sigma: need to be provided when Gaussian kernel is used.
    :return: a scalar representing the dual objective function value at alpha
    Hint: refer to the objective function of Eq. (47).
          You can try to call kernel_function.__name__ to figure out which kernel are used.
    """
    #########################################
    ## INSERT YOUR CODE HERE
    if kernel_function.__name__ == "linear_kernel":
        k = linear_kernel(train_X, train_X)
    else:
        k = Gaussian_kernel(train_X, train_X, sigma)
    
    one,m = np.shape(alpha) 
    a = 0
    for i in range(m):
        a += alpha[:,i]
    f = 0
    for i in range(m):
        for j in range(m):
            f += alpha[:,i]*alpha[:,j]*train_y[:,i]*train_y[:,j]*k[i,j]
    dual = a - 1/2 * f 
    return dual
        
        
    #########################################


# -------------------------------------------------------------------------
def primal_objective_function(alpha, train_y, train_X, b, C, kernel_function, sigma):
    """
    Compute the primal objective function value.
    When with linear kernel:
        The primal parameter w is recovered from the dual variable alpha.
    When with Gaussian kernel:
        Can't recover the primal parameter and kernel trick needs to be used to compute the primal objective function.

    alpha: 1 x m learned Lagrangian multipliers (the dual variables).
    train_y: 1 x m labels (-1 or 1) of training data.
    train_X: n x m training feature matrix.
    b: bias term
    C: regularization parameter of soft-SVM
    kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
    sigma: need to be provided when Gaussian kernel is used.

    :return: a scalar representing the primal objective function value at alpha
    Hint: you need to use kernel trick when come to Gaussian kernel. Refer to the derivation of the dual objective function Eq. (47) to check how to find
            1/2 ||w||^2 and the decision_function with kernel trick.
    """
    #########################################
    ## INSERT YOUR CODE HERE
    if kernel_function.__name__ == "linear_kernel":
        k = linear_kernel(train_X, train_X)
    else:
        k = Gaussian_kernel(train_X, train_X, sigma)
        
    n, m = np.shape(train_X)
#     z = np.zero([1, m])
    z = 0
    for i in range(m):
        z += alpha[:,i]*train_y[:,i]*k[i,:]
    z = z + b
    prim = np.sum(np.matmul(alpha.T, alpha)*k*np.matmul(train_y.T, train_y)) * 1/2+ C * np.sum(hinge_loss(z, train_y))
    return prim
    
    #########################################


def decision_function(alpha, train_y, train_X, b, kernel_function, sigma, test_X):
    """
    Compute the linear function <w, x> + b on examples in test_X, using the current SVM.

    alpha: 1 x m learned Lagrangian multipliers (the dual variables).
    train_y: 1 x m labels (-1 or 1) of training data.
    train_X: n x m training feature matrix.
    test_X: n x m2 test feature matrix.
    b: scalar, the bias term in SVM <w, x> + b.
    kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
    sigma: need to be provided when Gaussian kernel is used.

    :return: 1 x m2 vector <w, x> + b
    """
    #########################################
    ## INSERT YOUR CODE HERE
    if kernel_function.__name__ == "linear_kernel":
        k = linear_kernel(train_X, test_X)
    else:
        k = Gaussian_kernel(train_X, test_X, sigma)
    
    n,m = np.shape(train_X)
    n,m2 = np.shape(test_X)
    f = np.zeros([1, m2])
    for j in range(m):
        f += alpha[:,j]*train_y[:,j]*k[j,:]
    f = f + b
    return f
            
    #########################################
