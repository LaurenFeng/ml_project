'''
    Problem 1: Implement linear and Gaussian kernels and hinge loss
'''

import numpy as np
from sklearn.metrics.pairwise  import euclidean_distances


def linear_kernel(X1, X2):
    
    """
   

    X1: n x m1 matrix, each of the m1 column is an n-dim feature vector.
    X2: n x m2 matrix, each of the m2 column is an n-dim feature vector.
    
    Note that m1 may not equal m2

    :return: if both m1 and m2 are 1, return linear kernel on the two vectors; else return a m1 x m2 kernel matrix K,
            where K(i,j)=linear kernel evaluated on column i from X1 and column j from X2.
    """
    #########################################
    ## INSERT YOUR CODE HERE
    K = np.matmul(X1.T,X2)
    return K
    #########################################



def Gaussian_kernel(X1, X2, sigma=1):
    """
    Compute Gaussian kernel between two set of feature vectors.
    
    The constant 1 is not appended to the x's.
    
    For your convenience, please use euclidean_distances.

    X1: n x m1 matrix, each of the m1 column is an n-dim feature vector.
    X2: n x m2 matrix, each of the m2 column is an n-dim feature vector.
    sigma: Gaussian variance (called bandwidth)

    Note that m1 may not equal m2

    :return: if both m1 and m2 are 1, return Gaussian kernel on the two vectors; else return a m1 x m2 kernel matrix K,
            where K(i,j)=Gaussian kernel evaluated on column i from X1 and column j from X2

    """
    #########################################
    ## INSERT YOUR CODE HERE
    n,m1=np.shape(X1)
    n,m2=np.shape(X2)
    K = np.zeros([m1,m2])
    for i in range(m1):
         for j in range(m2):
            dist = euclidean_distances(X1[:, i].reshape(1,-1), X2[:, j].reshape(1,-1))
            K[i,j] = np.exp(-1*dist**2/(2*sigma**2))
    return K
    #########################################


def hinge_loss(z, y):
    """
    Compute the hinge loss on a set of training examples
    z: 1 x m vector, each entry is <w, x> + b (may be calculated using a kernel function)
    y: 1 x m label vector. Each entry is -1 or 1
    :return: 1 x m hinge losses over the m examples
    """
    #########################################
    ## INSERT YOUR CODE 
    loss = 1 - (z * y)
    zero = np.zeros_like(loss)
    hinge_loss = np.maximum(zero, loss)
    return hinge_loss
    #########################################
