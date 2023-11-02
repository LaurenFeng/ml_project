# -------------------------------------------------------------------------
'''
    Problem 1: Implement activation and loss functions.

    Note: A static method is also a method which is bound to the class and not the object of the class.
        @staticmethod is used before the function definition to indicate that the function is static.
'''

import numpy as np
from scipy.special import logsumexp
from scipy.special import xlogy, xlog1py

# --------------------------
class Activation():
    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def activate(Z):
        raise NotImplementedError

    @staticmethod
    def gradient(Z):
        raise NotImplementedError

# --------------------------
class Sigmoid(Activation):
    """
    Implement the sigmoid activation function and its gradient
    """

    @staticmethod
    def activate(Z):
        """
        Sigmoid of each element of Z.
        Z: n x m matrix, where each column of Z is the vector for an example of dimension n.
        :return: elementwise sigmoid of Z
        """
        #########################################
        ## INSERT YOUR CODE HERE
        S = 1 / (1 + np.exp(-1*Z))
        return S
                
       
        #########################################

    @staticmethod
    def gradient(Z):
        """
        Gradient of sigmoid at Z
        Z: n x m matrix, where each column of Z is the vector for an example of dimension n.
        :return: elementwise gradient of sigmoid at Z
        """
        #########################################
        ## INSERT YOUR CODE HERE
#         S = Sigmoid.activate(Z)
#         g = np.multiply((1-S),S)

        g = np.multiply(np.exp(-1*Z),1 / np.power(np.exp(-1*Z) + 1,2))
        return g
        #########################################

# --------------------------
class Softmax(Activation):
    """
    Implement the softmax activation function, for multi-class classification.
    """

    @staticmethod
    def activate(Z):
        """
        Transform each column of Z into a probability distribution through the Softmax mapping.
        Read this article

            https://blog.feedly.com/tricks-of-the-trade-logsumexp/

        to avoid numerical problem in calculating the denominator

        Z: n x m matrix, where each column of Z is the vector for an example of dimension n.
        :return: each column of Z get transformed to a probability distribution.
        """
        #########################################
        ## INSERT YOUR CODE HERE
        a = np.sum(np.exp(Z), axis=0)
        f =np.multiply(np.exp(Z), 1 / a)
        return f
        #########################################

    @staticmethod
    def gradient(Z):
        """
        No need to implement gradient of softmax wrt Z. The reason is that,
        cross_entropy_loss ( softmax(w^T x + b), Y) can be differentiated wrt w^T x + b = Z directly.
        This is implemented in the gradient of cross entropy.
        """
        raise NotImplementedError

# --------------------------
class Identity(Activation):
    """
        Implement the identify activation function, for regression problems.
        """

    @staticmethod
    def activate(Z):
        """
        Z: n x m matrix
        Z: n x m matrix, where each column of Z is the vector for an example of dimension n.
        :return: just return the argument and you don't need to do anything here.
        """
        #########################################
        ## DISREGARD THIS FUNCTION
        #########################################
        return Z

    @staticmethod
    def gradient(Z):
        """
        No need to implement the gradient and the reason is similar to the case of softmax activation.
        The gradient is found when when calculating the gradient of some loss function (e.g., MSE).
        """
        raise NotImplementedError


# --------------------------
class Tanh(Activation):
    """
    Implement the tanh activation function and its gradient
    """

    @staticmethod
    def activate(Z):
        """
        Z: n x m matrix, where each column of Z is the vector for an example of dimension n.
        :return: elementwise tanh of Z
        """
        #########################################
        ## INSERT YOUR CODE HERE
        f =np.multiply(np.exp(Z) - np.exp(-1*Z), 1 / (np.exp(Z) + np.exp(-1*Z)))
        return f
        #########################################

    @staticmethod
    def gradient(Z):
        """
        Gradient of tanh at Z
        Z: n x m matrix, where each column of Z is the vector for an example of dimension n.
        :return: elementwise gradient of tanh at Z
        """
        #########################################
        ## INSERT YOUR CODE HERE
        f = Tanh.activate(Z)
        g = 1 - np.power(f,2)
        return g
        #########################################

# --------------------------
class ReLU(Activation):
    """
    Implement the ReLU activation function and its gradient
    """
    @staticmethod
    def activate(Z):
        """
        Z: n x m matrix, where each column of Z is the vector for an example of dimension n.
        :return: elementwise tanh of Z
        """
        #########################################
        ## INSERT YOUR CODE HERE
        f = np.maximum (0, Z)
        return f
        #########################################

    @staticmethod
    def gradient(Z):
        """
        Gradient of ReLU at Z
        Z: n x m matrix, where each column of Z is the vector for an example of dimension n.
        :return: elementwise gradient of tanh at Z
        """
        #########################################
        ## INSERT YOUR CODE HERE
        g = (Z > 0) * 1
        return g
        #########################################

# --------------------------
class Loss():
    @staticmethod
    def loss(Y, Y_hat):
        raise NotImplementedError

    @staticmethod
    def gradient(Y, Y_hat):
        raise NotImplementedError

# --------------------------
class CrossEntropyLoss(Loss):

    """
    Define the cross entropy loss and its gradient with respect to the input linear term (see below)
    Cross entropy loss is defined here:

        https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy

    """
    @staticmethod
    def loss(Y, Y_hat):
        """
        Refer to
        https://stackoverflow.com/questions/50018625/how-to-handle-log0-when-using-cross-entropy
        to see how to avoid taking log of 0.
        Think: when and why log 0 happens?
        Y: k x m the ground truth labels of m training examples. Y[j, i]=1 if the i-th example has label j. k is the number of classes.
        Y_hat: k x m the multi-class or multi-label predictions on the m training examples. This is the output of the softmax function.
        :return: a scalar that is the averaged cross entropy loss
        """
        #########################################
        ## INSERT YOUR CODE HERE
#         loss=-np.sum(Y*np.log(Y_hat).T+(1-Y)*np.log(1-Y_hat).T)
#         return loss
#         Y_ = np.log(Y_hat).T
#         loss=-np.sum(Y*Y_)
#         return loss/float(Y_.shape[0])
#         N = Y_hat.shape[0]
#         loss = -np.sum(Y*np.log(Y_hat).T)/N
#         Yis1 = Y == 1
#         loss= -np.sum(np.multiply(Y, Y_),axis=0)
#         cost = np.mean(loss) 
#         cost = -(np.log(Y_hat[Yis1]).sum() + np.log(1 - Y_hat[~Yis1]).sum())/m
#         loss = xlogy(Y, Y_hat) + xlog1py(1 - Y, -Y_hat)
#         cost = -np.sum(loss)/m
        m = Y_hat.shape[1]
        Y_ = np.log(Y_hat+1e-15)
        loss= -np.sum(np.multiply(Y, Y_))
        cost = loss/m
        return cost
        #########################################

    @staticmethod
    def gradient(Y, Y_hat):
        """
        It is the gradient of cross entropy loss with respect to Z that is used to compute Y_hat, NOT wrt Y_hat.
        Y: k x m the ground truth labels of m training examples. Y[j, i]=1 if the i-th example has label j. k is the number of classes.
        Y_hat: k x m the multi-class or multi-label predictions on the m training examples
        :return: k x m vector, the gradients on the m training examples
        """
        #########################################
        ## INSERT YOUR CODE HERE
        g = Y_hat-Y
        return g
        #########################################

# --------------------------
class MSELoss(Loss):

    """
    Define the Mean Square Error loss and its gradient with respect to the input linear term (see below)
    """
    @staticmethod
    def loss(Y, Y_hat):
        """
        Y: k x m the ground truth k-values of m training examples.
        Y_hat: k x m the regression predictions on the m training examples
        :return: a scalar that is the averaged MSE loss
        """
        #########################################
        ## DISREGARD THIS FUNCTION
        m_loss = np.mean(np.power(Y-Y_hat,2))
        return m_loss
        #########################################

    @staticmethod
    def gradient(Y, Y_hat):
        """
        It is the gradient of MSE loss with respect to Y_hat.
        Y: k x m the ground truth k-values of m training examples.
        Y_hat: k x m the regression predictions on the m training examples
        :return: k x m vector, the gradients on the m training examples
        """
        #########################################
        ## DISREGARD THIS FUNCTION
        g = (Y-Y_hat)*2
        return g
        #########################################
