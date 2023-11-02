# -------------------------------------------------------------------------
'''
    Problem 3: SMO training algorithm

'''
from problem1 import *
from problem2 import *

import numpy as np

import copy

class SVMModel():
    """
    The class containing information about the SVM model, including parameters, data, and hyperparameters.

    DONT CHANGE THIS DEFINITION!
    """
    def __init__(self, train_X, train_y, C, kernel_function, sigma=1):
        """
            train_X: n x m training feature matrix. n: number of features; m: number training examples.
            train_y: 1 x m labels (-1 or 1) of training data.
            C: a positive scalar
            kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
            sigma: need to be provided when Gaussian kernel is used.
        """
        # data
        self.train_X = train_X
        self.train_y = train_y
        self.n, self.m = train_X.shape

        # hyper-parameters
        self.C = C
        self.kernel_func = kernel_function
        self.sigma = sigma

        # parameters
        self.alpha = np.zeros((1, self.m))
        self.b = 0
        
def train(model, max_iters = 10, record_every = 1, max_passes = 1, tol=1e-6):
    """
    SMO training of SVM
    model: an SVMModel
    max_iters: how many iterations of optimization
    record_every: record intermediate dual and primal objective values and models every record_every iterations
    max_passes: each iteration can have maximally max_passes without change any alpha, used in the SMO alpha selection.
    tol: numerical tolerance (exact equality of two floating numbers may be impossible).
    :return: 4 lists (of iteration numbers, dual objectives, primal objectives, and models)
    Hint: refer to subsection 3.5 "SMO" in notes.
    """
    #########################################
    ## INSERT YOUR CODE HERE
    kernel_function = model.kernel_func
    train_X = model.train_X
    train_y = model.train_y
    n = model.n
    m = model.m
    C = model.C
    sigma = model.sigma
    alpha = model.alpha
    b = model.b
    iters = []
    dual = []
    primal = []
    models = []
    if kernel_function.__name__ == "linear_kernel":
        k = linear_kernel(train_X, train_X)
    else:
        k = Gaussian_kernel(train_X, train_X, sigma)
        
    for it in range(max_iters):
        print("iter=%d"%it)
        num_passes = 0
        while (num_passes < max_passes):
            num_changes = 0
            for i in range(m):
                alpha_i = alpha[:,i]
                y_i = train_y[:,i]
                f_i = decision_function(alpha, train_y, train_X, b, kernel_function, sigma, train_X[:,i].reshape(-1,1))
                E_i = f_i - y_i
                #是否还需要训练（大于tol继续更新）
                if((y_i*E_i < -tol) and (alpha[:,i] < C)) or ((y_i*E_i > tol) and  (alpha[:,i] > 0)): 
                    j=i
                    while(j==i):
                        j = np.random.randint(m)  
                    alpha_j = alpha[:,j]
                    f_j = decision_function(alpha, train_y, train_X, b, kernel_function, sigma, train_X[:,j].reshape(-1,1))
                    y_j = train_y[:,j]
                    E_j = f_j - y_j
                    # y_i != y_j
                    # alpha_j_new must be upper bounded <= H = min{C, c - p} = min{C, C-(alpha1_old - alpha2_old)}
                    # lower bound of alpha_j_new L = max{0, -p} = max{0, -(alpha1_old - alpha2_old)}
                    # y_i != y_j
                    # alpha_j_new will be <= H = min{C, p} = min{C, alpha1_old + alpha2_old}
                    # lower bound of alpha_j_new L = max{0, p - C} = max{0, alpha1_old + alpha2_old - C}
                    if (y_i != y_j):
                        L = max(0, alpha_j - alpha_i)
                        H = min(C, C + alpha_j - alpha_i)
                    else:
                        L = max(0, alpha_j + alpha_i - C)
                        H = min(C, alpha_j + alpha_i)
                    if L==H:
                        continue
                    eta = 2 * k[i,j] - k[i,i] - k[j,j]
                    if eta >= 0:
                        continue
                    # according to Eq(93):alpha_j_new = alpha_j_old + y_js * (g1 - y1 - (g2 - y2)) / ( (K11 + K22 - 2 * K12)
                    alpha_j_new = alpha_j - train_y[:,j] *(E_i - E_j) / eta
                    # clip alpha2_new
                    if alpha_j_new > H:
                        alpha_j_new = H
                    if L > alpha_j_new:
                        alpha_j_new = L
                    alpha[:,j] = alpha_j_new
                    if (abs(alpha_j_new - alpha_j) < tol):
                        continue
                    # compute new b
                    # according to Eq(95)alpha_i_new = y_i * (p - y_j * alpha_j_new) = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)
                    alpha[:,i] += y_i*y_j*(alpha_j - alpha[:,j])
                    # update b
                    # Eq(96)b1 = b_old - E_i - y_i * (alpha_i_new - alpha_i_old) * K11 - y_j * (alpha_j_new - alpha_j_old) * K12
                    # Eq(97)b2 = b_old - E_j - y_i * (alpha_j_new - alpha_j_old) * K12 - y_j * (alpha_j_new - alpha_j_old) * K22
                    b1 = b - E_i - y_i*(alpha[:,i] - alpha_i)*k[i,i] - y_j*(alpha[:,j] - alpha_j)*k[i,j]
                    b2 = b - E_j - y_i*(alpha[:,i] - alpha_i)*k[i,j] - y_j*(alpha[:,j] - alpha_j)*k[j,j]
                    # range of b
                    if (0 < alpha[:,i]) and (C > alpha[:,i]):
                        b = b1
                    elif (0 < alpha_j_new) and (C > alpha_j_new):
                        b = b2
                    else:
                        b = (b1+b2)/2
                    # update iterator
                    num_changes += 1
                # one pass without any changing parameters
                if(num_changes == 0): num_passes +=1
                # at least one pair of alpha`s are changed
                else: num_passes = 0
                    
        if(it % record_every) == 0:
            iters.append(it)
            dual.append(dual_objective_function(alpha, train_y, train_X, kernel_function, sigma))
            primal.append(primal_objective_function(alpha, train_y, train_X, b, C, kernel_function, sigma))
            model.alpha = alpha
            model.b = b
            models.append(model)
            
    return iters, dual, primal, models
    #########################################


def predict(model, test_X):
    """
    Predict the labels of test_X
    model: an SVMModel
    test_X: n x m matrix, test feature vectors
    :return: 1 x m matrix, predicted labels
    """
    #########################################
    ## INSERT YOUR CODE HERE
    train_X = model.train_X
    train_y = model.train_y
    sigma = model.sigma
    alpha = model.alpha
    b = model.b
    kernel_function = model.kernel_func
    test_result = decision_function(alpha, train_y, train_X, b, kernel_function, sigma, test_X)
    result = np.sign(test_result)
    return result
    #########################################
