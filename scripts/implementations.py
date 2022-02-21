import numpy as np
from proj1_helpers import *


def compute_loss_MSE(y, tx, w):
    """
    Compute the MSE loss.
    
    It computes the MSE loss for a given set of labels y,
    given inputs and the weights of the model.
    
    Parameters
    ----------
    y : np.Array
        The true labels we are trying to fit.
    tx : np.Array
        The (N, F) dimension array of input values.
        N = number of data points
        F = number of features per data point
    w : np.Array
        The weight vector which represents the model.

    Returns
    -------
    float
        The MSE loss function value.
    """
    N = y.shape[0] # Determine the size of the data
    e = y - tx.dot(w)
    return e.T.dot(e) / (2 * N)


def compute_gradient_MSE(y, tx, w):
    """
    Compute the MSE gradient
    
    It computes the gradient for the linear regression model
    with MSE loss.
    
    Parameters
    ----------
    y : np.Array
        The true labels we are trying to fit.
    tx : np.Array
        The (N, F) dimension array of input values.
        N = number of data points
        F = number of features per data point
    w : np.Array
        The weight vector which represents the model.

    Returns
    -------
    np.Array
        The gradient vector which is of the same size
        as the given weight vector
    """
    N = y.shape[0] # Determine the size of the data
    e = y - tx.dot(w)
    return -tx.T.dot(e) / N

def compute_gradient_SGD_MSE(y, tx, w):
    """
    Compute the MSE gradient for SGD
    
    It computes the gradient for the linear regression model
    with MSE loss for Stochastic Gradient Descent.
    This is done seperately from the compute_gradient_MSE for
    gradient descent, because the GD works with y as a vector
    as opposed to SGD which works with y as a scalar.
    
    Parameters
    ----------
    y : float
        The true labels we are trying to fit.
    tx : np.Array
        The input vector for the given y label.
    w : np.Array
        The weight vector which represents the model.

    Returns
    -------
    np.Array
        The gradient vector which is of the same size
        as the given weight vector
    """
    e = y - tx.dot(w)
    return -tx.T.dot(e)

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Train weights for least squares with Gradient Descent.
    
    This function completes numerous iterations of 
    Gradient Descent on a vector of weights in order to
    minimize the MSE loss function for the given inputs
    and labels.
    
    Parameters
    ----------
    y : np.Array
        The true labels we are trying to fit.
    tx : np.Array
        The (N, F) dimension array of input values to
        train on.
        N = number of data points
        F = number of features per data point
    initial_w : np.Array
        The initial weight vector for the model.
    max_iters : int
        The number of Gradient Descent iterations
        i.e. the number of epochs
    gamma : float
        The learning rate
        i.e. step size

    Returns
    -------
    np.Array
        The trained weights of the model
    float
        The value of the last loss function
    """
    w = initial_w

    for i in range(max_iters):
        w = w - gamma * compute_gradient_MSE(y, tx, w)

    return w, compute_loss_MSE(y, tx, w)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Train weights for least squares with Stochastic
    Gradient Descent.
    
    This function completes numerous iterations of 
    Stochastic Gradient Descent on a vector of weights in order to
    minimize the MSE loss function for the given inputs
    and labels. The minibatch size is fixed to 1.
    
    Parameters
    ----------
    y : np.Array
        The true labels we are trying to fit.
    tx : np.Array
        The (N, F) dimension array of input values to
        train on.
        N = number of data points
        F = number of features per data point
    initial_w : np.Array
        The initial weight vector for the model.
    max_iters : int
        The number of Stochastic Gradient Descent
        iterations.
    gamma : float
        The learning rate
        i.e. step size

    Returns
    -------
    np.Array
        The trained weights of the model
    float
        The value of the last loss function
    """
    w = initial_w
    for i in range(max_iters):
        ind = np.random.randint(y.shape[0])
        w = w - gamma * compute_gradient_SGD_MSE(y[ind], tx[ind], w)

    return w, compute_loss_MSE(y, tx, w)


def least_squares(y, tx):
    """
    Determine the weights for the least squares linear 
    regressor using normal equations.
   
    This function produces the weights for the given labels
    and inputs using normal equations for the least squares
    linear regression.
    
    Parameters
    ----------
    y : np.Array
        The true labels we are trying to fit.
    tx : np.Array
        The (N, F) dimension array of input values to
        train on.
        N = number of data points
        F = number of features per data point

    Returns
    -------
    np.Array
        The trained weights of the model
    float
        The value of the last loss function
    """
    A = tx.T.dot(tx)
    B = tx.T.dot(y)
    w_opt = np.linalg.solve(A, B)
    return w_opt, compute_loss_MSE(y, tx, w_opt)


def ridge_regression(y, tx, lambda_):
    """
    Determine the weights for a ridge regressor using
    normal equations.
   
    This function produces the weights for the given labels
    and inputs using normal equations for rigde regression.
    This function is very similar to the least_squares
    function. However, it also includes a regularization
    feature.
    
    Parameters
    ----------
    y : np.Array
        The true labels we are trying to fit.
    tx : np.Array
        The (N, F) dimension array of input values to
        train on.
        N = number of data points
        F = number of features per data point
    lambda_ : float
        The regularization coefficient for ridge regression

    Returns
    -------
    np.Array
        The trained weights of the model
    float
        The value of the last loss function
    """
    N = tx.shape[0]
    w_length = tx.shape[1]
    aI = 2 * N * lambda_ * np.identity(w_length)
    A = tx.T.dot(tx) + aI
    B = tx.T.dot(y)
    w_opt = np.linalg.solve(A, B)
    return w_opt, compute_loss_MSE(y, tx, w_opt)


def sigmoid(x, epsilon=1e-9):
    """
    Calculate the sigmoid value for the given input
    
    Parameters
    ----------
    x : np.Array or float
        The input value for the sigmoid function.
    epsilon : float
        The minimum value that the sigmoid will output
        if the value for the given X is less then the
        output of the sigmoid will be clipped to epsilon
    Returns
    -------
    np.Array or float
        The sigmoid value for the given input.
    """
    thresh = -np.log(1/epsilon - 1) # Calculated from 1/(1+exp(-x)) = epsilon)
    mod_x = np.clip(x, a_min=thresh, a_max=None)
    return 1.0 / (1 + np.exp(-mod_x))


def compute_gradient_log_reg(y, x, w):
    """
    Compute the logistic regression gradient
    
    It computes the gradient for the logistic regression model.
    
    
    Parameters
    ----------
    y : np.Array
        The true labels we are trying to fit.
    tx : np.Array
        The (N, F) dimension array of input values.
        N = number of data points
        F = number of features per data point
    w : np.Array
        The weight vector which represents the model.

    Returns
    -------
    np.Array
        The gradient vector which is of the same size
        as the given weight vector
    """
    return x.T.dot(sigmoid(x.dot(w)) - y)


def compute_loss_log_reg(y, x, w):
    """
    Compute the logistic regression loss.
    
    It computes the logistic regression loss for 
    a given set of labels y, inputs 
    and the weights of the model.
    
    Parameters
    ----------
    y : np.Array
        The true labels we are trying to fit.
    tx : np.Array
        The (N, F) dimension array of input values.
        N = number of data points
        F = number of features per data point
    w : np.Array
        The weight vector which represents the model.

    Returns
    -------
    float
        The MSE loss function value.
    """
    temp = sigmoid(x.dot(w))
    return y.T.dot(np.log(np.clip(temp, a_min=1e-9, a_max=None))) + (1 - y).T.dot(np.log(np.clip(1 - temp, a_min=1e-9, a_max=None)))


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Train weights for logistic regression with Gradient
    Descent.
    
    This function completes numerous iterations of 
    Gradient Descent on a vector of weights in order to
    minimize the logistic regression loss function for
    the given inputs and labels.
    
    Parameters
    ----------
    y : np.Array
        The true labels we are trying to fit.
    tx : np.Array
        The (N, F) dimension array of input values to
        train on.
        N = number of data points
        F = number of features per data point
    initial_w : np.Array
        The initial weight vector for the model.
    max_iters : int
        The number of Gradient Descent iterations
        i.e. the number of epochs
    gamma : float
        The learning rate
        i.e. step size

    Returns
    -------
    np.Array
        The trained weights of the model
    float
        The value of the last loss function
    """
    w = initial_w

    for i in range(max_iters):
        w = w - gamma * compute_gradient_log_reg(y, tx, w)

    return w, compute_loss_log_reg(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Train weights for logistic regression with Gradient
    Descent with regularization.
    
    This function completes numerous iterations of 
    Gradient Descent on a vector of weights in order to
    minimize the logistic regression loss function for
    the given inputs and labels. 
    Unlike the logistic_regression function this one uses
    regularization term.
    
    Parameters
    ----------
    y : np.Array
        The true labels we are trying to fit.
    tx : np.Array
        The (N, F) dimension array of input values to
        train on.
        N = number of data points
        F = number of features per data point
    initial_w : np.Array
        The initial weight vector for the model.
    max_iters : int
        The number of Gradient Descent iterations
        i.e. the number of epochs
    gamma : float
        The learning rate
        i.e. step size

    Returns
    -------
    np.Array
        The trained weights of the model
    float
        The value of the last loss function
    """
    w = initial_w

    for i in range(max_iters):
        w = w - gamma * (compute_gradient_log_reg(y, tx, w) + 2 * lambda_ * w)

    return w, compute_loss_log_reg(y, tx, w) + w.T.dot(w).reshape((-1,))


def calculate_performance(weights, tX, y):
    """
    Calculate the performance for a given model.
    
    This function counts the true/false positives/negatives
    and calculates different performance evaluators for
    a given model
    
    
    Parameters
    ----------
    weights : np.Array
        The weight vector which represents the model.
    tX : np.Array
        The (N, F) dimension array of input values.
        N = number of data points
        F = number of features per data point
    y : np.Array
        The true labels we are trying to fit.

    Returns
    -------
    dict({
        TP : int
            True Positives
        FP : int
            False Positives
        TN : int
            True Negatives
        FN : int
            False Negatives
        precision : float
            Model precision
        recall : float
            Model recall
        accuracy : float
            Model accuracy
        fscore : float
            Model F1 score
    })
    """
    labels = predict_labels(weights, tX)
    TP = ((y==1)&(labels==1)).sum()
    FP = ((y==1)&(labels==-1)).sum()
    TN = ((y==-1)&(labels==-1)).sum()
    FN = ((y==-1)&(labels==1)).sum()
    precision = TP/(TP+FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    fscore = 2 * (precision*recall)/(precision + recall)
    perf = {
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'fscore': fscore
    }
    return perf