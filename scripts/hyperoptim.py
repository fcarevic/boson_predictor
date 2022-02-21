import numpy as np
from implementations import *
from featurization import *

MODELS = [least_squares_GD, least_squares_SGD, least_squares,
          ridge_regression, logistic_regression,
          reg_logistic_regression]

def hp_random_search(x_train, y_train, x_test, y_test,
                    gamma = 1e-4, lambda_ = 1e-1, model_num = None,
                    polynom_exp = 2, keep_num = 10,
                    train_max_iters = 1000, seed = 1234, 
                    search_max_iters = 1000,
                    expansion_mask = None,
                    keep_callback = None):

    """
    Performs polynomial feature expansion using a given expansion array,
    trains model and returns evaluated performance.

    Parameters
    ----------
    x_train : np.Array
        Contains unaugmented train features
    y_train : np.Array
        Contains train labels
    x_test : np.Array
        Contains unaugmented test features
    y_test : np.Array
        Contains validation labels
    gamma : float
        Parameter for least squares GD, least squares SGD,
        logistical regression and regularized logistical regression training methods
    lamba_ : float
        Parameter for regularization
    model_num: int
        Index of training method in MODEL array
        MODELS = [least_squares_GD, least_squares_SGD, least_squares,
          ridge_regression, logistic_regression,
          reg_logistic_regression]
    polynom_exp : int
        Degree of expansion
    keep_num : int
        Number of best models to keep
    train_max_iters : int
        Maximum number of iteration used in GD/SGD least squares, 
        logistic regression and regularized logistic regression training methods
    seed : int
        A seed for the random generator
    search_max_iters : int
        Maximum number of iterations of generating different
        expansion vectors
    expansion_mask : np.Array
        Array of 1 and 0 values where 0 masks the feature and it
        does not get expanded.
    Return
    -------
    dict
         report = {
            'id': ID,
            'weights': weights of the best fit model
            'loss': value of the loss function,
            'gamma': parameter used in training if specified 
            'lambda_': parameter used in training if specified 
            'max_iters': parameter used in training if specified 
            'initial_w': starting weight vector
            'expansion': expansion array used in feature augmentation
            'train_perf': performance description on train set
            'test_perf': performance description on test set
        }
    """
    
    # Init phase
    keep = []
    np.random.seed(seed)
    if(expansion_mask is None):
        expansion_mask = np.ones(x_train.shape[1])
    

    for i in range(search_max_iters):

        # Generation phase
        expansion = np.random.randint(0, polynom_exp+1, size=x_train.shape[1]) 
        expansion =np.power(expansion,  expansion_mask)
        fx_train = polynomial_feature_expansion(x_train, expansion)
        fx_test = polynomial_feature_expansion(x_test, expansion)
        model_ind = model_num if model_num else np.random.randint(0, len(MODELS))
        initial_w = np.random.normal(0, 1, fx_train.shape[1])

        kwargs = {
            'y': y_train,
            'tx': fx_train
        }

        # training phase
        if(model_ind in [3, 5]):
            kwargs.update({
                'lambda_': lambda_,
            })
        
        if(model_ind in [0, 1, 4, 5]):
            kwargs.update({
                'gamma': gamma,
                'max_iters': train_max_iters,
                'initial_w': initial_w,
            })

        w, loss = MODELS[model_ind](**kwargs)

        # Evaluation phase

        # Construct keep_obj
        keep_obj = {
            'id': i,
            'weights': w,
            'loss': loss,
            'gamma': gamma,
            'lambda_': lambda_,
            'max_iters': train_max_iters,
            'initial_w': initial_w,
            'expansion': expansion,
            'model_ind' : model_ind,
        }

        perf_train = calculate_performance(w, fx_train, y_train)
        perf_test = calculate_performance(w, fx_test, y_test)
        keep_obj.update({
            'train_perf': perf_train,
            'test_perf': perf_test
        })

        if(np.isnan(keep_obj['test_perf']['accuracy'])):
            continue

        keep.append(keep_obj)
        keep.sort(key=lambda x:x['test_perf']['accuracy'], reverse=True)

        if(len(keep) <= keep_num):
            keep_callback(keep, keep_obj)
        else:
            not_keeping = keep[-1]
            keep = keep[:-1]
            if(not_keeping['id'] != i):
                keep_callback(keep, keep_obj)

    return keep
    