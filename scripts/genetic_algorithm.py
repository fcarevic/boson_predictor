import numpy as np
import random as rand
import math as math
from implementations import *
from featurization import *




POPULATION_PARAMETER = 20
MAX_GENERATIONS = 10
SELECTION_PARAMETER =(int)(POPULATION_PARAMETER *20 /100)
MUTATION_PROBABILITY = 0.12
CROSSOVER_PROBABILITY = 0.8

POLY_DEGREE = 10
train = []
test = []
model_num = 2
MODELS = [least_squares_GD, least_squares_SGD, least_squares,
          ridge_regression, logistic_regression,
          reg_logistic_regression]
ID=0


def evaluate_expansion(train, test,
                    expansion,
                    gamma = 1e-4, lambda_ = 1e-1, model_num = None,
                    train_max_iters = 1000):
    
      """
    Performs polynomial feature expansion using given expansion array, trains model and returns evaluated performance.

    Parameters
    ----------
    train : touple or list
        Contains unaugmented train features and labels
    test : touple or list
        Contains unaugmented test features and labels
    
    gamma : float
        Parameter for least squares GD, least squares SGD, logistical regression and regularized logistical regression training methods
    lamba_ : float
        Parameter for regularization
    model_num: int
        Index of training method in MODEL array
        MODELS = [least_squares_GD, least_squares_SGD, least_squares,
          ridge_regression, logistic_regression,
          reg_logistic_regression]
    train_max_iters: int
        Maximum number of iteration used in GD/SGD least squares, logistic regression and regularized logistic regression training methods
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
    
        x_train = train[0]
        y_train = train[1]
        x_test = test[0]
        y_test= test[1]
            
        # Generation phase
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
        global ID
        
        # Construct report
        report = {
            'id': ID,
            'weights': w,
            'loss': loss,
            'gamma': gamma,
            'lambda_': lambda_,
            'max_iters': train_max_iters,
            'initial_w': initial_w,
            'expansion': expansion,
        }

        perf_train = calculate_performance(w, fx_train, y_train)
        perf_test = calculate_performance(w, fx_test, y_test)
        report.update({
            'train_perf': perf_train,
            'test_perf': perf_test
        })
        ID = ID +1
        return report
    








def mutation(individua):
      """
    Increases degree at randomly chosen index with MUTATION probability

    Parameters
    ----------
    individua: np.Array
        Array of degrees used in expansion
    Returns
    -------
    void
      """
    ind = rand.randint(0,len(individua)-1)
    if rand.random()< MUTATION_PROBABILITY:
        individua['expansion'][ind] += 1

def crossing_over(individua1,individua2):
      """
    Creates 2 new individuas by concatenating subparts of selected ones

    Parameters
    ----------
    individua1: np.Array
        First selected individua
    individua2: np.Array
        Second selected individua
        
    Returns
    -------
    list
        List of 2 created individuas
    """
    ind = rand.randint(4,len(individua1)-1)
    new_individua1 = np.concatenate([individua2[ind:] ,individua1[:ind]])
    new_individua2 = np.concatenate([individua1[ind:] ,individua2[:ind]])
    new_individuals= [new_individua1, new_individua2]
    return new_individuals

def selection(population):
       """
    Sorts initial population descending by cost_f and perserves 20% of initial population

    Parameters
    ----------
    population: np.Array
       Initial population 
       
        
    Returns
    -------
    np.Array
        Best 20% of the initial population by given cost_f criteria
    """
    population.sort(reverse=True, key=cost_f)
    return population[0:SELECTION_PARAMETER]

def cost_f(individua):
    """
    Evaluates quality of individua
    
    Parameters
    ----------
    individua: np.Array 
        
    Returns
    -------
    float 
        Numerical estimate of individua
    """
    
    return individua['accuracy']
    

def initial_population(max_degree, size, train, test, model_num):
    '''
    Creates initial population and evaluates each individua.
    Individua is array of degrees used in polynomial feature expansion
    
    Parameters
    ----------
    max_degree: int
        Maximum degree in feature expansion
    size: int
        Number of non-augmented features
    train: tuple
        Training dataset
    test: tuple
        Testing dataset
    
        
    Returns
    -------
    list 
        Initial population
    '''
    initial_pop=[]
    for i in range(POPULATION_PARAMETER):
        expansion = np.random.randint(0,high = max_degree, size = size)
        individua = {'accuracy': evaluate_expansion(train = train, test =test, expansion = expansion, model_num= model_num)['test_perf']['accuracy'],
                 'expansion':expansion}
        initial_pop.append(individua)
    return  initial_pop



def genetic_algo(train_, test_, model_num_, max_degree_ = 10):
     '''
    Optimization algorithm which finds best degree-array for polynomial feature expansion
    
    Parameters
    ----------
    train_: tuple
        Training dataset
    test_: tuple
        Testing dataset
    model_num_: int
        Index of training method in MODELS array
        MODELS = [least_squares_GD, least_squares_SGD, least_squares,
          ridge_regression, logistic_regression,
          reg_logistic_regression]
        
    max_degree_: int
       Maximum degree in degree-array
    
        
    Returns
    -------
    dict 
        See documentation for evaluate_expansion
    '''
    global train 
    global test
    global model_num
    model_num= model_num_
    test = test_
    train = train_
    
    population = initial_population(max_degree= max_degree_ , size = train_[0].shape[1], train= train, test= test,model_num= model_num)
    for generation in range(MAX_GENERATIONS):
        population = selection(population)


        while len(population)<POPULATION_PARAMETER: #fill rest of population

            ind1 = rand.randint(0, len(population)-1)
            ind2 = rand.randint(0, len(population)-1)
            childreen = crossing_over(individua1=population[ind1]['expansion'], individua2=population[ind2]['expansion'])
            
            entry1 = {'accuracy': evaluate_expansion(train = train, test =test, expansion = childreen[0], model_num= model_num)['test_perf']['accuracy'],
                      'expansion':childreen[0]}
            entry2 = {'accuracy': evaluate_expansion(train = train, test =test, expansion = childreen[1], model_num= model_num)['test_perf']['accuracy'],
                      'expansion':childreen[1]}
            
            population.append(entry1)
            population.append(entry2)
           
       
        for individua in population:
            mutation(individua)

       
    population.sort(reverse=True , key=cost_f)
    print('best')
    best = evaluate_expansion(train = train, test =test, expansion = population[0]['expansion'],model_num = model_num_)
    print(best)
    return best




