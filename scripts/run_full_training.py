from proj1_helpers import *
from implementations import *
from preprocessing import *
from visualization import *
from featurization import *
from hyperoptim import *

DATA_TRAIN_PATH = '../data/train.csv' 
DATA_TEST_PATH = '../data/test.csv'  
OUTPUT_PATH = './least_squeares21.csv' 

#defining seed so that every script execution returns same predictions
np.random.seed(1234)
def print_fscore(keep, keep_obj):
    print("FScore Test: " + str(keep_obj['test_perf']['fscore']))

'''
TRAINING PROCESS
'''

#load data
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

#extract data for testing
ratio = 0.1
ratio = int(0.1*tX.shape[0])
indices = np.random.permutation(tX.shape[0])

tX_test = tX[indices[:ratio]]
y_test  = y[indices[:ratio]]

tX = tX[indices[ratio:]]
y  = y[indices[ratio:]]


#split data by pri_jet_num
tX_split, y_split, idx_jet_num = split_dataset_by_jet_num(tX, y, remove_cols=True)
tX_0 = tX_split[0]
tX_1 = tX_split[1]
tX_2 = tX_split[2]

#clean data (fill NaNs, transform outliers)
tX_0 = clean_data(tX_0, fill_outliers=False)
tX_1 = clean_data(tX_1, fill_outliers=False)
tX_2 = clean_data(tX_2, fill_outliers= False)

y0 = y_split[0]
y1 = y_split[1]
y2 = y_split[2] 


#standardize data
tX_0, tX_0_means, tX_0_stds  = standardize_data(tX_0)
tX_1, tX_1_means, tX_1_stds  = standardize_data(tX_1)
tX_2, tX_2_means, tX_2_stds  = standardize_data(tX_2)

#split on train and validation sets 
ratio = 0.7
train0, validate0 = split_data(tX_0, y0, ratio)
train1, validate1 = split_data(tX_1, y1, ratio)
train2, validate2 = split_data(tX_2, y2, ratio)


#training models using random search
#note that np.random.seed is preset, so random search will always return same best models
best10_0=hp_random_search(train0[0], train0[1], validate0[0], validate0[1], gamma = 1e-4, model_num=2, keep_callback=print_fscore,search_max_iters= 200, polynom_exp=10)
weights0 = best10_0[0]['weights']
expansion0 = best10_0[0]['expansion']

best10_1=hp_random_search(train1[0], train1[1], validate1[0], validate1[1], gamma = 1e-4, model_num=2, keep_callback=print_fscore,search_max_iters= 200, polynom_exp=10)
weights1 = best10_1[0]['weights']
expansion1 = best10_1[0]['expansion']

best10_2=hp_random_search(train2[0], train2[1], validate2[0], validate2[1], gamma = 1e-4, model_num=2, keep_callback=print_fscore,search_max_iters= 200, polynom_exp=10)
weights2 = best10_2[0]['weights']
expansion2 = best10_2[0]['expansion']



#evaluation on test set

tX_split_test, y_split_test, idx_jet_num = split_dataset_by_jet_num(tX_test,y_test, remove_cols=True)
tX_0_test = tX_split_test[0]
tX_1_test = tX_split_test[1]
tX_2_test = tX_split_test[2]


tX_0_test = clean_data(tX_0_test)
tX_1_test = clean_data(tX_1_test)
tX_2_test = clean_data(tX_2_test)

#means and std are learned on training data
tX_0_test, _ , _  = standardize_data(tX_0_test, tX_0_means, tX_0_stds)
tX_1_test, _ , _   = standardize_data(tX_1_test, tX_1_means, tX_1_stds)
tX_2_test, _ , _   = standardize_data(tX_2_test, tX_2_means, tX_2_stds)

tX_0_test = polynomial_feature_expansion(tX_0_test, expansion0)
tX_1_test = polynomial_feature_expansion(tX_1_test, expansion1)
tX_2_test = polynomial_feature_expansion(tX_2_test, expansion2)

print(calculate_performance(weights0, tX_0_test, y_split_test[0]))
print(calculate_performance(weights1, tX_1_test, y_split_test[1]))
print(calculate_performance(weights2, tX_2_test, y_split_test[2]))



'''
PREDICTING test.csv DATA
'''


#load data
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

#spliting data by pri_jet_num
tX_split_test, y_split, idx_jet_num = split_dataset_by_jet_num(tX_test,np.ones(tX_test.shape[0]), remove_cols=True)
tX_0_test = tX_split_test[0]
tX_1_test = tX_split_test[1]
tX_2_test = tX_split_test[2]

#cleaning data
tX_0_test = clean_data(tX_0_test)
tX_1_test = clean_data(tX_1_test)
tX_2_test = clean_data(tX_2_test)


#means and std are learned on training data
#standardazing data
tX_0_test, _ , _  = standardize_data(tX_0_test, tX_0_means, tX_0_stds)
tX_1_test, _ , _   = standardize_data(tX_1_test, tX_1_means, tX_1_stds)
tX_2_test, _ , _   = standardize_data(tX_2_test, tX_2_means, tX_2_stds)


#polynomial feature expansion
#expansions are learned in training process
tX_0_test = polynomial_feature_expansion(tX_0_test, expansion0)
tX_1_test = polynomial_feature_expansion(tX_1_test, expansion1)
tX_2_test = polynomial_feature_expansion(tX_2_test, expansion2)

#predicting
y0_test = predict_labels(weights0,tX_0_test)
y1_test = predict_labels(weights1,tX_1_test)
y2_test = predict_labels(weights2,tX_2_test)

#rearranging ids to correspond to y array
y_pred = np.concatenate([y0_test, y1_test,y2_test],axis=0)
indices = np.concatenate(list(map(lambda x: x[0], idx_jet_num)),axis=0)
rearranged_ids = ids_test[indices]

#creating submission .csv
create_csv_submission(rearranged_ids, y_pred, OUTPUT_PATH)