import numpy as np

PRI_JET_NUM = 22
NaN_VALUE = -999

REPLACMENT_METHOD = np.median  #np.mean #test out which is best for NN

MEANINGLESS_COLS_BY_JET_NUM = {
    0: [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28,29],
    1: [4, 5, 6, 12, 22, 26, 27, 28],
    2: [22],
}

def replace_with_zeros(features):
    return 0

def get_data_indexes_by_jet_num(data):
    """
    Get the indexes of data with pri_jet_num

    This function returns three arrays of indexes
    that split the data based on pri_jet_num.

    Parameters
    ----------
    data: np.Array
        Matrix that has PRI_JET_NUM column
    Returns
    -------
    list
        List of three np.Arrays corresponding to
        the three groups of indexes split by different
        values of pri_jet_num
    """
    idx_jet_num_0 = np.where(data[:, PRI_JET_NUM] == 0)
    idx_jet_num_1 = np.where(data[:, PRI_JET_NUM] == 1)
    idx_jet_num_23 = np.where(data[:, PRI_JET_NUM] >= 2)
    
    return [idx_jet_num_0, idx_jet_num_1, idx_jet_num_23]

def split_dataset_by_jet_num(tX, y, remove_cols=False):
    """
    Split the dataset w.r.t. pri_jet_num
    
    This function splits the dataset into three subsets
    based on the pri_jet_num column and optionally removes
    meaningless columns. 

    Based on the pri_jet_num categorical value some 
    features become meaningless.

    If:
        pri_jet_num = 0   --> 11 features are meaningless
        pri_jet_num = 1   --> 8 features are meaningless
        pri_jet_num = 2,3 --> 1 feature is meaningless (pri_jet_num)
    
    Parameters
    ----------
        tX: np.Array
            input data
        y: np.Array
            output data
        remove_cols: bool
            Flag for optionally deleting meaningless columns
        
    Returns
    -------
        list
            three subsets of tX = [tX_0, tX_1, tX_23]
        list
            three subsets of y  = [y_0, y_1, y_23]
        np.Array
            indexes of original data split by pri_jet_num
    """

    idx_by_jet_num = get_data_indexes_by_jet_num(tX)
    
    tX_split = []
    y_split = []
    
    for i, idx_subset in enumerate(idx_by_jet_num):
        tX_subset = tX[idx_subset]
        y_subset = y[idx_subset]
        
        if remove_cols:
            tX_subset = np.delete(tX_subset, MEANINGLESS_COLS_BY_JET_NUM[i], 1)
            
        tX_split.append(tX_subset)
        y_split.append(y_subset)
    
    return tX_split, y_split, idx_by_jet_num

def replace_feature_NaN_values(x, replacment_method):
    """
    Replace Nan Values in feature column

    Returns cleaned feature column using the given 
    replacment method on feature x.
    The replacing method ignores NaN values during
    calculation of replacement value.

    Parameters
    ----------
        x: np.Array
            Feature column with NaN values
        replacing_method: function
            Function to be used for replacements

    Returns
    -------
    np.Array
        Feature column with NaN values replaced
    """
    x_non_nan = x[x != NaN_VALUE]
    feature_cleaned = np.copy(x)
    
    feature_cleaned[feature_cleaned == NaN_VALUE] = replacment_method(x_non_nan)
    
    return feature_cleaned

def remove_outliers(tX, y, k=3, sigma_rule=True):
    """
    Removes outliers from given data. For the outliers detection the Tukey's fences method is used. 
    A value is considered to be an outlier if it is outside the range [Q1 - k * (Q3 - Q1), Q3 + k * (Q3 - Q1)].
    
    Optionally, in case of sigma_rule=True instead of Tukey's fences method the k-sigma rule will be used
    to determine outliers (valid range = [mean - k * std, mean + k * std])

    Parameters
    ----------
        tX: np.Array
            input data matrix
        y: np.Array
            output data vector
        k: int
            range constant
        sigma_rule: bool
            if true the k-sigma rule will be used instead of the Tukey's fences method

    Returns
    -------
    (tX_acceptable, y_acceptable)
        Input and output data points with removed outliers
    """
    if sigma_rule:
        mean = np.mean(tX, axis=0)
        std = np.std(tX, axis=0)
        acceptable_idxs = ((tX >= mean - k * std) & (tX <= mean + k * std)).all(axis=1)
    
    else:
        Q1, Q3 = np.quantile(tX, [0.25, 0.75], axis=0) 
        IQR = Q3 - Q1
        acceptable_idxs = ((tX >= Q1 - k * IQR) & (tX <= Q3 + k * IQR)).all(axis=1)
    
    print("After removing outliers we are left with only: ", tX[acceptable_idxs].shape, " out of: ", tX.shape)
    tX_acceptable = tX[acceptable_idxs]
    y_acceptable = y[acceptable_idxs] 
    
    return tX_acceptable, y_acceptable
        

def clean_data(data, replacement_method= REPLACMENT_METHOD, fill_outliers=False):
    """
    Returns cleaned data. 
    
    Cleans the data matrix by column-wise replacement of NaN values.
    
    Optionally this function can also fill outliers. A value is considered to be an outlier
    if it is outside the range [mean - 3 * std, mean + 3 * std] (mean and std are in respect to feature).
    This function will fill the outliers with the lower/upper bound of the mentioned range.

    Parameters
    ----------
        data: np.Array
            data matrix
        replacement_method: function
            function which maps non-NaN features to desired output
        fill_outliers: bool
            flag for optionally filling the outliers

    Returns
    -------
    np.Array
        Data with NaN values replaces in all feature columns
    """
    data_cleaned = np.copy(data)
    feature_num = data.shape[1]
    
    for feature in range(feature_num):
        feature_values = data[:, feature]
        
        feature_values = replace_feature_NaN_values(feature_values, replacement_method)
        
        if fill_outliers:
            std = np.std(feature_values)
            mean = np.mean(feature_values)
            
            feature_values[feature_values < mean - 3 * std] = mean - 3 * std
            feature_values[feature_values > mean + 3 * std] = mean + 3 * std
        
        data_cleaned[:, feature] = feature_values
    
    return data_cleaned

def extract_PCA_components(numComponents, data):
    """
    Extract specified number of PCA components


    Parameters
    ----------
    numComponents : int
        Number of components to extract. 
        Must not be bigger than number of columns in data
    data : np.Array
        Data Matrix

    Returns
    -------
    np.Array
        Principle Components
    """
    assert numComponents <= data.shape[1], "Number of components must not be bigger than number of columns in data"
    covarianceMatrix = np.cov(data.T)
    # eigenvalues are sorted by default
    eigenvalues , eigenvectors = np.linalg.eig(covarianceMatrix)
    componentsPCA = eigenvectors.T[:numComponents].dot(data.T)
    return -componentsPCA.T
    
def standardize_data(data, data_means=None, data_stds=None):
    """
    Standardize data

    Parameters
    ----------
    data: np.Array
        Data Matrix
    data_means : np.Array
        Standardize column using given means
    data_stds : np.Array
        Standardize column using given stds

    Returns
    -------
    np.array
        Standardized data
    """
    standardized_data = []
    means = []
    stds =[]
    for columnNumber in range(data.shape[1]):
        column = data[:, columnNumber]
        
        mean = -1
        std = -1
        if(data_means is None):
            mean = np.mean(column)
            std = np.std(column)
        else :
            mean = data_means[columnNumber]
            std = data_stds[columnNumber]
        
        means.append(mean)
        stds.append(std)
        standardized_data.append( (column-mean)/std )
    
    return np.array(standardized_data).T, np.array(means), np.array(stds)
        
def split_data(tX, y, ratio):
    '''
    Splits data into training and test sets by given ratio

    Parameters
    ----------
    tX : np.Array 
        Feature matrix
    y : np.Array
        Class labels
    ratio : float
        Ratio by which to split

    Returns
    -------
    (tX_train, y_train)
        Train set touple for features and labels
    (tX_test, y_test)
        Test set touple for features and labels
    '''
    ratio = int(ratio* tX.shape[0])
    indices = np.random.permutation(tX.shape[0])
    train_indices = indices[:ratio]
    test_indices = indices[ratio:]
    
    tX_train  = tX[train_indices]
    y_train = y[train_indices]
    
    tX_test  = tX[test_indices]
    y_test = y[test_indices]
    
    return (tX_train, y_train) , (tX_test, y_test)