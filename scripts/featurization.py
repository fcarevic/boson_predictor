import numpy as np

def polynomial_feature_expansion(feats, expansion):
    """
    Expand/remove features for a given degrees
   
    This function converts a given dataset into an expanded
    one by expanding each feature of the feats array
    for to the degree given by the expansion array.
    If the expansion degree is zero then the feature is
    removed altogether.
    
    
    Parameters
    ----------
    feats : np.Array
        The input dataset of shape (N, F)
        N = number of data points
        F = number of features per data point
    expansion : np.Array
        The expansion degrees. Namely, it is of
        shape (N, F) and the values are integers
        corresponding to the largest degree of the polynomial
        expansion.

    Returns
    -------
    np.Array
        A new and expanded dataset with the same number of
        data points, but with a different number of features.
    """
    expansion = expansion.copy()
    feats_sets = []
    e = np.zeros(expansion.shape)
    while expansion.sum() > 0:
        e = np.minimum(e + 1, expansion)
        feats_sets.append((feats**e)[:, e>0])
        expansion *= (e!=expansion)
    return np.concatenate(feats_sets, axis=1)



def expand_one_hot(feats, feat_ind, num_of_classes):
    """
    Convert an categorical feature into  
    a one-hot representation
    
    Parameters
    ----------
    feats : np.Array
        The input dataset of shape (N, F)
        N = number of data points
        F = number of features per data point
    feat_ind : integer
        the index of the categorical feature
        that will be converted into a one-hot
        representation
    num_of_classes : integer
        the number of categories that the
        ordinal feature encompasses.
        i.e. the number of new features that
        will be used for the one-hot representation

    Returns
    -------
    np.Array
        A new dataset with the same number of
        data points, but with a different number 
        of features due to the conversion to 
        a one-hot feature.
    """
    expansion = np.array(feats[:,feat_ind], dtype=np.int32)[:, np.newaxis] == np.arange(0, num_of_classes)[np.newaxis, :]
    expansion = np.array(expansion, dtype=np.float)
    return np.concatenate([feats[:, :feat_ind], expansion, feats[:, feat_ind+1:]], axis=1)