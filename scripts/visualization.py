import numpy as np
import matplotlib.pyplot as plt
def scatter_plot_PCA_components(pca, y):
    '''
    Plots first and second PCA component on scatter plot
    Parameters:
    -----------
    pca : list or touple
        PCA components, there must be >= 2 components
    y: np.Array
        Data labels
    '''
    colors = ['#ff2110' if label == 1 else '#1345ff' for label in y]
    
    plt.figure(figsize=(20,10))
    plt.scatter(pca[:,0], pca[:,1], c=colors)
    plt.show()