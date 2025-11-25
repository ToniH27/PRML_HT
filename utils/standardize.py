# Idea of standardization was originally from PCA, but then built this utility function based from it. Sources of example codes and theory below:
# https://en.wikipedia.org/wiki/Principal_component_analysis
# https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491


import numpy as np

def standardize_data(X_data):

    X_data_std = [] # Standardized data

    for i in range(len(X_data)):


        X = X_data[i]
    

        # Standardization Z = (X-mu) / sigma for each column
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=1) # Sample standard deviation
        Z = (X - mu) / sigma
        X_data_std.append(Z)

    return X_data_std # Returns the standardized data as a numpy array