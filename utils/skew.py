import numpy as np

# Formulas for biased skewness values.
# https://en.wikipedia.org/wiki/Skewness
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html

# Function computes biased skewness values
def skew(x):
    """
    Function computes skewness values of each data column in numpy matrix array and returns the values as a numpy array

    x: numpy dim(N, M) matrix array
    returns: numpy (1, M) array
    """
    N = x.shape[0] # Sample size
    x_mean = np.mean(x, axis=0)
    m3 = np.sum((x - x_mean)**3, axis=0) / N # the biased sample third central moment.
    m2 = np.sum((x - x_mean)**2, axis=0) / N # the biased sample second central moment
    g1 = m3 / np.pow(m2, 3/2) # g1 = m_3 / m2^(3/2), 

    return g1