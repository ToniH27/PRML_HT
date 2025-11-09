# Source of feature extraction idea: https://github.com/yogeshgajjar/logistic_regression/blob/master/logistic_regression_classification.ipynb
import numpy as np
import pandas as pd

# Function takes the  and labels as input and forms a new pandas dataset 
def feature_extract(X_data, y_data):

    new_X_train = []
    # Extracting features from the previous training data and making a simpler dataset with class labels
    for i in range(len(X_data)):

        x = X_data[i]
        sum = x.sum()
        mean = x.mean()
        median = np.median(x)
        std = x.std()
        max = x.max()
        min = x.min()

        X = [sum, mean, median, std, max, min]
        new_X_train.append(X)



    # New dataset
    y_data = np.array(y_data).reshape(-1, 1)
    dataset = np.array(new_X_train)
    dataset = np.hstack((dataset, y_data))
    data = pd.DataFrame(dataset)
    data.columns = ['sum', 'mean','median', 'std', 'max', 'min', 'class'] # Set column names for plotting

    return data