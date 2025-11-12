# Sources of feature extraction ideas and some example codes:
# https://github.com/yogeshgajjar/logistic_regression/blob/master/logistic_regression_classification.ipynb
# https://www.geeksforgeeks.org/data-analysis/feature-engineering-for-time-series-data-methods-and-applications/#1-statistical-features
# https://towardsdatascience.com/applications-of-rolling-windows-for-time-series-with-python-1a4bbe44901d/


import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy import stats
from scipy.fftpack import fft

# Function takes the time series data and labels as input and forms a new pandas dataset 
def feature_extract(X_data, y_data):

    new_X_train = []
    # Extracting features from the time series data and making a simpler dataset with class labels
    for i in range(len(X_data)):

        X = [] # Feature sample list

        x = X_data[i] # Time series data sample
        x_len = len(x) # Length of the sample for indexing

        # X.append(np.mean(x[:x_len//5]))
        # X.append(np.mean(x[x_len//5:2*x_len//5]))
        # X.append(np.mean(x[2*x_len//5:3*x_len//5]))
        # X.append(np.mean(x[3*x_len//5:4*x_len//5]))
        # X.append(np.mean(x[4*x_len//5:-1]))

        # X.append(np.std(x[:x_len//5], ddof=1))
        # X.append(np.std(x[x_len//5:2*x_len//5], ddof=1))
        # X.append(np.std(x[2*x_len//5:3*x_len//5], ddof=1))
        # X.append(np.std(x[3*x_len//5:4*x_len//5], ddof=1))
        # X.append(np.std(x[4*x_len//5:-1], ddof=1))

        # X.append(np.max(x[:x_len//5]) - np.min(x[:x_len//5]))
        # X.append(np.max(x[x_len//5:2*x_len//5]) - np.min(x[x_len//5:2*x_len//5]))
        # X.append(np.max(x[2*x_len//5:3*x_len//5]) - np.min(2*x[x_len//5:3*x_len//5]))
        # X.append(np.max(x[3*x_len//5:4*x_len//5]) - np.min(3*x[x_len//5:4*x_len//5]))
        # X.append(np.max(x[4*x_len//5:-1]) - np.min(x[4*x_len//5:-1]))

        # Extract means of X in 8 subintervals
        X.append(np.mean(x[:x_len//8]))
        X.append(np.mean(x[x_len//8:2*x_len//8]))
        X.append(np.mean(x[2*x_len//8:3*x_len//8]))
        X.append(np.mean(x[3*x_len//8:4*x_len//8]))
        X.append(np.mean(x[4*x_len//8:5*x_len//8]))
        X.append(np.mean(x[5*x_len//8:6*x_len//8]))
        X.append(np.mean(x[6*x_len//8:7*x_len//8]))
        X.append(np.mean(x[7*x_len//8:-1]))

        # Extract stds of X in 8 subintervals
        X.append(np.std(x[:x_len//8], ddof=1))
        X.append(np.std(x[x_len//8:2*x_len//8], ddof=1))
        X.append(np.std(x[2*x_len//8:3*x_len//8], ddof=1))
        X.append(np.std(x[3*x_len//8:4*x_len//8], ddof=1))
        X.append(np.std(x[4*x_len//8:5*x_len//8], ddof=1))
        X.append(np.std(x[5*x_len//8:6*x_len//8], ddof=1))
        X.append(np.std(x[6*x_len//8:7*x_len//8], ddof=1))
        X.append(np.std(x[7*x_len//8:-1], ddof=1))

        # Extract max-mins of X in 5 subintervals
        X.append(np.max(x[:x_len//5]) - np.min(x[:x_len//5]))
        X.append(np.max(x[x_len//5:2*x_len//5]) - np.min(x[x_len//5:2*x_len//5]))
        X.append(np.max(x[2*x_len//5:3*x_len//5]) - np.min(2*x[x_len//5:3*x_len//5]))
        X.append(np.max(x[3*x_len//5:4*x_len//5]) - np.min(3*x[x_len//5:4*x_len//5]))
        X.append(np.max(x[4*x_len//5:-1]) - np.min(x[4*x_len//5:-1]))

        # Does not seem to improve accuracy. Probably because the intervals are already so small, so each max-min is probably very close to each other
        # X.append(np.max(x[:x_len//8]) - np.min(x[:x_len//8]))
        # X.append(np.max(x[x_len//8:2*x_len//8]) - np.min(x[x_len//8:2*x_len//8]))
        # X.append(np.max(x[2*x_len//8:3*x_len//8]) - np.min(x[2*x_len//8:3*x_len//8]))
        # X.append(np.max(x[3*x_len//8:4*x_len//8]) - np.min(x[3*x_len//8:4*x_len//8]))
        # X.append(np.max(x[4*x_len//8:5*x_len//8]) - np.min(x[4*x_len//8:5*x_len//8]))
        # X.append(np.max(x[5*x_len//8:6*x_len//8]) - np.min(5*x[x_len//8:6*x_len//8]))
        # X.append(np.max(x[6*x_len//8:7*x_len//8]) - np.min(6*x[x_len//8:7*x_len//8]))
        # X.append(np.max(x[7*x_len//8:-1]) - np.min(x[7*x_len//8:-1]))


        # Extract kurtosis of each column
        kurtosis = stats.kurtosis(x)
        # X.extend(kurtosis)
        # print(f'kurtosis: {kurtosis}')
        # Kurtosis values does not seem to help with the classification. My guess is that the values are too correlated with other features or too similar to each other for some samples

        # Extract skewness of each column
        skewness = skew(x)
        X.extend(skewness)
        # print(f'skewness: {skewness}')
        # Skewness values seem to help with the classification. I'm not sure if data splitting like for the mean, std, etc. is a good idea for skewness
 

        fft_x = fft(x) # Fast Fourier transform
        # print(fft_x)
        x_freqs = np.fft.fftfreq(len(fft_x)) # Frequencies
        # print(x_freqs)
        psd = np.abs(x_freqs)**2

        max_freq = x_freqs.max()
        # X.append(max_freq)

        # Top 5 peak frequencies
        max_freqs = x_freqs[np.argsort(psd)[-5:]]
        # print(max_freq)
        # All the columnn and sample frequencies are very close to each other, so not giving much information. I think that the given time series data is not ideal for FFT feature extraction
 
        # Add a new feature sample to the new dataset
        new_X_train.append(X)



    # New dataset
    y_data = np.array(y_data).reshape(-1, 1) # Reshape to fit the dataset
    dataset = np.array(new_X_train)
    dataset = np.hstack((dataset, y_data))
    data = pd.DataFrame(dataset)


    # Set column names for clarity
    # data.columns = ['mean', 'std', 'max-min', 'class']
    # data.columns = ['mean1','mean2', 'mean3', 'mean4', 'mean5', 'std1', 'std2', 'std3', 'std4','std5', 'maxmin1', 'maxmin2', 'maxmin3','maxmin4','maxmin5', 'kurtosis1', 'kurtosis2', 'kurtosis3', 'class']
    # data.columns = ['mean1','mean2', 'mean3', 'mean4', 'mean5', 'mean6', 'mean7', 'mean8', 'std1', 'std2', 'std3', 'std4','std5','std6','std7','std8', 'maxmin1', 'maxmin2', 'maxmin3','maxmin4','maxmin5','maxmin6','maxmin7','maxmin8', 'skewness1', 'skewness2', 'skewness3', 'class']
    data.columns = ['mean1','mean2', 'mean3', 'mean4', 'mean5', 'mean6', 'mean7', 'mean8', 'std1', 'std2', 'std3', 'std4','std5','std6','std7','std8', 'maxmin1', 'maxmin2', 'maxmin3','maxmin4','maxmin5', 'skewness1', 'skewness2', 'skewness3', 'class']
    # data.columns = ['mean1','mean2', 'mean3', 'mean4', 'mean5', 'mean6', 'mean7', 'mean8', 'std1', 'std2', 'std3', 'std4','std5', 'maxmin1', 'maxmin2', 'maxmin3','maxmin4','maxmin5', 'skewness1', 'skewness2', 'skewness3', 'class']
    # data.columns = ['mean1','mean2', 'mean3', 'mean4', 'mean5', 'std1', 'std2', 'std3', 'std4','std5', 'maxmin1', 'maxmin2', 'maxmin3','maxmin4','maxmin5', 'skewness1', 'skewness2', 'skewness3', 'class']
    # data.columns = ['mean1','mean2', 'mean3', 'mean4', 'mean5', 'std1', 'std2', 'std3', 'std4','std5', 'maxmin1', 'maxmin2', 'maxmin3','maxmin4','maxmin5', 'kurtosis1', 'kurtosis2', 'kurtosis3', 'skewness1', 'skewness2', 'skewness3', 'class']
    # data.columns = ['maxmin1','maxmin2', 'maxmin3', 'class']
    # data.columns = ['std1','std2', 'std3', 'class']
    # data.columns = ['mean1','mean2', 'mean3', 'std1', 'std2', 'std3', 'max-min1', 'max-min2', 'max-min3', 'class']
    # data.columns = ['mean', 'std', 'max', 'min', 'kurtosis_1', 'kurtosis_2', 'kurtosis_3', 'skewness_1', 'skewness_2', 'skewness_3', 'max_freq', 'class'] # Set column names for plotting
    # data.columns = ['mean', 'std', 'max', 'min', 'max_freq1', 'max_freq2', 'max_freq3', 'max_freq4', 'max_freq5', 'class'] # Set column names for plotting

    return data