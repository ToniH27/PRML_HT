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

        # Extract means of X in 8 subintervals
        # X.append(np.mean(x[:x_len//8]))
        # X.append(np.mean(x[x_len//8:2*x_len//8]))
        X.append(np.mean(x[2*x_len//8:3*x_len//8]))
        X.append(np.mean(x[3*x_len//8:4*x_len//8]))
        X.append(np.mean(x[4*x_len//8:5*x_len//8]))
        X.append(np.mean(x[5*x_len//8:6*x_len//8]))
        X.append(np.mean(x[6*x_len//8:7*x_len//8]))
        # X.append(np.mean(x[7*x_len//8:-1]))


        # Extract stds of X in 8 subintervals
        # X.append(np.std(x[:x_len//8], ddof=1))
        X.append(np.std(x[x_len//8:2*x_len//8], ddof=1))
        X.append(np.std(x[2*x_len//8:3*x_len//8], ddof=1))
        X.append(np.std(x[3*x_len//8:4*x_len//8], ddof=1))
        X.append(np.std(x[4*x_len//8:5*x_len//8], ddof=1))
        X.append(np.std(x[5*x_len//8:6*x_len//8], ddof=1))
        X.append(np.std(x[6*x_len//8:7*x_len//8], ddof=1))
        # X.append(np.std(x[7*x_len//8:-1], ddof=1))

        # Extract max-mins of X in 5 subintervals
        X.append(np.max(x[:x_len//5]) - np.min(x[:x_len//5]))
        X.append(np.max(x[x_len//5:2*x_len//5]) - np.min(x[x_len//5:2*x_len//5]))
        X.append(np.max(x[2*x_len//5:3*x_len//5]) - np.min(2*x[x_len//5:3*x_len//5]))
        X.append(np.max(x[3*x_len//5:4*x_len//5]) - np.min(3*x[x_len//5:4*x_len//5]))
        # X.append(np.max(x[4*x_len//5:-1]) - np.min(x[4*x_len//5:-1]))

        # Increasing number of max-mins subintervals does not seem to improve accuracy. Probably because the intervals are already so small, so each max-min is probably very close to each other



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
 

        # fft_x = fft(x) # Fast Fourier transform
        fft_x = np.fft.fft(x, axis=0)
        psd = np.abs(fft_x)**2
        psd_norm = psd / np.sum(psd, axis=0, keepdims=True)
        psd_entropy = stats.entropy(psd_norm, axis=0)
        # print(psd_entropy)
        # print(fft_x)
        x_freqs = np.fft.fftfreq(len(fft_x)) # Frequencies
        # print(x_freqs)
 

        max_freq = x_freqs.max()
        # X.append(max_freq)

        # Top 5 peak frequencies
        max_freqs = x_freqs[np.argsort(psd)[-5:]]
        # print(max_freq)
        # All the columnn and sample frequencies are very close to each other, so not giving much information. I think that the given time series data is not ideal for FFT feature extraction
    
        # Area under the curve features
        auc1 = np.trapezoid(x[:,1], x[:, 0], dx=5)
        # auc2 = np.trapezoid(x[:,2], x[:, 1], dx=5)
        auc3 = np.trapezoid(x[:,0], x[:, 2], dx=5)
        # print(auc1)
            
        
        X.append(auc1) 
        # X.append(auc2) 
        X.append(auc3)

        centroid = np.mean(x, axis=0)
        # print(centroid)
        X.append(centroid[0]) # xmean

        # Slopes
        slope = np.mean(np.diff(x[:,2]))
        # print(slope)
        # X.append(slope)

        # Length
        # X.append(x_len)


        # IQR
        p25 = np.percentile(x, 25, axis=0)
        p75 = np.percentile(x, 75, axis=0)
        iqr = p75 - p25
        # print(iqr)
        # X.extend(iqr[:1])


        rms = np.sqrt(np.mean(x**2, axis=0))
        # print(rms)
        # X.append(rms[0])

        # X.extend(psd_entropy[0::2])
        var = np.var(x, axis=0)
        # print(var)
        # X.extend(var[:1])

        # entropy = stats.entropy(x, axis=0)
        # print(entropy)

        X.append(np.trapezoid(x[:x_len//3, 1], x[:x_len//3, 0], dx=5))
        # X.append(np.trapezoid(x[:x_len//3, 0], x[:x_len//3, 2], dx=5))
        # X.append(np.trapezoid(x[x_len//3:2*x_len//3, 1], x[x_len//3:2*x_len//3, 0], dx=5))
        # X.append(np.trapezoid(x[x_len//3:2*x_len//3, 0], x[x_len//3:2*x_len//3, 2], dx=5))
        # X.append(np.trapezoid(x[2*x_len//3:-1, 1], x[2*x_len//3:-1, 0], dx=5))
        # X.append(np.trapezoid(x[2*x_len//3:-1, 0], x[2*x_len//3:-1, 2], dx=5))

        # aucs = [auc_1, auc_2, auc_3, auc_4, auc_5, auc_6]
        # X.extend(aucs[:])

        # Add a new feature sample to the new dataset
        new_X_train.append(X)



    # New dataset
    y_data = np.array(y_data).reshape(-1, 1) # Reshape to fit the dataset
    dataset = np.array(new_X_train)
    dataset = np.hstack((dataset, y_data))
    data = pd.DataFrame(dataset)


    # Set column names for seaborn and data splitting
  
    # data.columns = ['mean3', 'mean4', 'mean5', 'mean6', 'mean7', 'std2', 'std3', 'std4','std5','std6','std7', 'maxmin1', 'maxmin2', 'maxmin3','maxmin4', 'skewness1', 'skewness2', 'skewness3', 'auc1', 'auc3', 'xmean', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'class']
    data.columns = ['mean3', 'mean4', 'mean5', 'mean6', 'mean7', 'std2', 'std3', 'std4','std5','std6','std7', 'maxmin1', 'maxmin2', 'maxmin3','maxmin4', 'skewness1', 'skewness2', 'skewness3', 'auc1', 'auc3', 'xmean', 'a1', 'class']


    return data