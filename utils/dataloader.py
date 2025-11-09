# Code sources:
# https://stackoverflow.com/questions/33503993/read-in-all-csv-files-from-a-directory-using-python
# https://stackoverflow.com/questions/5137497/find-the-current-directory-and-files-directory
# https://stackoverflow.com/questions/9572490/find-index-of-last-occurrence-of-a-substring-in-a-string


import numpy as np
import os
import glob

# Loads data into numpy X_data matrix and y_data vector. Data folder needs to be at the same level as the utility folder
def load_data(datatype):

    X_data = []
    y_data = []

    # Directory path to the data
    directory_path = os.path.abspath(os.path.join(os.getcwd(),'..',datatype))

    # print(path)
    print(directory_path)

    for filename in glob.glob(os.path.join(directory_path, '*.csv')):

        # print(filename.rindex('_0'))
        # print(filename[filename.rindex('_0')-1])

        # Extracting the class labels from the filenames
        y = int(filename[filename.rindex('_0')-1])
        y_data.append(y)

        # Read the csv data file and convert into numpy array
        x = np.genfromtxt(filename, delimiter=',')
        X_data.append(x)


    # X_data = np.array(X_data[0])
    # X_data
    # print(X_data.shape)

    # Checking classes and their counts
    values, counts = np.unique(y_data, return_counts=True)

    print(f'The data has {len(X_data)} samples with {values} classes and respectively their counts {counts}.')

    return X_data, y_data