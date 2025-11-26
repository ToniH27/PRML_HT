"""
2.1 Function implementation
The classification function must have the name digit_classify and it must take parameters and
return values in the required way. The following details must be taken into account:

• Feature extraction: Possible function(s) responsible for feature extraction must be called within
the digit_classify function.

• Training: You may use the training data as you see fit. If the classifier requires training, it is not a
good idea to train the method every time the function digit_classify is called. Therefore, any
parameters set by the training process need to be loaded or hardcoded into the implementation.
If your classification technique requires extra parameters, the parameter values must be fixed (by
using default values chosen by you) or determined inside the digit_classify function.

• Inference: The classification function only needs to take in one sample (N x 3 matrix) to be
classified, possibly extract the features and produce a single class label. The computational com-
plexity of your implementation must be practical so that testing its performance with hundreds
of samples can be performed in a reasonable time.

• Environment for experiments: The digit_classify function and possible other related functions
must run in a standard environment. Both the filename and function name have to be correct.

• High-level functions: The use high-level functions of Matlab or Python library such as classify
is not allowed in your solution. You must make your own classifier implementing a classification
method.

• Visualisations or debug information: the submitted code should not produce any visualisations
or provide profound debug information. (It is good practice to include a debugging flag into the
code to enable or disable producing such info.)

Remember to properly comment your codes. Write also a help section to your codes that tells the
purpose of the function, usage, and explanation of the parameters. In Matlab, comments following the
first line of a function will show when the help command is used with the name of the function. You
can see an example, if you type following command in Matlab:
>> help mean
>> type mean"""

"""
The deadline of submitting the results of your work to Moodle is Friday, 5 December 2025 at 08:00
EET. The results consist of the document and the classifier implementation. The requirements are as
follows (STNUM is the student number of one of the group members):
1. The document is submitted in pdf format with filename STNUM .pdf.
2. The implementation is submitted as a single zip package and the file name of the package is
STNUM .zip. The package includes all relevant codes (feature extraction, training, classification 
and analysis of results). When the package contents are extracted, the result is a single directory
STNUM. This directory contains digit_classify Matlab function/wrapper and all the other
files (except the standard Matlab/Python library functions) needed to run the classification
function.
"""

###############################################################################################################
def standardize_data(X_data):
    X_data_std = [] # Standardized data
    for X in X_data: # (N,3)!
        # Standardization Z = (X-mu) / sigma for each column
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=1) # Sample standard deviation
        Z = (X - mu) / sigma
        X_data_std.append(Z)

    return X_data_std # Returns the standardized data as a numpy array
###############################################################################################################

###############################################################################################################
# Sources of feature extraction ideas and some example codes:
# https://github.com/yogeshgajjar/logistic_regression/blob/master/logistic_regression_classification.ipynb
# https://www.geeksforgeeks.org/data-analysis/feature-engineering-for-time-series-data-methods-and-applications/#1-statistical-features
# https://towardsdatascience.com/applications-of-rolling-windows-for-time-series-with-python-1a4bbe44901d/

import numpy as np
import pandas as pd
from scipy.stats import skew

# Function takes the time series data and labels as input and forms a new pandas dataset 
def feature_extract(X_data):
    new_X_train = []
    # Extracting features from the time series data and making a simpler dataset with class labels
    for i in range(len(X_data)):
        X = [] # Feature sample list

        x = X_data[i] # Time series data sample
        x_len = len(x) # Length of the sample for indexing

        # Extracts means of the selected intervals
        X.append(np.mean(x[2*x_len//8:3*x_len//8]))
        X.append(np.mean(x[3*x_len//8:4*x_len//8]))
        X.append(np.mean(x[4*x_len//8:5*x_len//8]))
        X.append(np.mean(x[5*x_len//8:6*x_len//8]))
        X.append(np.mean(x[6*x_len//8:7*x_len//8]))

        # Extracts standard deviations of the selected intervals
        X.append(np.std(x[x_len//8:2*x_len//8], ddof=1))
        X.append(np.std(x[2*x_len//8:3*x_len//8], ddof=1))
        X.append(np.std(x[3*x_len//8:4*x_len//8], ddof=1))
        X.append(np.std(x[4*x_len//8:5*x_len//8], ddof=1))
        X.append(np.std(x[5*x_len//8:6*x_len//8], ddof=1))
        X.append(np.std(x[6*x_len//8:7*x_len//8], ddof=1))

        # Extracts max-mins of the selected intervals
        X.append(np.max(x[:x_len//5]) - np.min(x[:x_len//5]))
        X.append(np.max(x[x_len//5:2*x_len//5]) - np.min(x[x_len//5:2*x_len//5]))
        X.append(np.max(x[2*x_len//5:3*x_len//5]) - np.min(2*x[x_len//5:3*x_len//5]))
        X.append(np.max(x[3*x_len//5:4*x_len//5]) - np.min(3*x[x_len//5:4*x_len//5]))

        # Computes skewness values of each spatial coordinate
        skewness = skew(x)
        X.extend(skewness)

        # Extracts area under the curve of the selected intervals
        auc1 = np.trapezoid(x[:,1], x[:, 0], dx=5)
        auc3 = np.trapezoid(x[:,0], x[:, 2], dx=5)
        X.append(auc1) 
        X.append(auc3)

        # Extracts the centroid of X coordinate
        centroid = np.mean(x, axis=0)
        X.append(centroid[0]) # xmean

        # Extracts the area under the curve in the selected interval
        X.append(np.trapezoid(x[:x_len//3, 1], x[:x_len//3, 0], dx=5))

        firstLastDist = x[0,:]-x[-1,:] # added euclidian distance
        eucDist = np.linalg.norm(firstLastDist)
        X.append(eucDist)

        new_X_train.append(X)

    # New dataset
    dataset = np.array(new_X_train)
    return dataset 
###############################################################################################################

###############################################################################################################
def MLR_test(W, X_te):
    N,D = X_te.shape
    # augmented weights
    x0 = np.ones((N,1)) # column of ones
    x_bias = np.concatenate([X_te,x0], axis=1) # (N,D+1), adding to the end
    z = x_bias @ W.T # logits, (N,K)
    predicted = z.argmax(axis=1) # taking the index value of the largest logit is enough for predictions
    return predicted
###############################################################################################################

import numpy as np
def digit_classify(X):
    # takes in a single Nx3 numpy matrix, outputs a single class label as an integer
    # TODO: "The classification function only needs to take in one sample (N × 3 matrix) to be
    #       classified, possibly extract the features and produce a single class label"
    # NOTE: current weight W is not optimal, was only trained with 200 steps.
    W=np.array([[-0.8852783215743817, -0.27316859985302333, -0.08630363766009544, 0.25377232423279567, 1.2796559492906614, -0.11180268487348793, 0.0967000879102813, 0.5446013128778743, 0.040607492088365095, 0.24375095677588574, -0.12384085786080207, 0.3923597746940668, 0.5195544401215683, -0.2948423654617961, -0.20880820208141407, -0.08861116334444613, -0.7152887332490395, 0.0377799148176448, -2.4886890381737943, -0.6916712222485508, -4.6462090608479495e-19, 0.7474448813761377, -2.5056112500379477, 0.09431467649596768], [1.49791099696911, -0.39924499816397613, -0.93939954467298, -1.0015028912067647, -1.108025465456538, 0.1355740388405263, 0.12829848962990614, -0.3428977177252697, -0.27347308045773044, -0.13944076871776717, -0.016406254315958307, 0.05633945723772773, -0.4531852623414588, 0.28079110887700376, 0.22180527013266813, 0.5845715313452091, 1.0391748103924123, 0.037668111285284966, 0.36516874227091267, 0.056274778146449544, -3.6672271911741034e-15, 0.4912198514001354, 0.7997898846745622, 0.4312602714085008], [-0.4772962251558372, 0.22914372275224926, -0.3408057473932633, -0.9644160607925931, 0.9955518461786496, -0.14449150402592278, 0.295473815018305, 0.8560990393688644, 0.030542905845591592, 0.746905948500615, 0.41486524488799925, -0.4711643061041469, 0.269123179108896, -0.002274569650110215, 0.09866150822029211, 0.7616856351740278, 0.675153098176035, -0.030062718927782468, -0.6463625251159358, 0.16664610164668758, 1.2959438918934997e-15, 2.4322835924885777, 0.6656534823900048, -0.040723457851094755], [0.22028032354095273, -0.5253358033395192, -0.3938560401581702, 0.4412786403567706, 0.15859699800986662, -0.04415557362680371, -0.215022832355385, -0.40700966009434486, -0.6660256932241367, -0.05865240599033547, -0.05478204657750843, 0.08837988075903959, 0.13980740825973123, -0.6787773933742244, -0.0035424554607038296, 0.24342757702051443, 0.27416923358419787, 0.05643047592428939, 2.922402555384082, 0.5475975584348309, 3.8715218358263186e-16, 0.9832107180856011, -0.4866939707206966, -0.22613269858982502], [-1.3001825053903604, 0.2573257199038256, 1.1089143096252705, 2.049769871833834, -0.024020793510463375, -0.31898649230799464, -0.18431037185852583, -0.39569217186862765, -0.5410771657735959, -0.6171719325805818, -0.33306730969299114, 0.305078887020996, 0.4692538633649293, 0.3082355124489839, 0.09017777318291038, -0.994788164799209, 0.6785924282435185, -0.25993691131300645, -0.5896398518724292, -0.14613535073424141, 3.932068226235777e-16, 0.8165394695910673, 0.7795851647551504, -0.29201813019375306], [0.46368222070446696, -1.1741770553282904, -1.0176966302131938, 0.12254634366635228, 0.19717445520451923, -0.15860447393822258, 0.2932046948013155, -0.14103744983294328, -0.615581969722191, -0.15839544402257122, 0.2540350755267943, -0.21647144402256666, 0.1381441433631646, -0.9042888578090744, 0.6945118158775663, 1.0847733628542595, 0.17392781942406904, 0.14345091807358318, 0.7351998414234372, 0.22434776848587645, 3.069695591640882e-16, -1.6648442199187634, 0.5947458059983329, 0.002128486668414594], [0.5677824081591084, -0.3474107459077371, -0.23192771386500216, -0.14844287673622433, -0.7166130294358866, 0.2253532206837762, -0.3635717095839083, -0.37273866180985016, 0.605848716229926, 0.07108395790131623, -0.3245781733732481, -0.1171255294246717, -0.8512932828462736, -0.0724199664195365, -0.19151487370578968, 0.7129414432742908, 0.8613712195289528, -0.31407385629219936, -1.9110981182919273, -0.8130429322019175, 4.9116442531001164e-17, -1.402845233137256, 0.5519777072984302, -0.22419551302610305], [-0.04606915925972589, 1.3584019745173226, 1.6433321124056182, -0.08241259546972486, -0.6250696964312923, 0.46110659269233034, 0.12882460955473993, 0.0036981336866670914, 0.4860266349162687, -0.16111650141853517, -0.04911951650962225, 0.33427001258109423, -0.2470675370328177, 0.6305984312347045, -0.6360568915600227, -0.16335941838006113, -1.4126692163688954, -0.48649056266674023, 1.4465938844760897, 0.1397962402698264, 1.610775167769133e-15, 1.1710504561575203, 0.13116963948260557, 0.26354371547913535], [0.016804047269081492, 0.5177044538768748, -1.0961260229869263, -1.1550728837778201, 0.8475618872498969, 0.09094290548483452, -0.164307988268997, 0.45882907003151513, 1.4546135122764805, 0.3434131995995828, -0.3781456887634002, 0.36034124174168036, 0.23982576922471438, 0.3415046558831556, 0.045094009827980064, -0.024256922014012774, -0.15424575070664595, 0.15583551161484982, 0.14086234188685218, 0.31061616143437754, -4.7601275335617024e-17, -1.4837538811765405, -1.05251755380612, 0.12153593720319528], [-0.05763378526241178, 0.35676133154227513, 1.3538689149187446, 0.48448012789337386, -1.0048121510994126, -0.13493602892903572, -0.015288794847732224, -0.20385189463388595, -0.5214813521789743, -0.2703770100476096, 0.6110395266787393, -0.7320079744832173, -0.22416272122245087, 0.39147344427090125, -0.11032795443348761, -2.1163838811305755, -1.4201849090246055, 0.6593991174840743, 0.025562168012711578, 0.20557089676666312, -3.2787098014812413e-16, -2.0903056348664832, 0.5219010899656794, -0.12971328759443818]],dtype=float)
    X_std = standardize_data([X])
    X_std_feat = feature_extract(X_std)
    label = MLR_test(W, X_std_feat)

    return int(label[0])

# placeholder experimentation to make sure 1 SINGLE (Nx3) matrix can be fed into these functions, this should probably be verified later to make sure it works as it is supposed to.
DF = pd.read_csv("stroke_0_0001.csv", header = None)
M = DF.to_numpy()
pred = digit_classify(M)
print(pred)

# NOTE: I think we should only change THIS function, IF there are bugs, and IF new implementations are needed...
# ...   IF new implementations are needed, the weights have to be updated. Also, we should check this function on
# ...   other devices too.

# just some quick checking with all samples, i.e: testing the entire model through this function setup with all the original data...
import pandas as pd
import numpy as np
import glob

doubleList = [] # each index corresponds to the class label from 0 to 9, where a separate list of each of the 100 sample matrices reside
for eachIndex in range(10):
    fileList = [] # loading each number's 100 samples into this in the below loop, appending to the doubleList
    fileName = "stroke_"+str(eachIndex)+"_*.csv" # the "*" sign means to take every file 

    for eachFile in glob.glob(fileName):
        df = pd.read_csv(eachFile, header = None)
        df = df.drop_duplicates().reset_index(drop = True) # using this to remove duplicate rows, comment out to have duplicates

        fileList.append(df.to_numpy())

    doubleList.append(fileList)

y_true = []
allPreds = []
for j in range(10):
    for i in range(100):
        sample = doubleList[j][i]
        prediction = digit_classify(sample)
        allPreds.append(prediction)
        y_true.append(j)

from sklearn.metrics import classification_report, confusion_matrix

CR_RF = classification_report(y_true, allPreds)
CM_RF = confusion_matrix(y_true, allPreds)
print(CR_RF)