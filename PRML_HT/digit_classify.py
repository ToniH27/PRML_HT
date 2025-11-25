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

        X.append(np.mean(x[2*x_len//8:3*x_len//8]))
        X.append(np.mean(x[3*x_len//8:4*x_len//8]))
        X.append(np.mean(x[4*x_len//8:5*x_len//8]))
        X.append(np.mean(x[5*x_len//8:6*x_len//8]))
        X.append(np.mean(x[6*x_len//8:7*x_len//8]))

        X.append(np.std(x[x_len//8:2*x_len//8], ddof=1))
        X.append(np.std(x[2*x_len//8:3*x_len//8], ddof=1))
        X.append(np.std(x[3*x_len//8:4*x_len//8], ddof=1))
        X.append(np.std(x[4*x_len//8:5*x_len//8], ddof=1))
        X.append(np.std(x[5*x_len//8:6*x_len//8], ddof=1))
        X.append(np.std(x[6*x_len//8:7*x_len//8], ddof=1))

        # Extract max-mins of X in 5 subintervals
        X.append(np.max(x[:x_len//5]) - np.min(x[:x_len//5]))
        X.append(np.max(x[x_len//5:2*x_len//5]) - np.min(x[x_len//5:2*x_len//5]))
        X.append(np.max(x[2*x_len//5:3*x_len//5]) - np.min(2*x[x_len//5:3*x_len//5]))
        X.append(np.max(x[3*x_len//5:4*x_len//5]) - np.min(3*x[x_len//5:4*x_len//5]))

        skewness = skew(x)
        X.extend(skewness)

        # Area under the curve features
        auc1 = np.trapz(x[:,1], x[:, 0], dx=5)
        auc3 = np.trapz(x[:,0], x[:, 2], dx=5)

        X.append(auc1) 
        X.append(auc3)

        centroid = np.mean(x, axis=0)

        X.append(centroid[0]) # xmean

        X.append(np.trapz(x[:x_len//3, 1], x[:x_len//3, 0], dx=5))
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
    W = np.array([[-1.1869342784721701, 0.07074614741131792, 0.38234576381920504, 0.4677763036695789, 1.4209105644410303, -0.36800793032319906, 0.01011225983913861, 0.5564018665116199, 0.06941789768677128, -0.14972733826461582, -0.2627740797966506, -0.6893761109499023, 0.07748641728591198, -0.36961395062780955, -0.7536021548086249, 0.28113591924646525, -1.2412882189003263, 0.23693035800705203, -3.021702848128081, -0.35453760059470796, -3.9437765689541624e-17, 1.5704146735950715, -0.2343960191418615], [1.124786963026856, -0.04018333369292667, -0.7563956726213857, -1.0133777423021801, -0.9970387355028446, -0.0007931752713607127, 0.08829590301039957, -0.09182578964566747, -0.0339434590857482, 0.013037113079129315, 0.15878359281522755, -0.04783787834610149, -0.3081057967459038, 0.35869041658378353, 0.30962187022175963, 0.42796632397178397, 0.6615897064798384, 0.21478864397499128, 0.6047022526239119, -0.1347192982592243, -1.645110474824714e-15, 0.24298520962018103, 0.5663871070270439], [-0.15882890840575106, 0.07558339492685993, -0.40072879252239546, -0.5967840987902022, 0.6332374510675682, 0.0020856506139828707, 0.2703780316502994, 0.5238514407370822, 0.05984954480322258, 0.4411610238144799, 0.33207597476791845, -0.13497745838792044, 0.2930791014074051, 0.07310299150009285, 0.21982479489550755, 0.497059202400616, 0.6320441523619809, 0.1184772176700621, -0.2796331035157069, 0.10867297231991976, 5.2290070402506845e-16, 1.9077042666824686, -0.011066684815104493], [0.14056945545996402, -0.39236066775420964, -0.2928095241698406, 0.3125008613048348, 0.15104787621434598, -0.06635812262239377, -0.1108403876467413, -0.24493491650153426, -0.4285092752943817, -0.007739881138612129, -0.07399293246962103, -0.025279115020062054, 0.11337451336851233, -0.5496531372901351, -0.019013473283560747, 0.22924644439328726, 0.26180324035820673, 0.010383419470983795, 2.3516309691778488, 0.4686574833842809, 3.413489402944934e-17, 0.9381399739716052, -0.19066392589546283], [-1.0040432506301293, 0.09156450200711196, 0.87510734085737, 1.653791667910658, 0.15450519729294945, -0.0379326475106245, -0.03535896655737707, -0.30665727840399915, -0.31226507997644665, -0.5040770691260917, 0.024773559174542457, 0.5235765509443355, 0.21479590121060985, 0.3230674614005693, 0.24629305676886726, -0.8544037325204705, 0.181235828447572, -0.23488233251168628, -0.377985350132186, -0.16924170657052992, 2.2219376001798443e-16, 0.7409007945572658, 0.016328562153769145], [0.245490784773251, -0.998760416197796, -0.7629612082262125, 0.21454563951629835, 0.18996210557055918, -0.033095232668095685, 0.28220542057414205, -0.08131408324666982, -0.46679215676612174, 0.058171816904891, 0.5854040760825169, 0.019190704272395206, 0.18440897559732178, -0.9097277919420522, 0.7191602150339008, 0.6858643631850984, 0.17211997240493535, 0.14528308381058616, 0.712855743259926, 0.20663266379306117, 1.6039361287785581e-16, -1.5100510253741848, 0.16336775746972396], [0.9276267626778687, -0.5909786529667593, -0.6092485611118783, -0.38129522476020394, -1.0116330871057626, 0.2036337584872686, -0.2717368620596574, -0.5334151605337708, 0.38269955019906604, 0.3867831168084599, -0.18528099856928884, 0.2746283150339894, -0.7606727465617709, -0.32658022393955294, -0.1273342130684257, 0.29198797376672536, 1.2994954028350378, -0.6103352432702472, -1.4535406015643708, -0.5883128844885356, 6.828893732332859e-17, -1.8753780182725828, 0.005869527256794016], [0.03145852242884344, 1.068390774183317, 1.13876344695927, -0.13580566150401088, -0.4597409794311574, 0.3361810531289729, 0.06487851128794063, 0.10467349035781962, 0.32588953762716666, -0.06509140966259694, -0.034085403039138, 0.3512333230609601, 0.029863458277615135, 0.565540540565786, -0.48065630858707337, 0.008327560458048378, -0.9042767941770306, -0.4595728784910738, 1.2752551844280366, 0.016378536174090758, 7.236895231501857e-16, 0.9197484965934994, 0.13201814306748766], [0.05080203480836236, 0.4828088426865099, -0.6169628958436459, -0.8291317838311189, 0.7668809428934158, 0.03213738517635057, -0.304291241134951, 0.4075002269860303, 1.1597253763443043, 0.11504229114255418, -0.7394282086360557, 0.3078163425729332, 0.16072606518414945, 0.6460479473018563, -0.28614559548354696, 0.1681034653802702, -0.1691464491773423, -0.07502797965610883, 0.058623512047872875, 0.3330555322262821, 1.0316948447082937e-17, -0.9485544310991069, -0.20953791642831848], [-0.17092808566709441, 0.23318940939657373, 1.0428901028595123, 0.3077800387863465, -0.8481313354401029, -0.06785073901090101, 0.006357331036806395, -0.3342797962609102, -0.7560719355378324, -0.2875596635575956, 0.19452441967055017, -0.5789746731806268, -0.004955889023848855, 0.1891257464474702, 0.1718518083112087, -1.7352875202818245, -0.893576840632874, 0.6539557109954401, 0.12979424180274876, 0.11341430201536451, -5.737013935670027e-17, -1.9859099402742115, -0.23830655069407125]], dtype=float)
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