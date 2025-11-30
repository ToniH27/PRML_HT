import numpy as np
from scipy.stats import skew


def standardize_data(X_data):
    """
    Scales a raw data matrix via the Z-scaling technique.

    :param X_data: Given raw data sample by the user.
    """
    X_data_std = [] # Standardized data
    for X in X_data:
        # Standardization Z = (X-mu) / sigma for each column
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=1) # Sample standard deviation
        Z = (X - mu) / sigma
        X_data_std.append(Z)
    return X_data_std # Returns the standardized data as a numpy array


# Function takes the time series data as an input
def feature_extract(X_data):
    """
    Performs feature extraction on a given sample matrix. Means, standard deviations, max-mins, areas under
    the curves of selected intervals. Skewness, X-centroid and euclidian distance.

    :param X_data: Given scaled data sample by the user.
    """
    if(X_data[0].shape[0] < 5): # if N is less than 5, return with all zeros as features
        x_return = np.zeros((1,23))
        return(x_return)

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


def MLR_test(W, X_te):
    """
    Classifies a given input sample. Returns an integer digit from 0 to 9.

    :param W: Trained multinomial logistic regression weights and biases for a 10 class digit classifier.
    :param X_te: Given test sample by the user.
    """
    N,D = X_te.shape
    # augmented weights
    x0 = np.ones((N,1)) # column of ones
    x_bias = np.concatenate([X_te,x0], axis=1) # (N,D+1), adding to the end
    z = x_bias @ W.T # logits, (N,K)
    predicted = z.argmax(axis=1) # taking the index value of the largest logit is enough for predictions
    return predicted


def digit_classify(X):
    """
    Classifies a given input sample. This function first scales the data sample, then extracts the features from it.
    Then classifies the sample via a pretrained multinomial logistic regression model. Returns an integer from 0 to 9.

    :param X: Given test sample by the user.
    """
    # takes in a single Nx3 numpy matrix, outputs a single class label as an integer
    W=np.array([[-0.8415864027201794, -0.281217170003975, 0.03317007432397439, 0.2235729259260053, 1.1769967660951686, -0.09539221980079436, 0.1172128408621826, 0.47655762756927694, 0.10704442977126037, 0.19523901041836922, -0.12752505340175752, 0.32987896472931183, 0.4917596038516059, -0.14996003007336214, -0.1913292226420432, 0.02107181086234435, -0.6911369922396219, 0.26007461041452284, -2.4562169346160303, -0.6205778658564683, 2.6737646734805186e-17, 0.7388823304773632, -2.5755762686132413, 0.06880163344594173], [1.1444706605187567, -0.26299424412025774, -0.7652706709597681, -0.9241698807011767, -1.1065637044672454, 0.08306825263263046, 0.2687064451880989, -0.31311282263531404, -0.12355952934148758, -0.13074028451819025, 0.07764128425147801, 0.015760666892979315, -0.3890628027829627, 0.40645973181238104, 0.03131649350027552, 0.6618240710850918, 1.0088972339325435, 0.035414342507773124, 0.5490397738573612, -0.051482210611522014, -6.593021161921417e-16, 0.3239261281572807, 0.9263013315391598, 0.384089695720931], [-0.4143469943331985, 0.31650314319421924, -0.28219718697077595, -0.9195997159693335, 0.9827377188184616, -0.19297147069467732, 0.2416961607173889, 0.8640740179095251, 0.1656193434835074, 0.7433002727201686, 0.3719122036383744, -0.45950765938348065, 0.09285276301963849, 0.14506066425014041, 0.0646359815005484, 0.8155077840641048, 0.8425633195014395, 0.2683121318099751, -0.6026863566721984, 0.16914866143826982, 1.1786621355478076e-16, 2.496772093399094, 0.6164832372889875, 0.007918519075168834], [0.19984494376735024, -0.5258090115206852, -0.4138381848319714, 0.4600773778879313, 0.20327346309278244, -0.03659487511597448, -0.18254211668042566, -0.34179881488055824, -0.683840452961375, -0.06461803868197775, -0.13337848694257, 0.18357389436332375, 0.2033958625468388, -0.5419954862354215, -0.08734396180244966, 0.12793917152266324, 0.3578875406482904, -0.0768270748414978, 2.882637737694954, 0.7035421980844315, -1.3065862772035857e-16, 1.0393390416607704, -0.5852397677603656, -0.2397990881105879], [-1.2472763990188145, 0.2537711537211554, 1.2003753206262235, 2.0592173227039585, 0.09215759192358985, -0.22747024137330424, -0.11461671162783178, -0.5354291230958993, -0.4006948747613514, -0.5764672424506925, -0.2978137670633116, 0.2872245359024846, 0.37902944876790556, 0.2153466652604462, 0.13198652945387826, -0.8919408577652423, 0.7271653790670519, -0.19006600813775126, -0.6248312635651037, -0.189423457777566, 5.294176772575762e-16, 0.8491818554029709, 0.8417830267912639, -0.24723603322555757], [0.35116283172827767, -1.1891541507095686, -1.010420988825338, 0.11736304587434383, 0.3400494404282227, -0.15996938604529362, 0.31752930067439916, -0.182888367494538, -0.5390575811382122, -0.039404358245021476, 0.3465085225550706, -0.4571809888465551, -0.017639561339688824, -0.9667618288979204, 0.8174137134495157, 1.0157471599927241, 0.019589111366951212, 0.28948999665756886, 0.8006235252835623, 0.17314245526053515, 3.7320452575686175e-16, -1.8265323013333175, 0.5139986493114579, 0.022480694272811505], [0.5388289419778881, -0.3272730115986134, -0.3031528848377989, -0.08957722421173446, -0.7343483243747579, 0.1548868404432365, -0.368609888333875, -0.2907538597528481, 0.4944141233383018, 0.11801853456947639, -0.30786649986140985, -0.18679330141672273, -0.8297061205919112, -0.11641082380960811, -0.18998996647207442, 0.5743380136670055, 0.8104169009055754, -0.574214156990444, -1.9435971805711467, -0.8273295358287291, 1.2590232260025471e-17, -1.342730927608303, 0.680943839720962, -0.1862975882345547], [0.12968358874293506, 1.3630726608246209, 1.2818224166648313, -0.24965991381835112, -0.5840393279675846, 0.5840598416935817, 0.08128791126781959, 0.08850183893979367, 0.3930764856584108, -0.12787065364320846, -0.057740619483930576, 0.6115119958273464, -0.006386931508253531, 0.5952661557046761, -0.7890498377128776, -0.12861106509338824, -1.2782970957843378, -0.8020947661443032, 1.3390377902760653, 0.14721886420859862, 1.6554614211845298e-16, 1.0828061566033653, 0.21886699031804427, 0.28757402899816903], [0.152790920547787, 0.4184491366908484, -1.1622747255732326, -1.1122925932413386, 0.852808952819899, 0.0842124236808179, -0.14922924685335642, 0.40199374389730097, 1.338735580349642, 0.31105481856655876, -0.22967512011312768, 0.400746455749563, 0.31414905149787525, 0.1705344592561041, 0.18681305046094396, 0.20177547108345864, -0.21057152523503428, -0.0025781566887890885, 0.010316277153704326, 0.26913415561132026, -8.306002214504078e-17, -1.3239418573836612, -1.2388298349324065, 0.15805462921929367], [-0.01357209121080185, 0.2346514935222553, 1.421786830383858, 0.435068655549696, -1.2230725763685337, -0.19382916542022124, -0.21143469521439925, -0.1671442404567395, -0.7517375243986967, -0.4285120587354822, 0.35793753642118287, -0.7252145638182471, -0.23839131346104753, 0.2424604927325733, 0.025547220264281304, -2.3976515594187626, -1.5865138721628596, 0.7924890814129473, 0.04567663115882972, 0.2266267354711307, -3.52341671624961e-16, -2.0377025193755562, 0.6012687963361376, -0.25558649116161486]],dtype=float)
    X_std = standardize_data([X])
    X_std_feat = feature_extract(X_std)
    label = MLR_test(W, X_std_feat)
    return int(label[0])