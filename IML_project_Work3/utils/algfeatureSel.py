import pandas as pd
import numpy as np
from skrebate import ReliefF
from sklearn.feature_selection import mutual_info_classif, chi2, mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def selectionIBLAlgorithm(name_dataset, method, k_best_features):
########################################################################################################################
# Function containing the feature-scoring method as well as the feature-selecting method.
#
# INPUTS:
# name_dataset: introduce the name of the dataset as it appears in the folder that contains it.
# method: choose the filtering method for scoring the different features. For now the Information Gain
#         (method = 'i_gain') and the Relief (method='relief') methods are included.
#
# OUTPUTS:
# feature_weights: an array of the same length as the number of features in the dataset, containing 1s for the useful
#                  features and 0s for the useless features.
########################################################################################################################
    # 1. Extract the test+train datasets for Fold-0 and join them:
    # 2. Normalise the dataset in [0,1] interval:
    # 3. Transforming of categorical values to one-hot encoded:
    # 4. Imputing all missing values:
    # 5. Select method and get "feature weights" array:
    if method=='i_gain':
        cols = selector.get_support(indices=True)
    elif method=='relief':
        cols =

    weights = np.zeros(X_test.shape[1])
    weights[cols] = 1.0

    return weights