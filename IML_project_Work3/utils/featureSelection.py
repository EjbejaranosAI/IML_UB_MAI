''' Feature selection'''
'''from .feature_selector import FeatureSelector'''
# This line is in initial .py

'''Description of variables'''
'''Local libraries'''
# from ib.ib import IB
# from utils.arff_parser import arff_to_df_normalized
'''import libraries'''
# from feature_selector import FeatureSelector


from sklearn.feature_selection import VarianceThreshold
import pandas as pd

# dataset for testsç
e = [[1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [1, 1, 0, 1]]

'''Function for remove low var features'''
# Features are in train and labels are in train_labels
def lowVarFeatures(data):
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)));
    sel.fit(data);
    transform = sel.transform(data)
    fit_transform = sel.fit_transform(data)
    return print(transform), print(fit_transform)


lowVarFeatures(e)
'''Function for unvaried feature selection'''
