''' Feature selection'''
'''from .feature_selector import FeatureSelector'''
#This line is in initial .py

'''Description of variables'''


'''Local libraries'''
from ib.ib import IB
from utils.arff_parser import arff_to_df_normalized
'''import libraries'''
from feature_selector import FeatureSelector
import pandas as pd
# Features are in train and labels are in train_labels
fs = FeatureSelector(data = train, labels = train_labels)