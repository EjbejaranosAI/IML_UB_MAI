''' Feature selection'''
import matplotlib
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
from skrebate import ReliefF
from sklearn.feature_selection import mutual_info_classif, chi2, mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# Load the dataset
# dataset='satimage'
seg_data = pd.read_csv('segmentation-all.csv')
#print(seg_data.shape)
#print(seg_data.head())
#print(seg_data['Class'].value_counts())

y = seg_data.pop('Class').values
X_raw = seg_data.values

X_tr_raw, X_ts_raw, y_train, y_test = train_test_split(X_raw, y,
                                                       random_state=42, test_size=1 / 2)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_tr_raw)
X_test = scaler.transform(X_ts_raw)

feature_names = seg_data.columns
#print(feature_names)
X_train.shape, X_test.shape
# ReliefF will produce scores for all features.
print(f'THis is y_train: {y_train}')
print(f'This is x_train:{X_train}')


def ReliefMethod(X_data, y_label,k_n):
    print('RESULTS OF RELIEF METHOD:')
    featuresWeights = ReliefF(n_features_to_select=k_n, n_neighbors=20, n_jobs=-1)
    featuresWeights.fit_transform(X_data, y_label)
    fs = featuresWeights.fit_transform(X_data, y_label)
    #featuresWeights.transform(X_train).shape
    print(f' The shape of the dataset before feature selection: {X_data.shape}')
    print(f' The shape of the dataset after feature selection: {fs.shape}')
    #print(f' The dataset dict after is: {featuresWeights.__dict__}')
    #print(f' Scores features after : {featuresWeights.feature_importances_}')
    #print(f' The top features after is: {featuresWeights.top_features_}')
    cols = np.array(featuresWeights.top_features_[0:k_n])
    print(f'The best {k_n} scores for choose the feature selection are: {cols}')
    weights = np.zeros(X_data.shape[1])
    weights[cols] = 1.0
    print(f'The weights for the feature selection are: {weights}')
    return featuresWeights, weights, cols





def iGain(X_data, y_label,k_n):
    print('RESULTS OF I-GAIN METHOD:')
    # 1. The mutual information score returned by mutual_info_classif is effectively an information gain score
    i_scores = mutual_info_classif(X_data, y_label)
    # 2. Save scores into dataframe
    df = pd.DataFrame({'Mutual Info.': i_scores, 'Feature': feature_names})
    df.set_index('Feature', inplace=True)
    df.sort_values('Mutual Info.', inplace=True, ascending=False)
    print('Features with respective scores:')
    print(df)
    # 3. Select k-best features
    selector = SelectKBest(mutual_info_classif, k=k_n)
    selector.fit(X_train, y_train)
    # 4. Get columns to keep and create new dataframe with those only
    cols = selector.get_support(indices=True)
    best_X = X_train[:, cols]
    best_feature_names = feature_names[cols]
    best_scores = selector.scores_[cols]
    print('Best 10 scores???')
    print(best_X.shape)
    # 7. Save selected features
    df2 = pd.DataFrame({'Mutual Info.': best_scores, 'Feature': best_feature_names})
    df2.set_index('Feature', inplace=True)
    df2.sort_values('Mutual Info.', inplace=True, ascending=False)
    print('Dataset with the 10 best scores')
    print(df2)
    # 8. Define weight array
    weights = np.zeros(X_test.shape[1])
    weights[cols] = 1.0
    print(weights)

# Feature Selection using ReliefF
ReliefMethod(X_train, y_train, 4)
# Feature Selection using I-Gain
iGain(X_train, y_train, 4)


