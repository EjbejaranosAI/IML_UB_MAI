import pandas as pd
from utils.arff_parser import arff_to_df_normalized
import numpy as np
from skrebate import ReliefF
from sklearn.feature_selection import mutual_info_classif, chi2, mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler


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
    # 1. Load the test+train datasets from Fold-0 and join them to generate the complete dataset:
    def _load_dataset(dataset_name, fold, mode):
        dataset_path = f"../datasetsCBR/{dataset_name}/{dataset_name}.fold.{fold:06}.{mode}.arff"
        X_data, classes, Y_labels = arff_to_df_normalized(dataset_path)

        return X_data, Y_labels, classes,

    def _preprocess_dataset(data, column_names):
        # remove columns that has more than 80% nans
        data = data.dropna(thresh=len(data) * 0.8, axis=1)
        data = data.fillna(data.mean())

        data_names = [name for name in column_names if name in data.columns]

        # scale and normalize
        scalar = StandardScaler()
        df_scaled = scalar.fit_transform(data)
        df_normalized = normalize(df_scaled)
        data = pd.DataFrame(df_normalized, columns=data_names)

        return data.to_numpy()

    X_data, classes, Y_labels = _load_dataset(name_dataset, 0, 'train')

    X_data = _preprocess_dataset(X_data, classes)
    Y_labels = Y_labels.to_numpy()
    Y_labels = np.array([x_class.decode("utf-8") for x_class in Y_labels])
    # 2. Normalise the dataset in [0,1] interval:
    # 3. Transforming of categorical values to one-hot encoded:
    # 4. Imputing all missing values:
    # 5. Select method and get "feature weights" array:
    if method == 'i_gain':
        selector = SelectKBest(mutual_info_classif, k=k_best_features)
        selector.fit(X_data, Y_labels)
        cols = selector.get_support(indices=True)
    elif method == 'relief':
        featuresWeights = ReliefF(n_features_to_select=k_best_features, n_neighbors=20, n_jobs=-1)
        featuresWeights.fit_transform(X_data, Y_labels)
        cols = np.array(featuresWeights.top_features_[0:k_best_features])

    weights = np.zeros(X_data.shape[1])
    weights[cols] = 1.0

    print(weights)
    return weights


selectionIBLAlgorithm('hypothyroid', 'relief', 4)
