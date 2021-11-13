from operator import itemgetter

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize

from utils import arff_parser
from sklearn.decomposition import PCA as PCA_sklearn


class PCA:
    def __init__(self, reduced_components):
        # number of components we want to reduce the data to
        self.reduced_components = reduced_components
        self.data_mean_sub = None
        self.new_components = None

    def fit(self, data):
        # subtract the mean vector from the dataset
        self.data_mean_sub = data - np.mean(data, axis=0)
        # calculate the covariance matrix of the dataset
        covariance_mat = np.cov(self.data_mean_sub, rowvar=False)
        # compute the eigen values and eigen vectors of the covariance matrix
        eigen_val, eigen_vec = np.linalg.eig(covariance_mat)
        eigen_vec = eigen_vec.transpose()
        print('Eigen values:',eigen_val)
        # return indexes of the eigen values in descending order (highest variability first)
        eigen_index = np.argsort(eigen_val)[::-1]
        # sort the eigen vectors by highest variability
        eigen_vec_sort = eigen_vec[eigen_index]
        # return declared "reduced_components" number of eigen vectors
        self.new_components = eigen_vec_sort[0:self.reduced_components]

    def transform(self):
        # reduce the dimensionality of the data bu creating a dot product
        data_reduced = np.dot(self.data_mean_sub, self.new_components.transpose())
        return data_reduced


content = "datasets/adult.arff"
df_normalized, data_num_idxs, data_cat_idxs, data_names, classes = arff_parser.arff_to_df_normalized(content)

df_normalized = df_normalized[list(itemgetter(*data_num_idxs)(data_names))].to_numpy(dtype='float32')
print(df_normalized)

model = PCA(2)
model.fit(df_normalized)
data_reduced = model.transform()

model_sklearn = PCA_sklearn(2)
model_sklearn.fit(df_normalized)
data_reduced_sklearn = model_sklearn.transform(df_normalized)

plt.figure()
plt.subplot(1, 3, 2)
plt.title('Dimensions reduced by PCA')
plt.gca().set_aspect('equal', adjustable='box')
plt.scatter(data_reduced[:, 0], data_reduced[:, 1], s = 0.1)
plt.subplot(1, 3, 1)
plt.title('First two dimensions')
plt.gca().set_aspect('equal', adjustable='box')
plt.scatter(df_normalized[:,0], df_normalized[:,1], s = 0.1)
plt.subplot(1, 3, 3)
plt.title('Dimensions reduced by PCA-sklearn')
plt.gca().set_aspect('equal', adjustable='box')
plt.scatter(data_reduced_sklearn[:, 0], data_reduced_sklearn[:, 1], s = 0.1)
plt.tight_layout()
plt.show()

