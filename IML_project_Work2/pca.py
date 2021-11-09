import numpy as np
from matplotlib import pyplot as plt
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


content = "datasets/vehicle.arff"
df_normalized, data_names_num, data_names_cat, data_names, class_names = arff_parser.arff_to_df_normalized(content)

model = PCA(2)
model.fit(df_normalized)
data_reduced = model.transform()

model_sklearn = PCA_sklearn(2)
model_sklearn.fit(df_normalized)
data_reduced_sklearn = model_sklearn.transform(df_normalized)

plt.figure()
plt.subplot(2, 2, 1)
plt.title('Dimensions reduced by PCA')
plt.scatter(data_reduced[:, 0], data_reduced[:, 1])
plt.subplot(2, 2, 2)
plt.title('First two dimensions')
plt.scatter(df_normalized[data_names[0]], df_normalized[data_names[1]])
plt.subplot(2, 2, 3)
plt.title('Dimensions reduced by PCA-sklearn')
plt.scatter(data_reduced_sklearn[:, 0], data_reduced_sklearn[:, 1])
plt.subplot(2, 2, 4)
plt.title('First two dimensions')
plt.scatter(df_normalized[data_names[0]], df_normalized[data_names[1]])
plt.show()
