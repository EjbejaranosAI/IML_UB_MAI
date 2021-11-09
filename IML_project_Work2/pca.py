import numpy as np
from matplotlib import pyplot as plt
from utils import arff_parser

class pca:
    def __init__(self, reduced_components):
        self.reduced_components = reduced_components
        self.data_mean_sub = None
        self.new_components = None


    def fit(self, data):
        self.data_mean_sub = data - np.mean(data, axis = 0)
        cov_mat = np.cov(self.data_mean_sub, rowvar=False)
        eigen_val, eigen_vec = np.linalg.eig(cov_mat)
        # return indexes of the eigen values in descending order (highest variability first)
        eigen_index = np.argsort(eigen_val)[::-1]
        eigen_val_sort = eigen_val[eigen_index]
        eigen_vec = eigen_vec.transpose()
        eigen_vec_sort = eigen_vec[eigen_index]
        self.new_components = eigen_vec_sort[0:self.reduced_components]

    def transform(self):
        data_reduced = np.dot(self.data_mean_sub, self.new_components.transpose())
        return data_reduced



content = "datasets/vehicle.arff"
df_normalized, data_names_num, data_names_cat, data_names, class_names = arff_parser.arff_to_df_normalized(content)

model = pca(2)
model.fit(df_normalized)
data_reduced = model.transform()
print(data_reduced)


plt.figure()
plt.scatter(data_reduced[:, 0], data_reduced[:, 1])
plt.show()
