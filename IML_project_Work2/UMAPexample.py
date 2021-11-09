from sklearn.datasets import fetch_openml
import umap
#import umap.umap_ as umap
from utils.arff_parser import arff_to_df_normalized
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
#%matplotlib inline    ONlY FOR JUPYTER NOTEBOOK
# Dimension reduction and clustering libraries
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import pandas as pd
CONTENT = "datasets/vehicle.arff"
#CONTENT = 'datasets/adult.arff'
df_normalized, data_num_names, data_cat_names, data_names, class_names = arff_to_df_normalized(CONTENT)

print(df_normalized.head())

X = df = df_normalized.to_numpy()
#UMAP CLUSTER

mnist = fetch_openml(X, version=1)
mnist.target = mnist.target.astype(int)


'''For visualization purposes we can reduce the data to 2-dimensions using UMAP. 
When we cluster the data in high dimensions we can visualize the result of that clustering. 
First, however, we’ll view the data colored by the digit that each data point 
represents – we’ll use a different color for each digit. This will help frame what follows.'''


#standard_embedding = umap.UMAP(random_state=42).fit_transform(mnist.data)
#plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=mnist.target.astype(int), s=0.1, cmap='Spectral');

#UMAP enhanced clustering

clusterable_embedding = umap.UMAP(
    n_neighbors=30,
    min_dist=0.0,
    n_components=2,
    random_state=42,
).fit_transform(X.data)


plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1],
            c=mnist.target, s=0.1, cmap='Spectral');