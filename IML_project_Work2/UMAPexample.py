import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import umap
from utils.arff_parser import arff_to_df_normalized


CONTENT = 'datasets/vehicle.arff'
# CONTENT = 'datasets/adult.arff'
df_normalized, data_num_names, data_cat_names, data_names, class_names = arff_to_df_normalized(CONTENT)

print(df_normalized.head())

X = df = df_normalized.to_numpy()

# Standard data
standard_embedding = umap.UMAP(random_state=42).fit_transform(X.data)
# PLot standar data
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=0.1, cmap='Spectral')
plt.title('STANDARD DATASET')
plt.show()

# Kmeans
kmeans_labels = cluster.KMeans(n_clusters=10).fit_predict(X.data)

# Plot Kmeans
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=0.1, cmap='Spectral')
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=kmeans_labels, s=0.1, cmap='Spectral')
plt.title('KMEANS DATASET')
plt.show()

# UMAP CLUSTER


'''For visualization purposes we can reduce the data to 2-dimensions using UMAP. 
When we cluster the data in high dimensions we can visualize the result of that clustering. 
First, however, we’ll view the data colored by the digit that each data point 
represents – we’ll use a different color for each digit. This will help frame what follows.'''



# mnist = fetch_openml('mnist_784', version=1)
# mnist.target = mnist.target.astype(int)
# UMAP enhanced clustering


clusterable_embedding = umap.UMAP(
    n_neighbors=30,
    min_dist=0.0,
    n_components=2,
    random_state=42,
).fit_transform(X.data)

print('THis is the shape after to UMAP: ', X.shape)
print('This is the shape before to embedding hte dataset: ', clusterable_embedding.shape)

# mnist = fetch_openml(X, version=1)
# mnist.target = mnist.target.astype(int)

plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], s=0.1, cmap='Spectral')
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=0.1, cmap='Spectral')
plt.title('UMAP Cluster')
plt.show()
print(len(standard_embedding))
print('THere is the end of cluster UMAP')
