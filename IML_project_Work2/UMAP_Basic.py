import numpy
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.io import arff
import umap

from Projects.IML_UB_MAI.IML_project_Work2.utils.arff_parser import arff_to_df_normalized


data = 'datasets/vehicle.arff'
#data = 'datasets/cmc.arff'
#data = 'datasets/adult.arff'   #Don't use this dataset yet, not is working
print(data)

df_normalized, data_num_names, data_cat_names, data_names, class_names = arff_to_df_normalized(data)

print(df_normalized.shape)
print(class_names.shape)
print(df_normalized.head)
print(class_names.head)
reducer = umap.UMAP()

print(data_names)

embedding = reducer.fit_transform(df_normalized)

embedding_max, embedding_min = embedding.max(), embedding.min()
embedding_normalize = (embedding - embedding_min)/(embedding_max - embedding_min)


embedding.shape
embedding_normalize.shape

classStr = class_names.to_string()

plt.scatter(
    embedding_normalize[:, 0],
    embedding_normalize[:, 1],
    #c=class_names,
    cmap='Spectral', s=5)
    #c = [sns.color_palette()[x] for x in classStr.map({"b": 0, "Chinstrap": 1, "Gentoo": 2})])
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.suptitle(f'UMAP REDUCER FOR {data}')
plt.show()

maxElement = numpy.amax(embedding)
maxElement_normalize = numpy.amax(embedding_normalize)

print(maxElement)
print(maxElement_normalize)


