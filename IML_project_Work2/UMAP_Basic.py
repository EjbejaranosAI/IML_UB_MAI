import numpy
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.io import arff
import umap


from Projects.IML_UB_MAI.IML_project_Work2.utils.arff_parser import arff_to_df_normalized


#data = 'datasets/vehicle.arff'
#data = 'datasets/cmc.arff'
data = 'datasets/adult.arff'   #Don't use this dataset yet, not is working
print(data)

df_normalized, data_names, classes = arff_to_df_normalized(data)

print(df_normalized.shape)
print(classes.shape)
print(df_normalized.head)
print(classes.head)
reducer = umap.UMAP()

print(data_names)

embedding = reducer.fit_transform(df_normalized)

embedding_max, embedding_min = embedding.max(), embedding.min()
embedding_normalize = (embedding - embedding_min)/(embedding_max - embedding_min)


embedding.shape
embedding_normalize.shape

classStr = classes.to_string()

# assign colours to every class
color = ['#FF3333', '#AAFF33', '#33FFCA', '#FF9133', '#33BDFF', '#0400FF', '#D400FF', '#FF008D', '#FFF633']

print('the color that we have are:',color)



for i, Class in enumerate(set(classes)):
    plt.scatter(embedding_normalize[:, 0][classes == Class],embedding_normalize[:, 1][classes == Class], cmap='Spectral', s=5, c = color[i])
plt.gca().set_aspect('equal', 'datalim')
plt.suptitle(f'UMAP REDUCER FOR {data}')



plt.show()



maxElement = numpy.amax(embedding)
maxElement_normalize = numpy.amax(embedding_normalize)

print(maxElement)
print(maxElement_normalize)


