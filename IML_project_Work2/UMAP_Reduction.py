import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap


sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})


from Projects.IML_UB_MAI.IML_project_Work2.utils.arff_parser import arff_to_df_normalized


CONTENT = 'datasets/vehicle.arff'
# CONTENT = 'datasets/adult.arff'
df_normalized, data_num_names, data_cat_names, data_names, class_names = arff_to_df_normalized(CONTENT)

print(df_normalized.head())

X = df = df_normalized.to_numpy()

'''Pandas DataFrame dropna() function is used to remove rows and columns with Null/NaN values.
 By default, this function returns a new DataFrame and the source DataFrame remains unchanged. 
 We can create null values using None, pandas.'''

#X_DROP = X.dropna()
#df.compactness.value_counts()
#PLot
#sns.pairplot(X, hue='compactness')


reducer = umap.UMAP(random_state=42)
reducer.fit(X.data)


embedding = reducer.transform(X.data)
# Verify that the result of calling transform is
# idenitical to accessing the embedding_ attribute
assert(np.all(embedding == reducer.embedding_))
embedding.shape


plt.scatter(embedding[:, 0], embedding[:, 1], c=X.target, cmap='Vehicles', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the Vehicles dataset', fontsize=24)





