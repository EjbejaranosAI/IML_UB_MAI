import numpy as np
from scipy.spatial import distance

labels = [1,2,3,2,3,4,4]
distances = [0.1,0.05, 0.2, 0.3, 0.3, 0.5, 0.8]
min_indices = [1, 0, 2, 3, 4, 5, 6]
votes_ind = []
distance = []
mean_distances = []
votes_dup = []
for l in set(labels):
    votes_ind.append([i for i, x in enumerate(labels) if x == l])

votes_len = [len(i) for i in votes_ind]
for v in votes_ind:
    if len(v) == max(votes_len):
        distance = []
        for iv in v:
            distance.append(distances[iv])
        mean_distances.append(np.mean(distance))
        votes_dup.append(v)
where_min = np.where(mean_distances ==  min(mean_distances))
final_index = votes_dup[where_min[0][0]]
print(labels[final_index[0]])
# print(votes_dup[np.where(mean_distances == min(mean_distances))])