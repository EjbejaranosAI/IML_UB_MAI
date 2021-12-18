import numpy as np
from scipy.spatial import distance
distance.cityblock()
labels = np.array([1,1,1,3,3,3,3])
distances = [0.05,0.1, 0.2, 0.3, 0.3, 0.5, 0.8]

def borda(distances, labels):
    k = len(distances)
    for i in range(k):
        duplicate = []
        scores = []
        scores_sum = []
        labels = labels[:k]
        lbls = list(set(labels))
        for kk in range(k):
            score = k - (kk + 1)
            scores.append(score)
        scores = np.array(scores)
        for l in lbls:
            ind = np.where(labels == l)
            scores_sum.append(np.sum(scores[ind]))
        print(lbls)
        print(scores_sum)
        for s in scores_sum:
            if s == max(scores_sum):
                duplicate = np.append(duplicate, 1)
        if len(duplicate) > 1:
            k = k - 1
        else:
            where = np.where(max(scores_sum))
            final_label = lbls[where[0][0]]
            return final_label

b = borda(distances, labels)
print(b)






