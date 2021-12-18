import numpy as np
labels = [1,2,2,2,4,4,4]
distances = [0.05,0.1, 0.2, 0.3, 0.3, 0.5, 0.8]
def plurality(distances, labels):
    k = len(distances)
    for kk in range(k):
        votes_ind = []
        labels = labels[:k]
        print(labels)
        for l in set(labels):
            votes_ind.append([i for i, x in enumerate(labels) if x == l])
        votes_len = [len(i) for i in votes_ind]
        print(votes_ind)
        duplicate = []
        for v in votes_ind:
            if len(v) == max(votes_len):
                duplicate.append(1)
        if len(duplicate)>1:
            k = k-1
        else:
            where = votes_len.index(max(votes_len))
            final_index = votes_ind[where][0]
            return labels[final_index]


print(plurality(distances, labels))
