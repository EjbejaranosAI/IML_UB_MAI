import statistics
import numpy as np
from math import sqrt


def HVDM(array, datapoint):
    distances = []
    if len(array) > 1:
        for row in array:
            s = []
            for i, r in enumerate(row):
                column = list(np.choose(i, array.T))
                st = statistics.stdev(column)
                subtract = abs(r - datapoint[i])
                s.append((subtract / 4 * st) ** 2)
            distances.append(sqrt(sum(s)))
    else:
        s = []
        for i, r in enumerate(array[0]):
            subtract = abs(r - datapoint[i])
            s.append(subtract)
        distances.append(sqrt(sum(s)))

    return distances

# print(HVDM(a,point))
