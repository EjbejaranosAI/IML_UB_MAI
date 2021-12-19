import statistics
import numpy as np
from math import sqrt

def HVDM(array,datapoint):
    distances = []
    for row in array:
        s = []
        for i, r in enumerate(row):
            column = list(np.choose(i, array.T))
            st = statistics.stdev(column)
            subtract = abs(r-datapoint[i])
            s.append((subtract/4*st)**2)
        distances.append(sqrt(sum(s)))
    return distances

