import numpy as np
import math
from Projects.IML_UB_MAI.IML_project_Work2.utils.arff_parser import arff_to_df_normalized
import time

tic = time.perf_counter()
#data = 'datasets/vehicle.arff'
data = 'datasetsCBR/bal/bal.fold.000000.train.arff'


df_normalized, data_names, classes = arff_to_df_normalized(data)

print(df_normalized.head)
print(classes)
print(classes.shape)

#First it is necessary to create the function to calculate the similarity
print(df_normalized.shape)

#This is for test the function
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#Similarity function for numeric-valued attributes
def similarityNum(x,y):
    sim = -(math.sqrt(sum((x-y)**2)))
    return print(f'The similarity for thenumeric-valued attributes : {sim}')
#Test similarity numeric-value
similarityNum(x,y)
#This is for test the function
a = np.array(['z','z','e','s','pirata'])
b = np.array(['','z','x','e','q'])

#Similarity function for Boolean and symbolic-valued attri-butes.
def similarityBoolean(x,y):
    sim = -(math.sqrt(sum(x != y)))

    return print(f'The similarity for the symbolic-valued attri-butes is: {sim}')

#test similarity boolean
similarityBoolean(a,b)


NewData = df_normalized.to_numpy()

# IB1 algorithm
# define IB1-function
def IB1(TrainingSet,classes):
    n = len(TrainingSet)
    CD = TrainingSet[0]  # CD = Concept Description
    CD_classes = classes[0]
    init = CD.shape
    classification = np.array([0])

    similarity = np.zeros(len(TrainingSet))
    for i in range(len(TrainingSet)):
        if (init)==CD.shape:
            m = 1
        else:
            m = CD.shape[0]
        sim = np.zeros(m)
        for j in range(m):
            sim[j]= similarityNum(TrainingSet[i],CD[j])
        Imax = np.argmax(sim)
        if classes[i] == CD_classes[Imax]:
            classification[i] = 1      # Correct
        else:
            classification[i] = 0     # Incorrect
        CD = np.append(CD, TrainingSet[i],axis=0)
        CD_classes = np.append(CD_classes,classes[i],axis=0)

    return CD, CD_classes, classification

IB1(NewData,classes)