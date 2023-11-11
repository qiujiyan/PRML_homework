import pandas as pd
from scipy.io import arff
from sklearn import preprocessing
import numpy as np
from random import shuffle

def loadData(filename):
    print("Using dataset: "+filename)
    if '.arff' in filename:
        data = arff.loadarff(filename)
        dataset = pd.DataFrame(data[0]).values
        normalizeData(dataset)
        #dataset[:,range(len(dataset[0])-1)] = preprocessing.normalize(dataset[:,range(len(dataset[0])-1)])               
    elif '.csv' in filename or '.data' in filename:
        df=pd.read_csv(filename, sep=',',header=None)
        dataset = df.values
        normalizeData(dataset)
        #dataset[:,range(len(dataset[0])-1)] = preprocessing.normalize(dataset[:,range(len(dataset[0])-1)])                     
    else:        
        print("Unknown format!!error")

    return dataset
def reverseInput(dataset):
    for i in range(len(dataset)):
        for j in range(3,len(dataset[i])):
            if(dataset[i][j] != "XXXXXXX"):
                dataset[i][j] = float(dataset[i][j])
            else:
                dataset[i][j] = float(nan)

    for i in range(len(dataset)):
        dataset[i] = dataset[i][::-1]
    return dataset

def normalizeData(dataset):
    max = list(dataset[0])
    min = list(dataset[0])

    for x in range(len(dataset)):
        for y in range(len(dataset[x])-1):
            if isinstance(dataset[x][y],float) == True:
                if dataset[x][y] > max[y]:
                    max[y] = dataset[x][y]
                if dataset[x][y] < min[y]:
                    min[y] = dataset[x][y]

    for x in range(len(dataset)):
        for y in range(len(dataset[x])-1):
            if isinstance(dataset[x][y],float) == True:
                dataset[x][y] = (dataset[x][y] - min[y])/ (max[y]-min[y])

def maxMin(dataset):
    max = list(dataset[0])
    min = list(dataset[0])

    for x in range(len(dataset)):
        for y in range(len(dataset[x])-1):
            if isinstance(dataset[x][y],float) == True:
                if dataset[x][y] > max[y]:
                    max[y] = dataset[x][y]
                if dataset[x][y] < min[y]:
                    min[y] = dataset[x][y]
    sub = []
    for i in range(len(max)):
        if isinstance(max[i],float) == True:
            sub.append(max[i] - min[i])
        else:
            sub.append("?")

    return sub


    


def getClasses(dataset):
    classes = dict.fromkeys(dataset[:,-1])
    
    return classes

def buildLookUp(dataset):
    lookup_table = dict.fromkeys(list(range(0,len(dataset[0])-1)))
    for i in range(0,len(dataset[0])-1):
        lookup_table[i] = {}

        for j in range (len(dataset)):
            if dataset[j][i] in lookup_table[i]:
                if isinstance(dataset[j][i],float) != True:
                    if dataset[j][-1] in lookup_table[i][dataset[j][i]]:
                        lookup_table[i][dataset[j][i]][dataset[j][-1]] += 1
                    else:
                        lookup_table[i][dataset[j][i]][dataset[j][-1]] = 1
            else:
                if isinstance(dataset[j][i],float) != True:
                    lookup_table[i][dataset[j][i]] = {}
                    lookup_table[i][dataset[j][i]][dataset[j][-1]] = 1

    soma = 0
    #for i in range(0,len(dataset[0])-1):
    for i in lookup_table:
            for j in lookup_table[i]:
                soma = sum(lookup_table[i][j].values())
                for k in lookup_table[i][j]:
                    lookup_table[i][j][k] = lookup_table[i][j][k] / float(soma) 
    return lookup_table

def calculateAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


def getRandomPrototypes(dataset,n):
    prototypes_book = {}
    feat_len = len(dataset[0])

    for i in range(len(dataset)):
        if dataset[i][-1] in prototypes_book:
            prototypes_book[dataset[i][-1]].append(dataset[i])
        else:
            prototypes_book[dataset[i][-1]] = []
            prototypes_book[dataset[i][-1]].append(dataset[i])
    prototypes = []

    for key in prototypes_book:
        rand_list = list(range(len(prototypes_book[key])))
        shuffle(rand_list)
        for i in range(n):
            if i <= len(rand_list):
                prototypes.append(prototypes_book[key][rand_list[i]])
    prototypes = pd.DataFrame(prototypes).values
    return prototypes

def getPrototypes(dataset):
    prototypes_book = {}
    feat_len = len(dataset[0])

    for i in range(len(dataset)):
        if dataset[i][-1] in prototypes_book:
            prototypes_book[dataset[i][-1]].append(dataset[i])
        else:
            prototypes_book[dataset[i][-1]] = []
            prototypes_book[dataset[i][-1]].append(dataset[i])
    prototypes = []
    
    for key in prototypes_book:
        temp = [0] * (len(prototypes_book[key][0])-1)
        #print len(temp)
        for i in range(len(prototypes_book[key])):
            for j in range(len(prototypes_book[key][i])-1):
                temp[j] += prototypes_book[key][i][j]
            #print temp
        temp = map(lambda x: x/len(prototypes_book[key]),temp)
        temp.append(key)
        prototypes.append(temp)

    #print prototypes
    prototypes = pd.DataFrame(prototypes).values

    #prototypes = np.asarray(prototypes,dtype=float)
        # for j in range(feat_len-1):
        #     col_len = len(list(zip(*prototypes_book[key])[j]))
        #     temp.append(sum(list(zip(*prototypes_book[key])[j])))
        #     temp[j] = temp[j]/col_len
        # temp.append(key)
        # prototypes.append(temp)

    return prototypes
