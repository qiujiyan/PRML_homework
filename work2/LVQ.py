import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import math

def lvq1(prototypes, trainSet, rate):
    prototypes_instances = prototypes[:,range(len(prototypes[0])-1)]
    # prototypes_class = map(str,prototypes[:,-1])
    prototypes_class = prototypes[:,-1]

    trainSet_instances = trainSet[:,range(len(trainSet[0])-1)]
    trainSet_class = trainSet[:,-1]
    
    #fit the prototypes
    knn = KNeighborsClassifier(n_neighbors=1)  
    knn.fit(prototypes_instances , prototypes_class)

    misses = 0
    epoch= 0
    max_iterations = 10
    #rate = 0.4
    while True:
        alfa_ = rate * (1 - (epoch / max_iterations))
        epoch += 1
        #alfa_ = 1/math.pow(2,epoch/6.0)
        
        for i in range(len(trainSet_instances)):
            neighbor_index = knn.kneighbors([trainSet_instances[i]], return_distance=False)
            neighbor_index = neighbor_index[0][0] 
            neighbor_class = prototypes_class[neighbor_index]

            if neighbor_class != trainSet_class[i]:
                misses += 1
                for x in range(len(prototypes_instances[neighbor_index])):
                    prototypes_instances[neighbor_index][x] -=  alfa_ * (trainSet_instances[i][x]-float(prototypes_instances[neighbor_index][x]))
            else:
                for x in range(len(prototypes_instances[neighbor_index])):
                    prototypes_instances[neighbor_index][x] +=  alfa_ * (trainSet_instances[i][x]-float(prototypes_instances[neighbor_index][x]))
            knn.fit(prototypes_instances , prototypes_class)
        #print misses,epoch
        if epoch == max_iterations:
            break
        misses = 0

    prototypes_class = np.reshape(prototypes_class,(len(prototypes_class), 1))
    prototypes= np.append(prototypes_instances, prototypes_class,axis=1)
    #prototypes = np.asarray(prototypes,dtype=float)

    return prototypes

def lvq2_1(prototypes, trainSet, rate, w):
    prototypes_instances = prototypes[:,range(len(prototypes[0])-1)]
    # for i in range(len(prototypes[:,-1])):
    #     prototypes[:,-1][i] = int(prototypes[:,-1][i])
    prototypes_class = prototypes[:,-1]
    # prototypes_class = prototypes_class.astype(np.int64)
    # print(prototypes_class)

    trainSet_instances = trainSet[:,range(len(trainSet[0])-1)]
    # for i in range(len(trainSet[:,-1])):
    #     trainSet[:,-1][i] = int(trainSet[:,-1][i])
    # trainSet_class = trainSet[:,-1]
    # trainSet_class = trainSet_class.astype(np.int64)
    trainSet_class = trainSet[:,-1]
    
    #fit the prototypes
    knn = KNeighborsClassifier(n_neighbors=2,n_jobs=-1)  
    knn.fit(prototypes_instances , prototypes_class)

    misses = 0
    epoch= 0
    max_iterations = 10
    #rate = 0.4
    #w = 0.6
    s = (1.0-w)/(1.0+w)

    while True:
        alfa_ = rate * (1 - (epoch / max_iterations))
        epoch += 1
        #alfa_ = 1/math.pow(2,epoch/6.0)
        
        for i in range(len(trainSet_instances)):
            print("{} / {}".format(i,len(trainSet_instances)))
            neighbor_dist, neighbor_index = knn.kneighbors([trainSet_instances[i]])
            neighbor_index = neighbor_index[0]
            neighbor_dist = neighbor_dist[0]
            neighbor_class = (prototypes_class[neighbor_index[0]],prototypes_class[neighbor_index[1]])
            
            if min(neighbor_dist[0]/(neighbor_dist[1]+0.00000000000001),neighbor_dist[1]/(neighbor_dist[0]+0.00000000000001)) > s:
                if neighbor_class[0] != neighbor_class[1]:
                    if neighbor_class[0] == trainSet_class[i]:
                        for x in range(len(prototypes_instances[neighbor_index[0]])):
                            prototypes_instances[neighbor_index[0]][x] +=  alfa_ * (trainSet_instances[i][x]-float(prototypes_instances[neighbor_index[0]][x]))
                        for x in range(len(prototypes_instances[neighbor_index[1]])):
                            prototypes_instances[neighbor_index[1]][x] -=  alfa_ * (trainSet_instances[i][x]-float(prototypes_instances[neighbor_index[1]][x]))
                        knn.fit(prototypes_instances , prototypes_class)
                    elif neighbor_class[1] == trainSet_class[i]:
                        for x in range(len(prototypes_instances[neighbor_index[0]])):
                            prototypes_instances[neighbor_index[0]][x] -=  alfa_ * (trainSet_instances[i][x]-float(prototypes_instances[neighbor_index[0]][x]))
                        for x in range(len(prototypes_instances[neighbor_index[1]])):
                            prototypes_instances[neighbor_index[1]][x] +=  alfa_ * (trainSet_instances[i][x]-float(prototypes_instances[neighbor_index[1]][x]))
                        knn.fit(prototypes_instances , prototypes_class)
                    else:
                        misses += 1

        if epoch == max_iterations:
            break
        misses = 0

    prototypes_class = np.reshape(prototypes_class,(len(prototypes_class), 1))
    prototypes= np.append(prototypes_instances, prototypes_class,axis=1)
    #prototypes = np.asarray(prototypes,dtype=float)

    return prototypes


def lvq3(prototypes, trainSet, rate, w, e):
    prototypes_instances = prototypes[:,range(len(prototypes[0])-1)]
    prototypes_class = map(str,prototypes[:,-1])

    trainSet_instances = trainSet[:,range(len(trainSet[0])-1)]
    trainSet_class = trainSet[:,-1]
    
    #fit the prototypes
    knn = KNeighborsClassifier(n_neighbors=2)  
    knn.fit(prototypes_instances , prototypes_class)

    misses = 0
    epoch= 0
    max_iterations = 10
    #rate = 0.4
    #w = 0.6
    s = (1.0-w)/(1.0+w)
    #e = 0.5
    while True:
        alfa_ = rate * (1 - (epoch / max_iterations))
        epoch += 1
        #alfa_ = 1/math.pow(2,epoch/6.0)
        
        for i in range(len(trainSet_instances)):
            neighbor_dist, neighbor_index = knn.kneighbors([trainSet_instances[i]])
            neighbor_index = neighbor_index[0]
            neighbor_dist = neighbor_dist[0]
            neighbor_class = (prototypes_class[neighbor_index[0]],prototypes_class[neighbor_index[1]])
            
            if min(neighbor_dist[0]/(neighbor_dist[1]+0.00000000000001),neighbor_dist[1]/(neighbor_dist[0]+0.00000000000001)) > s:
                if neighbor_class[0] != neighbor_class[1]:
                    if neighbor_class[0] == trainSet_class[i]:
                        for x in range(len(prototypes_instances[neighbor_index[0]])):
                            prototypes_instances[neighbor_index[0]][x] +=  alfa_ * (trainSet_instances[i][x]-float(prototypes_instances[neighbor_index[0]][x]))
                        for x in range(len(prototypes_instances[neighbor_index[1]])):
                            prototypes_instances[neighbor_index[1]][x] -=  alfa_ * (trainSet_instances[i][x]-float(prototypes_instances[neighbor_index[1]][x]))
                        knn.fit(prototypes_instances , prototypes_class)
                    elif neighbor_class[1] == trainSet_class[i]:
                        for x in range(len(prototypes_instances[neighbor_index[0]])):
                            prototypes_instances[neighbor_index[0]][x] -=  alfa_ * (trainSet_instances[i][x]-float(prototypes_instances[neighbor_index[0]][x]))
                        for x in range(len(prototypes_instances[neighbor_index[1]])):
                            prototypes_instances[neighbor_index[1]][x] +=  alfa_ * (trainSet_instances[i][x]-float(prototypes_instances[neighbor_index[1]][x]))
                        knn.fit(prototypes_instances , prototypes_class)
                    else:
                        misses += 1
                elif neighbor_class[0] == neighbor_class[1] == trainSet_class[i]:
                    for x in range(len(prototypes_instances[neighbor_index[0]])):
                        prototypes_instances[neighbor_index[0]][x] +=  e * alfa_ * (trainSet_instances[i][x]-float(prototypes_instances[neighbor_index[0]][x]))
                    for x in range(len(prototypes_instances[neighbor_index[1]])):
                        prototypes_instances[neighbor_index[1]][x] +=  e * alfa_ * (trainSet_instances[i][x]-float(prototypes_instances[neighbor_index[1]][x]))
                    knn.fit(prototypes_instances , prototypes_class)
                else:
                    misses += 1
        #print misses,epoch
        if epoch == max_iterations:
            break
        misses = 0

    prototypes_class = np.reshape(prototypes_class,(len(prototypes_class), 1))
    prototypes= np.append(prototypes_instances, prototypes_class,axis=1)
    #prototypes = np.asarray(prototypes,dtype=float)

    return prototypes