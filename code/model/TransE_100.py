# encoding:utf-8

import os
import cPickle
import time
import random
import math
import numpy as np

from copy import deepcopy

from sklearn import preprocessing

from numpy import *
from numpy import linalg as LA

def SGD(parameter_0,gradient,learningRate):
    parameter_1 = parameter_0 - learningRate*gradient


    return parameter_1


def normal(x, miu,sigma):

    return 1.0/(np.sqrt(2*np.pi)*sigma)*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));

def randn(miu,sigma,min,max):

    while(True):
        x = random.random()
        y = normal(x,miu,sigma)
        dScope = random.random()*normal(miu,miu,sigma)
        if dScope <= y:
            break

    return x

def norm(embedding):

    L = LA.norm(embedding)
    if L>1:
        embedding = embedding/float(L)

    return embedding


starttime = time.time()

nodeNumber = 1335

train_num = 2066
dimension = 100
nodenum_cascade = 10
margin = 1 #gamma
cascade_num = 2266      


user_embedding = np.zeros((nodeNumber*dimension)).reshape(nodeNumber,-1)
rel_embedding = np.zeros((cascade_num*dimension)).reshape(cascade_num,-1)

mean_low = float(-6) /math.sqrt(dimension)
mean_high = float(6) /math.sqrt(dimension)

for i in range(nodeNumber):
    for k in range(dimension):
        user_embedding[i][k] = randn(0.0,1.0/dimension,mean_low,mean_high)

    user_embedding[i] = norm(user_embedding[i])
    

for i in range(cascade_num):
    for k in range(dimension):
        rel_embedding[i][k] = randn(0.0,1.0/dimension,mean_low,mean_high)



print 'done!'
#train
print 'dimension is:',dimension

totaltimes = 100
remain_node = -1

index_file = 0


with open('../../data/cascade_digg') as f:
    for line in f:
        
        url_name = line.split()
        url_name_int = [int(x) for x in url_name]


        sourceNumber =  url_name_int[0]

        if index_file >(train_num - 1):
            infected = url_name_int[:remain_node]

        infected = url_name_int[1:] 
        cascadeSequence = [float('inf') for x in range(nodeNumber)]

        cascadeSequence[sourceNumber] = float(-1)
        time_infected = 0
        for it in infected:
            cascadeSequence[it] = float(time_infected)
            time_infected += 1 


        timestep = 0
        execution = 0

        # 设置参数
        learningRate = 0.01 

        not_infected = list(set(range(nodeNumber))^set(infected)) 
        not_infected_this_turn = deepcopy(not_infected)

        while timestep < totaltimes :
            timestep += 1
            for node_i_index,node_i in enumerate(infected):
            
                if node_i_index == (len(infected) - 1): 
                    candidate = not_infected_this_turn 
                else:
                    candidate = infected[(node_i_index+1):]
                    candidate.extend(not_infected_this_turn) 
                
                node_j = random.choice(candidate)
                    
                if cascadeSequence[node_i] < cascadeSequence[node_j]:
                    
                    

                    distancei =  (user_embedding[sourceNumber] + rel_embedding[index_file] - user_embedding[node_i])** 2
                    distance_i = sum(distancei)
                    distancej =  (user_embedding[sourceNumber] + rel_embedding[index_file] - user_embedding[node_j])** 2
                    distance_j = sum(distancej)

                    deltadistance = sqrt(distance_j) - sqrt(distance_i) #######distance_j-distance_i
                    first_d = deltadistance

                    itertion = 0
                    while itertion < 20:
                        if deltadistance >=margin:
                            break

                        execution += 1
                        
                        gradient = 2 * (user_embedding[node_i] - rel_embedding[index_file] - user_embedding[sourceNumber])
                        # print gradient
                        user_embedding[node_i] = SGD(user_embedding[node_i],gradient,learningRate)

                        gradient = 2 * ((rel_embedding[index_file] + user_embedding[sourceNumber]) - user_embedding[node_j])
                        # print gradient
                        user_embedding[node_j] = SGD(user_embedding[node_j],gradient,learningRate)
                        
                        gradient = 2* (user_embedding[node_j] - user_embedding[node_i])                                                
                        # print gradient
                        user_embedding[sourceNumber] = SGD(user_embedding[sourceNumber],gradient,learningRate)

                        gradient = 2 * (user_embedding[node_j] - user_embedding[node_i])
                        # print gradient
                        rel_embedding[index_file] = SGD(rel_embedding[index_file],gradient,learningRate)

                        distancei =  (user_embedding[sourceNumber] + rel_embedding[index_file] - user_embedding[node_i])** 2
                        distance_i = sum(distancei)
                        distancej =  (user_embedding[sourceNumber] + rel_embedding[index_file] - user_embedding[node_j])** 2
                        distance_j = sum(distancej)

                        deltadistance = sqrt(distance_j) - sqrt(distance_i) #######distance_j-distance_i

                        itertion += 1
                        distancei =  (user_embedding[sourceNumber] + rel_embedding[index_file] - user_embedding[node_i])** 2
                        distance_i = sum(distancei)
                        distancej =  (user_embedding[sourceNumber] + rel_embedding[index_file] - user_embedding[node_j])** 2
                        distance_j = sum(distancej)

                        deltadistance = sqrt(distance_j) - sqrt(distance_i) #######distance_j-distance_i
                    
                    user_embedding[sourceNumber] = norm(user_embedding[sourceNumber])
                    user_embedding[node_i] = norm(user_embedding[node_i])
                    user_embedding[node_j] = norm(user_embedding[node_j])

                    # if itertion != 0:
                    #     print 'the first is {0}'.format(first_d)
                    #     print deltadistance
                                

                                
        del not_infected_this_turn
        del not_infected
        print 'cascade is :', (index_file+1),'execution is :', execution
        index_file +=1

write_path = r'../../result/'

fileHandler2 = open(write_path + 'digg_nodelocation_TransE.pkl','w')
cPickle.dump(user_embedding,fileHandler2)
fileHandler2.close()


    
fileHandler3 = open(write_path + 'digg_rellocation_TransE.pkl','w')
cPickle.dump(rel_embedding,fileHandler3)
fileHandler3.close()


endtime = time.time()
usedtime = endtime-starttime
print usedtime,'finished!'
