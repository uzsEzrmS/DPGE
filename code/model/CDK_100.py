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


starttime = time.time()


nodeNumber = 1335

dimension = 100
margin = 1 #gamma
nodenum_cascade = 10
train_num = 2066


user_embedding = np.random.uniform(low=-1.0, high=1.0, size=(nodeNumber,1,dimension))


print 'done!'
#train
print 'dimension is:',dimension

totaltimes = 1000
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

        learningRate = 0.01 

        not_infected = list(set(range(nodeNumber))^set(infected)) 
        not_infected_this_turn = deepcopy(not_infected)

        while timestep < totaltimes :
            timestep += 1
            # pdb.set_trace()
            for node_i_index,node_i in enumerate(infected):
            
                if node_i_index == (len(infected) - 1): 
                    candidate = not_infected_this_turn 
                else:
                    candidate = infected[(node_i_index+1):]
                    candidate.extend(not_infected_this_turn) 
                
                node_j = random.choice(candidate)

                if cascadeSequence[node_i] < cascadeSequence[node_j]:
                    distancei =  (user_embedding[sourceNumber] - user_embedding[node_i])** 2
                    distance_i = sum(distancei)
                    distancej =  (user_embedding[sourceNumber] - user_embedding[node_j])** 2
                    distance_j = sum(distancej)

                    deltadistance = sqrt(distance_j) - sqrt(distance_i) #######distance_j-distance_i

                    itertion = 0
                    while itertion < 20:
                        if deltadistance >=margin:
                            break

                        execution += 1

                        
                        gradient = 2 * (user_embedding[node_i] - user_embedding[sourceNumber])
                        # print gradient
                        user_embedding[node_i] = SGD(user_embedding[node_i],gradient,learningRate)

                        gradient = 2 * (user_embedding[sourceNumber] - user_embedding[node_j])
                        # print gradient
                        user_embedding[node_j] = SGD(user_embedding[node_j],gradient,learningRate)
                        
                        gradient = 2* (user_embedding[node_j] - user_embedding[node_i])                                                
                        # print gradient
                        user_embedding[sourceNumber] = SGD(user_embedding[sourceNumber],gradient,learningRate)

                        distancei =  (user_embedding[sourceNumber] - user_embedding[node_i])** 2
                        distance_i = sum(distancei)
                        distancej =  (user_embedding[sourceNumber] - user_embedding[node_j])** 2
                        distance_j = sum(distancej)
                        deltadistance = sqrt(distance_j) - sqrt(distance_i) #######distance_j-distance_i
                        
                        itertion += 1 


        
        del not_infected_this_turn
        del not_infected
        print 'cascade is :', (index_file+1),'execution is :', execution
        index_file +=1

write_path = r'../../result/'
fileHandler2 = open(write_path + 'digg_nodelocation_CDK.pkl','w')
cPickle.dump(user_embedding,fileHandler2)
fileHandler2.close()

endtime = time.time()
usedtime = endtime-starttime
print usedtime,'finished!'
