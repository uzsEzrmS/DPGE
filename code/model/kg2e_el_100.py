# encoding:utf-8

import os

import cPickle
import time

import random
import math
import numpy as np

from copy import deepcopy

from numpy import *
from numpy import linalg as LA



def regularize(data_raw,parameter):
        
    data_max = float(data_raw.max())
    data_min = float(data_raw.min())
    data_range = data_max - data_min
    if isinstance(parameter,float):
        for item_index,item in enumerate(data_raw):
            if item>parameter :
                data_raw[item_index] = parameter
            elif item < -1 * parameter:
                data_raw[item_index] = -1 * parameter
            
        return data_raw
    elif isinstance(parameter,list):
        for item_index,item in enumerate(data_raw):
            if item>parameter[1]:
                data_raw[item_index] = parameter[1]
            elif item < parameter[0]:
                data_raw[item_index] = parameter[0]
            
        return data_raw
    else:
        print 'Not support input'
        return data_raw


print 'started!'

def SGD(parameter_0,gradient,learningRate):
    parameter_1 = parameter_0 - learningRate*gradient


    return parameter_1


def energyfunction(head,tail,user_embedding,user_variance,rel_embedding,rel_variance,dimension):
    # get the exceptation likelihood between two nodes

    
    score = 0

    means = user_embedding[head] - rel_embedding - user_embedding[tail]
    varis = user_variance[head] + user_variance[tail] + rel_variance

    for k in range(dimension):
        vari = user_variance[head][k] + user_variance[tail][k] + rel_variance[k]
        mean = user_embedding[head][k] - rel_embedding[k] - user_embedding[tail][k]
      
        score += math.log(vari)
        score += (mean)**2/vari

    return -0.5*score,means,varis



starttime = time.time()


# nodeNumber = 261826
nodeNumber = 1335

dimension = 100
margin = 1 #gamma
nodenum_cascade = 10
train_num = 2066

dimension = 100 #k
nodenum_cascade = 10
margin = 2 #gamma
variance_max = 5.000 
variance_min = 0.05 
cascadenumber = 2266    


variance_range = variance_max -variance_min

#mean vector and variance vector initialize

user_embedding = np.random.random(nodeNumber*dimension).reshape(nodeNumber,-1)
user_variance = np.ones((nodeNumber,dimension)).reshape(nodeNumber,-1) *0.33
rel_embedding = np.random.random(cascadenumber*dimension).reshape(cascadenumber,-1)
rel_variance = np.ones((cascadenumber,dimension)).reshape(cascadenumber,-1) *0.33


mean_range = 2.0
mean_low = -1.0


for i in xrange(nodeNumber):
    
    user_embedding[i] = mean_low + user_embedding[i]* mean_range
    user_embedding_l2 = LA.norm(user_embedding[i])
    
    user_embedding[i] = user_embedding[i]/user_embedding_l2


for i in xrange(cascadenumber):
    
    rel_embedding[i] = mean_low + rel_embedding[i]* mean_range
    rel_embedding_l2 = LA.norm(rel_embedding[i])
    
    rel_embedding[i] = rel_embedding[i]/rel_embedding_l2

print 'done!'
#train
print 'dimension is:',dimension


totaltimes = 1500 
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

        learningRate = 0.001 

        not_infected = list(set(range(nodeNumber))^set(infected)) 
        not_infected_this_turn = deepcopy(not_infected)
        
        while timestep < totaltimes:
            timestep += 1 

            for node_i_index,node_i in enumerate(infected):
            
                if node_i_index == (len(infected) - 1): 
                    candidate = not_infected_this_turn 
                else:
                    candidate = infected[(node_i_index+1):]
                    candidate.extend(not_infected_this_turn) 
                
                node_j = random.choice(candidate)
                    
                if cascadeSequence[node_i] < cascadeSequence[node_j]:

                    score_i,means_i,varis_i = energyfunction(sourceNumber,node_i,user_embedding,user_variance,rel_embedding[index_file],rel_variance[index_file],dimension)
                    
                    score_j,means_j,varis_j  = energyfunction(sourceNumber,node_j,user_embedding,user_variance,rel_embedding[index_file],rel_variance[index_file],dimension)

                    deltadistance = score_j - score_i 

                    itertion = 0
                    first_d = deltadistance
                    while itertion < 20:
                        if deltadistance >=margin:
                            break
                    
                        execution += 1
                        loss_pos_vari = 1./varis_i - (means_i/varis_i)**2

                        loss_pos_vari *= -1*0.5

                        gradient = loss_pos_vari

                        user_variance[sourceNumber] = SGD(user_variance[sourceNumber],gradient,learningRate)

                        user_variance[node_i] = SGD(user_variance[node_i],gradient,learningRate)

                        rel_variance[index_file] = SGD(rel_variance[index_file],gradient,learningRate)

                        loss_pos_mean = -1 * means_i/varis_i

                        gradient = loss_pos_mean

                        user_embedding[sourceNumber] = SGD(user_embedding[sourceNumber],gradient,learningRate)

                        user_embedding[node_i] = SGD(user_embedding[node_i],-1*gradient,learningRate)        

                        rel_embedding[index_file] = SGD(rel_embedding[index_file],-1*gradient,learningRate)



                        loss_neg_vari = 1./varis_j - (means_j/varis_j)**2

                        loss_neg_vari *= 0.5

                        gradient = loss_neg_vari

                        user_variance[sourceNumber] = SGD(user_variance[sourceNumber],gradient,learningRate)

                        user_variance[node_j] = SGD(user_variance[node_j],gradient,learningRate)

                        rel_variance[index_file] = SGD(rel_variance[index_file],gradient,learningRate)

                        loss_neg_mean = -1 * means_j/varis_j

                        gradient = loss_neg_mean

                        user_embedding[sourceNumber] = SGD(user_embedding[sourceNumber],-1*gradient,learningRate)

                        user_embedding[node_j] = SGD(user_embedding[node_j],gradient,learningRate)        

                        rel_embedding[index_file] = SGD(rel_embedding[index_file],gradient,learningRate)



                        rel_embedding_l2 = LA.norm(rel_embedding[index_file])
                            
                        rel_embedding[index_file] = rel_embedding[index_file]/rel_embedding_l2

                        rel_variance[index_file] = regularize(rel_variance[index_file],[variance_min,variance_max])

                        for i in [node_i,node_j,sourceNumber]:
                            user_embedding_l2 = LA.norm(user_embedding[i])
                            
                            user_embedding[i] = user_embedding[i]/user_embedding_l2
                            user_variance[i] = regularize(user_variance[i],[variance_min,variance_max])


                        itertion += 1
                        score_i,means_i,varis_i = energyfunction(sourceNumber,node_i,user_embedding,user_variance,rel_embedding[index_file],rel_variance[index_file],dimension)
                    
                        score_j,means_j,varis_j  = energyfunction(sourceNumber,node_j,user_embedding,user_variance,rel_embedding[index_file],rel_variance[index_file],dimension)

                        deltadistance = score_j - score_i    
                                                                
        del not_infected_this_turn
        del not_infected
        print 'cascade is :', (index_file+1),'execution is :', execution
        index_file += 1
write_path = r'../../result/'
fileHandler2 = open(write_path + 'digg_nodelocation_kg2e_el.pkl','w')
cPickle.dump(user_embedding,fileHandler2)
fileHandler2.close()

    
fileHandler3 = open(write_path + 'digg_rellocation_kg2e_el.pkl','w')
cPickle.dump(rel_embedding,fileHandler3)
fileHandler3.close()

fileHandler4 = open(write_path + 'digg_nodevariance_kg2e_el.pkl','w')
cPickle.dump(user_variance,fileHandler4)
fileHandler4.close()

fileHandler5 = open(write_path + 'digg_relvariance_kg2e_el.pkl','w')
cPickle.dump(rel_variance,fileHandler5)
fileHandler5.close()

endtime = time.time()
usedtime = endtime-starttime
print usedtime,'finished!'
