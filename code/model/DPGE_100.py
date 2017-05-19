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



def training_prob_triplets(head,tail,rel,head_variance,tail_variance):

    total_variance = head_variance**2 + tail_variance**2 + 1

    mixed_prob = 1e-100
    error_c = head + rel - tail
    
    mixed_prob = np.exp(-sum(np.abs(error_c))/total_variance)

    return mixed_prob


print 'started!'



starttime = time.time()


nodeNumber = 1335
dimension = 100 #k
nodenum_cascade = 10
margin = np.exp(2.0) #gamma

cascadenumber = 2266 
train_num = 2066      


#mean vector and variance vector initialize
user_embedding = (2 * np.random.randn(nodeNumber*dimension).reshape(nodeNumber,-1) -1) * math.sqrt(6.0/dimension) 
user_variance = np.zeros((nodeNumber,1)).reshape(nodeNumber,-1)
rel_embedding = (2 * np.random.random(cascadenumber*dimension).reshape(cascadenumber,-1) - 1) * math.sqrt(6.0/dimension) 

print 'done!'
#train
print 'dimension is:',dimension


totaltimes = 1000
remain_node = -1

index_file = 0
alpha = 0.01 # learning rate


with open('../../data/cascade_digg') as f:
    for line in f:
        
        url_name = line.split()
        url_name_int = [int(x) for x in url_name]


        sourceNumber =  url_name_int[0] #we define the first url as the source node

        #After the index_file is bigger than the train_num, we split the final node in the cascade.(Which is to be test)
        if index_file >(train_num - 1):
            infected = url_name_int[:remain_node]

        infected = url_name_int[1:] #Except the source node
        cascadeSequence = [float('inf') for x in range(nodeNumber)]

        cascadeSequence[sourceNumber] = float(-1)
        time_infected = 0
        for it in infected: 
            cascadeSequence[it] = float(time_infected)
            time_infected += 1 

        timestep = 0
        execution = 0

        not_infected = list(set(range(nodeNumber))^set(infected)) # extract the node not in the infected sequence
        not_infected_this_turn = deepcopy(not_infected)
        
        while timestep < totaltimes:

            timestep += 1 

            node_i = random.choice(infected) 
            node_i_index = infected.index(node_i)
            if node_i_index == (len(infected) - 1): #Which means the node chosen this time is the final node of cascade
                candidate = not_infected_this_turn # The candidate set is the nodes not infected
            else:
                candidate = infected[(node_i_index+1):]
                candidate.extend(not_infected_this_turn) 
            
            node_j = random.choice(candidate) # choose a node from the candidate set

            if cascadeSequence[node_i] < cascadeSequence[node_j]:

                prob_true = training_prob_triplets(user_embedding[sourceNumber],user_embedding[node_i],rel_embedding[index_file],user_variance[sourceNumber] ,user_variance[node_i])
                
                prob_false = training_prob_triplets(user_embedding[sourceNumber],user_embedding[node_j],rel_embedding[index_file],user_variance[sourceNumber] ,user_variance[node_j])


                if prob_true/prob_false < margin:

                    execution += 1

                    total_variance = user_variance[sourceNumber] **2 + user_variance[node_i] **2 + 1
                    
                    total_variance_f = user_variance[sourceNumber] **2 + user_variance[node_j] **2 + 1

                    srt_true = sum(abs(user_embedding[sourceNumber] + rel_embedding[index_file] - user_embedding[node_i]))
                    srt_false = sum(abs(user_embedding[sourceNumber] + rel_embedding[index_file] - user_embedding[node_j]))
                    prob_local_true = np.exp(-1*srt_true / total_variance)
                    prob_local_false = np.exp(-1*srt_false / total_variance_f)

                    thres = -0.01

                    user_variance[sourceNumber] += alpha *2 * prob_local_true /prob_true * srt_true/total_variance /total_variance *user_variance[sourceNumber] 
                
                    user_variance[sourceNumber] = max(-1*thres, min(thres, user_variance[sourceNumber]))

                    user_variance[node_i] += alpha * 2 *prob_local_true /prob_true * srt_true/total_variance /total_variance * user_variance[node_i]
                    user_variance[node_i] = max(thres, min(thres, user_variance[node_i]))

                    user_variance[sourceNumber] -= alpha *2 * prob_local_false/prob_false*srt_false/total_variance_f/total_variance_f*user_variance[sourceNumber]
                    user_variance[sourceNumber] = max(thres, min(thres, user_variance[sourceNumber]))

                    user_variance[node_j] -= alpha *2 * prob_local_false/prob_false * srt_false/total_variance_f/total_variance_f * user_variance[node_j] 
                    user_variance[node_j] = max(thres, min(thres, user_variance[node_j]))

                    sign_hrt = np.sign(user_embedding[sourceNumber] + rel_embedding[index_file] - user_embedding[node_i]) 
                    sign_hrt_f = np.sign(user_embedding[sourceNumber] + rel_embedding[index_file] - user_embedding[node_j]) 

                    user_embedding[sourceNumber] -= alpha * sign_hrt * prob_local_true/prob_true / total_variance

                    user_embedding[node_i] += alpha * sign_hrt * prob_local_true/prob_true/total_variance 
                
                    rel_embedding[index_file] -= alpha * sign_hrt * prob_local_true/prob_true/total_variance

                    user_embedding[sourceNumber] += alpha * sign_hrt_f * prob_local_false/prob_false/total_variance_f
                
                    user_embedding[node_j] -= alpha * sign_hrt_f /total_variance_f
                
                    rel_embedding[index_file] += alpha * sign_hrt_f * prob_local_false/prob_false/total_variance_f

                    rel_l2 = LA.norm(rel_embedding[index_file])
                    if rel_l2>1:
                        rel_embedding[index_file] = rel_embedding[index_file]/rel_l2

                    user_l2 = LA.norm(user_embedding[sourceNumber])
                    if user_l2 >1:
                        user_embedding[sourceNumber] = user_embedding[sourceNumber]/ user_l2

                    user_l2 = LA.norm(user_embedding[node_i])
                    if user_l2 >1:
                        user_embedding[node_i] = user_embedding[node_i]/ user_l2

                    user_l2 = LA.norm(user_embedding[node_j])
                    if user_l2 >1:
                        user_embedding[node_j] = user_embedding[node_j]/ user_l2
                                
                                
        del not_infected_this_turn
        del not_infected
        print 'cascade is :', (index_file+1),'execution is :', execution
        index_file +=1


write_path = r'../../result/'

fileHandler2 = open(write_path + 'digg_nodelocation_DPGE.pkl','w')
cPickle.dump(user_embedding,fileHandler2)
fileHandler2.close()

    
fileHandler3 = open(write_path + 'digg_rellocation_DPGE.pkl','w')
cPickle.dump(rel_embedding,fileHandler3)
fileHandler3.close()

fileHandler4 = open(write_path + 'digg_nodevariance_DPGE.pkl','w')
cPickle.dump(user_variance,fileHandler4)
fileHandler4.close()

endtime = time.time()
usedtime = endtime-starttime
print usedtime,'finished!'
