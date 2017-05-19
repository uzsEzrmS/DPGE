#encoding:utf-8:
import cPickle
import time
import re
import os
import pdb
import numpy as np
import math
import sys, getopt


# 目前共分三大类，之后需要细分
def softmax(inputlength):
    input_array = np.array(range(1,(1+inputlength)))
    output_array = np.exp(input_array)/sum(np.exp(input_array))
    return output_array

def energyfunction_el(head,tail,user_embedding,user_variance,rel_embedding,rel_variance,dimension):
    # get the exceptation likelihood between two nodes

    
    score = 0

    means = user_embedding[head] - rel_embedding - user_embedding[tail]
    varis = user_variance[head] + user_variance[tail] + rel_variance

    for k in range(dimension):
        vari = user_variance[head][k] + user_variance[tail][k] + rel_variance[k]
        mean = user_embedding[head][k] - rel_embedding[k] - user_embedding[tail][k]
      
        score += math.log(vari)
        score += (mean)**2/vari

    return -0.5*score

def training_prob_triplets(head,tail,rel,head_variance,tail_variance):

    total_variance = head_variance**2 + tail_variance**2 + 1

    mixed_prob = 1e-100
    error_c = head + rel - tail
    
    mixed_prob = np.exp(-sum(np.abs(error_c))/total_variance)

    return mixed_prob

def set_parameter():

    global nodeNumber
    global nodenum_cascade

    nodenum_cascade = 10 
    nodeNumber = 1335

def calcutedistance(infected_node,user_embedding,method,argv,user_variance = None,rel_embedding = None,rel_variance = None):
    # we aim at predictiong the eventual node with the metric    
    print 'dimension is:' ,dimension
 
    candidate_node = list(set(range(nodeNumber))^set(infected_node)) #candidate sets are the nodes which do not be infected.
    target_nodes = infected_node

    target_node_weight = softmax(len(target_nodes))     

    if method == 'CDK_random':

        distance_this_method = {}
        for node in candidate_node:
            dis = 0
            for target_node_index,target_node in enumerate(target_nodes):
                for k in xrange (dimension):
                    dis += target_node_weight[target_node_index]*((user_embedding[target_node][0][k] - user_embedding[node][0][k]) **2)

            distance_this_method[node] = np.sqrt(dis)
        distance_this_method= sorted(distance_this_method.iteritems(), key=lambda d:d[1])

        return distance_this_method


    elif argv[1] == 'TransE' or (argv[1] == 'CDK' and argv[2] !='random'):

        distance_this_method = {}

        for node in candidate_node:
            dis = 0
            for target_node_index,target_node in enumerate(target_nodes):
                for k in xrange (dimension):
                    dis += target_node_weight[target_node_index]*((user_embedding[target_node][k] + rel_embedding[k] - user_embedding[node][k]) **2)

            distance_this_method[node] = np.sqrt(dis)
        distance_this_method= sorted(distance_this_method.iteritems(), key=lambda d:d[1])

        return distance_this_method
   
            
    
    elif argv[1] == 'DPGE':

        distance_this_method = {}

        for node in candidate_node:
            proi = 0.0
            for target_node_index,target_node in enumerate(target_nodes):
                proi += target_node_weight[target_node_index]*training_prob_triplets(user_embedding[target_node],user_embedding[node],rel_embedding,user_variance[target_node],user_variance[node])
            distance_this_method[node] = proi

        distance_this_method= sorted(distance_this_method.iteritems(), key=lambda d:d[1],reverse=True)

        return distance_this_method


    elif argv[1] == 'KG2EEL':

        distance_this_method = {}

        for node in candidate_node:
            score_i = 0.0
            for target_node_index,target_node in enumerate(target_nodes):
                score_i += target_node_weight[target_node_index] * energyfunction_el(target_node,node,user_embedding,user_variance,rel_embedding,rel_variance,dimension)
            distance_this_method[node] = score_i

        distance_this_method= sorted(distance_this_method.iteritems(), key=lambda d:d[1])


        return distance_this_method


def training_prob_triplets(head,tail,rel,head_variance,tail_variance):

    total_variance = head_variance**2 + tail_variance**2 + 1

    mixed_prob = 1e-100
    error_c = head + rel - tail
    
    mixed_prob = np.exp(-sum(np.abs(error_c))/total_variance)

    return mixed_prob
    
def loadpkl(pkl_file):

    pkl_file = open(pkl_path+'/'+pkl_file,'r')
    embedding  = cPickle.load(pkl_file)
    pkl_file.close()
    return embedding


def get_result(user_embedding,method,argv,user_variance = None,rel_embedding = None,rel_variance = None):

    print 'The method in get_result function:{0}'.format(method)
    
    train_num = 2066
    string_buff = []

    with open('../../data/cascade_digg') as f:
        for index_file, line in enumerate(f):
            if index_file >(train_num - 1):
            
                url_name = line.split()
                url_name_int = [int(x) for x in url_name]

                sourceNumber =  url_name_int[0]                     
                infected = url_name_int
    

                correct_node = infected[-1]
                infected_node = infected[:-1]


                hit_10 = 0
                hit_5 = 0
                hit_3 = 0
                hit_1 = 0
                countright = 0
                    
                  
                if  method == 'CDK_random':

                    distance = calcutedistance(infected_node,user_embedding,method,argv)


                elif argv[1] == 'TransE' or (argv[1] == 'CDK' and argv[2] !='random'):


                    distance = calcutedistance(infected_node,user_embedding,method,argv,rel_embedding = rel_embedding[index_file])
	            
	                    
	            
                elif argv[1] == 'DPGE':

                    distance = calcutedistance(infected_node,user_embedding,method,argv,user_variance = user_variance,rel_embedding = rel_embedding[index_file])


                elif argv[1] == 'KG2EEL':

                    distance = calcutedistance(infected_node,user_embedding,method,argv,user_variance = user_variance,rel_embedding = rel_embedding[index_file],rel_variance = rel_variance[index_file])



                hit_num = 10
                distance_keys = [item[0] for item in distance]

                meanap = 0.0
                rank_all = 0

                
                rank_all = distance_keys.index(correct_node)+1

                meanap = 1.0/(rank_all)

                if correct_node in distance_keys[:10]: 
                    hit_10 += 1
                if correct_node in distance_keys[:5]: 
                    hit_5 += 1
                if correct_node in distance_keys[:3]: 
                    hit_3 += 1
                if correct_node in distance_keys[:1]: 
                    hit_1 += 1


                string_buff.append(str(index_file) + '\t' + str(meanap) + '\t' + str(hit_10) +'\t'+ str(hit_5) +'\t'+ str(hit_3) +'\t'+ str(hit_1) +'\t' +str(rank_all)  +'\n')
                print 'file: {3},meanap : {0};rank : {1};hit_10 : {2};hit_5 : {4};hit_3 : {5};hit_1 : {6}'.format(meanap,rank_all,hit_10,index_file,hit_5,hit_3,hit_1)


    fileHandler4 = open(method + '_' + str(dimension)+'_time.txt','a') 
    
    fileHandler4.write(''.join(string_buff))
    fileHandler4.close()


def test(argv):
    
    if len(argv) == 0:
        print 'no input'
    else:
        len_argv = len(argv)
        method = argv[1] + '_' + argv[2]  
    print 'started!'
    starttime = time.time()
    set_parameter()
    global pkl_path
    global dimension 
    write_path = r'../../result/'
    pkl_path = r'../../result/'
    dimension = 100

    if argv[1] == 'TransE' or (argv[1] == 'CDK' and argv[2] != 'random'):
        user_embedding = loadpkl(argv[3])
        rel_embedding = loadpkl(argv[4])
        print 'The method tested is {0}'.format(method)
            
        get_result(user_embedding,method,argv,rel_embedding = rel_embedding)
    elif argv[1] == 'CDK' and argv[2] == 'random':
        
        user_embedding = loadpkl(argv[3])

        print 'The method tested is {0}'.format(method)

        get_result(user_embedding,method,argv)
    elif argv[1] == 'DPGE':
        
        user_variance = loadpkl(argv[5])
        user_embedding = loadpkl(argv[3])
        rel_embedding = loadpkl(argv[4])

        print 'The method tested is {0}'.format(method)

        get_result(user_embedding,method,argv,user_variance = user_variance,rel_embedding = rel_embedding)

    elif argv[1] == 'KG2EEL':

        rel_variance = loadpkl(argv[6])
        user_variance = loadpkl(argv[5])
        user_embedding = loadpkl(argv[3])
        rel_embedding = loadpkl(argv[4])

        print 'The method tested is {0}'.format(method)

        get_result(user_embedding,method,argv,user_variance = user_variance,rel_embedding = rel_embedding,rel_variance = rel_variance)

    endtime = time.time()
    exetime = endtime - starttime
    print exetime

if __name__ == "__main__":
    # argv parameters: Model, method
    test(sys.argv)
