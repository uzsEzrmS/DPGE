#encoding:utf-8:
import cPickle
import time
import re
import os
import pdb
import sys
import numpy as np
import math
import random


pkl_clu = {'average_rank':{'random':{}},'nb_app':{'random':{}},
            'ic':{'random':{}}}

def set_parameter():

    global nodeNumber 
    global cascadenumber
    global pkl_path


    nodeNumber = 1335
    cascadenumber = 2266 # 2066 training and 200 test
    pkl_path = r'../../result/'




def getpkl(path):
    #return .pkl from the target path
    doc_list = os.listdir(path)
    doc_pkl = filter(ispkl,doc_list) 

    return doc_pkl
#@profile
def ispkl(string):
    string_split = string.split('.')
    if len(string_split) == 2:
        if string.split('.')[1] == 'pkl':
            return True
        else:
            return False
    else:
        return False


def calcutedistance(infected_node,user_embedding,method):
    # we aim at predictiong the eventual node with the metric
    
    print 'The method in calcutedistance function:{0}'.format(method)

    candidate_node = list(set(range(nodeNumber))^set(infected_node))  #candidate sets are the nodes which do not be infected.
     

    if method == 'average_rank':

        distance_this_method = {}
        target_node = infected_node[-1] 

        for node in candidate_node:
            if user_embedding.has_key(node):
                distance_this_method[node] = user_embedding[node]
                print user_embedding[node]
            else:
                distance_this_method[node] = nodeNumber
            
        distance_this_method= sorted(distance_this_method.iteritems(), key=lambda d:d[1])

        return distance_this_method


    elif method == 'nb_app' :

        distance_this_method = {}
        target_node = infected_node[-1] 

        for node in candidate_node:
            dis = 0
            if user_embedding.has_key(node):
                distance_this_method[node] = user_embedding[node]
                print user_embedding[node]
            else:
                distance_this_method[node] = 0.001
                
        distance_this_method= sorted(distance_this_method.iteritems(), key=lambda d:d[1])

        return distance_this_method
            
    
    elif method == 'ic':

        distance_this_method = {}
        result = []
        target_node = infected_node[-1]

        for node in candidate_node:
            node_pair = str(target_node) + ' ' +str(node)
            if user_embedding.has_key(node_pair):
                proi =  user_embedding[node_pair]
            else:
                proi =  random.uniform(0,0.001)
            distance_this_method[node] = proi
        node_chosen = distance_this_method.keys()
        node_pro = distance_this_method.values()
        time = 0
        node_chosen_cur = random_pick(node_chosen,node_pro)
        result = [node_chosen_cur]
        while (node_chosen_cur != target_node and time <= 100):
            node_chosen_cur = random_pick(node_chosen,node_pro)
            result.append(node_chosen_cur)
            time += 1
        return result 


# @profile
def loadpkl(pkl_file):

    pkl_file = open(pkl_path+'/'+pkl_file+'.pkl','r')
    embedding  = cPickle.load(pkl_file)
    pkl_file.close()
    return embedding

# @profile
def return_embedding(filename):
    #Judge the embedding.pkl with the string
    if len(filename) == 0:
        return None

    file = filename.split('.')[0] 
    file_split = file.split('_')
    if 'meanrank' in file_split:
        pkl_clu['average_rank']['random']= loadpkl(file)

    elif 'nb' in file_split:
        pkl_clu['nb_app']['random'] = loadpkl(file)

    elif 'pairs' in file_split:
        pkl_clu['ic']['random'] = loadpkl(file)

def random_pick(some_list,probabilities):
    x=random.uniform(0,1)
    cumulative_probability=0.0
    for item,item_probability in zip(some_list,probabilities):
        cumulative_probability+=item_probability
        if x < cumulative_probability: break
    return item

def get_result(filename,user_embedding,method):

    print 'The method in get_result function:{0}'.format(method)   
    string_buff = []
    start_pos = 2066
    
    with open('../../data/cascade_digg') as f:
        for index_file, line in enumerate(f):
            
            remain_node = -1
            countright = 0
            if index_file >= start_pos:
                
                url_name = line.split()
                url_name_int = [int(x) for x in url_name]

                infected = url_name_int[::]

                correct_node = infected[remain_node]
                infected_node = infected[:-1]


                hit_10 = 0
                hit_5 = 0
                hit_3 = 0
                hit_1 = 0
                countright = 0

                if method == 'average_rank':

                    distance = calcutedistance(infected_node,user_embedding,method)


                elif method == 'nb_app':

                    distance = calcutedistance(infected_node,user_embedding,method)
                        
                
                elif method == 'ic':

                    distance = calcutedistance(infected_node,user_embedding,method)
                

                hit_num = 10
                if method == 'ic':
                    distance_keys = distance
                else:
                    distance_keys = [item[0] for item in distance]

                meanap = 0.0
                rank_all = 0

                try:
                    rank_all = distance_keys.index(correct_node)+1
                except:
                    rank_all = nodeNumber #in such situation, the node is not predicted, we set the number of nodes as its rank.

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
                # string_buff += str(index_file) + '\t' + str(meanap) + '\t' + str(hit_10) +'\t'+ str(hit_5) +'\t'+ str(hit_3) +'\t'+ str(hit_1) +'\t' +str(rank_all)  +'\n'
                print 'file: {3},meanap : {0};rank : {1};hit_10 : {2};hit_5 : {4};hit_3 : {5};hit_1 : {6}'.format(meanap,rank_all,hit_10,index_file,hit_5,hit_3,hit_1)

    fileHandler4 = open(method + '.txt','a')
    
    fileHandler4.write(''.join(string_buff))
    fileHandler4.close()


def test():
    print 'started!'
    starttime = time.time()
    set_parameter()
    pklpath  = r'../../result/'
    doc_pkl = getpkl(pklpath)


    for pkl in doc_pkl:
        return_embedding(pkl)

    for key_i in pkl_clu.keys():
        if key_i == 'average_rank':
            for key_j in pkl_clu[key_i].keys():
                if key_j == 'random': 
                    method = 'average_rank'

                user_rank = pkl_clu[key_i][key_j]

                filename = key_i+key_j 
                print 'The method tested is {0}'.format(filename)
                
                get_result(filename,user_rank,method)

        elif key_i == 'nb_app':
            for key_j in pkl_clu[key_i].keys():
                if key_j == 'random': 
                    method = 'nb_app'

                user_number = pkl_clu[key_i][key_j]

                filename = key_i+key_j 
                print 'The method tested is {0}'.format(filename)

                get_result(filename,user_number,method)


        elif key_i == 'ic':
            for key_j in pkl_clu[key_i].keys():
                if key_j == 'random': 
                    method = 'ic'
                pair_pro = pkl_clu[key_i][key_j]

                filename = key_i+key_j 
                print 'The method tested is {0}'.format(filename)

                get_result(filename,pair_pro,method)



    endtime = time.time()
    exetime = endtime - starttime
    print exetime

if __name__ == "__main__":
    test()