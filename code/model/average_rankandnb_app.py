#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import scipy.io as sio
#import matplotlib.pyplot as plt
import numpy as np  
import re
import copy
import math
import cPickle
import collections
import pdb
# Player = collections.namedtuple('Player', 'score name')
# d = {'John':5, 'Alex':10, 'Richard': 7}
# worst = sorted(Player(v,k) for (k,v) in d.items())
# player = best[1]
# player.name
# 'Richard'
# player.score
# 7

# re_splittab = re.compile(r'\t')
re_splitdou = re.compile(',')
userrank = {}

def get_rank_user(filename, num_nodes,cascade_nums):
    rank_user = {}
    train_num = 2066
    remain_node = -1

    node_num = 0
    
    with open (filename) as f:
        for index_line,line in enumerate(f):
            userrank[index_line] = []
            url_name = line.split()
            url_name_int = [int(x) for x in url_name]

            if index_line >(train_num - 1):
                infected = url_name_int[:remain_node]

            infected = url_name_int[:] 
            node_num += len(infected)
            for index_node,node in enumerate(infected):
                if rank_user.has_key(node):
                    rank_user[node].append(index_node+1)
                else:
                    rank_user[node] = [index_node+1]
    averange_node = float(node_num)/cascade_nums
    return rank_user,averange_node

def get_result(rank,num_nodes,cascade_nums,averange_node):
    
    averagerank = {}
    cascade_occur = {}

    for key,value in rank.iteritems():
        mean_rank = np.mean(value)

        cascade_occur[key] = float(value[0])/cascade_nums
        cascade_not_occur = cascade_nums - len(value)
        mean_rank += (cascade_not_occur*(averange_node+10))
        averagerank[key] = mean_rank
    
    return averagerank,cascade_occur





if  __name__ =='__main__':

    file =  '../../data/cascade_digg'
    num_nodes = 1335
    cascade_nums = 2266

    rank_user,averange_node= get_rank_user(file,num_nodes,cascade_nums)
    averagerank,cascade_occur = get_result(rank_user,num_nodes,cascade_nums,averange_node)



    write_path = r'../../result/'

    fileHandler = open(write_path + 'meme_meanrank_3000.pkl','w')
    cPickle.dump(averagerank,fileHandler)
    fileHandler.close()

    fileHandler = open(write_path + 'meme_nb_app_3000.pkl','w')
    cPickle.dump(cascade_occur,fileHandler)
    fileHandler.close()
