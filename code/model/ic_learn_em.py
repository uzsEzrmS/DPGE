#encoding:utf-8:
import cPickle
import time
import re
import os
import pdb
import sys
import numpy as np

graph = r'../../data/digg_graph'

pairs_plus = {} #the occurance of pairs:v->w
pairs_minus = {} #the occurance of v without w
node2node = {}
with open(graph) as f:
    for line in f:
        line_split = line.split()
        pairs_plus[' '.join(line_split)] = 0
        pairs_minus[' '.join(line_split)] = 0
        try:
            node2node[line_split[0]].append(line_split[1])
        except:
            node2node[line_split[0]] = [line_split[1]]

cascade = r'../../data/cascade_digg'
with open(cascade) as f:
    for line_index, line in enumerate(f):
        if (line_index < 2066):
            line_split = line.split()
        else:
            line_split = line.split()[:-1]
        for index,node_i in enumerate(line_split):
            if index == 0:
                continue
            else:
                pair_this_turn = line_split[(index-1)]+' '+node_i
                num = pairs_plus[pair_this_turn]
                num +=1
                pairs_plus[pair_this_turn] = num
                # If 1->2 occurs，1->othernode will not occur in this cascade
                for node_j in node2node[line_split[(index-1)]]:
                    if node_j != node_i:
                        pair_this_turn = line_split[(index-1)]+' '+node_j
                        num = pairs_minus[pair_this_turn]
                        num +=1
                        pairs_minus[pair_this_turn] = num

pairs_pro = {}

for key in pairs_minus.keys():
    pairs_pro[key] = float(pairs_plus[key])/(pairs_plus[key]+pairs_minus[key])

write_path = r'../../result/'
file = open(write_path+'meme_pairs_pro_em.pkl','w')
cPickle.dump(pairs_pro,file)
file.close()
