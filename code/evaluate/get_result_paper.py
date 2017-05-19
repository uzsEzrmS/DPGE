# encoding:utf-8

import numpy as np
import pdb
import math
import sys

from numpy import *

def get_result(argv):

    filename = argv[1]  
        
    with open(filename + '.txt') as f:
        
        meanap = 0.0
        hits10 = 0.0
        hits5 = 0.0
        hits3 = 0.0
        hits1 = 0.0
        meanrank = 0.0
        cascade_num = float(argv[2])
        train_time = 1

        for line in f:
            line_split = line.split()
            meanap += float(line_split[1])
            hits10 += float(line_split[2])
            hits5 += float(line_split[3])
            hits3 += float(line_split[4])
            hits1 += float(line_split[5])
            meanrank += float(line_split[6])
            
        meanap/= cascade_num * train_time
        hits10/= cascade_num * train_time
        hits5/= cascade_num * train_time
        hits3/= cascade_num * train_time
        hits1/= cascade_num * train_time

        meanrank/= cascade_num * train_time

        print '{0}\'s meanap:{1:.4} hits10: {2:.4%} hits5: {3:.4%} hits3: {4:.4%} hits1: {5:.4%}  meanrank: {6:.4} '.format(filename,
            meanap,hits10,hits5,hits3,hits1,meanrank)

if __name__ == "__main__":
    get_result(sys.argv)
