# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 11:28:14 2022

@author: Administrator
"""

from interval import Interval

import numpy as np
from os import listdir
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd     
# -*- coding=utf-8 -*-
import numpy as np

def new_concat(data):
    lable=np.zeros((data.shape[0],1))
    for i in range(data.shape[0]):
      #  for j in range(data.shape[1]):
            if data[i][4]-data[i][7] >0:#left>right
                lable[i]=1
            else:#left<right
                lable[i]=0
    return lable
def root_pos(lable,data): 
     distance=data[1:,3:]-data[:-1,3:]
     for i in range(distance.shape[0]):  
         if lable[i]<0.8:
                 data[i+1][0]=data[i][0]-distance[i][3]
                 data[i+1][1]=data[i][1]-distance[i][4]
                 data[i+1][2]=data[i][2]-distance[i][5]      
         else:
                 data[i+1][0]=data[i][0]-distance[i][0]
                 data[i+1][1]=data[i][1]-distance[i][1]
                 data[i+1][2]=data[i][2]-distance[i][2]
     return data[:,:3]
def root_cal(seq_pos,lable):
    foot=np.concatenate((seq_pos[:,0],seq_pos[:,3],seq_pos[:,7]),axis=-1)
    foot_lable=lable[:,0]
    root_tra=root_pos(foot_lable,foot)
    return root_tra
def test_new_concat(data):
    lable=[]
    for i in range(data.shape[0]):
        if data[i][4]-data[i][7] >0:#left>right
            lable.append(1)
        else:#left<right
            lable.append(0)
    return lable

    

    
