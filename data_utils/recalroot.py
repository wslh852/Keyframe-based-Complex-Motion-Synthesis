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

def concat(data):
    left_high=[]
    right_high=[]
    left_concat=[]
    right_concat=[]
    cor=[]
    cor1=[]
    for i in range(data.shape[0]):
        if data[i][4]-data[i][7] >0:
            right_high.append(i)
        else:
            left_high.append(i)
    if right_high[0]==0:
        for i in range(len(right_high)-1):
           if right_high[i+1]-right_high[i]==1:
              continue;
           else:
               if left_concat==[]:
                   left_concat.append([0,right_high[i]])
                   cor.append(right_high[i+1])
                   
               else:
                   left_concat.append([cor[-1],right_high[i]])
                   cor.append(right_high[i+1])
        left_concat.append([cor[-1],right_high[-1]])
        for i in range(len(left_high)-1):
           if left_high[i+1]-left_high[i]==1:
              continue;
           else:
               if right_concat==[]:
                   right_concat.append([left_concat[0][-1],left_high[i]])
                   cor1.append(left_high[i+1])
                   
               else:
                   right_concat.append([cor1[-1],left_high[i]])
                   cor1.append(left_high[i+1])
        right_concat.append([cor1[-1],left_high[-1]])
    return left_concat,right_concat
def new_concat(data):
    lable=[]
    for i in range(data.shape[0]):
        if data[i][4]-data[i][7] >0:#left>right
            lable.append(1)
        else:#left<right
            lable.append(0)
    return lable
def root_pos(lable,data): 
     distance=np.zeros((data.shape[0]+1,6))
     for i in range(data.shape[0]-1):
        distance[i]=data[i+1][3:]-data[i][3:]#左脚
     root=np.zeros(3)
     for i in range(data.shape[0]):
             if lable[i]==0:
                 data[i][0]=root[0]-distance[i][0]
                 data[i][1]=root[1]-distance[i][1]
                 data[i][2]=root[2]-distance[i][2]
                 root[0]=data[i][0]
                 root[1]=data[i][1]
                 root[2]=data[i][2]
                # distance[i+1]=distance[i+1]+distance[i]
             if lable[i]==1:    
                 data[i][0]=root[0]-distance[i][3]
                 data[i][1]=root[1]-distance[i][4]
                 data[i][2]=root[2]-distance[i][5]
                 root[0]=data[i][0]
                 root[1]=data[i][1]
                 root[2]=data[i][2]
               #  distance[i+1]=distance[i+1]+distance[i]
                 
     return data[:,:3]
     
def root_concat(lable,data):
    index=np.zeros((3))
    for i in range(data.shape[0]-2):
       if lable[i]!=lable[i+1]:
           index[0]+=data[i][0]-data[i+2][0]
           index[1]+=data[i][1]-data[i+2][1]
           index[2]+=data[i][2]-data[i+2][2]
           data[i]=data[i-1]
           
       else:
           data[i]=data[i]+index
    return data
def write_data(bvh_filename,data,root):
   bvh_file = open(bvh_filename, "r")
   lines = bvh_file.readlines()
   bvh_file.close()
   l = [lines.index(i) for i in lines if 'MOTION' in i]
   data_start=l[0]
   #data_start = lines.index('MOTION\n')
   first_frame  = data_start + 3
   
   num_params = len(lines[first_frame].split(' ')) 
   num_frames = len(lines) - first_frame
                                     
   #data= np.zeros((num_frames,num_params))
   f= open("bvh数据.txt","w")
   for i in range(num_frames):
       strnum=[]
       for j in range(6):
           strnum.append(0)
          # strnum.append(str(root[i][j]))
       line = lines[first_frame + i].split(' ')
       line = line[6:len(line)]
       line=strnum+line
       for i in line:
           f.write(str(i)+" ")
       f.write('\n')
   f.close()
def root_cal(input):
    data=np.concatenate((input[:,:3],input[:,9:12],input[:,21:24]),axis=-1)
    lable=new_concat(data)
    root_tra=root_pos(lable,data)
  #  root=root_concat(lable,root_tra)
    return root_tra



#data_raw = pd.read_csv("motion_019_original_0_pos.csv")
#data=np.loadtxt("motion_019_original_0_pos.csv",delimiter=",",skiprows=1, usecols=(1,2,3,10,11,12,19,20,21))
#lable=new_concat(data)
#root_tra=root_pos(lable,data)
#root_concat=root_concat(lable,root_tra)
#bvh_filename="../train_data_bvh/motion_019_original.bvh"
#write_data(bvh_filename,data,root_concat)
#bvh_filename1="../train_data_bvh/motion_019_original.bvh"
#write_data(bvh_filename1,data,root_concat)
#print('over')


    
