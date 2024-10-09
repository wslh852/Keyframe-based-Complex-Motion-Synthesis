# !/usr/bin/env python
#
# Copyright 2020 Siyuan Wang.
#

import os
import time
import yaml
import math
import random
import argparse
import torch
import numpy as np
from config import Config
import matplotlib.pyplot as plt
from data_utils.skeleton import Skeleton
from data_utils.animation_data import get_positions
from torch.utils.data import DataLoader
from prediction_network import test_prediction, divide_data, DanceDataset
from reconstruction.get_result import save_test_result_to_bvh
from data_utils.animation_data import get_vel_factor
from data_utils.constants import GLOBAL_INFO_DIRECTORY, TEST_OUT_DIRECTORY
from data_utils.quaternion_frame import convert_quaternion_frames_to_euler_frames
from data_utils.animation_data import calculate_quaternions#AnimaitonData
from data_utils.animation_data import get_positions
import torch.nn.functional as F 
random_select = True
static_begin_frame = 500
static_test_file = 1
test_interval = 300
window_size = 30






def get_npz_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith('.npy')]

def fast_npss(gt_seq, pred_seq):
    """
    Computes Normalized Power Spectrum Similarity (NPSS).

    This is the metric proposed by Gropalakrishnan et al (2019).
    This implementation uses numpy parallelism for improved performance.

    :param gt_seq: ground-truth array of shape : (Batchsize, Timesteps, Dimension)
    :param pred_seq: shape : (Batchsize, Timesteps, Dimension)
    :return: The average npss metric for the batch
    """
    # Fourier coefficients along the time dimension
    gt_fourier_coeffs = np.real(np.fft.fft(gt_seq, axis=1))
    pred_fourier_coeffs = np.real(np.fft.fft(pred_seq, axis=1))

    # Square of the Fourier coefficients
    gt_power = np.square(gt_fourier_coeffs)
    pred_power = np.square(pred_fourier_coeffs)

    # Sum of powers over time dimension
    gt_total_power = np.sum(gt_power, axis=1)
    pred_total_power = np.sum(pred_power, axis=1)

    # Normalize powers with totals
    gt_norm_power = gt_power / gt_total_power[:, np.newaxis, :]
    pred_norm_power = pred_power / pred_total_power[:, np.newaxis, :]

    # Cumulative sum over time
    cdf_gt_power = np.cumsum(gt_norm_power, axis=1)
    cdf_pred_power = np.cumsum(pred_norm_power, axis=1)

    # Earth mover distance
    emd = np.linalg.norm((cdf_pred_power - cdf_gt_power), ord=1, axis=1)

    # Weighted EMD
    power_weighted_emd = np.average(emd, weights=gt_total_power)

    return power_weighted_emd

def load_data(data_path):
    """
    data(npy): [frame_num, 215]  215 = 23 * 3 + 3 + 1 + 4 + 23 * 3 + 23 * 3
    include:
        joint_pos        # [joint_num, 3] - [23, 3]
        root_pos         # [3]
        root_rot         # [1]
        contact          # [4]
        velocity         # [joint_num, 3] - [23, 3]
        acceleration     # [joint_num, 3] - [23, 3]
    ps:
        joint_num: 24
    """
    all_files = get_npz_files(data_path)
    # train_data_num = int(len(all_files) * train_data_proportion)
    data_set = []
    data_name = []
    file_num = len(all_files)
    for i, bvh in enumerate(all_files):
        strs = bvh.split("\\")
        data_name.append(strs[-1][:-4])
        data = np.load(bvh)
        print("load file %s (%d/%d)" % (strs[-1][:-4], i, file_num))
        print("  shape:", data.shape)
        data_set.append(data[:, :])
    print()
    print("data file num:", len(data_name), len(data_set))
    return data_set, data_name





def draw_root_trajectory(pred_root_pos, true_root_pos, test_section, time_str):
    pred_pos_y = pred_root_pos[:, 1]
    pred_pos_x = pred_root_pos[:, 0]
    pred_pos_z = pred_root_pos[:, 2]

    true_pos_y = true_root_pos[:, 1]
    true_pos_x = true_root_pos[:, 0]
    true_pos_z = true_root_pos[:, 2]

    plt.plot(pred_pos_x, pred_pos_z, color='green', label='pred xz')
    plt.scatter(pred_pos_x, pred_pos_z)

    plt.plot(true_pos_x, true_pos_z, color='red', label='true xz')
    plt.scatter(true_pos_x, true_pos_z)
    plt.legend()

    plt.savefig('test_out/root_xz_' + time_str + '_' + '.jpg')
    plt.close()

    x = np.arange(0, pred_root_pos.shape[0], 1)
    plt.plot(x, pred_pos_y, color='green', label='pred y')
    plt.scatter(x, pred_pos_y)

    plt.plot(x, true_pos_y, color='red', label='true xz')
    plt.scatter(x, true_pos_y)
    plt.legend()
    plt.savefig('test_out/root_y_' + time_str + '_' + '.jpg')
    plt.close()


def draw_vel_factor(positions, true_vel_factor, test_section, time_str):
    def draw_one_vel_factor(pre_vel, true_vel, index):
        x = np.arange(0, pre_vel.shape[0], 1)
        plt.plot(x, pre_vel, color='green', label='pred y')
        plt.scatter(x, pre_vel)

        plt.plot(x, true_vel, color='red', label='true xz')
        plt.scatter(x, true_vel)
        plt.legend()
        plt.savefig('test_out/vel_' + time_str + '_' + str(index) + '.jpg')
        plt.close()

    temp_positions = np.concatenate([[positions[0]], positions], axis=0)  # [T + 1, J, 3]
    velocity = temp_positions[1:] - temp_positions[:-1]  # [T, J, 3]

    pred_vel_factor = get_vel_factor(velocity)

    for i in range(pred_vel_factor.shape[1]):
        draw_one_vel_factor(pred_vel_factor[:, i], true_vel_factor[:, i], i)


def generate_test_data(raw_data,  test_section):
    frame_num = len(raw_data)
    interval = test_section["interval"]
    frame_sum = 1

    if frame_sum + 10 >= frame_num:
        print("not enough frames(%d - %d)" % (frame_sum, frame_num))

    begin_frame = static_begin_frame
    
    test_section["begin_frame"] = begin_frame
    print("choose test data, test_file: %d, begin_frame: %d" % (test_section["test_file"], begin_frame))
    gt_data = raw_data[begin_frame:begin_frame+test_interval].reshape(test_interval//window_size,window_size,-1)
    keyframe = np.concatenate((gt_data[:,:1],gt_data[:,-1:]),axis=1)

    return  keyframe ,gt_data 




def getL2Q(true_positions, test_positions):
    # positions [B, T, 24 * 3]
    batch_size = true_positions.shape[0]
    seq_len = true_positions.shape[1]
    z = np.sum(np.linalg.norm(true_positions - test_positions, axis=-1)) / seq_len / batch_size
    return z


def get_vel_factor_for_batch(positions, vel_factor_dim):
    # [B, T, 24 * 3]
    batch_size = positions.shape[0]
    seq_len = positions.shape[1]
    joint_num = int(positions.shape[2] / 3)

    temp_positions = np.concatenate([positions[:, :1], positions], axis=1)
    velocity = temp_positions[:, 1:] - temp_positions[:, :-1]

    weight = [1, 1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 1, 1, 1, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
    parts = [0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 3, 3, 3, 3, 4, 4, 4, 4, 0, 0]
    weight_sum = []
    for part in range(5):
        p_sum = 0
        for j in range(joint_num):
            if parts[j] == part:
                p_sum += weight[j]
        weight_sum.append(p_sum)

    vel_factor = np.zeros((batch_size, seq_len, vel_factor_dim))
    for part in range(vel_factor_dim):
        for j in range(joint_num):
            if parts[j] == part:
                vel_factor[..., part] += weight[j] / weight_sum[part] * \
                                         pow(pow(velocity[..., j * 3], 2) +
                                             pow(velocity[..., j * 3 + 1], 2) +
                                             pow(velocity[..., j * 3 + 2], 2), 0.5)
    return vel_factor




def root_pos(lable,data, window_size, concat_foot_r,concat_foot_l): 
     distance=np.zeros((data.shape[0],data.shape[1],6))
     for i in range(data.shape[1]-1):
        distance[:,i,:3]=data[:,i+1,concat_foot_r]-data[:,i,concat_foot_r]#左脚
        distance[:,i,3:]=data[:,i+1,concat_foot_l]-data[:,i,concat_foot_l]
     data_c=np.zeros((data.shape[0],window_size,3))
     root=np.zeros((data.shape[0],3))
     for i in range(data.shape[0]):
         for j in range(data.shape[1]):
             if lable[i][j][0]<0.5:
                 data_c[i][j][0]=root[i][0]-distance[i][j][0]
                 data_c[i][j][1]=root[i][1]-distance[i][j][1]
                 data_c[i][j][2]=root[i][2]-distance[i][j][2]
                 root[i][0]=data_c[i][j][0]
                 root[i][1]=data_c[i][j][1]
                 root[i][2]=data_c[i][j][2]
                # distance[i+1]=distance[i+1]+distance[i]
             if lable[i][j][0]>0.5:    
                 data_c[i][j][0]=root[i][0]-distance[i][j][3]
                 data_c[i][j][1]=root[i][1]-distance[i][j][4]
                 data_c[i][j][2]=root[i][2]-distance[i][j][5]
                 root[i][0]=data_c[i][j][0]
                 root[i][1]=data_c[i][j][1]
                 root[i][2]=data_c[i][j][2]
               #  distance[i+1]=distance[i+1]+distance[i]
                 
     return data_c
def smooth(seq, width):
     seq_rep=np.repeat(seq, width//2,axis=0)
     seq_in=np.concatenate((seq_rep[:width//2],seq,seq_rep[-width//2:]),axis=0)
     out=np.zeros((seq.shape[0],seq.shape[1]))
     for i in range(seq.shape[0]):
         i=i+width//2
         out[i-width//2]=np.mean(seq_in[i-width//2:i+width//2],axis=0)
     return out
def root_cal(data,foot_pre ,window_size, concat_foot_r,concat_foot_l ):
    #concat_lab= foot_pre[0][:,0]
    root_cal=root_pos(foot_pre,data,window_size, concat_foot_r,concat_foot_l)
    return root_cal

def root_concat(lable,data):
    index=np.zeros((3))
    for i in range(data.shape[0]-1):
       if lable[i]!=lable[i+1]:
           index[0]+=data[i][0]-data[i+1][0]
           index[1]+=data[i][1]-data[i+1][1]
           index[2]+=data[i][2]-data[i+1][2]
           data[i]=data[i-1]     
       else:
           data[i]=data[i]+index
    return data
def write2bvh(data, output_file_path):
    input_bvh = './test_out/testhead.bvh'
    with open(input_bvh, 'r') as f:
            lines = f.readlines()
            
    for i in range(test_interval):
        out_str = ''
        for num  in data[i]:
            out_str = out_str + ' ' +str(num)
        out_str =out_str + '\n'
        #data_str = data_str.replace('[', '').replace(']', '\n')
        lines[144+i] = out_str
    with open(output_file_path, 'w') as f:
        for line in lines:
            f.write(line)

def test_prediction_network(args):
    config = Config()
    mean_std_data = np.load(GLOBAL_INFO_DIRECTORY + "mean_std.npz")
    mean, std = mean_std_data["mean"], mean_std_data["std"]  # 215, 215

    data_set, data_name = load_data(args.data_path)
    data_num = len(data_name)
    test_section = {"test_file": 0, "interval": [], "key_frame": [], "begin_frame": 0}
    if random_select:
        # [a, b]
        train_num = int(data_num * config.train_data_proportion)
        test_file = random.randint(train_num, data_num - 1)
    else:
        test_file = static_test_file
    test_section["test_file"] = test_file
    test_section["test_file_name"] = data_name[test_file]
    if test_file < int(data_num * config.train_data_proportion):
        use_train_data = True
    else:
        use_train_data = False

    test_section["interval"] = test_interval
    test_section["use_train_data"] = use_train_data
    skeleton = Skeleton()
    # choose data
    def loss_mse(a,b):
            root_error=torch.pow(a-b, 2)
            error = torch.sqrt(torch.sum(root_error, dim=-1) )
            error = torch.mean(error, dim=0)
            return error
        
    def cal_pos(predict_seq):
            root_0 = np.zeros((test_interval,3))
            predict_seq=np.concatenate((root_0,predict_seq),axis=-1)
            euler=convert_quaternion_frames_to_euler_frames(predict_seq)
            euler=euler[:,3:].reshape(euler.shape[0],config.num_joints,3)
            quaternion = calculate_quaternions(euler, skeleton.rotation_order)
            joint_pos = get_positions(quaternion, skeleton)
            return joint_pos,quaternion,euler
    
    print("Choose file %s (%d/%d) for testing." % (data_name[test_file], test_file, data_num))
    #loss_fn3 = torch.nn.MSELoss(reduction='mean')
   
    data = data_set[static_test_file]

    keyframe , gt_seq = generate_test_data(data,  test_section)

    root_gt = gt_seq[:,:,:3]
    gt_seq = gt_seq[:,:,3:3+4*config.num_joints+2]
        
    predict_model_path = config.model_dir + args.predict_model_path
    predict_seq,foot_pre = test_prediction( keyframe ,  gt_seq, predict_model_path ,config)
        
    predict_seq = smooth(predict_seq.cpu().detach().numpy().reshape(test_interval,-1),config.smooth_T).reshape(test_interval//config.window_size,config.window_size,-1)
        
    gt_seq=torch.from_numpy(gt_seq[:,:4*config.num_joints])

    pre_pos,quaternion,euler=cal_pos(predict_seq.reshape(test_interval,-1))
    gt_pos,quaternion_gt,euler_gt=cal_pos(gt_seq.reshape(test_interval,-1))
    quaternion =quaternion.reshape(test_interval,-1)
    root_pre = root_cal(pre_pos.reshape(1,test_interval,config.num_joints,3),foot_pre.reshape(1,test_interval,2) ,test_interval,  config.concat_foot_r, config.concat_foot_l)
    root_pre= root_pre.reshape(test_interval,-1)
    euler = euler.reshape((test_interval,-1))
        
    out_bvh_data = np.concatenate((root_pre,euler),axis=1)
    write2bvh(out_bvh_data, config.out_path)
    
    root_gt=root_gt-root_gt[0]
    root_gt=torch.from_numpy(root_gt)
    root_pre=torch.from_numpy(root_pre)
    #    gt_root=

    root_los=loss_mse(torch.from_numpy(root_gt.reshape((test_interval,-1))),torch.from_numpy(root_pre))
    
    res_loss=loss_mse(torch.from_numpy(pre_pos.reshape((test_interval,-1))),torch.from_numpy(gt_pos.reshape((test_interval,-1))))

    #K_res_loss=loss_mse(gt_seq,predict_seq)
    
    npss=fast_npss(torch.from_numpy(gt_seq[:,:,:96].reshape((test_interval,-1))),torch.from_numpy(predict_seq.reshape((test_interval,-1))))

    print('root_loss',root_los)
    print('res_loss',res_loss)
    print('npss_loss',npss)
    
def parse_args():
    parser = argparse.ArgumentParser("test")
    parser.add_argument("--test", type=str,default="prediction")
    parser.add_argument("--data_path", type=str,default="./data/Cyprus_out/")
    parser.add_argument("--predict_model_path", type=str, default="")
    return parser.parse_args()


# --test prediction --data_path data/Cyprus_out/ --predict_model_path 2021.04.14/
# --test prediction --data_path data/Cyprus_out/
if __name__ == '__main__':
    args = parse_args()
    if args.test == "prediction":
        print("Test Prediction Network!")
        # evaluation_prediction(args, 150)
        test_prediction_network(args)
