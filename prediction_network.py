# !/usr/bin/env python
#
# Copyright 2020 Siyuan Wang.
#
import time 
import datetime
import torch
import time
import math
import random
import numpy as np
import torch.nn as nn
from utils import get_latest_weight_file, save_model_loss_info, save_loss_pic, load_dict
from config import Config
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from predition_loss import ReconstructionLoss
import trajectory as tra
import model_STtansformer as m

from data_utils.skeleton import Skeleton
from data_utils.quaternion_frame import convert_quaternion_frames_to_euler_frames
from data_utils.animation_data import calculate_quaternions#AnimaitonData
from data_utils.animation_data import get_positions
import torch.nn.functional as F 


def filt(seq):
    for j in range(seq.shape[0]):
        for i in range(seq.shape[1]-2):
            i=i+1
            seq[j][i]=torch.mean(seq[j][i-1:i+2],dim=0)
    return seq


def Key_frame_embeding(batch, seq_len,lab): # +25*4
    embeding=torch.zeros([batch, seq_len,lab])
    for i in range(seq_len):
        if i==0 or i==seq_len-1:
            embeding[:,i,1]=1
        else:
            embeding[:,i,0]=1
    return embeding

def slerp(a,b,t):
    cosa=a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3]
    if (cosa<0):
        b[0]=-b[0]
        b[1]=-b[1]
        b[2]=-b[2]
        b[3]=-b[3]
        cosa=-cosa
    k0=0
    k1=0
    if(cosa>0.9995):
         k0=1-t
         k1=t
    else:
        sina=math.sqrt(1-cosa*cosa)
        tht=math.atan2(sina,cosa)
        k0=math.sin((1-t)*tht)/sina
        k1=math.sin(t*tht)/sina
    c=k0*a+k1*b
    return c
def Interpolation(first_frame,last_frame,frame_number,out):
    _first_frame=first_frame[:,:].reshape(out.shape[0],24,4)
    _last_frame=last_frame[:,:].reshape(out.shape[0],24,4)
    out=out.reshape(out.shape[0],out.shape[1],24,4)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            for k in range(24):
                out[i][j][k]=slerp(_first_frame[i][k], _last_frame[i][k],1/(frame_number-1)*j)
    out=out.reshape(out.shape[0],out.shape[1],-1)
    #np.save("out",out)
    return out

def root_Interpolation(seq):
    out=np.zeros((seq.shape[0],seq.shape[1],seq.shape[2]))
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            out[i][j]=seq[i][0]+(seq[i][-1]-seq[i][0])*j/(seq.shape[1]-1)
    return out

class Prediction(nn.Module):
    def __init__(self, hparams):
        super(Prediction, self).__init__()
        self.skeleton = Skeleton()
        self.hparams = hparams
        self.conv1d_in=nn.Conv1d(96, 256, 3, padding=1)
        self.conv1d_out=nn.Conv1d(256, 96, 3, padding=1)
        self.Key_frame_embeding = nn.Linear(2, 256)
        self.Foot_concat_embeding = nn.Linear(2, 256)
        self.Root_embeding=nn.Linear(3, 256)
        self.dropout_out = nn.Dropout(p=0.2)
        self.foot_line_in=nn.Linear(2, 256)
        self.foot_line_out=nn.Linear(256, 2)
        #self.Key_frame_embeding = nn.Conv1d(2, 256, 3, padding=1)
        #self.Foot_concat_embeding = nn.Conv1d(2, 256, 3, padding=1)
        #self.Root_embeding=nn.Conv1d(3, 256, 3, padding=1)
   
        self.prelu = nn.PReLU()

        self.model_dir = hparams.model_dir
    

    def root_cal(self,pre_frame,foot_lab):
        out=np.zeros((pre_frame.shape[0],pre_frame.shape[1],3))
        joint_pos_out=np.zeros((pre_frame.shape[0],pre_frame.shape[1],72))
        pre_frame=pre_frame.cpu().detach().numpy()
        foot_lab=foot_lab.cpu().detach().numpy()
        for i in range(pre_frame.shape[0]):
            euler=convert_quaternion_frames_to_euler_frames(pre_frame[i,:,:99])
            euler=euler[:,3:].reshape(euler.shape[0],24,3)
            quaternion = calculate_quaternions(euler, self.skeleton.rotation_order)
            joint_pos = get_positions(quaternion, self.skeleton)
            root=tra.root_cal(joint_pos,foot_lab[i])
            #joint_pos=joint_pos.reshape(joint_pos.shape[0],-1)
            out[i]=root
            joint_pos=joint_pos.reshape(joint_pos.shape[0],-1)
            joint_pos_out[i]=joint_pos
        out =torch.from_numpy(out)
        joint_pos_out =torch.from_numpy(joint_pos_out)
       # joint_pos =joint_pos.unsqueeze(1)
        return out,joint_pos_out

    def stata_cov1D(self,x):
        x=x.transpose(1,2)
        conv1d_out=self.conv1d_in(x)
        _conv1d_out=conv1d_out.transpose(1,2)
        return _conv1d_out
    def Key_frame_cov1D(self,x):
        #x=x.transpose(1,2)
        key_frame_embeding=self.Key_frame_embeding(x)#128,30,256
        #_key_frame_embeding=key_frame_embeding.transpose(1,2)
        return key_frame_embeding
    def Root_embeding_cov1D(self,x):
        #x=x.transpose(1,2)
        Root_embeding_out=self.Root_embeding(x)
        #_Root_embeding_out=Root_embeding_out.transpose(1,2)
        return Root_embeding_out
    def Foot_concat_embeding_cov1D(self,x):
        #x=x.transpose(1,2)
        Foot_concat_embeding_out=self.Foot_concat_embeding(x)
        #_Foot_concat_embeding_out=Foot_concat_embeding_out.transpose(1,2)
        return Foot_concat_embeding_out
    def forward(self, x1_input, key_frame_embeding):
        Stata_conv1d_out=self.stata_cov1D(x1_input)#128,99,30
       # foot_in=self.foot_line_in(x1_input[:,:,-2:])
        Key_frame_embeding_out=self.Key_frame_cov1D(key_frame_embeding)
        #Root_embeding_out=self.Root_embeding_cov1D(x1_input[:,:,:3])
        #Foot_concat_embeding_out=self.Foot_concat_embeding_cov1D(x1_input[:,:,-2:])
        
        transformer_input=Stata_conv1d_out+Key_frame_embeding_out#+foot_in#128,30,256
        transformer_output=self.slerp_transformer(transformer_input)#128,30,256
        transformer_output=transformer_output.transpose(1,2)#128,256,30
       # foot_pre=self.foot_line_out(self.dropout_out(transformer_output.transpose(1,2)))
        out=self.conv1d_out(self.dropout_out(transformer_output))#128,99,30
        out=out.transpose(1,2)#128,30,99
        out=out+x1_input
        return out


class DanceDataset(Dataset):
    def __init__(self, train_x):
        self.train_data = train_x

    def __len__(self):
        return self.train_data.shape[0]

    def __getitem__(self, item):
        return torch.tensor(self.train_data[item], dtype=torch.float32)


def get_train_data(data, config):
   
    state_derivative_input = data[:, :, 3:99]
    Interpolation_motion=np.zeros((state_derivative_input.shape[0],state_derivative_input.shape[1],96))
    Interpolation_motion=Interpolation(state_derivative_input[:,0],state_derivative_input[:,-1],state_derivative_input.shape[1],Interpolation_motion)
    train_x = np.concatenate((Interpolation_motion,data[:,:,99:101],data), axis=-1)#222=149+73
    return train_x
def slerp2pos(Interpolation_motion):
    sk=Skeleton()
    _Interpolation_motion=Interpolation_motion.reshape(Interpolation_motion.shape[0],24,4)
    joint_pos=np.zeros((Interpolation_motion.shape[0],24,3))
    for i in range(Interpolation_motion.shape[0]):
        joint_pos[i] = get_positions(_Interpolation_motion[i], sk)
    joint_pos_out=joint_pos.reshape(Interpolation_motion.shape[0],-1)
    return joint_pos_out
def divide_data(data, window, win_step):
    divided_data = []
    for i, unit in enumerate(data):
        frame_num = len(unit)
        index = 0
        for start_frame in range(0, frame_num - window + 1, win_step):
            end_frame = start_frame + window - 1
            index += 1
            divided_data.append(unit[start_frame:end_frame + 1])
    return np.array(divided_data)


def get_random_root_pos_factor(time_factor):
    random_factor = np.zeros(time_factor.shape)
    seq_len = time_factor.shape[0]
    delta = 1 / seq_len
    random_scale = delta
    last_value = 0.0
    for i in range(seq_len):
        if i == 0:
            random_factor[i] = random.uniform(0.0, delta + random_scale)
        elif i == seq_len - 1:
            random_factor[i] = 1.0
        else:
            min_value = max(delta * (i + 1) - random_scale, last_value)
            max_value = min(delta * (i + 1) + random_scale, delta * (i + 2))
            random_factor[i] = random.uniform(min_value, max_value)
        last_value = random_factor[i]
    return random_factor


def get_time_label(win):
    time_label = []
    for i in range(1, win):
        time_label.append(i / (win - 1))

    return np.array(time_label)


def get_teacher_forcing_ratio(it, config):
    if config.sampling_type == "teacher_forcing":
        return 1.0
    elif config.sampling_type == "schedule":
        if config.schedule_sampling_decay == "exp":
            scheduled_ratio = config.ss_exp_k ** it
        elif config.schedule_sampling_decay == "sigmoid":
            if it / config.ss_sigmoid_k > 700000:
                scheduled_ratio = 0.0
            else:
                scheduled_ratio = config.ss_sigmoid_k / \
                                  (config.ss_sigmoid_k + math.exp(it / config.ss_sigmoid_k))
        else:
            scheduled_ratio = config.ss_linear_k - config.ss_linear_c * it
        scheduled_ratio = max(config.schedule_sampling_limit, scheduled_ratio)
        return scheduled_ratio
    else:
        return 0.0


def test_schedule_sample(config):
    max_it = 150000
    ratio_list = []
    for i in range(1, max_it):  # [1, max_it - 1]
        ratio = get_teacher_forcing_ratio(i, config)
        ratio_list.append(ratio)

    x = np.arange(1, max_it, 1)
    plt.plot(x, ratio_list, color='green')
    plt.show()
from data_utils.skeleton import Skeleton
skeleton = Skeleton()
def cal_pos(predict_seq,window_size,num_joint):
        root_0=np.zeros((predict_seq.shape[0],window_size,3))
        predict_seq=np.concatenate((root_0,predict_seq),axis=-1)
        out=np.zeros((predict_seq.shape[0],predict_seq.shape[1],num_joint,3))
        for i in range(predict_seq.shape[0]):
            euler=convert_quaternion_frames_to_euler_frames(predict_seq[i])
            euler=euler[:,3:].reshape(euler.shape[0],num_joint,3)
            quaternion = calculate_quaternions(euler, skeleton.rotation_order)
            joint_pos = get_positions(quaternion, skeleton)
            out[i]=joint_pos
        return out
def root_cal(data,foot_pre ,window_size, concat_foot_r,concat_foot_l ):
    #concat_lab= foot_pre[0][:,0]
    root_cal=root_pos(foot_pre,data,window_size, concat_foot_r,concat_foot_l)
    return root_cal
def root_pos(lable,data, window_size, concat_foot_r,concat_foot_l): 
     distance=np.zeros((data.shape[0],data.shape[1],6))
     for i in range(data.shape[1]-1):
        distance[:,i,:3]=data[:,i+1,concat_foot_r]-data[:,i,concat_foot_r]#左脚
        distance[:,i,3:]=data[:,i+1,concat_foot_l]-data[:,i,concat_foot_l]
     data_c=np.zeros((data.shape[0],window_size,3))
     root=np.zeros((data.shape[0],3))
     for i in range(data.shape[0]):
         for j in range(data.shape[1]):
             if lable[i][j][0]==0:
                 data_c[i][j][0]=root[i][0]-distance[i][j][0]
                 data_c[i][j][1]=root[i][1]-distance[i][j][1]
                 data_c[i][j][2]=root[i][2]-distance[i][j][2]
                 root[i][0]=data_c[i][j][0]
                 root[i][1]=data_c[i][j][1]
                 root[i][2]=data_c[i][j][2]
                # distance[i+1]=distance[i+1]+distance[i]
             if lable[i][j][0]==1:    
                 data_c[i][j][0]=root[i][0]-distance[i][j][3]
                 data_c[i][j][1]=root[i][1]-distance[i][j][4]
                 data_c[i][j][2]=root[i][2]-distance[i][j][5]
                 root[i][0]=data_c[i][j][0]
                 root[i][1]=data_c[i][j][1]
                 root[i][2]=data_c[i][j][2]
               #  distance[i+1]=distance[i+1]+distance[i]
                 
     return data_c
def train_one_iteration(model, rec_criterion, log_out, train_x, train_y, 
                                    optimizer, loss_dict,  config):
    device = config.device
    _train_x1 = train_x.to(device)
    _train_y = train_y.to(device)
  
    key_frame_embeding = Key_frame_embeding(_train_x1.shape[0],_train_x1.shape[1],2).to(device)
    input_data=_train_x1[:,:,:4*config.num_joints].reshape(_train_x1.shape[0],_train_x1.shape[1],config.num_joints,4)
    foot_emb=train_x[:,:,-2:].to(device)
    
    pre_frame,foot_pre = model.forward(input_data , key_frame_embeding, foot_emb)
    
    pre_pos = cal_pos(pre_frame.cpu().detach().numpy(), config.window_size, config.num_joints)
    gt_pos=cal_pos(_train_y[:,:,3:3+4*config.num_joints].cpu().detach().numpy(), config.window_size, config.num_joints)
    
    root_pre = root_cal(pre_pos, _train_y[:,:,-2:],  config.window_size,  config.concat_foot_r, config.concat_foot_l)
    root_pre=torch.from_numpy(root_pre).cuda()
    gt_pos=torch.from_numpy(gt_pos).cuda()
    pre_pos=torch.from_numpy(pre_pos).cuda()
    
    root=train_y[:,:,:3]
    root=root-root[:,0:1,:]
    root=root.cuda()

    optimizer.zero_grad()

    rec_loss,foot_loss,pos_loss = rec_criterion(pre_frame, _train_y[:,:,3:-2],foot_pre,_train_y[:,:,-2:],gt_pos,pre_pos)
    total_loss = config.rec_loss_weights*rec_loss + config.foot_loss_weights*foot_loss + pos_loss*config.pos_loss_weights#+ root_loss*config.root_loss_weights
    total_loss.backward()
    optimizer.step()
    

    
    if True:
        loss_dict["rec_loss"].append(rec_loss.item())
       # loss_dict["root_loss"].append(root_loss.item())
      #  loss_dict["vec_loss"].append(vec_loss.item())
        loss_dict["total_loss"].append(total_loss.item()),
        print("rec_loss:", rec_loss.detach().cpu().numpy(),
            #  " root_loss:", root_loss.detach().cpu().numpy(),
              "total_loss:", total_loss.detach().cpu().numpy(),
              )


def train_prediction(data_set, parents, gt_bone_length, config):

    start_time = time.asctime(time.localtime(time.time()))
    print("start_time:", start_time)
    start = time.time()
   # torch.cuda.is_available()
    device = config.device
    print("train on", device, torch.cuda.is_available())

    model = m.STTransformer(num_frame = config.window_size,
            num_joints = config.num_joints,
            in_chans = config.in_chans,
            embed_dim_ratio = config.embed_dim_ratio,
            depth = config.depth,
            num_heads = config.num_heads,
            mlp_ratio = config.mlp_ratio,
            qkv_bias = config.qkv_bias,
            qk_scale = config.qk_scale,
            drop_rate = config.drop_rate,
            attn_drop_rate = config.attn_drop_rate,
            drop_path_rate = config.drop_path_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    rec_criterion = ReconstructionLoss().to(device)
    
    model.train()
    
    loss_dict = {"iteration": [], "rec_loss": [], "pos_loss": [],  "root_loss": [],"total_loss": [], "contact_loss": [],"foot_loss":[],"vec_loss":[]}

    latest_info_file, latest_it = get_latest_weight_file(config.model_dir)
    print("lateset_file:", latest_info_file)
    if latest_info_file is not None:
        checkpoint = torch.load(latest_info_file)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss_dict = load_dict(config.model_dir + 'loss_%07d' % latest_it)
        print("Read the information of iteration %d successfully. \nContinue training..." % latest_it)
        
    window_size = config.window_size
    window_step = config.window_step 
    train_data = divide_data(data_set, window_size, window_step)
    train_x_y = get_train_data(train_data, config)
    for it in range(config.max_iteration):
              print("train data shape:", train_data.shape, "train_x_y:", train_x_y.shape, " iteration:", it)
              train_loader = DataLoader(DanceDataset(train_x_y), batch_size=config.batch_size,shuffle=True)
              for i, _data in enumerate(train_loader):  # batch
                log_out = False
                if i % config.log_freq == 0 or i == latest_it + 1:
                    #print("Iteration: %08d/%08d, transition length: %d" % (i, config.max_iteration, _data.shape[1]))
                    log_out = True
                    loss_dict["iteration"].append(i)
                train_x = _data[:, :, :4*config.num_joints+2]
               # vel_factor_seq = _data[..., train_x_dim:train_x_dim + config.vel_factor_dim]
                train_y = _data[:, :, 4*config.num_joints+2: ]
                #train_y_pos = _data[:, :, -72:]
                train_one_iteration(model, rec_criterion,
                                    log_out, train_x, train_y,
                                    optimizer, loss_dict,  config)
            
                cur_loss = loss_dict["total_loss"][len(loss_dict["total_loss"]) - 1]
                if cur_loss < 0:
                    min_loss = cur_loss
                    save_model_loss_info(it, model, optimizer, loss_dict, config, min_loss)
                elif it % config.save_freq == 0:
                    save_model_loss_info(it, model, optimizer, loss_dict, config, cur_loss)
                    print('save')

           


def smooth(seq,width):
    seq_rep=np.repeat(seq, width//2,axis=0)
    seq_in=np.concatenate((seq_rep[:width//2],seq,seq_rep[-width//2:]),axis=0)
    out=np.zeros((seq.shape[0],seq.shape[1]))
    for i in range(seq.shape[0]):
        i=i+width//2
        out[i-width//2]=np.mean(seq_in[i-width//2:i+width//2],axis=0)
    return out

def test_one_interval(model,  config, slerp,foot_emb):

    device = config.device
    _interpolation=slerp.to(device)
    _train_x1 = _interpolation.to(device)
    key_frame_embeding=Key_frame_embeding(_interpolation.shape[0],_interpolation.shape[1],2).to(device)
    input=_train_x1.reshape(_train_x1.shape[0],_train_x1.shape[1],24,4)
    #foot_emb=_interpolation[:,:,-2:].to(device)
    pre_frame,foot_pre= model.forward(input,key_frame_embeding,foot_emb)

    return pre_frame ,foot_pre


def test_prediction(keyframe, target, model_path, config):

    device = config.device
    model = m.STTransformer(num_frame = config.window_size,
            num_joints = config.num_joints,
            in_chans = config.in_chans,
            embed_dim_ratio = config.embed_dim_ratio,
            depth = config.depth,
            num_heads = config.num_heads,
            mlp_ratio = config.mlp_ratio,
            qkv_bias = config.qkv_bias,
            qk_scale = config.qk_scale,
            drop_rate = config.drop_rate,
            attn_drop_rate = config.attn_drop_rate,
            drop_path_rate = config.drop_path_rate).to(device)

    latest_info_file, latest_it = get_latest_weight_file(model_path)
    if latest_info_file:
        print("Load latest file", latest_info_file)
        checkpoint = torch.load(latest_info_file,map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        #print(checkpoint['model']['lstm1.weight_hh_l0'])
    else:
        print("Error: No model parameters file in", model_path)
        exit(-1)

    model.eval()

    seq_len = config.window_size

    predict_seq = np.zeros( ( target.shape[0] ,target.shape[1], 4*config.num_joints))
    foot_pre_out = np.zeros( (target.shape[0] ,target.shape[1], 2))
    slerp_out = np.zeros( (target.shape[0] ,target.shape[1], 4*config.num_joints))
  
    slerp = Interpolation(keyframe[ :,:1,3:99], keyframe[ :,-1:,3:99],seq_len, slerp_out)
    
    foot_c = target[:,:,-2:]
    foot_c = torch.tensor(foot_c, dtype=torch.float32).to(device)


    slerp = torch.tensor( slerp, dtype=torch.float32).to(device)

    pred_seq,foot_pre = test_one_interval(model, config, slerp, foot_c)

    return pred_seq,foot_pre