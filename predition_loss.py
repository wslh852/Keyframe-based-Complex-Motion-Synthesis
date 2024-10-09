import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        #self.velocity_dim = config.velocity_dim

    def forward(self, pre_frame, train_y,foot_pre,foot_tar,pos_pre,pos_gt):
        MSE_loss = nn.MSELoss()
        CEL_loss=torch.nn.CrossEntropyLoss()
        #CEL_loss=torch.nn.CrossEntropyLoss()
        rec_loss = MSE_loss(pre_frame[:,:,:],train_y)
        #root_loss = MSE_loss(root,root_pre)
        pos_loss = MSE_loss(pos_pre,pos_gt)
        #root_loss=MSE_loss(root,root_pre)rec_loss = MSE_loss(pre_frame[:,:,:],_train_y)
        #foot_pre =  foot_pre.permute(0, 2, 1)
        foot_loss=  MSE_loss(foot_pre, foot_tar)
        #foot_pre=foot_pre.permute(0, 2, 1)
       # foot_loss=  CEL_loss(foot_pre, _train_y[:,:,99].long())
        return  rec_loss,foot_loss,pos_loss


class BoneLoss(nn.Module):
    def __init__(self, gt_bone_length, parents, _mean, _std, config):
        super(BoneLoss, self).__init__()
        self.gt_bone_length = gt_bone_length
        self.parents = parents
        self._mean = _mean
        self._std = _std
        self.device = config.device
        self.pos_dim = config.pos_dim

    def calculate_bone_length_for_seq(self, seq):
        # AddBackward0 [batch_size, T, size]
        src_seq = seq[..., :self.pos_dim] * self._std[:self.pos_dim] + self._mean[:self.pos_dim]

        # ViewBackward  [batch_size, T, J-1, 3]
        new_seq = src_seq.view(src_seq.shape[0], src_seq.shape[1], int(src_seq.shape[2] / 3), 3)

        root_pos = torch.tensor([[0, 0, 0]], dtype=torch.float32).to(self.device)
        root_positions = torch.unsqueeze(torch.unsqueeze(root_pos, 0), 0)
        root_positions = root_positions.repeat(src_seq.shape[0], src_seq.shape[1], 1, 1)
        # CatBackward [batch_size, T, J, 3]
        positions = torch.cat((root_positions, new_seq), 2)

        # [200, 6, 23]
        result_list = torch.empty((src_seq.shape[0], src_seq.shape[1], int(src_seq.shape[2] / 3)),
                                  dtype=torch.float32).to(self.device)
        index = 0
        for joint, parent in enumerate(self.parents):
            if parent == -1:
                continue
            # [200, 6, 3] SelectBackward
            joint_pos = positions[:, :, joint]
            parent_pos = positions[:, :, parent]
            # [200, 6] SubBackward0
            delta_x = joint_pos[..., 0] - parent_pos[..., 0]
            delta_y = joint_pos[..., 1] - parent_pos[..., 1]
            delta_z = joint_pos[..., 2] - parent_pos[..., 2]
            # [200, 6]  PowBackward0
            length_temp = (delta_x ** 2 + delta_y ** 2 + delta_z ** 2) ** 0.5
            result_list[..., index] = length_temp
            index += 1
        return result_list

    def forward(self, predict_seq, _train_x1, _train_x2):
        train_bone_length = self.calculate_bone_length_for_seq(predict_seq)
        _, gt_bone_length = torch.broadcast_tensors(train_bone_length, self.gt_bone_length)

        MSE_loss = nn.MSELoss()
        bone_loss = MSE_loss(train_bone_length, gt_bone_length)

        return bone_loss * 2


class VelocityLoss(nn.Module):
    def __init__(self, _mean, _std, config):
        super(VelocityLoss, self).__init__()
        self._mean = _mean
        self._std = _std
        self.device = config.device
        self.root_pos_dim = config.root_pos_dim
        self.pos_dim = config.pos_dim
        self.velocity_dim = config.velocity_dim
        self.vel_factor_dim = config.vel_factor_dim

    def calculate_velocity(self, src_pos_seq, src_init_pos):
        """
        :param pos_seq: the position of predict sequence [Batch_size, seq_length, J * 3]
        :param init_pos:  the position of initial frame
        :return:
        """
        # [batch_size, T + 1, J * 3]    grad_fn=<CatBackward>
        temp_positions = torch.cat((torch.unsqueeze(src_init_pos, 1), src_pos_seq), 1)
        velocity = temp_positions[:, 1:] - temp_positions[:, :-1]
        return velocity

    def get_vel_factor(self, velocity):
        batch_size = velocity.shape[0]
        seq_len = velocity.shape[1]
        joint_num = int(velocity.shape[-1] / 3)
        weight = [1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 1, 1, 1, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 1]
        parts = [1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 3, 3, 3, 3, 4, 4, 4, 4, 0, 0, 0]
        weight_sum = []

        for part in range(5):
            p_sum = 0
            for j in range(joint_num):
                if parts[j] == part:
                    p_sum += weight[j]
            weight_sum.append(p_sum)

        vel_factor = torch.empty((batch_size, seq_len, self.vel_factor_dim), dtype=torch.float32).to(self.device)
        for i in range(seq_len):
            factor = torch.zeros((batch_size, self.vel_factor_dim), dtype=torch.float32).to(self.device)
            for part in range(5):
                for j in range(joint_num):
                    if parts[j] == part:
                        factor[:, part: part + 1] = factor[:, part: part + 1] + weight[j] / weight_sum[part] * \
                                                    pow(pow(velocity[:, i:i + 1, j * 3], 2) +
                                                        pow(velocity[:, i:i + 1, j * 3 + 1], 2) +
                                                        pow(velocity[:, i:i + 1, j * 3 + 2], 2), 0.5)
            vel_factor[:, i, :] = factor

        return vel_factor

    def forward(self, predict_seq, gt):
        # velocity
        MSE_loss = nn.MSELoss()
        velocity_loss=MSE_loss(predict_seq, gt)
        return velocity_loss*50


class ContactLoss(nn.Module):
    def __init__(self, _mean, _std, config):
        super(ContactLoss, self).__init__()
        self._mean = _mean
        self._std = _std
        self.root_pos_dim = config.root_pos_dim
        self.pos_dim = config.pos_dim
        self.contact_dim = config.contact_dim
        self.velocity_dim = config.velocity_dim
        self.left_feet = config.left_foot
        self.right_feet = config.right_foot
        self.vel_factor_dim = config.vel_factor_dim
        self.contact_loc = self.contact_dim + self.velocity_dim + self.vel_factor_dim

    def calculate_foot_vels(self, src_pos_seq, src_init_pos, left_foot, right_foot):
        # [batch_size, T + 1, J * 3]    grad_fn=<CatBackward>
        temp_positions = torch.cat((torch.unsqueeze(src_init_pos, 1), src_pos_seq), 1)

        left_foot0_vel = (temp_positions[:, 1:, left_foot[0] * 3:(left_foot[0] * 3 + 3)]
                          - temp_positions[:, :-1, left_foot[0] * 3:(left_foot[0] * 3 + 3)]) ** 2
        left_foot0_vel = torch.sum(left_foot0_vel, -1, keepdim=True)
        left_foot1_vel = (temp_positions[:, 1:, left_foot[1] * 3:(left_foot[1] * 3 + 3)]
                          - temp_positions[:, :-1, left_foot[1] * 3:(left_foot[1] * 3 + 3)]) ** 2
        left_foot1_vel = torch.sum(left_foot1_vel, -1, keepdim=True)
        right_foot0_vel = (temp_positions[:, 1:, right_foot[0] * 3:(right_foot[0] * 3 + 3)]
                           - temp_positions[:, :-1, right_foot[0] * 3:(right_foot[0] * 3 + 3)]) ** 2
        right_foot0_vel = torch.sum(right_foot0_vel, -1, keepdim=True)
        right_foot1_vel = (temp_positions[:, 1:, right_foot[1] * 3:(right_foot[1] * 3 + 3)]
                           - temp_positions[:, :-1, right_foot[1] * 3:(right_foot[1] * 3 + 3)]) ** 2
        right_foot1_vel = torch.sum(right_foot1_vel, -1, keepdim=True)
        feet_vel = torch.cat((left_foot0_vel, left_foot1_vel, right_foot0_vel, right_foot1_vel), -1)
        return feet_vel  # [batch_size, seq_size, 4]

    def forward(self, predict_seq, _train_y):
        MSE_loss = nn.MSELoss()
        contact_loss=MSE_loss(predict_seq[:,:,-2:],_train_y[:,:,99:101])
     #   init_pos = _train_x1[:, 0, :self.pos_dim + self.root_pos_dim]
       # src_pos_seq = (predict_seq[..., :self.pos_dim + self.root_pos_dim] *
      #                 self._std[:self.pos_dim + self.root_pos_dim] + self._mean[:self.pos_dim + self.root_pos_dim])
     #   src_init_pos = (init_pos *
     #                   self._std[:self.pos_dim + self.root_pos_dim] + self._mean[:self.pos_dim + self.root_pos_dim])
      #  feet_vels = self.calculate_foot_vels(src_pos_seq, src_init_pos, self.left_feet,
      #                                       self.right_feet)

     #   feet_contact = torch.abs(predict_seq[..., -4:] *
     #                            self._std[-self.contact_loc:-(self.velocity_dim + self.vel_factor_dim)] + \
      #                           self._mean[-self.contact_loc:-(self.velocity_dim + self.vel_factor_dim)])
     #   contact_loss = torch.mean(torch.sum(torch.sum(feet_contact * feet_vels, dim=-1), dim=-1))
        return contact_loss*8


class KeyframeLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.root_pos_dim = config.root_pos_dim
        self.root_rot_dim = config.root_rot_dim
        self.pos_dim = config.pos_dim
        self.key_num = config.key_num

    def forward(self, predict_seq,  _train_x1, gt_seq,_mean, _std):
        key_frame1 =  _train_x1[:, 0]
        key_frame2 = gt_seq[:, -1,:101]
        predict_pos = predict_seq

        num = predict_pos.shape[1]
        MSE_loss = nn.MSELoss()
        loss = torch.zeros([]).to(self.device)
        
        if num <= self.key_num * 2:
            for i in range(num):
                t = (i + 1) / (num + 1)
                pos = predict_pos[:, i, :]
                loss = loss + (1 - t) * MSE_loss(pos, key_frame1) + t * MSE_loss(pos, key_frame2)
        else:
            
            for i in range(num -1, num):
                loss = loss + MSE_loss(predict_pos[:, i, :3], key_frame2[:,:3])
                loss = loss + MSE_loss(predict_pos[:, i, 3:], key_frame2[:,3:])
        return loss 


class SmoothLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
       

    def forward(self, predict_seq):
       
        # pos_seq  SliceBackward
        seq_num = len(predict_seq[0])
        batch_size = len(predict_seq)
        root_pos_item = torch.zeros([]).to(self.device)
       
        MSE_loss = nn.MSELoss()
        for idx in range(seq_num-1):
                 pos_temp = torch.sum(torch.pow( predict_seq[:, idx+1, :] -  predict_seq[:, idx , :], 2)) \
                                / batch_size / seq_num
               
            # AddBackward0
                 root_pos_item = root_pos_item + pos_temp
          
        loss = root_pos_item   # DivBackward0
        return loss * 1.5
