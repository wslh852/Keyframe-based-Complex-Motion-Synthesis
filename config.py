import torch


class Config:
    # train info
    use_cude = True
    model_dir = "/home/lh/lihao/code/Result"
    device = torch.device("cuda:0" if use_cude and torch.cuda.is_available() else "cpu")
    train_data_proportion = 0.8
    log_freq = 200
    save_freq = 20
    loss_pic_freq = 10000
    
    window_size = 30
    window_step = 3
    # general attribute
    batch_size = 12
    learning_rate = 0.0001
    max_iteration = 500000
    noise_factor = 0.001

    # data structure

    win_step_factor = 0.1  # window step: math.ceil(win_step_factor * p_num)

    concat_foot_r = 3
    concat_foot_l = 7
    # prediction network

    num_joints=24 
    in_chans=4
    embed_dim_ratio=32
    depth=4
    num_heads=8
    mlp_ratio=2.0
    qkv_bias=True
    qk_scale=None,
    drop_rate=0.0
    attn_drop_rate=0.0
    drop_path_rate=0.2
    
    rec_loss_weights = 1
    foot_loss_weights = 1
    pos_loss_weights = 0.00001
    root_loss_weights = 0.00001
    # schedule sampling
    sampling_type = "schedule"
    schedule_sampling_decay = "exp"  # "exp" "sigmoid"  "linear"
    schedule_max_iteration = 100000
    ss_exp_k = 0.99995
    ss_sigmoid_k = 6000
    ss_linear_k = 1
    ss_linear_c = 0.000015
    schedule_sampling_limit = 0

    out_path = './test_out/outputdata.bvh'
    smooth_T = 10