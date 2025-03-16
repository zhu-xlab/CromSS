import os
import json
import torch
import argparse
import numpy as np
from pathlib import Path

from utils.get_args_common import get_common_args, get_common_pretrain_args


CLS_DICT = {9:{0:'Water', 1:'Trees', 2:'Grass', 3:'Flooded vegetation', 4:'Crops', 5:'Shrub&Scrub', 6:'Built area', 7:'Bareland', 8:'Ice&Snow'}}


def get_args(multi_modal=False, use_gt=False): # multi-modality
    parser = argparse.ArgumentParser(description='Pre-Train the UNet on images and target masks')
    # paths of input data
    if use_gt:
        parser.add_argument('--train_path', type=str, default='E:/Datasets/DFC2020/lmdb/DFC2020_s12_validation.lmdb')
        parser.add_argument('--test_path', type=str, default='E:/Datasets/DFC2020/lmdb/DFC2020_s12_test.lmdb')
        parser.add_argument('--label_type', dest='lbl_t', type=str, default='dw', choices=['dw', 'osm'], help='Type of label')
        parser.add_argument('--input_type', dest='it', type=str, default='s12', choices=['s1', 's2', 's12'], help='Type of input data')
        parser.add_argument('--s1_norm', action='store_true', default=False, help='Whether using (x-u)/std normalization for s1')
        parser.add_argument('--s2_norm', action='store_true', default=False, help='Whether using ./10000 normalization for s2')
    else:
        parser.add_argument('--data_path', type=str, default=r'C:\liuchy\Research\Projects\20230412_NL_Pretrain\Codes\data_prepare\toy_data_SSL4EO')
        parser.add_argument('--data_name_s1', dest='dn1', type=str, default='0k_251k_uint8_s1.lmdb')
        parser.add_argument('--data_name_s2', dest='dn2', type=str, default='0k_251k_uint8_s2c.lmdb')
        parser.add_argument('--data_name_label', dest='dnl', type=str, default='dw_labels.lmdb')
        parser.add_argument('--input_type', dest='it', type=str, default='s12', choices=['s1', 's2', 's12'], help='Type of input data')
        parser.add_argument('--norm', action='store_true', default=False, help='Whether using ./10000 normalization')
    # multi-modality settings
    if multi_modal:
        parser.add_argument('--fusion_type', dest='fuse_type', type=str, choices=['mid', 'late'], default='mid', 
                            help='Type of fusion method')
        parser.add_argument('--use_common_feature', dest='use_com_feat', action='store_true', # default=False,
                            help='Whether to use concatenated features as input')
        parser.add_argument('--consist_loss_type', dest='cl_type', type=str, choices=['mse', 'ce', 'kl', 'none'], default='ce', 
                            help='Type of consistency loss used in multi-modal training')
        parser.add_argument('--consist_loss_weight', dest='cl_weight', type=float, default=1,
                            help='Weight of consistency loss')
        # sample selection settings
        parser.add_argument('--sample_selection', dest='sc', action='store_true', default=False,
                            help='Whether to use sample selection by confidence.')
        parser.add_argument('--sample_selection_rmup_func', dest='sc_rmup_func', type=str, choices=['linear', 'exp', 'none'], default='none',
                            help='Function used for sample selection ramp-up')
        parser.add_argument('--sample_selection_rmup_len', dest='sc_rmup_len', type=int, default=100,
                            help='Length of sample selection ramp-up')
        parser.add_argument('--sample_selection_conf_prop', dest='sc_conf_prop', type=float, default=0.5,
                            help='Proportion of confident samples to be selected')
        parser.add_argument('--sample_selection_unc_prop', dest='sc_unc_prop', type=float, default=0.5,
                            help='Proportion of uncertain samples to be selected')
        
    # get common arguments
    parser = get_common_args(parser)
    parser = get_common_pretrain_args(parser)
    
    # initialize arguments
    args = parser.parse_args()
    
    # computing environment parameters
    args = get_device_info(args)
    
    # set saving dir
    args = set_log_files(args, multi_modal)
    
    # set class dict for logging images
    args.cls_dict = CLS_DICT[args.n_classes]
    return args
    

def set_log_files(args, multi_modal=False):
    
    # set unet_type according to model settings: unet_gt or unet_ns(_rm)
    lbl_t = 'gt' if args.experiment=='gt_labels' else 'ns'
    args.unet_type = 'PT'
    # learning rate tags
    ea = int(np.ceil(np.log10(1/args.lr)))
    eb = int(args.lr*10**ea)
    # set group and job types
    if multi_modal:
        assert args.it == 's12', f'Please set input_type to \'s12\' when using multi-modal data!'   
        mtag = 3 if args.use_com_feat else 2
        args.unet_type += f'_{args.fuse_type[0]}{mtag}_{args.frm}_{args.mt}_{args.it}_{lbl_t}'
        args.job_type = f'{args.loss_type}_{args.cl_type}_lr_{eb}e{ea}' 
        if args.ls: args.job_type += f'_ls{int(args.lsf*100)}'
        if args.sc:
            args.job_type += f'_c{int(args.sc_conf_prop*100)}_u{int(args.sc_unc_prop*100)}'
            if args.sc_rmup_func != 'none':
                args.job_type += f'_up{args.sc_rmup_func[0]}{args.sc_rmup_len}'
    else:
        assert args.it != 's12', f'Please set input_type to \'s1\' or \'s2\' when using single-modal data!' 
        args.unet_type += f'_{args.frm}_{args.mt}_{args.it}_{lbl_t}'
        args.job_type = f'{args.loss_type}_lr_{eb}e{ea}' 
        if args.ls: args.job_type += f'_ls{int(args.lsf*100)}'
    # if args.schd>0: args.job_type += f'_sch{args.schd}'
    if args.schdt =='rop':
        args.schd = int(args.schd)
        assert args.schd>0, f'Please provide a positive value for ReduceLROnPlateau scheduler! Current: {args.schd}' 
        args.job_type += f'_sch_{args.schdt}{args.schd}'
    elif args.schdt == 'exp':
        assert args.schd>0 and args.schd<1, f'Invalid schd value (={args.schd}) for the exponential lr scheduler.'
        tmp = int(args.schd*1000)
        args.job_type += f'_sch_{args.schdt}{tmp}'
    # # resume option
    # if args.resume:
    #     if args.resume in ['imagenet','ssl','swsl']: 
    #         args.job_type += f'_{args.resume[:3]}'
    #     else:
    #         args.job_type += '_res'
    # embed batch size into job type
    args.job_type += f'_b{args.bs}'
    
    # saving dirs
    # parent dir
    args.pdir = os.path.join(args.save_dir, args.unet_type, args.job_type, f"mcr_{args.mcr}_seed_{args.seed}")
    # checkpoint saving dir
    args.checkpoints_dir = os.path.join(args.pdir,"checkpoints")
    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    
    # statistic recording in txt files
    # record params into txt file
    with open(f'{args.pdir}/params.txt', 'w') as ft:
        json.dump(args.__dict__, ft)
    # # record system outputs to txt file
    # if args.print_to_log:
    #     sys.stdout = open(f'{args.pdir}/log.txt', 'w')
    
    # output the model type
    print(f"{args.mt} trained with {args.experiment} ({args.unet_type})\n")
    
    return args

 
def get_device_info(args):
    if args.slurm:
        args.slurm = "SLURM_JOB_ID" in os.environ
        
    if args.slurm:
        args.nodes = int(os.environ["SLURM_NNODES"])
        args.devices = int(os.environ["SLURM_GPUS_ON_NODE"])
        args.strategy = 'ddp'
        args.current_device = int(os.environ["SLURM_LOCALID"])
        print(f'Slurm found with {args.nodes} nodes, {args.devices} devices per node (current:{args.current_device}).')
    else:    
        args.nodes = 1
        # gpu
        if args.accelerator == 'gpu':
            args.devices = torch.cuda.device_count()
            if args.devices == 0:
                args.accelerator = 'cpu'
                print('No GPU found, switch to CPU mode.')
            elif args.devices == 1:
                args.current_device = 0
                print('Uising 1 GPU for training!')
            else:
                args.current_device = int(os.environ["LOCAL_RANK"])
                print(f'No slurm found! Uising {args.devices} local GPUs for training (current:{args.current_device})!')
        # cpu
        if args.accelerator == 'cpu':
            args.devices = 1
            args.current_device = 0
            print('Uising CPU for training!')
            # args.devices = os.cpu_count()
            # print(f'Uising {args.devices} CPU cores for training!')

    ndevs = args.nodes * args.devices
    args.strategy = 'ddp' if ndevs>1 else 'auto'
    return args
