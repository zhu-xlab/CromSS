import os
import sys
import json
import torch
import argparse
import numpy as np
from pathlib import Path

from utils.get_args_common import get_common_args, get_common_finetune_args

CLS_DICT = {8:{0:'Water', 1:'Forest', 2:'Grassland', 3:'Wetland', 4:'Cropland', 5:'Shrubland', 6:'Urban/Built area', 7:'Barren'},
            9:{0:'Water', 1:'Trees', 2:'Grass', 3:'Flooded vegetation', 4:'Crops', 5:'Shrub & Scrub', 6:'Built area', 7:'Bare land', 8:'Ice & Snow'},
            13:{0:'Urban', 1:'Arable land', 2:'Forest', 3:'Industrial', 4:'Artifical veg', 5:'Dump site', 6:'Pasture', 7:'Permanent crops', 8:'water',
                9: 'Open space', 10:'Shrub', 11:'wetland', 12:'Coastal'},
        }


def get_args_dwosm():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    # data settings
    parser.add_argument('--train_path', type=str, default='E:/Datasets/DFC2020/lmdb/DFC2020_s12_validation.lmdb')
    parser.add_argument('--test_path', type=str, default='E:/Datasets/DFC2020/lmdb/DFC2020_s12_test.lmdb')
    parser.add_argument('--label_type', dest='lbl_t', type=str, default='dw', choices=['dw', 'osm'], help='Type of label')
    parser.add_argument('--input_type', dest='it', type=str, default='s2', choices=['s1', 's2', 's12'], help='Type of input data')
    parser.add_argument('--s1_norm', action='store_true', default=False, help='Whether using (x-u)/std normalization for s1')
    parser.add_argument('--s2_norm', action='store_true', default=False, help='Whether using ./10000 normalization for s2')
    
    # get common arguments
    parser = get_common_args(parser)
    parser = get_common_finetune_args(parser)
    
    # initialize arguments
    args = parser.parse_args()
    
    # computing environment parameters
    args = get_device_info(args)
    
    # set saving dir
    args = set_log_files(args)
    
    # set class dictionary for logging images
    args.cls_dict = CLS_DICT[args.n_classes]
    
    return args


def get_args_dfc2020():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    # data settings
    parser.add_argument('--train_path', type=str, default='E:/Datasets/DFC2020/lmdb/DFC2020_s12_validation.lmdb')
    parser.add_argument('--test_path', type=str, default='E:/Datasets/DFC2020/lmdb/DFC2020_s12_test.lmdb')
    parser.add_argument('--input_type', dest='it', type=str, default='s2', choices=['s1', 's2', 's12'], help='Type of input data')
    parser.add_argument('--s1_norm', action='store_true', default=False, help='Whether using (x-u)/std normalization for s1')
    parser.add_argument('--s2_norm', action='store_true', default=False, help='Whether using ./10000 normalization for s2')
    
    # get common arguments
    parser = get_common_args(parser)
    parser = get_common_finetune_args(parser)
    
    # initialize arguments
    args = parser.parse_args()
    
    # computing environment parameters
    args = get_device_info(args)
    
    # set saving dir
    args = set_log_files(args)
    
    # set class dictionary for logging images
    args.cls_dict = CLS_DICT[args.n_classes]
    
    return args


def get_args_oscd():
    parser = argparse.ArgumentParser(description='Train the UNet for change detection')
    # data settings
    parser.add_argument('--data_dir', type=str, default='E:/Datasets/DFC2020/lmdb/DFC2020_s12_validation.lmdb')
    parser.add_argument('--value_discard', action='store_false', default=True,
                        help='Whether to discard the extreme values according to mean and 2*std')
    parser.add_argument('--patch_size', type=int, default=96)
    
    # get common arguments
    parser = get_common_args(parser)
    parser = get_common_finetune_args(parser)
    
    # initialize arguments
    args = parser.parse_args()
    
    # computing environment parameters
    args = get_device_info(args)
    
    # set saving dir
    args = set_log_files(args)
    
    # set class dictionary for logging images
    args.cls_dict = CLS_DICT['CD']
    
    return args


def set_log_files(args):
    # check n_classes variable
    if args.dataset=='DFC':
        assert args.n_classes==8, f'{args.n_classes} is invalid (should be 8 for the DFC2020 dataset)!'
    elif args.dataset=='DW':
        assert args.n_classes==9, f'{args.n_classes} is invalid (should be 9 for the DW dataset)!'
        assert args.lbl_t=='dw', f'Please provide a valid label type for the DW dataset (current:{args.lbl_t})!'
    elif args.dataset=='OSM':
        assert args.n_classes in [9,13], f'{args.n_classes} is invalid (should be 9 or 13 for the OSM dataset)!'
        tr_size = args.train_path.split('_')[-2]
        args.dataset += (tr_size[0].upper()+args.lbl_t[0]) 
    # set unet_type according to model settings: unet_gt or unet_ns(_rm)
    if 'CD' in args.dataset:
        args.unet_type = f'{args.dataset}_{args.mt}_{args.n_channels}b_{args.prt_tag}'    
    else:
        if args.mt.startswith('vit'):
            mt_splits = args.mt.split('_')
            mt_str = f'{mt_splits[0]}{mt_splits[1][0]}p{args.patch_size}'
        else:
            mt_str = args.mt
        args.unet_type = f'{args.dataset}_{args.frm}_{mt_str}_{args.n_channels}b_{args.n_classes}cls_{args.prt_tag}'
    # for s1 as inputs, record number of s2 bands in pretraining
    if 'fs1' in args.prt_tag:
        args.unet_type += f'b{args.ws2b}'
    if 'nss' in args.prt_tag or 'fs' in args.prt_tag:
        if 'fs' in args.prt_tag:
            assert f'PT_{args.prt_tag[0]}' in args.wpath, f'Pretraining tag ({args.prt_tag}) is not compatible with the given loaded weights ({args.wpath})'
        # label smoothing
        if args.wls: 
            if len(args.wlsf)==1:
                if args.wls_prior_type=='none':
                    tmp = f'_ls{int(args.wls*100)}'
                else:
                    tmp = f'_ls{int(args.wlsf[0]*100)}{args.wls_prior_type}'
            else:
                assert len(args.wlsf)==2, f'More than 2 label smoothing factors {args.wlsf} are provided!'
                tmp = f'_ls{int(args.wlsf[0]*100)}{args.wls_prior_type}{int(args.wlsf[1]*100)}'
            assert tmp in args.wpath, f'Please provide valid label smoothing settings for logging loaded weights (current:{tmp[1:]})'            
            args.unet_type += tmp
        # sample selection settings
        if args.wss:
            tmp = f'_ssp{int(args.wss_prop*100)}_{args.wss_ctype}'
            if args.wss_rm_func != 'none':
                tmp += f'_rm{args.wss_rm_func[0]}{args.wss_nep}'
            assert tmp in args.wpath, f'Please provide valid sample selection settings for logging loaded weights (current:{tmp[1:]})'            
            args.unet_type += tmp
        # lr scheduler in the pretraining stage
        if args.wschdt != 'none':
            if args.wschdt =='rop':
                args.wschdv = int(args.wschdv)
                tmp = f'_sch_{args.wschdt}{args.wschdv}'
            elif args.wschdt == 'exp':
                tmp = f'_sch_{args.wschdt}{int(args.wschdv*1000)}'
            else:
                tmp = f'_sch_{args.wschdt[:3]}'
            assert tmp in args.wpath, f'Please provide valid sample selection settings for logging loaded weights (current:{tmp[1:]})'            
            args.unet_type += tmp
        # batch size used in the pretraining stage
        args.unet_type += f'_b{args.wbs}'
        assert f'epoch={args.wep-1}' in args.wpath, f'Please provide a valid epoch index for logging loaded weights (current:{args.wep})' 
        args.unet_type += f'_{args.wep}'
    if args.prt_tag != 'rand': args.unet_type += f'_{args.wload_tag[:3]}'
    ea = int(np.ceil(np.log10(1/args.lr)))
    eb = int(np.round(args.lr*10**ea))
    if args.eft_tag=='eft':
        assert args.eft_ep>0 and args.eft_ep<args.epochs, f'Please provide a valid epoch index for triggering encoder finetuning (0-{args.epochs-1})!' 
        # fea = int(np.ceil(np.log10(1/args.eft_lr)))
        # feb = int(args.eft_lr*10**fea)
        args.job_type = f'{args.eft_tag}{args.eft_ep}_{args.dft_tag}_lr{eb}e{ea}' # f'{args.eft_tag}{args.eft_ep}_{feb}e{fea}_{args.dft_tag}_lr{eb}e{ea}'
    else:
        args.job_type = f'{args.eft_tag}_{args.dft_tag}_lr{eb}e{ea}'
    if args.schdt!='none': 
        args.job_type += f'_{args.schdt}'
        if args.schdt=='step':
            for dv in args.schdv: args.job_type += f'_{int(dv)}'
        elif args.schdt=='exp':
            args.job_type += f'_{int(args.schdv[0]*100)}'
        elif args.schdt=='cos':
            args.job_type += f'_{int(args.schdv[0])}'
            if len(args.schdv)>1: 
                ea = int(np.ceil(np.log10(1/args.schdv[1])))
                eb = int(args.schdv[1]*10**ea)
                args.job_type += f'_f{eb}e{ea}'  # final_lr setting
    # norm
    if (args.it=='s1' and args.s1_norm) or (args.it=='s2' and args.s2_norm): 
        args.job_type += '_norm'
    # # resume option
    # if args.resume:
    #     if args.resume in ['imagenet','ssl','swsl']: 
    #         args.job_type += f'_{args.resume[:3]}'
    #     else:
    #         args.job_type += '_res'
    
    # saving dirs
    # for test only
    if args.only_test: 
        # parent dir
        if args.ts_wpath is None:
            args.pdir = os.path.join(args.save_dir, args.unet_type) 
            ep = 0 
        else:
            args.pdir = os.path.join(args.save_dir, args.unet_type, args.job_type, f"mcr_{args.mcr}_seed_{args.seed}_bs_{args.bs}")
            ep_str = args.ts_wpath.split('epoch=')[1].split('-')[0]
            ep = int(ep_str)
        # test dir
        args.pdir = os.path.join(args.pdir, f'test_ep{ep}')
        Path(args.pdir).mkdir(parents=True, exist_ok=True)
        
        # log outputs to txt file
        acc_txt = os.path.join(args.pdir, 'outputs.txt')
        sys.stdout = open(acc_txt, 'w')
    # for fine-tuning training
    else:
        # parent dir
        args.pdir = os.path.join(args.save_dir, args.unet_type, args.job_type, f"mcr_{args.mcr}_seed_{args.seed}_bs_{args.bs}")
        # checkpoint saving dir
        args.checkpoints_dir = os.path.join(args.pdir,"checkpoints")
        Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    
        # statistic recording in txt files
        # record params into txt file
        with open(f'{args.pdir}/params.txt', 'w') as ft:
            json.dump(args.__dict__, ft)
        
        # output the model type
        print(f"{args.mt} trained with {args.prt_tag} weights ({args.unet_type})\n")
    
    return args

 
def get_device_info(args):
    if args.slurm:
        args.slurm = "SLURM_JOB_ID" in os.environ
        
    if args.slurm:
        args.nodes = int(os.environ["SLURM_NNODES"])
        args.devices = int(os.environ["SLURM_TASKS_PER_NODE"])
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
