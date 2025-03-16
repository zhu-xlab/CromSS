import datetime
import os
import sys
import numpy as np
from random import sample

# appending self-defined package path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
from torch.utils.data import DataLoader, random_split, Subset

# import lightening
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from py_scripts_SSL4EO.get_args_finetune import get_args_dwosm as get_args
from models.pl_finetune_mask import UNET_PL_FT

# dataset
from dataset.dwosm_dataset_lmdb import DWOSMDataset, SegAugTransforms
import albumentations as A


# # # # # # # # # # main function # # # # # # # # # # 
def main():
    t0 = datetime.datetime.now().replace(microsecond=0)

    ###### 1 - parameter setting ######
    global args
    args = get_args()
    # resume option
    if args.resume:
        ckpt_path = args.resume
        wandb_resume = True
    else:
        ckpt_path = None
        wandb_resume = False
    # Setting all the random seeds to the same value.
    # This is important in a distributed training setting. 
    # Each rank will get its own set of initial weights. 
    # If they don't match up, the gradients will not match either,
    # leading to training that may not converge.
    pl.seed_everything(args.seed)
    # setting logger
    os.environ["WANDB_MODE"] = args.wmode
    wandb_logger = WandbLogger(project=args.pjn, entity=args.entity, 
                               group=args.unet_type, job_type=args.job_type,
                               name=f'{args.prt_tag}_m{args.mcr}_s{args.seed}_b{args.bs}_{args.wload_tag[:3]}',
                               save_dir=args.pdir,
                               resume=wandb_resume) 
    
    
    ###### 2 - load data ######
    # datasets
    if args.mt.startswith('vit'):
        trans_tr = A.Compose([A.RandomCrop(width=args.im_size, height=args.im_size),
                              A.Flip(p=0.5),
                              A.Rotate(p=0.2),
                             ])
        trans_ts = A.Compose([A.CenterCrop(width=args.im_size, height=args.im_size)])
        trans_ts = SegAugTransforms(trans_ts)
    else:
        img_size = 480 if args.frm in ['unet', 'linknet', 'fpn', 'unetplusplus', 'manet'] else 496
        trans_tr = A.Compose([A.RandomCrop(width=img_size, height=img_size), # (width=496, height=496),
                              A.Flip(p=0.5),
                              A.Rotate(p=0.2),
                             ])
        trans_ts = None

    def get_dataset(data_path, trans=None):
        dataset = DWOSMDataset(data_path, lbl_type=args.lbl_t, mode=args.it, n_bands=args.n_channels, 
                               transform=trans, s1_normalize=args.s1_norm, s2_normalize=args.s2_norm)
        return dataset
    train_dataset = get_dataset(args.train_path, SegAugTransforms(trans_tr))
    test_dataset = get_dataset(args.test_path, trans_ts)
    print(f'Images are loaded with {train_dataset[0]["img"].shape[0]} bands.')
    # dataloader: no matther what devices is used, here directly utilize DataLoader since pl will take of DDP cases
    def get_dataloader(dataset, batch_size, shuffle):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                 num_workers=args.num_workers, drop_last=True)
        return data_loader
    test_loader = get_dataloader(test_dataset, args.tbs, False)
    # for training dataloader
    if args.ntr > 0:
        sequence = list(np.arange(len(train_dataset)))
        train_inds = sample(sequence, args.ntr)
        train_dataset = Subset(train_dataset, train_inds)
    if args.val == 0:
        # version 1 - do not use validation set
        train_loader = get_dataloader(train_dataset, args.bs, True)
        n_batch_tr, n_batch_ts = int(len(train_dataset)/args.bs), int(len(test_dataset)/args.tbs)
        print(f'len/n_batch of train, test:{len(train_dataset)}/{n_batch_tr}, {len(test_dataset)}/{n_batch_ts}')
    else:
        # version 2 - seperate a small part of data from the training set to construct the validation set
        n_val = int(len(train_dataset) * args.val)
        n_train = len(train_dataset) - n_val
        train_set, val_set = random_split(train_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
        # construct dataloaders
        train_loader = get_dataloader(train_set, args.bs, True)
        val_loader = get_dataloader(val_set, args.tbs, False)
        n_batch_tr, n_batch_val, n_batch_ts = int(len(train_set)/args.bs), int(len(val_set)/args.tbs), int(len(test_dataset)/args.tbs)
        print(f'len/n_batch of train, val, test:{len(train_set)}/{n_batch_tr}, {len(val_set)}/{n_batch_val}, {len(test_dataset)}/{n_batch_ts}')
        
    
    ###### 3 - define the model ######
    torch.set_float32_matmul_precision('high')
    model = UNET_PL_FT(args)
    
    
    ###### 4 - train the model ######
    print('\nStart training...')
    callbacks = [ModelCheckpoint(dirpath = args.checkpoints_dir, every_n_epochs = args.save_inv, 
                                 # save_weights_only=True, 
                                 save_top_k=-1,  # Save all models
                                 )]
    if args.schdt!='none': callbacks += [LearningRateMonitor(logging_interval='epoch')]
    trainer = pl.Trainer(accelerator=args.accelerator,
                         devices=args.devices,
                         strategy=args.strategy,
                         num_nodes=args.nodes,
                         precision='16-mixed' if args.amp else '32-true',
                         max_epochs=args.epochs, 
                         logger=wandb_logger,
                         enable_progress_bar=True,
                         callbacks=callbacks)  
    if args.val == 0:
        trainer.fit(model, train_loader, test_loader, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
        # test the model
        trainer.test(dataloaders=test_loader, verbose=True)
    
    t1 = datetime.datetime.now().replace(microsecond=0)
    print(f'Training is finished|Total spent time:{t1-t0}!' )
    
    
    
if __name__ == '__main__':
    main()


