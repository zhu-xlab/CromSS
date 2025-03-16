import datetime
import os
import sys

# appending self-defined package path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
from torch.utils.data import DataLoader, random_split

# import lightening
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from py_scripts_SSL4EO.get_args_pretrain import get_args
from models.pl_pretrain import UNET_PL

# dataset
from dataset.ssl4eo_dataset_lmdb import SSL4EODataset, SegAugTransforms, split_for_train_and_val_sets
import albumentations as A


# # # # # # # # # # main function # # # # # # # # # # 
def main():
    t0 = datetime.datetime.now().replace(microsecond=0)
    
    ###### 1 - parameter setting ######
    global args
    args = get_args()
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
                               name=f'm{args.mcr}_s{args.seed}',
                               save_dir=args.pdir,)
    
    
    ###### 2 - load data ######
    # file directories
    lmdb_im1 = os.path.join(args.data_path, args.dn1) if args.it=='s1' else None
    lmdb_im2 = os.path.join(args.data_path, args.dn2) if args.it=='s2' else None
    print(f'Using {args.it} data for single-modal training!')
    lmdb_lb = os.path.join(args.data_path, args.dnl)
    # transforms
    trans = A.Compose([A.RandomCrop(width=256, height=256),
                       A.Flip(p=0.5),
                       A.Rotate(p=0.2),
                       # A.RandomBrightnessContrast(p=0.2),
                       # A.GaussianBlur(0.1, 2.0, p=0.2),
                       ])
    # datasets
    dataset = SSL4EODataset(lmdb_file_ns=lmdb_lb,
                            lmdb_file_s1=lmdb_im1, 
                            lmdb_file_s2=lmdb_im2, 
                            mode=args.it,
                            n_bands=args.n_channels, 
                            season=None,
                            transform=SegAugTransforms(trans), 
                            normalize=args.norm)
    # dataloader: no matther what devices is used, here directly utilize DataLoader since pl will take of DDP cases
    def get_dataloader(dataset, batch_size, shuffle):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                 num_workers=args.num_workers, drop_last=True)
        return data_loader
    if args.val == 0:
        # version 1 - do not use validation set
        train_loader = get_dataloader(dataset, args.bs, True)
        n_batch_tr = int(len(dataset)/args.bs)
        print(f'len/n_batch of train (no validation set):{len(dataset)}/{n_batch_tr}')
    else:
        # version 2 - seperate a small part of data from the training set to construct the validation set
        train_set, val_set = split_for_train_and_val_sets(dataset, args.val, seed=args.seed)
        # construct dataloaders
        train_loader = get_dataloader(train_set, args.bs, True)
        val_loader = get_dataloader(val_set, args.tbs, False)
        n_batch_tr, n_batch_val = int(len(train_set)/args.bs), int(len(val_set)/args.tbs)
        print(f'len/n_batch of train, val:{len(train_set)}/{n_batch_tr}, {len(val_set)}/{n_batch_val}')
        
    
    ###### 3 - define the model ######
    torch.set_float32_matmul_precision('high')
    model = UNET_PL(args)
    
    
    ###### 4 - train the model ######
    print('\nStart training...')
    callbacks = [ModelCheckpoint(dirpath = args.checkpoints_dir, every_n_epochs = args.save_inv, 
                                 save_weights_only=True, save_top_k=-1,  # Save all models
                                 )]
    if args.schd>0: callbacks += [LearningRateMonitor(logging_interval='epoch')]
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
        trainer.fit(model, train_loader)
    else:
        trainer.fit(model, train_loader, val_loader)
    
    t1 = datetime.datetime.now().replace(microsecond=0)
    print(f'Training is finished|Total spent time:{t1-t0}!' )
    
    
if __name__ == '__main__':
    main()


