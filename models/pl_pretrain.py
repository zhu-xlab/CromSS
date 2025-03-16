import os
import sys
import wandb
# appending self-defined package path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import torch.nn as nn
from torch import optim

# import segmentation models pytorch package
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

# import lightening
import pytorch_lightning as pl

# self-defined modules
import utils.evaluates as evl


KEY_IMG = 'img'
KEY_GT = 'gt'
KEY_NS = 'ns'
KEY_METRICS = ['iou', 'f1', 'precise', 'recall', 'oa']


# # # # for model loading # # # # 
def modify_module_state_dict_keys(msd):
    prefix = 'module.'
    for k in list(msd.keys()):
        if k.startswith(prefix):
            newk = k[len(prefix):]
            msd[newk] = msd[k]
            del msd[k]
    return msd

def send_optim_tensor_gpu(optimizer,device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                if k == 'step': continue
                state[k] = v.to(device=device)
    return
# # # # for model loading # # # # 

class UNET_PL(pl.LightningModule):
    # 1. training setup
    def __init__(self, args):
        """
        Initializes a UNet model with the specified architecture and loss functions.

        Args:
            args: An object containing the following attributes:
                - mt: A string representing the name of the encoder to use (e.g. mobilenet_v2 or efficientnet-b7).
                - resume: A string representing the path to a checkpoint file to resume training from, or one of the following strings: 'imagenet', 'ssl', 'swsl' to use pre-trained weights for encoder initialization.
                - prt: A boolean indicating whether the resumed model is a pre-trained one for initialization or for continued training.
                - n_channels: An integer representing the number of input channels (e.g. 1 for gray-scale images, 3 for RGB, etc.).
                - n_classes: An integer representing the number of output channels (i.e. the number of classes in the dataset).
                - loss_type: A string representing the type of loss function(s) to use, where 'c' stands for Cross Entropy, 'd' stands for Dice, and 'f' stands for Focal loss.
                - if_mask: A boolean indicating whether the dataset contains masks.
                - experiment: A string representing the type of experiment label to use, where 'ns_labels' stands for non-seismic labels and 'gt_labels' stands for ground truth labels.
                - gt_avail: A boolean indicating whether ground truth labels are available when training with noisy labels.
                - cal_tr: A boolean indicating whether to calculate training metrics.
                - opt: A string representing the optimizer to use (e.g. Adam, SGD, etc.).
                - lr: A float representing the learning rate.
                - schd: A string representing the learning rate scheduler to use (e.g. StepLR, CosineAnnealingLR, etc.).
                - batch_to_wandb: An integer representing the number of batches to log to Weights & Biases.
        """
        super().__init__()
        # pl.seed_everything(args.seed)
        self.last_epoch = 0
        # # # # model architecture # # # #
        self.n_classes = args.n_classes
        self._get_model(args)

        # # # # segmentation loss # # # #
        self._get_loss(args.loss_type, args.n_classes, args.if_mask)
        
        # # # # other hyperparameters # # # # 
        self.opt = args.opt
        self.lr = args.lr
        self.schd = args.schd
        self.if_mask = args.if_mask
        self.log_batch = args.batch_to_wandb
        assert args.experiment in ['ns_labels','gt_labels'], 'Experiment label type is wrong!'
        self.experiment = args.experiment
        self.gt_avail = args.gt_avail
        self.cal_tr = args.cal_tr
        self.current_device = args.current_device
        self.display_inv = args.display_inv
        self.cls_dict = args.cls_dict
        
        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()
    
     
    # 2. forward propagation
    def forward(self, x):
        out = self.net(x).squeeze(axis=1)
        return out
    
    
    # 3. Optimizer
    def configure_optimizers(self):
        # optimizer
        if self.opt=='adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-8)
            print('Adam is used as optimizer!')
        elif self.opt=='sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
            print('SGD is used as optimizer!')
        else:
            raise ValueError("Please provide correct optimizer type (adam or sgd)!")
        # resume optimizer state
        if self.opt_w is not None:
            try:
                optimizer.load_state_dict(self.opt_w)
            except:
                raise TypeError("The type of optimizer stored in ckpt file is not compitable to the used optimizer!")
            # # send optimizer tensor to gpu
            # if args.opt=='adam': send_optim_tensor_gpu(optimizer,device)
            print(f"Training is resumed at epoch {self.last_epoch} from '{self.last_epoch}'!")
        # scheduler
        if self.schd>0: 
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=self.schd, factor=0.5)  # goal: maximize Dice score
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "v_miou"}
        else:
            return optimizer
    
    
    # 4. training step
    def training_step(self, batch, batch_idx):
        x = batch[KEY_IMG]
        if self.experiment=='ns_labels': 
            y_ns = batch[KEY_NS].squeeze(axis=1)
            if self.gt_avail: y_gt = batch[KEY_GT].squeeze(axis=1)
            y = y_ns
        else:
            y_gt = batch[KEY_GT].squeeze(axis=1)
            y = y_gt
        # prediction
        pred_logits = self.forward(x).squeeze(axis=1)
        if self.if_mask: 
            discard_masks = batch['mask_discard'].squeeze(axis=1)
            y[discard_masks!=0] = self.n_classes
        # calculate losses
        loss = torch.tensor(0.0, device=self.device)
        for lk in self.losses:
            L = self.losses[lk]
            loss_ = L(pred_logits, y)
            loss += loss_
            self.log(f'tr_loss_{lk}', loss_, on_epoch=True, on_step=self.log_batch, sync_dist=True)
        self.log('tr_loss', loss, on_epoch=True, on_step=self.log_batch, sync_dist=True)
        # training accuracy
        if self.cal_tr:
            self._get_batch_acc(pred_logits, y, label_type='tr')
            if self.experiment == 'ns_labels' and self.gt_avail:
                self._get_batch_acc(pred_logits, y_gt, label_type='trg') 
        # training image examples  
        if batch_idx==0 and self.current_device==0 and self.current_epoch%self.display_inv==0:
            if self.experiment=='gt_labels':
                self._log_imgs('tr', x, pred_logits, y_gt=y_gt)
            else:
                self._log_imgs('tr', x, pred_logits, y_ns=y_ns, y_gt=y_gt if self.gt_avail else None)
        return loss
    
    
    # 5. validation step
    def validation_step(self, batch, batch_idx):
        x = batch[KEY_IMG]
        if self.experiment=='ns_labels': 
            y_ns = batch[KEY_NS].squeeze(axis=1)
            if self.gt_avail: y_gt = batch[KEY_GT].squeeze(axis=1)
            y = y_ns
        else:
            y_gt = batch[KEY_GT].squeeze(axis=1)
            y = y_gt
        # prediction
        pred_logits = self.forward(x).squeeze(axis=1)
        # accuracy
        self._get_batch_acc(pred_logits, y, label_type='v')
        if self.experiment == 'ns_labels' and self.gt_avail:
            self._get_batch_acc(pred_logits, y_gt, label_type='vg')
        # log images
        if batch_idx in [0,5,10] and self.current_device==0 and self.current_epoch%self.display_inv==0:
            if self.experiment=='gt_labels':
                self._log_imgs('v', x, pred_logits, y_gt=y_gt)
            else:
                self._log_imgs('v', x, pred_logits, y_ns=y_ns, y_gt=y_gt if self.gt_avail else None)
    
    
    # 6. test step
    def test_step(self, batch, batch_idx):
        x = batch[KEY_IMG]
        pred_logits = self.forward(x).squeeze(axis=1)
        if self.experiment=='ns_labels': 
            y_ns = batch[KEY_NS].squeeze(axis=1)
            self._get_batch_acc(pred_logits, y_ns, label_type='tsn')
            if self.gt_avail: 
                y_gt = batch[KEY_GT].squeeze(axis=1)
                self._get_batch_acc(pred_logits, y_gt, label_type='ts')
        else:
            y_gt = batch[KEY_GT].squeeze(axis=1)
            self._get_batch_acc(pred_logits, y_gt, label_type='ts')
        
    
    # auxiliary functions
    # a.1. get model
    def _get_model(self, args):
        if args.resume in ['imagenet','ssl','swsl']: assert args.n_channels==3, \
            'Pretrained encoder weights in smp package only support 3 bands as input!'
        
        self.net = smp.Unet(encoder_name=args.mt,           # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                            encoder_weights=args.resume if args.resume in ['imagenet','ssl','swsl'] else None,  # pre-trained weights for encoder initialization
                            in_channels=args.n_channels,    # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                            classes=self.n_classes,         # model output channels (number of classes in your dataset)
                            )
        print(f"Construct UNet model with {args.mt} as backbone using smp package!")
        
        # load weights from a checkpoint file
        self.opt_w = None
        if args.resume: 
            if args.resume in ['imagenet','ssl','swsl']:
                print(f"Pre-trained model is loaded from smp package (type:{args.resume})!")
            else:
                assert os.path.isfile(args.resume), 'Given model checkpoint loading path is invalid!'
                checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
                msd = modify_module_state_dict_keys(checkpoint['model_state_dict'])
                self.net.load_state_dict(msd)
                # initialize or resume
                if args.prt:
                    print(f"Pre-trained model is loaded from '{args.resume}'!")
                else:
                    self.opt_w = checkpoint['optimizer_state_dict']
                    self.last_epoch = checkpoint['epoch']
        else:
            print('No pretrained weights, will start from scratch.')
    
    # a.2. get losses
    def _get_loss(self, loss_type, n_classes, if_mask):
        self.losses = {}
        # Cross Entropy
        if 'c' in loss_type:
            if n_classes>1:
                if if_mask:
                    self.CELoss = nn.CrossEntropyLoss(ignore_index=n_classes)  
                else:
                    self.CELoss = nn.CrossEntropyLoss()
                print("CrossEntropy loss is used!")
            else:
                self.CELoss = nn.BCEWithLogitsLoss()
                print("BinaryCrossEntropy loss is used!")
            self.losses['c'] = self.CELoss
        else:
            self.CELoss = None  
        # Dice
        if 'd' in loss_type:
            self.DLoss = DiceLoss(mode='multiclass' if n_classes>1 else 'binary', 
                                  from_logits=True, 
                                  ignore_index=n_classes if if_mask else None)
            print("Dice loss is used!")
            self.losses['d'] = self.DLoss
        else:
            self.DLoss = None
        # Focal loss
        if 'f' in loss_type:
            self.FLoss = FocalLoss(mode='multiclass' if n_classes>1 else 'binary', 
                                   ignore_index=n_classes if if_mask else None)
            print("Focal loss is used!")
            self.losses['f'] = self.FLoss
        else:
            self.FLoss = None
        assert len(self.losses)>0, 'No loss is used!'
        
    # a.3. get batch accuracy
    def _get_batch_acc(self, pred_logits, y, label_type='tr'):
        log_batch = self.log_batch if 'tr' in label_type else False
        metrics = evl.evaluate_batch_pl(pred_logits, y, self.n_classes)
        for mk in metrics:
            metr = metrics[mk]
            if torch.prod(torch.tensor(metr.size()))>1:
                for i in range(len(metr)-1):
                    self.log(f'{label_type}_{mk}_{i}', metr[i], on_epoch=True, on_step=log_batch, sync_dist=True)
                self.log(f'{label_type}_m{mk}', metr[-1], on_epoch=True, on_step=log_batch, sync_dist=True)
            else:
                self.log(f'{label_type}_{mk}', metr, on_epoch=True, on_step=log_batch, sync_dist=True)
    
    # a.4. log images
    def _log_imgs(self, prefix, x, pred_logits, y_gt=None, y_ns=None):
        mask_data = pred_logits[0].detach().cpu().numpy().argmax(axis=0)
        class_labels = self.cls_dict
        masks={'prediction': {'mask_data': mask_data, 'class_labels': class_labels},}
        if y_gt is not None:
            masks['ground truth'] = {'mask_data': y_gt[0].detach().cpu().numpy(), 'class_labels': class_labels}
        if y_ns is not None:
            masks['noisy labels'] = {'mask_data': y_ns[0].detach().cpu().numpy(), 'class_labels': class_labels}
        if x.shape[1]==3:
            mask_img = wandb.Image(x[0].detach().cpu().numpy().transpose((1,2,0)), masks=masks)
        elif x.shape[1]>3:
            mask_img = wandb.Image(x[0,[3,2,1]].detach().cpu().numpy().transpose((1,2,0)), masks=masks)
        elif x.shape[1]<3:
            mask_img = wandb.Image(x[0,0].detach().cpu().numpy(), masks=masks)
        else:
            print(f'Please provide correct number of channels for image logging! Current number of channels is {x.shape[1]}.')
        wandb.log({f'{prefix}_masks': mask_img})
        
    # # a.4. define wandb metrics
    # def _define_wandb_metrics(self):
    #     log_metrics = ['tr_loss']+[f'tr_loss_{lk}' for lk in self.losses]+\
    #                     [f'{lb}_{metr}_{c}' for lb in ['tr','trg','v','vg','ts','tsn'] for metr in ['iou','f1','precise','recall'] for c in range(self.n_classes)]+\
    #                     [f'{lb}_m{metr}' for lb in ['tr','trg','v','vg','ts','tsn'] for metr in ['iou','f1','precise','recall']]+\
    #                     [f'{lb}_oa' for lb in ['tr','trg','v','vg','ts','tsn']]
    #     for lm in log_metrics:
    #         wandb.define_metric(lm, summary='mean')
    
    
                

