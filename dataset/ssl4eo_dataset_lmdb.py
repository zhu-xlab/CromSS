import os
import random
import pickle
import numpy as np
import albumentations as A
from fastai.vision.all import *
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split
import lmdb, torch

# for generating label smoothing masks
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

import matplotlib as mpl
import matplotlib.pyplot as plt

### band statistics: mean & std
# calculated from 50k subset
S1_MEAN = [-12.54847273, -20.19237134]
S1_STD = [5.25697717, 5.91150917]

S2A_MEAN = [752.40087073, 884.29673756, 1144.16202635, 1297.47289228, 1624.90992062, 2194.6423161, 2422.21248945, 2517.76053101, 2581.64687018, 2645.51888987, 2368.51236873, 1805.06846033]
S2A_STD = [1108.02887453, 1155.15170768, 1183.6292542, 1368.11351514, 1370.265037, 1355.55390699, 1416.51487101, 1474.78900051, 1439.3086061, 1582.28010962, 1455.52084939, 1343.48379601]

S2C_MEAN = [1605.57504906, 1390.78157673, 1314.8729939, 1363.52445545, 1549.44374991, 2091.74883118, 2371.7172463, 2299.90463006, 2560.29504086, 830.06605044, 22.10351321, 2177.07172323, 1524.06546312]
S2C_STD = [786.78685367, 850.34818441, 875.06484736, 1138.84957046, 1122.17775652, 1161.59187054, 1274.39184232, 1248.42891965, 1345.52684884, 577.31607053, 51.15431158, 1336.09932639, 1136.53823676]

BANDS_3 = [3,2,1]
BANDS_9 = [1,2,3,4,5,6,7,11,12]
BANDS_12 = [0,1,2,3,4,5,6,7,8,9,11,12] # remove B10 - cirrus band

# # # # label smoothing mask parameters
NCLS = 9
TMP_SIGMA = 3
TMP_TRUNC = 4   # radius = int(truncate * sigma + 0.5)

SPA_SIGMA = 3
SPA_TRUNC = 4   # radius = int(truncate * sigma + 0.5)

def get_smoothed_maps(map_to_smooth, sigma, truncate):
    smoothed_maps = []
    for c in range(NCLS):
        smoothed_map = gaussian_filter(map_to_smooth[:,:,c], sigma=sigma, truncate=truncate)  # radius = int(truncate * sigma + 0.5)
        smoothed_maps.append(smoothed_map)
    smoothed_maps = np.stack(smoothed_maps, axis=0)
    smoothed_maps = smoothed_maps/np.sum(smoothed_maps, axis=0, keepdims=True)
    return smoothed_maps

# # # # functions for reading data from lmdb
def get_band_info(n_bands):
    # band ids
    if n_bands == 3:
        band_ids = BANDS_3
    elif n_bands == 9:
        band_ids = BANDS_9
    elif n_bands == 12:
        band_ids = BANDS_12    
    else:
        band_ids = np.arange(13)
    return band_ids


def normalize(img):
    img = img.astype(np.float32)
    
    # get band info
    n_bands = img.shape[0]
    band_ids = get_band_info(n_bands)
    
    # normalize band by band
    img_new = []
    for b in band_ids:
        im = img[b]
        mean = S2C_MEAN[b]
        std = S2C_STD[b]
        min_value = mean - 2 * std
        max_value = mean + 2 * std
        
        # normalize
        im = min_value + im*(max_value - min_value)/255.
        im /= 10000.0
        
        # concatenate
        img_new.append(im[None])
    
    # concatenate
    img_new = np.concatenate(img_new, axis=0)
    
    return img_new
                

# get subset of given dataset    
class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


# construct subset with randomly generated indexes
def random_subset(dataset, f_or_num, seed=None):
    if f_or_num <=0:
        raise ValueError('Please provide valid subset size!')
    elif f_or_num <= 1:
        n_select = int(len(dataset)*f_or_num)
    else:
        n_select = int(f_or_num)
    indices_all = np.arange(len(dataset))
    if seed is not None: random.seed(seed)
    random.shuffle(indices_all)
    indices = indices_all[:n_select]
    return Subset(dataset, indices)


# split all the data into training and validation sets
def split_for_train_and_val_sets(dataset, nval, seed=None):
    if nval <=0:
        raise ValueError('Please provide valid subset size!')
    elif nval <= 1:
        n_val = int(len(dataset)*nval)
    else:
        n_val = int(nval)
    n_tr = len(dataset)-n_val
    train_set, val_set = random_split(dataset, [n_tr, n_val], generator=torch.Generator().manual_seed(seed))
    return train_set, val_set


class SSL4EODataset(Dataset):
    def __init__(self, lmdb_file_ns, lmdb_file_s1=None, lmdb_file_s2=None,
                 mode='s2', n_bands=13, season=None, normalize=False, 
                 label_smooth=False, smooth_prior_type='t',  # 't': temporal, 's': spatial, 'ts': temporal & spatial
                 s1_transform=None, s2_transform=None, transform=None):
        if n_bands not in [3,9,12,13]:
            n_bands = 13
        self.n_bands = n_bands
        self.season = season
        self.mode = mode
        if '1' in mode:
            assert lmdb_file_s1 is not None, "Please provide the path to S1 data!"
            assert os.path.exists(lmdb_file_s1), f"LMDB file for s1 ({lmdb_file_s1}) does not exist!"
            self.env_s1 = lmdb.open(lmdb_file_s1, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        if '2' in mode:
            assert lmdb_file_s2 is not None, "Please provide the path to S2 data!"
            assert os.path.exists(lmdb_file_s2), f"LMDB file for s2 ({lmdb_file_s2}) does not exist!"
            self.env_s2 = lmdb.open(lmdb_file_s2, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        # self.lmdb_file_s1 = lmdb_file_s1
        # self.lmdb_file_s2 = lmdb_file_s2
        # self.lmdb_file_ns = lmdb_file_ns
        self.s1_transform = s1_transform
        self.s2_transform = s2_transform
        self.transform = transform
        self.normalize = normalize
        self.ls = label_smooth
        if self.ls: assert smooth_prior_type in ['t','s','ts'], "Please provide valid label smoothing type!"
        self.ls_type = smooth_prior_type
        
        self.env_ns = lmdb.open(lmdb_file_ns, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env_ns.begin(write=False) as txn_ns:
            self.length = txn_ns.stat()['entries'] 
        
        if self.transform is None:
            t = ToTensorV2()
            def base_transform(x):
                img, ns = x
                if ns.flags['WRITEABLE'] is False:
                    lbl_writable = np.copy(ns)
                    lbl_writable.setflags(write=True)
                    tens = t(image=img.transpose(1,2,0), mask=lbl_writable)
                else:
                    tens = t(image=img.transpose(1,2,0), mask=ns)
                return tens['image'], tens['mask']
            self.transform = base_transform

    def __getitem__(self, index):
        # get data
        # ns labels
        with self.env_ns.begin(write=False) as txn_ns:
            data_ns = txn_ns.get(str(index).encode())
        ns_bytes, ns_shape, ns_id = pickle.loads(data_ns)
        ns = np.frombuffer(ns_bytes, dtype=np.uint8).reshape(ns_shape)
        # ***** generate temporal label smoothing mask ***** #
        if self.ls:
            lbl_tensor = torch.from_numpy(ns.copy()).long()
            lbl_one_hot = F.one_hot(lbl_tensor, num_classes=NCLS).numpy().astype(np.float32)
        if self.ls and 't' in self.ls_type:
            # calculate temporal frequencies
            lbl_temp_freq = np.sum(lbl_one_hot, axis=0)/lbl_one_hot.shape[0]
            # smooth temporal frequency map with Gaussian kernel & normalize smoothed temporal frequency map
            temporal_prior = get_smoothed_maps(lbl_temp_freq, sigma=TMP_SIGMA, truncate=TMP_TRUNC)
            if self.ls_type == 't':
                smooth_prior = temporal_prior
        # *****
        
        # s1
        if '1' in self.mode:
            with self.env_s1.begin(write=False) as txn_im:
                data = txn_im.get(str(index).encode())
            img_bytes, img_shape, img_id_s1 = pickle.loads(data)
            img_s1 = np.frombuffer(img_bytes, dtype=np.uint8).reshape(img_shape)
            assert img_id_s1==ns_id, f"S1 image id ({img_id_s1}) is not equal to mask id ({ns_id})"
        # s2
        if '2' in self.mode:
            with self.env_s2.begin(write=False) as txn_im:
                data = txn_im.get(str(index).encode())
            img_bytes, img_shape, img_id_s2 = pickle.loads(data)
            img_s2 = np.frombuffer(img_bytes, dtype=np.uint8).reshape(img_shape)
            assert img_id_s2==ns_id, f"S2 image id ({img_id_s2}) is not equal to mask id ({ns_id})"
        
        # get specific bands for s2
        if '2' in self.mode:
            if self.n_bands == 3:
                img_s2 = img_s2[:,BANDS_3]
            elif self.n_bands == 9:
                img_s2 = img_s2[:,BANDS_9]
            elif self.n_bands == 12:
                img_s2 = img_s2[:,BANDS_12]
        
        # randomly select a season patch
        if self.season in [0,1,2,3]:
            season = self.season
        else:
            season = np.random.choice([0,1,2,3])
        if '1' in self.mode:
            img_s1 = img_s1[season] 
            # S1 /255 to [0,1]
            img_s1 = img_s1.astype(np.float32)/255
        if '2' in self.mode:
            img_s2 = img_s2[season] 
            # S2 normalize by /10000 or /255 to [0,1]
            if self.normalize:
                img_s2 = normalize(img_s2)
            else:
                img_s2 = img_s2.astype(np.float32)/255
        ns = ns[season].astype(np.float32)
        # ***** generate spatial label smoothing mask ***** #
        if self.ls and 's' in self.ls_type:
            spatial_prior = get_smoothed_maps(lbl_one_hot[season], sigma=SPA_SIGMA, truncate=SPA_TRUNC)
            if self.ls_type == 's':
                smooth_prior = spatial_prior  # shape: (NCLS, H, W)
            else:
                smooth_prior = (temporal_prior + spatial_prior)/2  # shape: (NCLS, H, W)
        # *****
        
        # separate transform for s1 and s2
        if '1' in self.mode and self.s1_transform is not None:
            img_s1 = self.s1_transform(img_s1)
        if '2' in self.mode and self.s2_transform is not None:
            img_s2 = self.s2_transform(img_s2)
        
        # concatenate s1 and s2
        if self.mode == 's1':
            img = img_s1
        elif self.mode == 's2':
            img = img_s2
        elif self.mode == 's12':
            img = np.concatenate([img_s1, img_s2], axis=0)  # shape: (n_bands, H, W)
            
        # combine smoothed prior with img for transformation
        if self.ls:
            img = np.concatenate([smooth_prior, img], axis=0)  # shape: (NCLS+n_bands, H, W)
        
        # transform
        img, ns = self.transform((img, ns))                
        
        if self.ls:
            smooth_prior = img[:NCLS]
            img = img[NCLS:]
            return {'img': img, 'ns': ns.long(), 'smooth_prior': smooth_prior}
        else:
            return {'img': img, 'ns': ns.long()}
    
    def __len__(self):
        return self.length
    

# augmentation class for (img,label) pairs
class SegAugTransforms(ItemTransform):
    def __init__(self, aug): 
        self.aug = aug
        self.ToTensor = ToTensorV2()
    def encodes(self, x):
        img, ns = x
        # for albumentations to work correctly, the channels must be at the last dimension
        data_aug = self.aug(image=img.transpose(1,2,0), mask=ns)
        if data_aug['mask'].flags['WRITEABLE'] is False:
            lbl_writable = np.copy(data_aug['mask'])
            lbl_writable.setflags(write=True)
        else: 
            lbl_writable = data_aug['mask']
        tens = self.ToTensor(image=data_aug['image'], mask=lbl_writable)
        return tens['image'], tens['mask']
    
    
if __name__ == '__main__':
    selected_i = 0
    mode = 's2'
    s = 3
    fp_ns = r'/p/project/hai_dm4eo/liu_ch/data/SSL4EO-S2/dw_labels.lmdb'
    fp_s1 = r'/p/project/hai_dm4eo/liu_ch/data/SSL4EO-S2/0k_251k_uint8_s1.lmdb'
    fp_s2 = r'/p/project/hai_dm4eo/liu_ch/data/SSL4EO-S2/0k_251k_uint8_s2c.lmdb'
    trans = A.Compose([A.RandomCrop(width=256, height=256),
                           A.Flip(),
                           A.Rotate(p=0.2),
                        #    A.RandomBrightnessContrast(p=0.2),
                        #    A.GaussianBlur(0.1, 2.0, p=0.2),
                           # A.ColorJitter(p=0.5)
                           ])
    
    train_dataset = SSL4EODataset(
        lmdb_file_ns  = fp_ns,
        lmdb_file_s1 = fp_s1,
        lmdb_file_s2 = fp_s2,
        mode = mode,
        season=s,
        transform  = None, # SegAugTransforms(trans),
        normalize = False,
        label_smooth=True,
        smooth_prior_type='ts',
        )
    
    train_dataset_nt = SSL4EODataset(
        lmdb_file_ns  = fp_ns,
        lmdb_file_s1 = fp_s1,
        lmdb_file_s2 = fp_s2,
        mode = mode,
        season=s,
        transform  = None, # SegAugTransforms(trans), # None,
        normalize = False,
        label_smooth=True,
        smooth_prior_type='t',
        )
    print(len(train_dataset), len(train_dataset_nt))
    
    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=0)
    train_loader_nt = DataLoader(train_dataset_nt, batch_size=4, num_workers=0)
    
    for i, (s2c,s2c_nt) in enumerate(zip(train_loader,train_loader_nt)):
        if i == selected_i:
            img, ns, lsm = s2c['img'], s2c['ns'], s2c['smooth_prior']
            img_nt, ns_nt, lsm_nt = s2c_nt['img'], s2c_nt['ns'], s2c_nt['smooth_prior']
            print(img.shape,img.dtype,img.min(),img.max(),ns.shape,ns.dtype,np.unique(ns))
            print(lsm.shape,lsm.dtype,lsm.min(),lsm.max(),np.percentile(lsm,84),torch.numel(lsm),(lsm>0).sum()/torch.numel(lsm),(lsm==1).sum()/torch.numel(lsm))
            a = lsm.sum(axis=1)
            print(a.shape,a.dtype,a.min(),a.max())
            print(img_nt.shape,img_nt.dtype,img_nt.min(),img_nt.max(), ns_nt.shape,ns_nt.dtype,np.unique(ns_nt))
            print(lsm_nt.shape,lsm_nt.dtype,lsm_nt.min(),lsm_nt.max(),np.percentile(lsm_nt,84),torch.numel(lsm_nt),(lsm_nt>0).sum()/torch.numel(lsm_nt),(lsm_nt==1).sum()/torch.numel(lsm_nt))
            b = lsm_nt.sum(axis=1)
            print(b.shape,b.dtype,b.min(),b.max())
            ind = 1
            # if '2' in mode:
            #     plt.figure(figsize=(10,10))
            #     plt.subplot(221)
            #     plt.imshow((img.numpy()[ind,[5,4,3]]).transpose(1,2,0))
            #     plt.subplot(222)
            #     plt.imshow(ns.numpy()[ind],norm=mpl.colors.Normalize(vmin=0,vmax=8))
            #     plt.subplot(223)
            #     plt.imshow((img_nt.numpy()[ind,[5,4,3]]).transpose(1,2,0))
            #     plt.subplot(224)
            #     plt.imshow(ns_nt.numpy()[ind],norm=mpl.colors.Normalize(vmin=0,vmax=8))
            
            # if '1' in mode:
            #     plt.figure(figsize=(10,10))
            #     plt.subplot(221)
            #     plt.imshow(img.numpy()[ind,0])
            #     plt.subplot(222)
            #     plt.imshow(img.numpy()[ind,1])
            #     plt.subplot(223)
            #     plt.imshow(img_nt.numpy()[ind,0])
            #     plt.subplot(224)
            #     plt.imshow(img_nt.numpy()[ind,1])
            
            break
