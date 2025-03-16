# -*- coding: utf-8 -*-
"""
Crop original images into small patches and save them into lmdb files for ViT-backboned models
@author: liu_ch
"""

import os
import lmdb
import pickle
import datetime
import numpy as np
from pathlib import Path


SDTYPE = {'s1':np.float64, 's2':np.uint16, 'lbl':np.uint8}

def crop_image_BWH(img, patch_size=256, patch_bias=0, ori_im_size=256):
    # # # # # crop big images into small patches # # # # #
    n_patch = int((ori_im_size-patch_bias)/patch_size)
    imgs = []
    for i in range(n_patch):
        for j in range(n_patch):
            if len(img.shape) == 2:
                im = img[patch_bias+i*patch_size:patch_bias+(i+1)*patch_size, patch_bias+j*patch_size:patch_bias+(j+1)*patch_size]
            else:
                im = img[:, patch_bias+i*patch_size:patch_bias+(i+1)*patch_size, patch_bias+j*patch_size:patch_bias+(j+1)*patch_size]
            imgs.append(im)
    return imgs


def create_s12_lmdb_file(dir_source_lmdb:str, dir_lmdb:str, map_size:int, 
                         split:str='validation', write_interv:int=1000,
                         patch_size=256, patch_bias=0):
    '''
    create lmdb file for given split from single tif files

    Parameters
    ----------
    dir_data : str
        directory of data.
    dir_lmdb : str
        directory of lmdb folder.
    map_size : int
        map size of lmdb dataset.
    split : str, optional
        for which split the lmdb is created. The default is 'validation'.
    write_interv : int, optional
        writting inverval during the creation. The default is 1000.

    Returns
    -------
    None.

    '''
    # source data paths
    fp_lmdb_src = os.path.join(dir_source_lmdb,f'DFC2020_s12_{split}.lmdb')
    env_src = lmdb.open(fp_lmdb_src, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    tnx_src = env_src.begin(write=False)
    nf = int(tnx_src.stat()['entries']/3)  

    # open lmdb
    if not os.path.exists(dir_lmdb):
        env = lmdb.open(dir_lmdb, map_size=map_size)
    else:
        env = lmdb.open(dir_lmdb, map_size=map_size, writemap=True) # continuously write to disk
    txn = env.begin(write=True)
    
    # read tif data and write it into lmdb
    for i in range(nf):
        # save to lmdb
        for k in ['s1','s2', 'lbl']:
            key = f'{k}_{i}'
            data_ = tnx_src.get(key.encode())
            data_bytes, data_shape = pickle.loads(data_)
            data = np.frombuffer(data_bytes, dtype=SDTYPE[k]).reshape(data_shape)
            imgs = crop_image_BWH(data, patch_size, patch_bias)
            for j, im in enumerate(imgs):
                key_ = f'{k}_{i*len(imgs)+j}'
                if k == 'lbl': im = im.astype(np.uint8)
                obj = (im.tobytes(), im.shape)
                txn.put(key_.encode(), pickle.dumps(obj)) 
        
        if (i+1)%write_interv == 0:
            txn.commit()
            txn = env.begin(write=True)
            
    # commit
    txn.commit()
    env.sync()
    env.close()
    env_src.close()
    
    return



if __name__ == '__main__':
    dir_data = r'/p/project1/hai_dm4eo/liu_ch/data/DFC2020'
    lmdb_dn = 'lmdb_small'
    data_dn = 'lmdb_raw'
    img_type = 's12'
    patch_size = 128
    patch_bias = 0
    # lmdb directory
    lmdb_pd = os.path.join(dir_data,lmdb_dn)
    Path(lmdb_pd).mkdir(parents=True, exist_ok=True)
    MSIZ = {'s2':{'validation':1932735283,'test':9234179686},
            's12':{'validation':3221225472, 'test':15032385536},}
    msize_dict = MSIZ[img_type]
    for split in ['validation']:    
        t0 = datetime.datetime.now().replace(microsecond=0)
        
        dir_lmdb = os.path.join(lmdb_pd,f'DFC2020_{img_type}_{split}.lmdb')
        map_size = msize_dict[split]
        # crop data to new lmdb
        create_s12_lmdb_file(os.path.join(dir_data,data_dn), 
                             dir_lmdb, map_size, 
                             split=split, write_interv=500,
                             patch_size=patch_size, patch_bias=patch_bias)
    
        t1 = datetime.datetime.now().replace(microsecond=0)
        print(f'Writting {split} data into lmdb uses {t1-t0}s.')
    
    
    