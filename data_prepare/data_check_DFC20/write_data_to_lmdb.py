# -*- coding: utf-8 -*-
"""
Create lmdb file for DFC2020 dataset
@author: liu_ch
"""

import os
import lmdb
import pickle
import rasterio
import datetime
import numpy as np
from pathlib import Path


dfc_lbls = [1,2,4,5,6,7,9,10]
dw_lbls =  [1,5,2,3,4,6,7,0]


def convert_label_annotations(lbl):
    lbl_new = np.ones(lbl.shape)*8
    for dfc_i, dw_i in zip(dfc_lbls, dw_lbls):
        lbl_new[lbl==dfc_i] = dw_i
    n_check = np.sum(lbl_new==8)
    return lbl_new, n_check


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


def create_s_lmdb_file(dir_data:str, dir_lmdb:str, map_size:int, 
                       split:str='validation', img_type='s2',
                       write_interv:int=1000,
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
    # raw data paths
    pimg = os.path.join(dir_data,split,f'{img_type}_{split}')
    plbl = os.path.join(dir_data,split,f'dfc_{split}')
    nf = len(os.listdir(pimg))
    
    # open lmdb
    if not os.path.exists(dir_lmdb):
        env = lmdb.open(dir_lmdb, map_size=map_size)
    else:
        env = lmdb.open(dir_lmdb, map_size=map_size, writemap=True) # continuously write to disk
    txn = env.begin(write=True)
    
    # read tif data and write it into lmdb
    for i in range(nf):
        # get file names
        # e.g., fname_img = 'ROIs0000_test_s2_0_p0.tif'
        #       fname_img = 'ROIs0000_test_dfc_0_p0.tif
        fname_img = f'ROIs0000_{split}_{img_type}_0_p{i}.tif'
        fp_img = os.path.join(pimg,fname_img)
        fname_lbl = f'ROIs0000_{split}_dfc_0_p{i}.tif'
        fp_lbl = os.path.join(plbl,fname_lbl)
        
        # read data from disk
        with rasterio.open(fp_img) as dataset:
            img = dataset.read().squeeze() # s1-dtype:float64 (2,256,256)
                                           # s2-dtype:uint16  (13,256,256)
        with rasterio.open(fp_lbl) as dataset:
            lbl = dataset.read().squeeze() # dtype:uint16 (13,256,256)
        
        # convert annotations for labels
        lbl, n_check = convert_label_annotations(lbl)
        if n_check>0:
            print(f'{fname_lbl} has {n_check} pixels with unknown label type!')
    
        # save to lmdb
        for k, img_ in zip(['img', 'lbl'],[img,lbl]):
            imgs = crop_image_BWH(img_, patch_size, patch_bias)
            for j, im in enumerate(imgs):
                key = f'{k}_{i*len(imgs)+j}'
                obj = (im.tobytes(), im.shape)
                txn.put(key.encode(), pickle.dumps(obj))  
        
        if (i+1)%write_interv == 0:
            txn.commit()
            txn = env.begin(write=True)
            
    # commit
    txn.commit()
    env.sync()
    env.close()
    
    return


def create_s12_lmdb_file(dir_data:str, dir_lmdb:str, map_size:int, 
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
    # raw data paths
    ps1 = os.path.join(dir_data,split,f's1_{split}')
    ps2 = os.path.join(dir_data,split,f's2_{split}')
    plbl = os.path.join(dir_data,split,f'dfc_{split}')
    nf = len(os.listdir(ps1))
    
    # open lmdb
    if not os.path.exists(dir_lmdb):
        env = lmdb.open(dir_lmdb, map_size=map_size)
    else:
        env = lmdb.open(dir_lmdb, map_size=map_size, writemap=True) # continuously write to disk
    txn = env.begin(write=True)
    
    # read tif data and write it into lmdb
    for i in range(nf):
        # get file names
        # e.g., fname_img = 'ROIs0000_test_s2_0_p0.tif'
        #       fname_img = 'ROIs0000_test_dfc_0_p0.tif
        fname_s1 = f'ROIs0000_{split}_s1_0_p{i}.tif'
        fname_s2 = f'ROIs0000_{split}_s2_0_p{i}.tif'
        fname_lbl = f'ROIs0000_{split}_dfc_0_p{i}.tif'
        fp_s1 = os.path.join(ps1,fname_s1)
        fp_s2 = os.path.join(ps2,fname_s2)
        fp_lbl = os.path.join(plbl,fname_lbl)
        
        # read data from disk
        with rasterio.open(fp_s1) as dataset:
            s1 = dataset.read().squeeze()  # dtype:float64 (2,256,256)
        with rasterio.open(fp_s2) as dataset:
            s2 = dataset.read().squeeze()  # dtype:uint16 (13,256,256)
        with rasterio.open(fp_lbl) as dataset:
            lbl = dataset.read().squeeze() # dtype:uint16 (256,256)
        
        # convert annotations for labels (reassign labels according to DW categories)
        lbl, n_check = convert_label_annotations(lbl)
        if n_check>0:
            print(f'{fname_lbl} has {n_check} pixels with unknown label type!')
    
        # save to lmdb
        for k, img in zip(['s1','s2', 'lbl'],[s1,s2,lbl]):
            imgs = crop_image_BWH(img, patch_size, patch_bias)
            for j, im in enumerate(imgs):
                key = f'{k}_{i*len(imgs)+j}'
                if k == 'lbl': im = im.astype(np.uint8)
                obj = (im.tobytes(), im.shape)
                txn.put(key.encode(), pickle.dumps(obj)) 
        
        if (i+1)%write_interv == 0:
            txn.commit()
            txn = env.begin(write=True)
            
    # commit
    txn.commit()
    env.sync()
    env.close()
    
    return



if __name__ == '__main__':
    dir_data = 'DFC2020'
    lmdb_dn = 'lmdb_small'
    data_dn = 'raw_data'
    img_type = 's12' # ['s1','s2','s12']
    patch_size = 128
    patch_bias = 0
    # lmdb directory
    lmdb_pd = os.path.join(dir_data,lmdb_dn)
    Path(lmdb_pd).mkdir(parents=True, exist_ok=True)
    MSIZ = {'s2':{'validation':1932735283,'test':9234179686},
            's12':{'validation':3221225472, 'test':15032385536},}
    msize_dict = MSIZ[img_type]
    for split in ['test']:    
        t0 = datetime.datetime.now().replace(microsecond=0)
        
        dir_lmdb = os.path.join(lmdb_pd,f'DFC2020_{img_type}_{split}.lmdb')
        map_size = msize_dict[split]
        # write data to lmdb
        if img_type == 's12':
            create_s12_lmdb_file(os.path.join(dir_data,data_dn), 
                                 dir_lmdb, map_size, 
                                 split=split, write_interv=500,
                                 patch_size=patch_size, patch_bias=patch_bias)
        else:
            create_s_lmdb_file(dir_data, dir_lmdb, map_size, 
                               split=split, img_type=img_type, 
                               write_interv=500,
                               patch_size=patch_size, patch_bias=patch_bias)
    
        t1 = datetime.datetime.now().replace(microsecond=0)
        print(f'Writting {split} data into lmdb uses {t1-t0}s.')
    
    
    