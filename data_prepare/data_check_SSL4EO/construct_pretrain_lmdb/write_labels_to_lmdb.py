# -*- coding: utf-8 -*-
"""
Write labels into lmdb (single process)
- read from label.tif files
- write into lmdb
- record missing image ids according to csv

Created on Fri Jun 16 16:17:03 2023

@author: liu_ch
"""

import os
import lmdb
import argparse
import datetime
import pickle
import rasterio
import numpy as np
import pandas as pd
from PIL import Image

# Global variables
sub_nlist = ['sub1','sub2','sub3','sub4']
img_s = 264

def get_args():
    parser = argparse.ArgumentParser(description='Read and rewrite label data into lmdb')
    parser.add_argument('--file-name-of-dw-ids', dest='fn_dwids', default = 'dw_complete_ids.csv',
                        help='file name of noisy label ids')
    parser.add_argument('--directory-of-noisy-labels', dest='fp_x', default = 'dw_noisy_labels',
                        help='directory path of noisy labels')
    parser.add_argument('--output-lmdb-file-path', dest='dir_lmdb', default = 'dw_labels.lmdb',
                        help='output lmdb file')
    parser.add_argument('--check-start-id', type=int, dest='si', default=0)
    parser.add_argument('--check-end-id', type=int, dest='ei', default=-1)
    
    return parser.parse_args()


def write_files_to_lmdb(df_dm, fp_x, dir_lmdb, start_id=0, end_id=-1):
    unx_ids = []
    if end_id<0: end_id = df_dm.shape[0]
    print(f'Writing files from {start_id} to {end_id}')
    t0 = datetime.datetime.now().replace(microsecond=0)
    
    # open lmdb
    if not os.path.exists(dir_lmdb):
        env = lmdb.open(dir_lmdb, map_size=161061273600)
    else:
        env = lmdb.open(dir_lmdb, map_size=161061273600, writemap=True) # continuously write to disk
    txn = env.begin(write=True)
    
    # read and write
    unx_id = []
    for i in range(start_id, end_id):
        fi = df_dm.loc[i,'ids']
        fn_i = f'{fi:07d}'
        ind = df_dm.loc[i,'ind']
        subs = df_dm.loc[i,sub_nlist]
        
        # start reading files
        seas = []
        fail_load = 0
        # 2.1 - read data
        for sub in subs:
            # read data
            try:
                # read data from disk
                fp = os.path.join(fp_x,fn_i,sub,'label.tif')
                with rasterio.open(fp) as dataset:
                    img = dataset.read().squeeze().astype(np.uint8) # int16
                img = Image.fromarray(img)
                # resize - upsampling
                img = img.resize((img_s,img_s),Image.Resampling.BICUBIC)
                img = np.array(img)
                img = np.clip(img, 0, 8).astype(np.uint8)
            except:
                # record fail times for current image
                fail_load += 1
                break
            
            # gather seasonal data
            if fail_load > 0:
                break
            else:
                seas.append(img[None])
        
        # 2.2 - write extracted files to lmdb or record unextracted ones
        if fail_load == 0:
            seas = np.concatenate(seas,axis=0)
            obj = (seas.tobytes(), seas.shape, fi)        
            txn.put(str(ind).encode(), pickle.dumps(obj))   
            unx_id.append(-1)
        else:
            unx_id.append(ind)
            print(f'{i}-{ind}-{fi} fails!')
        
        if i%1000 == 0:
            txn.commit()
            txn = env.begin(write=True)
            t = datetime.datetime.now().replace(microsecond=0)
            print(f'The {i}th file have been written in lmdb at {t}')
            
    # commit
    txn.commit()
    env.sync()
    env.close()
    
    t1 = datetime.datetime.now().replace(microsecond=0)
    print(f'Writting is finished using {t1-t0}s')

    # gather unextracted ids
    unx_ids = np.array(unx_id)
    print(f'number of total processed ids: {len(unx_ids)}')
    unx_ids_f = np.unique(unx_ids)
    if len(unx_ids_f)==1 and unx_ids_f[0]==-1:
        print('All the files have been successfully read and rewritten.')
    else:
        nuns = len(unx_ids_f)-1 if -1 in unx_ids_f else len(unx_ids_f)
        print(f'number of unsuccessfully read files: {nuns}')
        print(f'number of successfully read files: {np.sum(unx_ids==-1)}')
        print(f'check: {nuns}+{np.sum(unx_ids==-1)}={nuns+np.sum(unx_ids==-1)}')

    return unx_ids_f

if __name__ == '__main__':
    args = get_args()

    # 1 - read noisy label data ids
    fn_unxids = r'unfound_label_ids.csv'
    df_dw = pd.read_csv(args.fn_dwids) # names=["ind", "ids", "sub1","sub2","sub3","sub4"]
    df_dw['ind'] = np.arange(df_dw.shape[0])
    
    # 2 - read extracted files to lmdb and record unextracted ones
    t0 = datetime.datetime.now().replace(microsecond=0)
    print(f'The reading and writing process starts at {t0}')
    
    unx_ids = write_files_to_lmdb(df_dw, args.fp_x, args.dir_lmdb, 
                                  start_id=args.si,
                                  end_id=args.ei)
    
    t1 = datetime.datetime.now().replace(microsecond=0)
    print(f'The reading and writing process ends at {t1} (total:{t1-t0})')

    # 4 - save unextracted file ids
    if len(unx_ids)>1:
        unx_ids = np.delete(unx_ids, np.where(unx_ids == -1))
        df_unxIDs = pd.DataFrame(unx_ids, columns=['unxID'])
        df_unxIDs.to_csv(fn_unxids)
    