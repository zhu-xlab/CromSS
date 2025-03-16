# -*- coding: utf-8 -*-
"""
Read data from single tif files or directly from tar.gz into lmdb (single process)
- first try to read from tif
- if fails, fetch data from tar.gz (will be very slow when tar.gz is large)

@author: liu_ch
"""

import os, lmdb, argparse, tarfile, datetime, io, pickle, rasterio
import numpy as np
import pandas as pd
import PIL.Image as Image
from tqdm import tqdm


BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    # paths of input data
    parser.add_argument('--file-name-of-dw-ids', dest='fn_dwids', default = 'dw_complete_ids.csv',
                        help='file name of noisy label ids')
    parser.add_argument('--directory-of-images', dest='fd_xal', default = 'SSL4EO_S12',
                        help='directory path of SSL4EO-S12 image data (containing extracted data directory and the original tar.gz file)')
    parser.add_argument('--directory-of-extracted-images', dest='fd_xal', default = '0k_251k_uint8_jpeg_tif/s2c',
                        help='directory path of SSL4EO-S12 image data (containing uncompressed data directory and tar.gz file)')
    parser.add_argument('--name-of-compressed-file', dest='fn_compress', default = 's2c_uint8.tar.gz',
                        help='name of the original compressed file')
    parser.add_argument('--output-lmdb-file-path', dest='dir_lmdb', default = '0k_251k_uint8_s2c.lmdb',
                        help='output lmdb file')
    parser.add_argument('--check-already-extracted-files', dest='ck_xal', action='store_false', default=True)
    parser.add_argument('--delete-extracted-files', dest='del_xal', action='store_false', default=True)
    parser.add_argument('--number-of-encoded-files', type=int, dest='nendf', default=-1)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    img_s = 264
    
    # 1 - read noisy label data ids
    df_dw = pd.read_csv(args.fn_dwids, names=["ids", "sub1", "sub2", "sub3", "sub4"])
    ids_dw = df_dw.to_numpy()[:,0]
    if args.nendf<0 or args.nendf>len(ids_dw): args.nendf = len(ids_dw)
    print(f'Encoding {args.nendf} files into lmdb')
    
    # 2 - create or open lmdb folder    
    if not os.path.exists(args.dir_lmdb):
        env = lmdb.open(args.dir_lmdb, map_size=1099511627776)
    else:
        env = lmdb.open(args.dir_lmdb, map_size=1099511627776, writemap=True) # continuously write to disk
    txn = env.begin(write=True)
    
    # 3 - open tar.gz file for reading
    t0 = datetime.datetime.now().replace(microsecond=0)
    print(f'tar file was opened at {t0}')
    tar = tarfile.open(os.path.join(args.fd_xal, args.fn_compress), "r:gz")
    
    # 4 - read and write data to lmdb one by one
    for i in tqdm(range(args.nendf), desc='Creating LMDB'):
        fi = ids_dw[i]
        fn_i = f'{fi:07d}'
        seas = []
        # 4.1 - read data
        for sub in df_dw.loc[i,['sub1','sub2','sub3','sub4']]:
            bands = []
            for b in BANDS:
                # read data
                fn = os.path.join(args.fn_x,fn_i,sub,f'{b}.tif')
                try:
                    # read data from disk
                    fp = os.path.join(args.fd_xal,fn)
                    with rasterio.open(fp) as dataset:
                        img = dataset.read().squeeze()
                    img = Image.fromarray(img)
                except:
                    # read data from tar
                    content = tar.extractfile(fn).read()
                    img = Image.open(io.BytesIO(bytearray(content)))
                # resize - upsampling
                img = img.resize((img_s,img_s),Image.Resampling.BICUBIC)
                img = np.array(img)
                bands.append(img[None])
            bands = np.concatenate(bands,axis=0)
            seas.append(bands[None])
        seas = np.concatenate(seas,axis=0)
        
        # 4.2 - write into lmdb 
        obj = (seas.tobytes(), seas.shape, fi)        
        txn.put(str(i).encode(), pickle.dumps(obj))            

        if i % 100 == 0:
            txn.commit()
            txn = env.begin(write=True)
            t1 = datetime.datetime.now().replace(microsecond=0)
            print(f'{i} files have been written in lmdb at {t1}')
    txn.commit()

    env.sync()
    env.close()
    tar.close()
    
    t2 = datetime.datetime.now().replace(microsecond=0)
    print(f'Process finished at {t2} - total:{t2-t0}')
            
        