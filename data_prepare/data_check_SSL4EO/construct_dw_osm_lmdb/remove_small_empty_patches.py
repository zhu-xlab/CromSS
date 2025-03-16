'''
Remove small empty patches from the original lmdb files after cropping for ViT inputs
'''
import lmdb
import os
import numpy as np
import pickle

src_dir =  'SSL4EO-S2'
src_lmdb_folder = 'downstream_lmdb_small'
dst_lmdb_folder = 'downstream_lmdb_small_remove'

src_lmdb_path = os.path.join(src_dir, src_lmdb_folder)
dst_lmdb_path = os.path.join(src_dir, dst_lmdb_folder)
os.makedirs(dst_lmdb_path, exist_ok=True)
src_files = os.listdir(src_lmdb_path)

for src_file in src_files:
    fp_src = os.path.join(src_lmdb_path, src_file)
    fp_dst = os.path.join(dst_lmdb_path, src_file)

    with lmdb.open(fp_src, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False) as env_src:
        with lmdb.open(fp_dst, map_size=env_src.info()['map_size']) as env_dst:
            with env_src.begin(write=False) as txn_src:
                with env_dst.begin(write=True) as txn_dst:
                    if 'osm' in src_file:
                        key_prefix = 'osm'
                        keys = ['s1', 's2', 'osm', 'dw']
                        n = int((txn_src.stat()['entries'])/4)
                    else:
                        key_prefix = 'dw'
                        keys = ['s1', 's2', 'dw']
                        n = int((txn_src.stat()['entries'])/3)
                    print(f'Original {src_file} has {n} patches in total')
                    
                    ind = 0
                    n_empty = 0
                    for i in range(n): 
                        key_lbl = f'{key_prefix}_{i}'
                        data_lbl = txn_src.get(key_lbl.encode())
                        lbl_bytes, lbl_shape, lbl_fid = pickle.loads(data_lbl)
                        lbl = np.frombuffer(lbl_bytes, dtype=np.uint8).reshape(lbl_shape)
                        if (lbl==0).sum()==lbl.size: 
                            n_empty += 1
                            continue
                        # resave the non-empty patches
                        for key in keys:
                            key_org = f'{key}_{i}'
                            key = f'{key}_{ind}'
                            data = txn_src.get(key_org.encode())
                            txn_dst.put(key.encode(), data)
                        ind += 1
    print(f'{src_file} has {n_empty} empty patches removed, with {ind} non-empty patches left')

# # downstream_lmdb_smaller
# Original osm_test_lmdb has 10000 patches in total
# osm_test_lmdb has 2822 empty patches removed, with 7178 non-empty patches left
# Original osm_train_lmdb has 34300 patches in total
# osm_train_lmdb has 9501 empty patches removed, with 24799 non-empty patches left
# Original dw_train_lmdb has 89350 patches in total
# dw_train_lmdb has 1963 empty patches removed, with 87387 non-empty patches left
# Original dw_test_lmdb has 8500 patches in total
# dw_test_lmdb has 72 empty patches removed, with 8428 non-empty patches left