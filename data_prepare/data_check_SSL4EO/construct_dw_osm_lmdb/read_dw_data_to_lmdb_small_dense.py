'''
Construct densely cropped patches with overlap for generating test prediction maps
'''
import os
import lmdb
import pickle
import rasterio
import numpy as np
from tqdm import tqdm


S1_MEAN = [-12.54847273, -20.19237134]
S1_STD = [5.25697717, 5.91150917]

S2A_MEAN = [752.40087073, 884.29673756, 1144.16202635, 1297.47289228, 1624.90992062, 2194.6423161, 2422.21248945, 2517.76053101, 2581.64687018, 2645.51888987, 2368.51236873, 1805.06846033]
S2A_STD = [1108.02887453, 1155.15170768, 1183.6292542, 1368.11351514, 1370.265037, 1355.55390699, 1416.51487101, 1474.78900051, 1439.3086061, 1582.28010962, 1455.52084939, 1343.48379601]

S2C_MEAN = [1605.57504906, 1390.78157673, 1314.8729939, 1363.52445545, 1549.44374991, 2091.74883118, 2371.7172463, 2299.90463006, 2560.29504086, 830.06605044, 22.10351321, 2177.07172323, 1524.06546312]
S2C_STD = [786.78685367, 850.34818441, 875.06484736, 1138.84957046, 1122.17775652, 1161.59187054, 1274.39184232, 1248.42891965, 1345.52684884, 577.31607053, 51.15431158, 1336.09932639, 1136.53823676]


# show single image examples
########################## s1 ##########################
# fp_s1 = 'data_s12/dw_train/image_s1/s1_-0.0520777954_32.9493499204-20190816.tif'

# with rasterio.open(fp_s1) as src:
#     img_s1 = src.read()
#     img_s1 = np.transpose(img_s1, (1, 2, 0))
#     print(img_s1.shape, img_s1.dtype, img_s1.min(), img_s1.max())
#     # (510, 510, 2) float32 -51.266907 1.3400639

# head = np.percentile(img_s1, 1, axis=(0, 1))
# tail = np.percentile(img_s1, 99, axis=(0, 1))
# print(head, tail) # [-20.08683107 -43.63033806] [ -6.45329768 -18.027113  ]

# img_head = np.ones_like(img_s1) * np.array(head)[None, None, :]
# img_tail = np.ones_like(img_s1) * np.array(tail)[None, None, :]
# img_s1 = np.clip(img_s1, img_head, img_tail)

# for i in range(2):
#     img = img_s1[:, :, i]
#     head_rate = np.sum(img < head[i]) / img.size
#     tail_rate = np.sum(img > tail[i]) / img.size
#     print(i, head_rate, tail_rate) # 0/1 0.01 0.01


########################## s2 ##########################
# fp_s2 = 'data_s12/dw_train/image_s2/s2_-0.0520777954_32.9493499204-20190816.tif'
# with rasterio.open(fp_s2) as src:
#     img_s2 = src.read()
#     img_s2 = np.transpose(img_s2, (1, 2, 0))
#     print(img_s2.shape, img_s2.dtype, img_s2.min(), img_s2.max())
#     # (510, 510, 12) uint16 0 10000


########################## dw ##########################
# fp_dwtr = 'data_s12/dw_train/label_dw/dw_-0.0520777954_32.9493499204-20190816.tif'
# with rasterio.open(fp_dwtr) as src:
#     img_dw = src.read()
#     img_dw = np.transpose(img_dw, (1, 2, 0))
#     print(img_dw.shape, img_dw.dtype, img_dw.min(), img_dw.max())
#     # (510, 510, 1) uint8 0 1

# fp_dwts = 'data_s12/dw_test/label_dw/dw_-0p6481384406_6p8231873947-20190226.tif'
# with rasterio.open(fp_dwts) as src:
#     img_dw = src.read()
#     img_dw = np.transpose(img_dw, (1, 2, 0))
#     print(img_dw.shape, img_dw.dtype, img_dw.min(), img_dw.max())
#     # (510, 510, 1) uint8 0 1


########################## osm ##########################
# dosm = 'data_s12/osm_train_large/label_osm'
# fnames_osm = os.listdir(dosm)
# for f in fnames_osm:
#     fp = os.path.join(dosm, f)
#     with rasterio.open(fp) as src:
#         img_osm = src.read()
#         img_osm = np.transpose(img_osm, (1, 2, 0))
#         if img_osm.max() == 13:
#             print(f, img_osm.shape, img_osm.dtype, img_osm.min(), img_osm.max())
        
# fp_osmtr = 'data_s12/osm_train_large/label_osm/osm_-75.1378532186_19.9946367890-20190107.tif'
# with rasterio.open(fp_osmtr) as src:
#     img_osm = src.read()
#     img_osm = np.transpose(img_osm, (1, 2, 0))
#     print(img_osm.shape, img_osm.dtype, img_osm.min(), img_osm.max(), np.unique(img_osm))
#     # (510, 510, 1) uint8 0 1


def s1_exclude_outliers(img, head_rate=1, tail_rate=99):
    head = np.percentile(img, head_rate, axis=(0, 1))
    img_head = np.ones_like(img) * np.array(head)[None, None, :]
    tail = np.percentile(img, tail_rate, axis=(0, 1))
    img_tail = np.ones_like(img) * np.array(tail)[None, None, :]
    img = np.clip(img, img_head, img_tail)
    return img
    
    
def s1_normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def s2_normalize(img):
    img = img.astype(np.float32) / 10000.0 * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def write_tif_to_lmdb(tif_dir, lmdb_dir, map_size_G):
    env = lmdb.open(lmdb_dir, map_size=int(map_size_G*1024**3), writemap=True)
    with env.begin(write=True) as txn:
        subdirs = os.listdir(tif_dir)
        fnames = [f[3:] for f in os.listdir(os.path.join(tif_dir, 'image_s1')) if f.endswith('.tif')]
        for subdir in subdirs:
            desc_head = f"{tif_dir}/{subdir}"
            # write s1 to lmdb
            if 's1' in subdir:
                for fi, fname in tqdm(enumerate(fnames), desc=desc_head):
                    # read s1
                    fp = os.path.join(tif_dir, subdir, f's1_{fname}')
                    with rasterio.open(fp) as src:
                        img = src.read()
                        img = np.transpose(img, (1, 2, 0))
                    # exclude outliers: 1% and 99%
                    img = s1_exclude_outliers(img)
                    # normalize
                    img_norm = [s1_normalize(img[:,:,i], S1_MEAN[i], S1_STD[i]) for i in range(2)]
                    img_norm = np.dstack(img_norm)
                    # write to lmdb    
                    if fi==0: print(desc_head, ":", img_norm.shape, img_norm.dtype, img_norm.min(), img_norm.max())
                    obj = (img_norm.tobytes(), img_norm.shape, f's1_{fname}')       
                    ind = f's1_{fi}'
                    txn.put(str(ind).encode(), pickle.dumps(obj)) 

            # write s2 to lmdb
            if 's2' in subdir:
                for fi, fname in tqdm(enumerate(fnames), desc=desc_head):
                    # read s2
                    fp = os.path.join(tif_dir, subdir, f's2_{fname}')
                    with rasterio.open(fp) as src:
                        img = src.read()
                        img = np.transpose(img, (1, 2, 0))
                    # normalize
                    img = s2_normalize(img)
                    # write to lmdb
                    if fi==0: print(desc_head, ":", img.shape, img.dtype, img.min(), img.max())
                    obj = (img.tobytes(), img.shape, f's2_{fname}')
                    ind = f's2_{fi}'
                    txn.put(str(ind).encode(), pickle.dumps(obj))
            
            # write dw label to lmdb
            if 'dw' in subdir:
                for fi, fname in tqdm(enumerate(fnames), desc=desc_head):
                    # read label
                    fp = os.path.join(tif_dir, subdir, f'dw_{fname}')
                    with rasterio.open(fp) as src:
                        img = src.read().squeeze()
                    if len(img.shape)>2: img = img[1]
                    # write to lmdb
                    if fi==0: print(desc_head, ":", img.shape, img.dtype, img.min(), img.max())
                    obj = (img.tobytes(), img.shape, f'dw_{fname}')
                    ind = f'dw_{fi}'
                    txn.put(str(ind).encode(), pickle.dumps(obj))
            
            # write osm label to lmdb
            if 'osm' in subdir:
                for fi, fname in tqdm(enumerate(fnames), desc=desc_head):
                    # read label
                    fp = os.path.join(tif_dir, subdir, f'osm_{fname}')
                    with rasterio.open(fp) as src:
                        img = src.read().squeeze()
                    # write to lmdb
                    if fi==0: print(desc_head, ":", img.shape, img.dtype, img.min(), img.max())
                    obj = (img.tobytes(), img.shape, f'osm_{fname}')
                    ind = f'osm_{fi}'
                    txn.put(str(ind).encode(), pickle.dumps(obj))
    env.close()
    return 


def write_original_tif_to_lmdb(tif_dir, lmdb_dir, map_size_G):
    env = lmdb.open(lmdb_dir, map_size=int(map_size_G*1024**3), writemap=True)
    with env.begin(write=True) as txn:
        subdirs = os.listdir(tif_dir)
        fnames = [f[3:] for f in os.listdir(os.path.join(tif_dir, 'image_s1')) if f.endswith('.tif')]
        for subdir in subdirs:
            desc_head = f"{tif_dir}/{subdir}"
            # write s1 to lmdb
            if 's1' in subdir:
                for fi, fname in tqdm(enumerate(fnames), desc=desc_head):
                    # read s1
                    fp = os.path.join(tif_dir, subdir, f's1_{fname}')
                    with rasterio.open(fp) as src:
                        img = src.read().squeeze()
                        # img = np.transpose(img, (1, 2, 0))
                    # write to lmdb    
                    if fi==0: print(desc_head, ":", img.shape, img.dtype, img.min(), img.max())
                    obj = (img.tobytes(), img.shape, f's1_{fname}')       
                    ind = f's1_{fi}'
                    txn.put(str(ind).encode(), pickle.dumps(obj)) 

            # write s2 to lmdb
            if 's2' in subdir:
                for fi, fname in tqdm(enumerate(fnames), desc=desc_head):
                    # read s2
                    fp = os.path.join(tif_dir, subdir, f's2_{fname}')
                    with rasterio.open(fp) as src:
                        img = src.read().squeeze()
                        # img = np.transpose(img, (1, 2, 0))
                    # write to lmdb
                    if fi==0: print(desc_head, ":", img.shape, img.dtype, img.min(), img.max())
                    obj = (img.tobytes(), img.shape, f's2_{fname}')
                    ind = f's2_{fi}'
                    txn.put(str(ind).encode(), pickle.dumps(obj))
            
            # write dw label to lmdb
            if 'dw' in subdir:
                for fi, fname in tqdm(enumerate(fnames), desc=desc_head):
                    # read label
                    fp = os.path.join(tif_dir, subdir, f'dw_{fname}')
                    with rasterio.open(fp) as src:
                        img = src.read().squeeze()
                    if len(img.shape)>2: img = img[1]
                    # write to lmdb
                    if fi==0: print(desc_head, ":", img.shape, img.dtype, img.min(), img.max())
                    obj = (img.tobytes(), img.shape, f'dw_{fname}')
                    ind = f'dw_{fi}'
                    txn.put(str(ind).encode(), pickle.dumps(obj))
            
            # write osm label to lmdb
            if 'osm' in subdir:
                for fi, fname in tqdm(enumerate(fnames), desc=desc_head):
                    # read label
                    fp = os.path.join(tif_dir, subdir, f'osm_{fname}')
                    with rasterio.open(fp) as src:
                        img = src.read().squeeze()
                    # write to lmdb
                    if fi==0: print(desc_head, ":", img.shape, img.dtype, img.min(), img.max())
                    obj = (img.tobytes(), img.shape, f'osm_{fname}')
                    ind = f'osm_{fi}'
                    txn.put(str(ind).encode(), pickle.dumps(obj))
    env.close()
    return 

# # # # # prepare small patch lmdb # # # # #
def write_original_tif_to_lmdb_in_small_patch(tif_dir, lmdb_dir, map_size_G, patch_size):
    env = lmdb.open(lmdb_dir, map_size=int(map_size_G*1024**3), writemap=True)
    with env.begin(write=True) as txn:
        subdirs = os.listdir(tif_dir)
        fnames = [f[3:] for f in os.listdir(os.path.join(tif_dir, 'image_s1')) if f.endswith('.tif')]
        for subdir in subdirs:
            desc_head = f"{tif_dir}/{subdir}"
            # write s1 to lmdb
            if 's1' in subdir:
                for fi, fname in tqdm(enumerate(fnames), desc=desc_head):
                    # read s1
                    fp = os.path.join(tif_dir, subdir, f's1_{fname}')
                    with rasterio.open(fp) as src:
                        img = src.read().squeeze()
                        # img = np.transpose(img, (1, 2, 0))
                    # write to lmdb    
                    # # # # # pad and crop the image # # # # #
                    imgs = pad_and_crop_image_BWH(img, patch_size)
                    if fi==0: print(desc_head, ":", img.shape, img.dtype, patch_size, len(imgs))
                    # # # # # pad and crop the image # # # # #
                    for i, img in enumerate(imgs):
                        obj = (img.tobytes(), img.shape, f's1_{fname}_{fi}_{i}')
                        ind = f's1_{int(fi*len(imgs)+i)}'
                        txn.put(str(ind).encode(), pickle.dumps(obj))       
                        
            # write s2 to lmdb
            if 's2' in subdir:
                for fi, fname in tqdm(enumerate(fnames), desc=desc_head):
                    # read s2
                    fp = os.path.join(tif_dir, subdir, f's2_{fname}')
                    with rasterio.open(fp) as src:
                        img = src.read().squeeze()
                        # img = np.transpose(img, (1, 2, 0))
                    # write to lmdb
                    # # # # # pad and crop the image # # # # #
                    imgs = pad_and_crop_image_BWH(img, patch_size)
                    if fi==0: print(desc_head, ":", img.shape, img.dtype, patch_size, len(imgs))
                    # # # # # pad and crop the image # # # # #
                    for i, img in enumerate(imgs):
                        obj = (img.tobytes(), img.shape, f's2_{fname}_{fi}_{i}')
                        ind = f's2_{int(fi*len(imgs)+i)}'
                        txn.put(str(ind).encode(), pickle.dumps(obj))   
            
            # write dw label to lmdb
            if 'dw' in subdir:
                for fi, fname in tqdm(enumerate(fnames), desc=desc_head):
                    # read label
                    fp = os.path.join(tif_dir, subdir, f'dw_{fname}')
                    with rasterio.open(fp) as src:
                        img = src.read().squeeze()
                    if len(img.shape)>2: img = img[1]
                    # write to lmdb
                    # # # # # pad and crop the image # # # # #
                    imgs = pad_and_crop_image_BWH(img, patch_size)
                    if fi==0: print(desc_head, ":", img.shape, img.dtype, patch_size, len(imgs))
                    # # # # # pad and crop the image # # # # #
                    for i, img in enumerate(imgs):
                        obj = (img.tobytes(), img.shape, f'dw_{fname}_{fi}_{i}')
                        ind = f'dw_{int(fi*len(imgs)+i)}'
                        txn.put(str(ind).encode(), pickle.dumps(obj))   
            
            # write osm label to lmdb
            if 'osm' in subdir:
                for fi, fname in tqdm(enumerate(fnames), desc=desc_head):
                    # read label
                    fp = os.path.join(tif_dir, subdir, f'osm_{fname}')
                    with rasterio.open(fp) as src:
                        img = src.read().squeeze()
                    # write to lmdb
                    # # # # # pad and crop the image # # # # #
                    imgs = pad_and_crop_image_BWH(img, patch_size)
                    if fi==0: print(desc_head, ":", img.shape, img.dtype, patch_size, len(imgs))
                    # # # # # pad and crop the image # # # # #
                    for i, img in enumerate(imgs):
                        obj = (img.tobytes(), img.shape, f'osm_{fname}_{fi}_{i}')
                        ind = f'osm_{int(fi*len(imgs)+i)}'
                        txn.put(str(ind).encode(), pickle.dumps(obj))   
    env.close()
    return 


def pad_and_crop_image_BWH(img, patch_size=256):
    # # # # # pad the image to 512x512 by copying the first and last rows/columns # # # # #
    if img.shape[-1] == 510:
        if len(img.shape) == 2:
            img = np.pad(img, ((1, 1), (1, 1)), 'edge')
        else:
            img = np.pad(img, ((0, 0), (1, 1), (1, 1)), 'edge')
    # # # # # crop big images into small patches # # # # #
    n_patch = np.ceil(img.shape[1] / patch_size).astype(int)
    patch_interv = np.ceil((img.shape[1] - patch_size)/(n_patch-1))
    imgs = []
    for i in range(n_patch):
        si = int(i * patch_interv) if i < n_patch-1 else img.shape[1] - patch_size
        ei = int(si + patch_size)
        for j in range(n_patch):
            sj = int(j * patch_interv) if j < n_patch-1 else img.shape[1] - patch_size
            ej = int(sj + patch_size)
            if len(img.shape) == 2:
                im = img[si:ei, sj:ej]
            else:
                im = img[:, si:ei, sj:ej]
            imgs.append(im)
    return imgs
    

if __name__ == '__main__':
    patch_size = 224
    lmdb_dir_name = f'data_lmdb_small_dense/patch{patch_size}'
    pdir = '/p/project1/hai_ws4eo/liu_ch/DW/test'
    splits = ['osm_test', 'dw_test'] # ['osm_test', 'dw_test']
    map_size_Gs = [8, 10]   # 106, 1666, 1219, 3574, 340
    # lmdb_dir_name = 'data_lmdb_small' if save_small else 'data_lmdb'
    if not os.path.exists(os.path.join(pdir, lmdb_dir_name)):
        os.makedirs(os.path.join(pdir, lmdb_dir_name))
    for i, sp in enumerate(splits):
        tif_dir = os.path.join(pdir, 'data_s12', sp)
        lmdb_dir = os.path.join(pdir, lmdb_dir_name, f'{sp}_lmdb') 
        map_size_G = map_size_Gs[i]
        write_original_tif_to_lmdb_in_small_patch(tif_dir, lmdb_dir, map_size_G, patch_size)