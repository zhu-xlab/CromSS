"""
download osm labels from https://osmlanduse.org/#12/8.7/49.4/0/ via Web Map Service (WMS) 
and save as tif for the training set

To reduce the misalignment, 
    - we first fetch a biger patch (2 times of the original patch) from WMS in its CRS
    - we then transform the patch to the target CRS and crop to the original patch
"""
import os
import datetime
import skimage
import requests
import rasterio
import numpy as np
import pandas as pd
import multiprocessing

from io import BytesIO
from pathlib import Path
from owslib.wms import WebMapService as WMS

from rasterio import transform
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

# import matplotlib.pyplot as plt

WCRS = 3857
WURL = 'https://maps.heigit.org/osmlanduse/service'
WLAYER = 'osmlanduse:osm_lulc'

SIZE_EXPAND = 2
SIZE_BUFF = 10


def get_color_coding_from_legend():
    wms = WMS(WURL)
    layer = wms.contents[WLAYER]
    
    # get legend url
    for k1 in layer.styles:
        for k2 in layer.styles[k1]:
            if k2=='legend': break
        if k2=='legend': break
    url_lg = layer.styles[k1][k2]

    # fetch legend data -> an image with color bulk and names
    response = requests.get(url_lg)
    with BytesIO(response.content) as memfile:
        with rasterio.open(memfile) as dataset:
            legend_array = dataset.read()
    
    # get color mapping dictionary
    la = legend_array.transpose(1,2,0)[:,:,:3]
    la_sr = np.sum(la[:,:,0],axis=0)
    la_single_col = la[:,la_sr!=la_sr[0]][:,0]
    la_colors_, inds = np.unique(la_single_col,axis=0,return_index=True)
    sort_inds = np.argsort(inds)
    la_colors = la_colors_[sort_inds,:]
    
    color_mapping = {}
    for i in range(la_colors.shape[0]):
        color_mapping[i] = la_colors[i]
    return color_mapping
        

def fetch_data_from_wms(bbox, region_size, style='default', format='image/GeoTIFF'):
    wms = WMS(WURL)
    params = {"layers": [WLAYER],
              "bbox": bbox, # Left, bottom, right, top
              "styles": [style],
              "format": format,
              "size": region_size,
              "srs": CRS.from_epsg(WCRS),}
    response = wms.getmap(**params, transparent=False)
    
    # Convert the image data to a NumPy array using rasterio
    with BytesIO(response.read()) as memfile:
        with rasterio.open(memfile) as dataset:
            img_array = dataset.read()
    return img_array.transpose(1,2,0)


def convert_rgb_to_single_band_label(img_array, color_mapping):
    # convert image data to float type
    imgf = img_array.astype(float)
    
    # find which classes are contained in current patch
    dists = []
    for i,k in enumerate(color_mapping):
        col_code = color_mapping[k].astype(float)
        dist = ((imgf-col_code[None,None])**2).sum(axis=2)
        dists.append(dist)
    dists_all = np.dstack(dists)
    
    # get final one-digit labels
    la_final = np.argmin(dists_all,axis=2)
    return la_final.astype(dtype=img_array.dtype)


def erosion_dilation_scikit(image, n_classes, kernel_size=3):
    nimages = [np.zeros_like(image)]
    for c in range(n_classes):
        if c!=0:
            img_c = (image==c).astype(image.dtype)
            img_c = skimage.morphology.binary_erosion(img_c).astype(image.dtype)
            if np.sum(img_c)>0:
                # img_c2 = skimage.morphology.binary_erosion(img_c).astype(image.dtype)
                # if np.sum(img_c2)>0:
                img_c = skimage.morphology.binary_dilation(img_c).astype(image.dtype)
            nimages.append(img_c)
    nimages = np.dstack(nimages)
    img_new = np.argmax(nimages,axis=2).astype(image.dtype)
    dup = nimages.sum(axis=2)>1
    img_new[dup] = 0
            
    return img_new


def transform_to_target_crs_and_crop(img_array, bbox, regsize, dst_bbox, dst_regsize, dst_crs):
    # Calculate the transform matrix for georeferencing from bounding boxes
    # src
    src_transform = transform.from_bounds(bbox[0],bbox[1],bbox[2],bbox[3],regsize[0],regsize[1])
    # dst
    dst_transform = transform.from_bounds(dst_bbox[0],dst_bbox[1],dst_bbox[2],dst_bbox[3],dst_regsize[0],dst_regsize[1])
    
    # Reproject and crop the image
    dst_array = np.empty((dst_regsize[0],dst_regsize[1]), dtype=img_array.dtype)
    reproject(source=img_array,
              destination=dst_array,
              dst_transform=dst_transform,
              dst_crs=CRS.from_epsg(dst_crs),
              src_transform=src_transform,
              src_crs=CRS.from_epsg(WCRS),
              resampling=Resampling.bilinear
              )       
    return dst_array  


def save_geotiff(lbl_array, bbox, out_path):
    # Define the pixel size based on the bounding box and image dimensions
    pixel_width = (bbox[2] - bbox[0]) / lbl_array.shape[1]
    pixel_height = (bbox[3] - bbox[1]) / lbl_array.shape[0]

    # Calculate the transform matrix for georeferencing: given left top and pixel sizes.
    transform = from_origin(bbox[0], bbox[-1], pixel_width, pixel_height)

    # Create the GeoTIFF file and write the data
    tiff_params = {"driver":   'GTiff',
                   "height":    lbl_array.shape[0],
                   "width":     lbl_array.shape[1],
                   "count":     1,  # Single band image
                   "dtype":     lbl_array.dtype,
                   "crs":       CRS.from_epsg(WCRS),  # Use the specified CRS
                   "transform": transform}
    with rasterio.open(out_path,'w', **tiff_params) as dst:
        dst.write(lbl_array, 1)  # Write the data to the first band
    return  


def get_and_save_single_label_image(bbox, dst_bbox, dst_regsize, dst_crs, color_mapping, out_path):
    try:
        # get source/expanded bounding boxes and region sizes
        left, bottom, right, top = bbox
        bbox_h, bbox_w = top - bottom, right - left
        h_buff, w_buff = bbox_h/SIZE_BUFF, bbox_w/SIZE_BUFF
        bbox_expand = [left-h_buff, bottom-h_buff, right+w_buff, top+w_buff]
        regsize_expand = [int(dst_regsize[j]*(1+2/SIZE_BUFF)*SIZE_EXPAND) for j in range(2)]
        try: 
            # fetch from expanded region
            img_array = fetch_data_from_wms(bbox_expand, regsize_expand, 
                                            style='default', format='image/GeoTIFF')
        except:
            # fetch from original region
            img_array = fetch_data_from_wms(bbox, regsize_expand, 
                                            style='default', format='image/GeoTIFF')
        # rgb -> single band
        lbl_array = convert_rgb_to_single_band_label(img_array, color_mapping)
        lbl_array = erosion_dilation_scikit(lbl_array, len(color_mapping), kernel_size=3)
        
        # transform to target crs and crop to original region
        osm = transform_to_target_crs_and_crop(lbl_array, bbox_expand, regsize_expand, dst_bbox, dst_regsize, dst_crs)
        osm = erosion_dilation_scikit(osm, len(color_mapping), kernel_size=3)
    
        # save as GeoTiff
        if np.sum(osm>0)>0:
            save_geotiff(osm, bbox, out_path)
            return
        else:
            fname = os.path.basename(out_path)
            dw_id = 'dw_'+fname[4:-4]
            return dw_id
    except Exception as e:
        fname = os.path.basename(out_path)
        dw_id = 'dw_'+fname[4:-4]
        print(f"Error occured at {dw_id}: {e}")
        return dw_id


def parallel_single_func(args):
    a, b, c, d, e, f = args
    return get_and_save_single_label_image(a, b, c, d, e, f)


if __name__ == '__main__':
    t0 = datetime.datetime.now().replace(microsecond=0)
    pdir = '/p/scratch/hai_dm4eo/liu_ch/test/data_v2/download_osm'
    save_dir = os.path.join(pdir, 'dw_train_osm')
    fp_csv = '/p/scratch/hai_dm4eo/liu_ch/test/data_v1/download_osm/center_voids_expert.csv'
    # fp_csv = r'C:\liuchy\Research\Projects\20230412_NL_Pretrain\Codes\data_prepare\toy_data_SSL4EO\annotated\center_voids_expert_v1.csv'
    fp_empty = os.path.join(pdir, 'empty_osm_files.csv')
    
    # read meta data
    meta = pd.read_csv(fp_csv, header=0)
    n_sam = meta.shape[0]
    
    # construct subdirectories
    hmsph = np.unique(np.array(meta.loc[:,'hemisphere']))
    biome = np.unique(np.array(meta.loc[:,'biome']))
    for hm in hmsph:
        for bi in biome:
            sub_dir = os.path.join(save_dir,hm,str(bi))
            Path(sub_dir).mkdir(parents=True, exist_ok=True)
    
    # preparation - get color coding from legend 
    color_mapping = get_color_coding_from_legend()
    
    # preparation - input of single compute function 
    bboxes = [[meta.loc[i,f'bbox{j}'] for j in range(1,5)] for i in range(n_sam)]
    dst_bboxes = [[meta.loc[i,f'bound{j}'] for j in range(1,5)] for i in range(n_sam)]
    dst_crses = [meta.loc[i,'crs'] for i in range(n_sam)]
    dst_regsizes = [[meta.loc[i,'width'],meta.loc[i,'height']] for i in range(n_sam)]
    out_paths = [os.path.join(save_dir, meta.loc[i,'hemisphere'], str(meta.loc[i,'biome']), 
                              f"osm_{meta.loc[i,'dw_id'][3:]}.tif") for i in range(n_sam)]
    
    data = [(bboxes[i], dst_bboxes[i], dst_regsizes[i], dst_crses[i], color_mapping, out_paths[i]) for i in range(n_sam)]
    
    # parallel computing
    # get number of cpu cores
    num_processes = multiprocessing.cpu_count()  
    
    if num_processes>1:
        # Create a pool of worker processes
        print(f'{num_processes} cores are used!')
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(parallel_single_func, data)
        # record empty files
        empty_files = []
        for r in results: 
            if r: 
                empty_files.append(r)
    else:
        # single process
        print('No parallel processes are triggered!')
        for d in [data[1]]:
            r = get_and_save_single_label_image(d[0],d[1],d[2],d[3],d[4],d[5])
            if r: 
                empty_files.append(r)
    print(f"{len(empty_files)} files have no corresponding osm labels.")
    
    # write to excel
    df = pd.DataFrame(empty_files)
    # Write the DataFrame to an Excel file
    df.to_csv(fp_empty, index=False, header=False)
    
    t1 = datetime.datetime.now().replace(microsecond=0)
    print(f"Finished! Used {t1-t0}s.")
    
    