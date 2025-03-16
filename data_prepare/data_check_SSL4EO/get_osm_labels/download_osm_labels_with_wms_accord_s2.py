# -*- coding: utf-8 -*-
"""
download osm labels via wms service according to bounding boxes of s2 images tranformed to wms crs
Test (discarded) script due to severe misalignment
"""

import os
import skimage
import requests
import rasterio
import numpy as np

from owslib.wms import WebMapService as WMS
from io import BytesIO

from rasterio.crs import CRS
from rasterio.transform import from_origin
from pyproj import Transformer

# import matplotlib.pyplot as plt

WCRS = 3857
WURL = 'https://maps.heigit.org/osmlanduse/service'
WLAYER = 'osmlanduse:osm_lulc'


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


def get_geoinfo_from_s2(tif_file, target_crs=3857):
    with rasterio.open(tif_file) as dataset:
        # get original crs 
        ocrs = dataset.crs
        # get image width and height
        h, w = dataset.height, dataset.width
    
    # crs transformer
    transformer = Transformer.from_crs(ocrs, CRS.from_epsg(3857))
    
    # lower left corner
    x01, y01 = dataset.xy(w,0)
    x1, y1 = transformer.transform(y01, x01)
    # upper right corner
    x02, y02 = dataset.xy(0,h)
    x2, y2 = transformer.transform(y02, x02)
    
    # output 1 - region size
    region_size = (h, w)
    # output 2 - bounding box
    bbox = (x1, y1, x2, y2)
    return region_size, bbox
        

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


def save_geotiff(lbl_array, bbox, out_path):
    # Define the pixel size based on the bounding box and image dimensions
    pixel_width = (bbox[2] - bbox[0]) / lbl_array.shape[1]
    pixel_height = (bbox[1] - bbox[3]) / lbl_array.shape[0]

    # Calculate the transform matrix for georeferencing
    transform = from_origin(bbox[0], bbox[1], pixel_width, pixel_height)

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




if __name__ == '__main__':
    fp_img = r'C:\liuchy\Research\Projects\20230412_NL_Pretrain\Codes\data_prepare\toy_data_SSL4EO\annotated\dw_train_s2c'
    fp_lbl = r'C:\liuchy\Research\Projects\20230412_NL_Pretrain\Codes\data_prepare\toy_data_SSL4EO\annotated\dw_train_label'
    fp_mso = r'C:\liuchy\Research\Projects\20230412_NL_Pretrain\Codes\data_prepare\toy_data_SSL4EO\annotated\dw_train_osm'
    
    # preparation - get color coding from legend 
    color_mapping = get_color_coding_from_legend()
    
    # fetch data one by one
    folders = os.listdir(fp_img)
    for fd in folders:
        tif_file = os.path.join(fp_img,fd,'B2.tif')
        # 1 - get image size and bounding box from sentinel-2 data 
        region_size, bbox = get_geoinfo_from_s2(tif_file, target_crs=CRS)
        
        # 2 - fetch data using wms
        img_array = fetch_data_from_wms(bbox, region_size, 
                                        style='default', format='image/GeoTIFF')
        
        # 3 - convert RGB to single band label image
        lbl_array = convert_rgb_to_single_band_label(img_array, color_mapping)
        
        # 4 - save as geotiff
        out_path = os.path.join(fp_mso,f'osm_{fd[3:]}.tif')
        lbl_array = np.flip(lbl_array,axis=0)
        # before saving, apply erosion and dilation operators to smooth the mask
        lbl_array = erosion_dilation_scikit(lbl_array, len(color_mapping), kernel_size=3)
        save_geotiff(lbl_array, bbox, out_path)