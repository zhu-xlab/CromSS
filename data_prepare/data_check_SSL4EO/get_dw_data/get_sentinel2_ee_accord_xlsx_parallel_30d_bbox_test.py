'''
script for fetching sentinel-2 data from earth engine for the test set
each image is separately saved as tif
Final used version: cropping images based on bounding box & checking crs consistency
'''
import os
import ee
import rasterio
import numpy as np
import pandas as pd
import multiprocessing

from skimage.transform import resize
from datetime import datetime, timedelta
from pathlib import Path

from rasterio.crs import CRS
from rasterio.transform import from_origin


TIME_BUFF = 15
CLOUD = 20


def separate_df(df):
    '''
    separate df into sub df based on 'hemisphere' and 'biome' columns
    for multiprocessing based on each region
    defined for fetching training data
    [UNUSED] in this srcipt
    '''
    sub_dataframes = {}
    
    # Group the DataFrame by the 'hemisphere' column
    group_hemi = df.groupby('hemisphere')

    # in each sub df, further group by 'biome'
    for hemi, ghemi in group_hemi:
        group_bio = ghemi.groupby('biome')
        for biome, group in group_bio:
            group.reset_index(drop=True, inplace=True)
            sub_dataframes[f'{hemi}_{biome}'] = group
    return sub_dataframes


def clip_img_with_POI_and_buff(POI, img_buff, t_start, t_end, after=True):
    '''
    clip s2 images on ee using center points as POIs and corresponding buffer size
    based on the official function given by dw team
    [UNUSED] in this srcipt
    '''
    # s2 collection
    s2 = ee.ImageCollection('COPERNICUS/S2')
    
    # Filter the collection to mid-October 2021 intersecting the Point Of Interest.
    s2 = s2.filterBounds(POI).filterDate(t_start, t_end)

    # Fileter cloud coverage
    s2 = s2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD))

    # Grab the first image in the filtered collection. Dynamic World uses a subset
    # of Sentinel-2 bands, so we'll want to select down to just those.
    s2_image = s2.first() if after else s2.median()
    s2_image = s2_image.select('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7','B8', 'B8A',
                               'B9', 'B10', 'B11', 'B12')

    # Resample the data so that the bands all map 1 pixel -> 10m. We'll use B2 (red)
    # for a reference projection.
    s2_image = s2_image.toFloat().resample('bilinear').reproject(
        s2_image.select('B2').projection());

    # This creates an ee.Feature with a property named "array" that we'll grab later.
    s2_image_sample = s2_image.toArray().sampleRectangle(POI.buffer(img_buff*10))

    # get data array
    image = np.array(s2_image_sample.getInfo()['properties']['array'])
    # Note this shape isn't exactly 400 a side (2 * 2km of 10m pixels) since the
    # "buffer" we used earlier was in a different (geographic) projection than the pixels.

    return image


def clip_img_with_bbox(bbox, crs, t_start, t_end, download_l2a=False):
    '''
    clip s2 images on ee using bounding box and crs
    specifically, to further avoid co-registration problem, we double check the crs of fetched images
    only fetch from the images with the same crs as required
    in this case, the fetched images should be of exactly the same size as the target/DW-label images
    '''
    # s2 collection
    if download_l2a:
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') # ('COPERNICUS/S2_SR')
    else:
        s2 = ee.ImageCollection('COPERNICUS/S2')
    
    # Filter the collection to specific dates and region
    boundbox = ee.Geometry.Rectangle(bbox, proj=f'EPSG:{crs}',evenOdd=False)
    s2 = s2.filterBounds(boundbox).filterDate(t_start, t_end)

    # Fileter cloud coverage
    s2 = s2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD))

    # Grab the image one by one in the filtered collection
    s2_images = s2.toList(s2.size())
    n_imgs = s2_images.length().getInfo()
    for j in range(n_imgs):
        s2_image = ee.Image(s2_images.get(j))
        b2 = s2_image.select('B2')
        dst_crs = b2.projection().getInfo()["crs"]
        if dst_crs.split(':')[-1]==str(crs):
            break
        if j == n_imgs-1:
            raise ValueError("CRS info is inconsistent!")
    
    # select 
    if download_l2a:
        s2_image = s2_image.select('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 
                                   'B8A', 'B9', 'B11', 'B12')
    else:
        s2_image = s2_image.select('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 
                                   'B8A', 'B9', 'B10', 'B11', 'B12')

    # Resample the data so that the bands all map 1 pixel -> 10m. We'll use B2 (red)
    # for a reference projection.
    s2_image = s2_image.toFloat().resample('bilinear').reproject(
        s2_image.select('B2').projection());

    # This creates an ee.Feature with a property named "array" that we'll grab later.
    s2_image_sample = s2_image.toArray().sampleRectangle(boundbox)

    # get data array
    image = np.array(s2_image_sample.getInfo()['properties']['array'])
    
    return image
    

def get_single_s2_data_from_ee(ds, download_l2a=False):  
    '''
    get s2 data from ee
    '''
    # get bounding box and crs
    bbox = [ds[f'bound{b}'] for b in range(1,5)]
    crs = ds['crs']

    # get extract data time
    date_str = str(ds['date'])
    # Parse the original date string to a datetime object
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    date_obj_after = date_obj + timedelta(days=TIME_BUFF)
    date_obj_before = date_obj + timedelta(days=-TIME_BUFF)
    # Format the datetime object to the desired format
    formatted_date_string = date_obj.strftime("%Y-%m-%d")
    formatted_date_string_after = date_obj_after.strftime("%Y-%m-%d")
    formatted_date_string_before = date_obj_before.strftime("%Y-%m-%d")
    
    # fetch data
    try:
        try:
            # current day + 15 days
            image = clip_img_with_bbox(bbox, crs, formatted_date_string, formatted_date_string_after, download_l2a)
        except:
            # current day - 15 days
            image = clip_img_with_bbox(bbox, crs, formatted_date_string_before, formatted_date_string, download_l2a)
        return image
    except Exception as e:
        error_msg = f"Pulling error: {e}"
        # with open(os.path.join(save_dir,f"error_log_{ds['hemisphere']}_{ds['biome']}.txt"), 'a') as file:
        #     file.write(f"{base_name}: {e}")  # Write the error message to the file
        return error_msg


def save_geotiff(lbl_array, bbox, dst_crs, out_path):
    '''
    save fetched s2 data to geotiff
    '''
    # Define the pixel size based on the bounding box and image dimensions
    pixel_width = (bbox[2] - bbox[0]) / lbl_array.shape[1]
    pixel_height = (bbox[3] - bbox[1]) / lbl_array.shape[0]

    # Calculate the transform matrix for georeferencing: given left top and pixel sizes.
    transform = from_origin(bbox[0], bbox[-1], pixel_width, pixel_height)

    # Create the GeoTIFF file and write the data
    tiff_params = {"driver":   'GTiff',
                   "height":    lbl_array.shape[0],
                   "width":     lbl_array.shape[1],
                   "count":     lbl_array.shape[2],  # Single band image
                   "dtype":     lbl_array.dtype,
                   "crs":       CRS.from_epsg(dst_crs),  # Use the specified CRS
                   "transform": transform}
    with rasterio.open(out_path,'w', **tiff_params) as dst:
        for i in range(lbl_array.shape[2]):
            dst.write(lbl_array[:,:,i], i+1)
    return


def get_s2_for_each_tif(args):
    '''
    fetch s2 data for each tif
    basic function for multiprocessing
    '''
    # parameters
    ds, save_dir, download_l2a = args
    
    # get data properties
    h, w = ds[['height','width']]
    h, w = int(h), int(w)
    did = ds['dw_id']
    
    # get image from ee
    image = get_single_s2_data_from_ee(ds, download_l2a)
    if type(image) is str:
        err_msg = image
    else:
        uni = np.unique(image)
        if len(uni)<13*9:
            err_msg = 'Too many invalid values'
        else:
            # interpolat to target size
            if not image.shape[:2]==(h,w):
                err_msg = f'Resize image from {image.shape[:2]} to {(h,w)}'
                image = resize(image, (h, w), anti_aliasing=True)
            else:
                err_msg = 0
            # save image
            if download_l2a:
                tif_path = os.path.join(save_dir, f's2a_{did[3:]}.tif')
            else:
                tif_path = os.path.join(save_dir, f's2_{did[3:]}.tif')    # change to s2c?
            bbox = [ds[f"bound{j}"] for j in range(1,5)]
            dst_crs = ds["crs"]
            save_geotiff(image.astype(np.uint16), bbox, dst_crs, tif_path)
    
    return {'dw_id':did, 'err':err_msg}


if __name__ == '__main__':
    label_file_path = '/p/project1/hai_ws4eo/liu_ch/DW/test/data_test/center_voids_test.csv'
    save_dir = '/p/project1/hai_ws4eo/liu_ch/DW/test/data_test/download_s2a_harmonized'
    # label_file_path = r'C:\liuchy\Research\Projects\20230412_NL_Pretrain\Codes\data_prepare\toy_data_SSL4EO\data_test\center_voids_test.csv'
    # save_dir = r'C:\liuchy\Research\Projects\20230412_NL_Pretrain\Codes\data_prepare\toy_data_SSL4EO\data_test\download_s12'
    df_labels = pd.read_csv(label_file_path,header=0)
    n_core = 1
    download_l2a = True
    
    # # Trigger the authentication flow.
    # ee.Authenticate()
    
    # initialization of ee
    ee.Initialize()
    
    # 1 - check whether the storage directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 2 - prepare data
    df = pd.read_csv(label_file_path, header=0)
    data = [(df.loc[i], save_dir, download_l2a) for i in df.index]
        
    # 3 - get core number
    # get number of cpu cores
    n_CPU = multiprocessing.cpu_count()  
    num_processes = n_core if n_core>0 else n_CPU
    
    # 4 - fetch data from ee
    t0 = datetime.now().replace(microsecond=0)
    get_s2_for_each_split = get_s2_for_each_tif
    err_msgs = []
    dw_ids = []
    if num_processes>1:
        # Create a pool of worker processes
        print(f'{num_processes} cores out of {n_CPU} are used!')
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(get_s2_for_each_split, data)
        # record failed files
        for r in results: 
            dw_ids.append(r['dw_id'])
            err_msgs.append(r['err'])
    else:
        # single process
        print('No parallel processes are triggered!')
        for d in data:
            results = get_s2_for_each_split(d)
            dw_ids.append(results['dw_id'])
            err_msgs.append(results['err'])
    
    # 5 - output failed file names
    outputs = [[did, err] for (did, err) in zip(dw_ids, err_msgs) if err!=0]
    if len(outputs)>0:
        df_outs = pd.DataFrame(outputs, columns=['dw_id', 'error_message'])
        df_outs = df_outs.sort_values(by=['error_message'])
        df_outs.to_csv(os.path.join(save_dir, 'Error_messages_for_s2_data_pulling.csv'), index=False)
        print(f'{df_outs.shape[0]} files have error message.')
    else:
        print('All the data have been successfully fetched!')
        
    t1 = datetime.now().replace(microsecond=0)
    print(f'Data pulling is finished using {t1-t0}!')
    
    
    
    
    
    


