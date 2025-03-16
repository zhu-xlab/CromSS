'''
script for fetching sentinel-2 data from earth engine for the training set
each image is separately saved as tif
Test (discarded) version: cropping images based on POI and buffer size
                          severe misalignment between images and labels exists
'''
import os
import ee
import h5py
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
TYPE = 'Expert'
BANDS = 13
SAVE_TYPE = 'tif' # 'tif' or 'h5'


def separate_df(df):
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
    

def get_single_s2_data_from_ee(ds, save_dir):
    # get buffer size for extracting region polygons later
    h, w = int(ds['height']), int(ds['width'])
    img_b = int(max(h//2,w//2))

    # get coordinates of center point
    x, y = ds['center_lon'], ds['center_lat']
    # define the center point in the DRC
    POI = ee.Geometry.Point([x, y])
    
    # get dw_id and extract data time
    base_name = ds['dw_id']
    date_str = base_name.split('-')[-1]
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
            image = clip_img_with_POI_and_buff(POI, img_b, formatted_date_string, formatted_date_string_after)
        except:
            # current day - 15 days
            image = clip_img_with_POI_and_buff(POI, img_b, formatted_date_string_before, formatted_date_string, after=False)
    except Exception as e:
        with open(os.path.join(save_dir,f"error_log_{ds['hemisphere']}_{ds['biome']}.txt"), 'a') as file:
            file.write(f"{base_name}: {e}")  # Write the error message to the file
        return
    else:
        # interpolat to target size
        if not image.shape[:2]==(h,w):
            image = resize(image, (h, w), anti_aliasing=True)
        return image.astype(np.uint16)


def get_s2_for_each_split_h5(args):
    ee.Initialize()
    t0 = datetime.now().replace(microsecond=0)
    df, save_dir = args
    hemi, bio = df.loc[0,['hemisphere', 'biome']]
    print(f'Start fetching data for {hemi}-{bio} split!')
    
    # target h5 file path
    fp = os.path.join(save_dir, f'dw_train_s2_{TYPE}_{hemi}_{bio}.h5')
    
    # get data properties
    h, w = df.loc[0,['height','width']]
    h, w = int(h), int(w)
    n_sam = df.shape[0]
    
    # get data from ee
    fail_ids = []
    with h5py.File(fp, 'w') as hf:
        hf.create_dataset('img', shape=(n_sam, h, w, BANDS), dtype='uint16')
        hf.create_dataset('dw_id', data=np.array(df.loc[:,'dw_id']))
        hf.create_dataset('indicator', data=np.zeros(n_sam), dtype='uint8')
        
        # get data from ee
        for i in range(n_sam):
            ds = df.loc[i]
            image = get_single_s2_data_from_ee(ds, save_dir)
            if type(image) is type(None):
                fail_ids.append(df.loc[i,'dw_id'])
            else:
                uni = np.unique(image)
                if len(uni)<13*9:
                    fail_ids.append(df.loc[i,'dw_id'])
                else:
                    hf['img'][i] = image
                    hf['indicator'][i] = 1
    
    # print fetching results
    t1 = datetime.now().replace(microsecond=0)
    print(f'{hemi}-{bio} data fetching finishes ({t1-t0}s)!')
    n_fails = len(fail_ids)
    print(f'{n_sam} in total, {n_sam-n_fails} succeed, {n_fails} fail!')
    
    return fail_ids


def save_geotiff(lbl_array, bbox, dst_crs, out_path):
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


def get_s2_for_each_split_tif(args):
    ee.Initialize()
    t0 = datetime.now().replace(microsecond=0)
    df, save_dir = args
    hemi, bio = df.loc[0,['hemisphere', 'biome']]
    print(f'Start fetching data for {hemi}-{bio} split!')
    
    # target h5 file path
    fpdir = os.path.join(save_dir, TYPE, hemi, str(bio))
    Path(fpdir).mkdir(parents=True, exist_ok=True)
    
    # get data properties
    h, w = df.loc[0,['height','width']]
    h, w = int(h), int(w)
    n_sam = df.shape[0]
    
    # get data from ee
    fail_ids = []
    for i in range(n_sam):
        ds = df.loc[i]
        image = get_single_s2_data_from_ee(ds, save_dir)
        if type(image) is type(None):
            fail_ids.append(df.loc[i,'dw_id'])
        else:
            uni = np.unique(image)
            if len(uni)<13*9:
                fail_ids.append(df.loc[i,'dw_id'])
            else:
                tif_path = os.path.join(fpdir, f's2_{df.loc[i,"dw_id"][3:]}.tif')
                bbox = [df.loc[i,f"bound{j}"] for j in range(1,5)]
                dst_crs = df.loc[i,"crs"]
                save_geotiff(image, bbox, dst_crs, tif_path)
    
    # print fetching results
    t1 = datetime.now().replace(microsecond=0)
    print(f'{hemi}-{bio} data fetching finishes ({t1-t0}s)!')
    n_fails = len(fail_ids)
    print(f'{n_sam} in total, {n_sam-n_fails} succeed, {n_fails} fail!')
    
    return fail_ids


if __name__ == '__main__':
    label_file_path = '/p/scratch/hai_dm4eo/liu_ch/test/data_v1/download_s2/center_voids_expert.csv'
    save_dir = '/p/scratch/hai_dm4eo/liu_ch/test/data_v2/download_s2'
    df_labels = pd.read_csv(label_file_path,header=0)
    
    # # Trigger the authentication flow.
    # ee.Authenticate()
    
    # 1 - separate df according to hemisphere and biome
    sub_dfs = separate_df(df_labels)
    n_sdfs = len(sub_dfs)
    
    # 2 - check whether the storage directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 3 - prepare data
    data = [(sub_dfs[k], save_dir) for k in sub_dfs]
        
    # 4 - get core number
    # get number of cpu cores
    n_CPU = multiprocessing.cpu_count()  
    num_processes = min(n_CPU, len(sub_dfs))
    
    # 5 - fetch data from ee
    get_s2_for_each_split = get_s2_for_each_split_h5 if SAVE_TYPE == 'h5' else get_s2_for_each_split_tif
    if num_processes>1:
        # Create a pool of worker processes
        print(f'{num_processes} cores out of {n_CPU} are used!')
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(get_s2_for_each_split, data)
        # record failed files
        fails = []
        for r in results: fails += r
    else:
        # single process
        print('No parallel processes are triggered!')
        fails = []
        for d in data:
            results = get_s2_for_each_split(d)
            fails += results
    
    # 6 - output failed file names
    if len(fails)>0:
        df_fails = pd.DataFrame(fails)
        df_fails.to_csv(os.path.join(save_dir, 'fnames_of_no_s2_data_fetched.csv'))
        print(f'{df_labels.shape[0]-len(df_fails)} data have been successfully fetched with {len(df_fails)} fialed!')
    else:
        print('All the data have been successfully fetched!')
    
    
    
    
    
    


