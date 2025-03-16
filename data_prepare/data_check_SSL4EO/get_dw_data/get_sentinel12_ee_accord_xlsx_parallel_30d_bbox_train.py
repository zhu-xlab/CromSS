'''
script for fetching sentinel-2 data from earth engine for the training set
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
TYPE = 'Experts'


def separate_df(df):
    '''
    separate df into sub df based on 'hemisphere' and 'biome' columns
    for multiprocessing based on each region
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


def clip_img_with_bbox_s2(bbox, crs, t_start, t_end):
    '''
    clip s2 images on ee using bounding box and crs
    specifically, to further avoid co-registration problem, we double check the crs of fetched images
    only fetch from the images with the same crs as required
    in this case, the fetched images should be of exactly the same size as the target/DW-label images
    '''
    # s2 collection
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
    
    return image, s2_image.select('B2')
    

def get_single_s2_data_from_ee(ds):  
    # get bounding box and crs
    bbox = [ds[f'bound{b}'] for b in range(1,5)]
    crs = ds['crs']

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
            image, s2_example = clip_img_with_bbox_s2(bbox, crs, formatted_date_string, formatted_date_string_after)
        except:
            # current day - 15 days
            image, s2_example = clip_img_with_bbox_s2(bbox, crs, formatted_date_string_before, formatted_date_string)
        return image, s2_example
    except Exception as e:
        error_msg = f"S2 Pulling error: {e}"
        # with open(os.path.join(save_dir,f"error_log_{ds['hemisphere']}_{ds['biome']}.txt"), 'a') as file:
        #     file.write(f"{base_name}: {e}")  # Write the error message to the file
        return error_msg
    

def clip_img_with_bbox_s1(bbox, crs, t_start, t_end, s2_example, after):
    '''
    clip s2 images on ee using bounding box and crs
    specifically, to further avoid co-registration problem, we double check the crs of fetched images
    only fetch from the images with the same crs as required
    in this case, the fetched images should be of exactly the same size as the target/DW-label images
    '''
    # s1 collection
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    
    # Filter the collection to specific dates and region
    boundbox = ee.Geometry.Rectangle(bbox, proj=f'EPSG:{crs}',evenOdd=False)
    s1 = s1.filterBounds(boundbox).filterDate(t_start, t_end)

    # Filter the content
    s1 = s1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    s1 = s1.filter(ee.Filter.eq('instrumentMode', 'IW'))
    
    n_imgs = s1.toList(s1.size()).length().getInfo()
    if n_imgs>0:
        if after:
            s1_image = s1.sort('system:time_start', False).first()  # get most recent
        else:
            # Grab the image one by one in the filtered collection
            s1_images = s1.toList(s1.size())
            for j in range(n_imgs):
                s1_image = ee.Image(s1_images.get(j))
                vv = s1_image.select('VV')
                dst_crs = vv.projection().getInfo()["crs"]
                if dst_crs.split(':')[-1]==str(crs):
                    break
                if j == n_imgs-1:
                    # raise ValueError("CRS info is inconsistent!")
                    s1_image = ee.Image(s1_images.get(0))
    else:
        raise ValueError("No data fetched!")
    
    # select 
    s1_image = s1_image.select('VV', 'VH')
    
    # resample s2 to s1 size
    s1_image = s1_image.toFloat().resample('bilinear').reproject(
        s2_example.projection())

    # This creates an ee.Feature with a property named "array" that we'll grab later.
    s1_image_sample = s1_image.toArray().sampleRectangle(boundbox)

    # get data array
    image = np.array(s1_image_sample.getInfo()['properties']['array'])
    
    return image


def get_single_s1_data_from_ee(ds, s2_example):  
    # get bounding box and crs
    bbox = [ds[f'bound{b}'] for b in range(1,5)]
    crs = ds['crs']

    # get dw_id and extract data time
    base_name = ds['dw_id']
    date_str = base_name.split('-')[-1]
    # Parse the original date string to a datetime object
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    date_obj_after = date_obj + timedelta(days=TIME_BUFF)
    date_obj_before = date_obj + timedelta(days=-TIME_BUFF)
    # Format the datetime object to the desired format
    # formatted_date_string = date_obj.strftime("%Y-%m-%d")
    formatted_date_string = s2_example.date()
    formatted_date_string_after = date_obj_after.strftime("%Y-%m-%d")
    formatted_date_string_before = date_obj_before.strftime("%Y-%m-%d")
    
    # fetch data
    try:
        try:
            # current day + 15 days
            image = clip_img_with_bbox_s1(bbox, crs, formatted_date_string, formatted_date_string_after, s2_example, after=True)
        except:
            # current day - 15 days
            image = clip_img_with_bbox_s1(bbox, crs, formatted_date_string_before, formatted_date_string, s2_example, after=False)
        return image
    except Exception as e:
        error_msg = f"Pulling error: {e}"
        # with open(os.path.join(save_dir,f"error_log_{ds['hemisphere']}_{ds['biome']}.txt"), 'a') as file:
        #     file.write(f"{base_name}: {e}")  # Write the error message to the file
        return error_msg


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


def get_s12_for_each_split_tif(args):
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
    
    # get s12 from ee
    fail_ids = []
    rshp_ids = []
    rshp_ids_s1 = []
    err_msgs = []
    for i in range(n_sam):
        success = True
        ds = df.loc[i]
        did = df.loc[i,'dw_id']
        # get s2 from ee
        out_s2 = get_single_s2_data_from_ee(ds)
        try:
            s2_image, s2_example = out_s2
            uni = np.unique(s2_image)
            if len(uni)<13*9:
                fail_ids.append([hemi, bio, did])
                err_msg = 'S2: Too many invalid values'
                success = False
            else:
                # interpolat to target size
                if not s2_image.shape[:2]==(h,w):
                    err_msg = f'S2: Resize image from {s2_image.shape[:2]} to {(h,w)}'
                    s2_image = resize(s2_image, (h, w), anti_aliasing=True)
                    rshp_ids.append([hemi, bio, did])
                else:
                    err_msg = 0
        except:       
            if type(out_s2) is str:
                fail_ids.append([hemi, bio, did])
                err_msg = out_s2
                success = False
            else:
                print(f"{hemi}-{bio}-{did}:unknown error with {type(out_s2)}-{out_s2}")
                err_msg = 'S2: unknown error!'
                success = False
            
        # get s1 from ee
        if success:
            s1_image = get_single_s1_data_from_ee(ds, s2_example)
            if type(s1_image) is str:
                fail_ids.append([hemi, bio, did])
                err_msg = s1_image
                success = False
            else:
                uni = np.unique(s1_image)
                if len(uni)<2*9:
                    fail_ids.append([hemi, bio, did])
                    err_msg = 'S1: Too many invalid values'
                    success = False
                else:
                    # interpolat to target size
                    if not s1_image.shape[:2]==(h,w):
                        err_msg = f'S1: Resize image from {s1_image.shape[:2]} to {(h,w)}'
                        s1_image = resize(s1_image, (h, w), anti_aliasing=True)
                        rshp_ids_s1.append([hemi, bio, did])
                    else:
                        err_msg = 0
        
        # save images
        if success:
            for img, tag, it in zip((s1_image, s2_image), ('s1','s2'), (np.float32, np.int16)):
                tif_path = os.path.join(fpdir, f'{tag}_{did[3:]}.tif')
                bbox = [ds[f"bound{j}"] for j in range(1,5)]
                dst_crs = ds["crs"]
                save_geotiff(img.astype(it), bbox, dst_crs, tif_path)
        
        # record error messages
        err_msgs.append(err_msg)
    
    # output failed file names
    outputs = [[did, err] for (did, err) in zip(df['dw_id'], err_msgs) if err!=0]
    if len(outputs)>0:
        df_outs = pd.DataFrame(outputs, columns=['dw_id', 'error_message'])
        df_outs = df_outs.sort_values(by=['error_message'])
        df_outs.to_csv(os.path.join(save_dir, 'Error_messages_for_s2_data_pulling.csv'), index=False)
        print(f'{df_outs.shape[0]} files have error message.')
    else:
        print('All the data have been successfully fetched!')
        
    # print fetching results
    t1 = datetime.now().replace(microsecond=0)
    print(f'{hemi}-{bio} data fetching finishes ({t1-t0}s)!')
    n_fails = len(fail_ids)
    n_rshp = len(rshp_ids)
    print(f'{n_sam} in total, {n_sam-n_fails} succeed ({n_rshp} reshaped), {n_fails} fail!')
    
    return {'fail':fail_ids, 'reshape':rshp_ids}


if __name__ == '__main__':
    label_file_path = '/p/project/hai_ws4eo/liu_ch/DW/test/data_v2/center_voids_expert.csv'
    save_dir = '/p/project/hai_ws4eo/liu_ch/DW/test/data_v2/download_s12'
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
    get_s12_for_each_split = get_s12_for_each_split_tif
    fails = []
    rshps = []
    if num_processes>1:
        # Create a pool of worker processes
        print(f'{num_processes} cores out of {n_CPU} are used!')
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(get_s12_for_each_split, data)
        # record failed files
        for r in results: 
            fails += r['fail']
            rshps += r['reshape']
    else:
        # single process
        print('No parallel processes are triggered!')
        for d in data:
            results = get_s12_for_each_split(d)
            fails += results['fail']
            rshps += results['reshape']
    
    # 6 - output failed file names
    if len(rshps)>0:
        df_rshps = pd.DataFrame(rshps, columns=['hemisphere', 'biome', 'dw_id'])
        df_rshps.to_csv(os.path.join(save_dir, 'fnames_of_reshaped_s2_data_fetched.csv'), index=False, header=0)
        print('Total {len(df_rshps)} were reshaped!')
    if len(fails)>0:
        df_fails = pd.DataFrame(fails, columns=['hemisphere', 'biome', 'dw_id'])
        df_fails.to_csv(os.path.join(save_dir, 'fnames_of_no_s2_data_fetched.csv'), index=False)
        print(f'{df_labels.shape[0]-len(df_fails)} data have been successfully fetched with {len(df_fails)} fialed!')
    else:
        print(f'All the data have been successfully fetched ({len(df_rshps)} reshaped)!')
    
    
    
    
    
    


