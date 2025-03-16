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

from dateutil.relativedelta import relativedelta

TIME_BUFF = 60
CLOUD = 50
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
    
    # # get bounding box
    # coords = s2_image_sample.geometry().bounds().getInfo()['coordinates']
    # coords = np.array(coords).squeeze()
    # left, right = np.unique(coords[:,0])
    # bottom, top = np.unique(coords[:,1])
    # bbox = [left, bottom, right, top]
    # return image, bbox
    return image


def clip_img_with_bbox(bbox, crs, t_start, t_end, l2a=False):
    '''
    clip s2 images on ee using bounding box and crs
    specifically, to further avoid co-registration problem, we double check the crs of fetched images
    only fetch from the images with the same crs as required
    in this case, the fetched images should be of exactly the same size as the target/DW-label images
    '''
    # s2 collection
    if l2a:
        s2 = ee.ImageCollection('COPERNICUS/S2_SR')
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
    get_j = 0
    cloud_coverage = 100
    for j in range(n_imgs):
        s2_image = ee.Image(s2_images.get(j))
        b2 = s2_image.select('B2')
        dst_crs = b2.projection().getInfo()["crs"]
        # if dst_crs.split(':')[-1]==str(crs):
        try:
            s2_image_sample = s2_image.toArray().sampleRectangle(region=boundbox, defaultValue=0)
            c_ = s2_image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
            if c_<cloud_coverage:
                cloud_coverage = c_
                get_j = j
        except:
            continue
            # break
        # if j == n_imgs-1:
        #     raise ValueError("CRS info is inconsistent!")
    s2_image = ee.Image(s2_images.get(get_j))
    
    # select 
    if l2a:
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
    
    return image, cloud_coverage
    

def get_single_s2_data_from_ee(ds, save_dir, download_l2a=False):  
    # get bounding box and crs
    bbox = [ds[f'bound{b}'] for b in range(1,5)]
    crs = ds['crs']

    # get dw_id and extract data time
    base_name = ds['dw_id']
    date_str = base_name.split('-')[-1]
    # Parse the original date string to a datetime object
    date_obj = datetime.strptime(date_str, "%Y%m%d") # + relativedelta(years=1)
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
            image, cloud_cpverage = clip_img_with_bbox(bbox, crs, formatted_date_string, formatted_date_string_after, l2a=download_l2a)
        except:
            # current day - 15 days
            image, cloud_cpverage = clip_img_with_bbox(bbox, crs, formatted_date_string_before, formatted_date_string, l2a=download_l2a)
        return (image, cloud_cpverage)
    except Exception as e:
        with open(os.path.join(save_dir,f"error_log_{ds['hemisphere']}_{ds['biome']}.txt"), 'a') as file:
            file.write(f"{base_name}: {e}")  # Write the error message to the file
        return


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
    df, save_dir, download_l2a = args
    hemi, bio = df.loc[0,['hemisphere', 'biome']]
    print(f'Start fetching data for {hemi}-{bio} split!')
    
    # target h5 file path
    fpdir = save_dir # os.path.join(save_dir, TYPE, hemi, str(bio))
    Path(fpdir).mkdir(parents=True, exist_ok=True)
    
    # get data properties
    h, w = df.loc[0,['height','width']]
    h, w = int(h), int(w)
    n_sam = df.shape[0]
    
    # get data from ee
    fail_ids = []
    rshp_ids = []
    for i in range(n_sam):
        ds = df.loc[i]
        did = df.loc[i,'dw_id']
        image_ = get_single_s2_data_from_ee(ds, save_dir, download_l2a)
        if type(image_) is type(None):
            fail_ids.append([hemi, bio, did])
        else:
            image, cloud_coverage = image_
            print(f'{did}:', cloud_coverage)
            uni = np.unique(image)
            if len(uni)<13*9:
                fail_ids.append([hemi, bio, did])
            else:
                # interpolat to target size
                if not image.shape[:2]==(h,w):
                    image = resize(image, (h, w), anti_aliasing=True)
                    rshp_ids.append([hemi, bio, did])
                # save image
                if download_l2a:
                    tif_path = os.path.join(fpdir, f's2a_{df.loc[i,"dw_id"][3:]}.tif')
                else:
                    tif_path = os.path.join(fpdir, f's2_{df.loc[i,"dw_id"][3:]}.tif')  # change to s2c?
                bbox = [df.loc[i,f"bound{j}"] for j in range(1,5)]
                dst_crs = df.loc[i,"crs"]
                save_geotiff(image.astype(np.uint16), bbox, dst_crs, tif_path)
    
    # print fetching results
    t1 = datetime.now().replace(microsecond=0)
    print(f'{hemi}-{bio} data fetching finishes ({t1-t0}s)!')
    n_fails = len(fail_ids)
    n_rshp = len(rshp_ids)
    print(f'{n_sam} in total, {n_sam-n_fails} succeed ({n_rshp} reshaped), {n_fails} fail!')
    
    return {'fail':fail_ids, 'reshape':rshp_ids}


if __name__ == '__main__':
    label_file_path = '/p/project1/hai_ws4eo/liu_ch/DW/test/data_v2/center_voids_expert_s2a_lack_v2.csv'
    # r'C:\liuchy\Research\Projects\20230412_NL_Pretrain\Codes\data_prepare\toy_data_SSL4EO\annotated\center_voids_expert_v1.csv'
    # '/p/scratch/hai_dm4eo/liu_ch/test/data_v1/download_s2/center_voids_expert.csv'
    save_dir = '/p/project1/hai_ws4eo/liu_ch/DW/test/data_v2/s2a_lack'
    # r'C:\liuchy\Research\Projects\20230412_NL_Pretrain\Codes\data_prepare'
    # '/p/scratch/hai_dm4eo/liu_ch/test/data_v2/download_s2'
    df_labels = pd.read_csv(label_file_path,header=0)
    download_l2a = True
    
    # # Trigger the authentication flow.
    # ee.Authenticate()
    
    # 1 - separate df according to hemisphere and biome
    sub_dfs = separate_df(df_labels)
    n_sdfs = len(sub_dfs)
    
    # 2 - check whether the storage directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 3 - prepare data
    data = [(sub_dfs[k], save_dir, download_l2a) for k in sub_dfs]
        
    # 4 - get core number
    # get number of cpu cores
    n_CPU = multiprocessing.cpu_count()  
    num_processes = min(n_CPU, len(sub_dfs))
    
    # 5 - fetch data from ee
    get_s2_for_each_split = get_s2_for_each_split_tif
    fails = []
    rshps = []
    if num_processes>1:
        # Create a pool of worker processes
        print(f'{num_processes} cores out of {n_CPU} are used!')
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(get_s2_for_each_split, data)
        # record failed files
        for r in results: 
            fails += r['fail']
            rshps += r['reshape']
    else:
        # single process
        print('No parallel processes are triggered!')
        for d in data:
            results = get_s2_for_each_split(d)
            fails += results['fail']
            rshps += results['reshape']
    
    # 6 - output failed file names
    if len(rshps)>0:
        df_rshps = pd.DataFrame(rshps, columns=['hemisphere', 'biome', 'dw_id'])
        df_rshps.to_csv(os.path.join(save_dir, 'fnames_of_reshaped_s2_data_fetched.csv'), index=False, header=0)
    if len(fails)>0:
        df_fails = pd.DataFrame(fails, columns=['hemisphere', 'biome', 'dw_id'])
        df_fails.to_csv(os.path.join(save_dir, 'fnames_of_no_s2_data_fetched.csv'), index=False)
        print(f'{df_labels.shape[0]-len(df_fails)} data have been successfully fetched ({len(rshps)} reshaped) with {len(df_fails)} fialed!')
    else:
        print(f'All the data have been successfully fetched ({len(df_rshps)} reshaped)!')
    
    
    
    
    
    


