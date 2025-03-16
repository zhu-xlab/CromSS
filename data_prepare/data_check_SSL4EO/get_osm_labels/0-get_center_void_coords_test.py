# -*- coding: utf-8 -*-
"""
Get coordinate info from each test patch and save to csv file
including: 
    - basic info: dw_id, crs, date
    - center void locations
    - original bounding box and the transformed one for fetching data from wms
    - region size
"""

import os
import rasterio
import csv
import pandas as pd
import geopandas as gpd

from tqdm import tqdm
from rasterio.crs import CRS
from pyproj import Transformer
from shapely.geometry import Point

pdir = r'C:\liuchy\Research\Projects\20230412_NL_Pretrain\Codes\data_prepare\toy_data_SSL4EO\data_test'
fdn_dw = 'dw_release_labels'
fn_dw_meta = 'meta.csv'
fcoords = os.path.join(pdir, 'center_voids_test.csv')

# get file names
fd_dw = os.path.join(pdir, fdn_dw)
fnames = os.listdir(fd_dw)

# read meta data
meta = pd.read_csv(os.path.join(pdir,fn_dw_meta), header=0)

locs = []
with open(fcoords,'w',newline='') as fc:
    writer = csv.writer(fc)
    header = ['dw_id', 'crs', 'date',
              'center_lon', 'center_lat',
              'bound1', 'bound2', 'bound3', 'bound4',
              'height', 'width', 
              'bbox1', 'bbox2', 'bbox3', 'bbox4']
    writer.writerow(header)  
    for f in tqdm(fnames):
        if f[-3:] == 'tif':
            did = f[6:-4]
            tif_file = os.path.join(fd_dw, f)
            sys_str = meta[meta['filename']==did]['s2_system_index'].item()
            date = sys_str[:8]
            
            with rasterio.open(tif_file) as dataset:
                # # getting center void locations
                transformer = Transformer.from_crs(dataset.crs, CRS.from_epsg(4326))
                row, col = dataset.height // 2, dataset.width // 2
                x, y = dataset.xy(row, col) 
                lat, lon = transformer.transform(x, y)
                locs.append([lon,lat])
                
                # # getting bounding box for fetching data from wms
                transformer2 = Transformer.from_crs(dataset.crs, CRS.from_epsg(3857))
                h, w = dataset.height, dataset.width
                left, bottom, right, top = dataset.bounds
                # left bottom corner
                x1, y1 = transformer2.transform(left, bottom)
                # right top corner
                x2, y2 = transformer2.transform(right, top)
                # output 1 - region size
                region_size = (h, w)
                # output 2 - bounding box
                bbox = (x1, y1, x2, y2)
                
                data = [did, dataset.crs.to_epsg(), date, lon, lat] + \
                    list(dataset.bounds) + list(region_size) + list(bbox)
                
                writer.writerow(data)  
                    
# plot
points =  [Point((lon,lat)) for lon, lat in locs]
df = {'geometry': points}
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
# Create a GeoDataFrame with the world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# plot all the points
ax1 = gdf.plot(ax=world.plot(figsize=(20, 12)), marker='o', color='red', markersize=2)
ax1.axis('off')
                    
                
    