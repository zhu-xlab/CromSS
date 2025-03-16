# -*- coding: utf-8 -*-
"""
Get coordinate info from each training patch and save to csv file
including: 
    - basic info: dw_id, hemisphere, biome, crs
    - center void locations
    - original bounding box and the transformed one for fetching data from wms
    - region size
"""

import os
import rasterio
import csv
import geopandas as gpd

from tqdm import tqdm
from rasterio.crs import CRS
from pyproj import Transformer
from shapely.geometry import Point

split = 'expert' # 'non_expert'

if split=='expert':
    directory = r'D:\Datasets\DW\dw_original_labels\train_set\Experts' # r'D:\Datasets\DW\Non_expert\WorkForce'
elif split=='non_expert':
    directory = r'D:\Datasets\DW\Non_expert\WorkForce'
else:
    raise ValueError("Please provide correct split name!")

subdir = ['EH', 'WH']
pdir = r'C:\liuchy\Research\Projects\20230412_NL_Pretrain\Codes\data_prepare\toy_data_SSL4EO\annotated'
fcoords = os.path.join(pdir, f'center_voids_{split}.csv')

locs = []
with open(fcoords,'w',newline='') as fc:
    writer = csv.writer(fc)
    header = ['dw_id', 'hemisphere', 'biome', 'crs',
              'center_lon','center_lat','bound1','bound2','bound3','bound4',
              'height', 'width', 'bbox1', 'bbox2', 'bbox3', 'bbox4']
    writer.writerow(header)  
    for sd in tqdm(subdir):
        sdir = os.path.join(directory, sd)
        reg_dirs = os.listdir(sdir)
        for rd in reg_dirs:
            fnames = os.listdir(os.path.join(sdir, rd))
            for f in fnames:
                if f[-3:] == 'tif':
                    tif_file = os.path.join(sdir, rd, f)
                    
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
                        
                        data = [f[:-4], sd, rd, dataset.crs.to_epsg(), lon, lat] + list(dataset.bounds) + list(region_size) + list(bbox)
                        
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
                    
                
    