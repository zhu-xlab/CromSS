# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 08:40:43 2023

@author: liu_ch
"""

import os
import pandas as pd

import geopandas as gpd
from shapely.geometry import Point

pdir = r'C:\liuchy\Research\Projects\20230412_NL_Pretrain\Codes\data_prepare'

feall = os.path.join(pdir,r'toy_data_SSL4EO\annotated\center_voids_expert.csv')
fepart = os.path.join(pdir,r'data_check_SSL4EO\dw_GT\data_statistics_train_expert.xlsx')

eall = pd.read_csv(feall,header=0)
epart = pd.read_excel(fepart, header=0)
a = pd.merge(eall, epart, how='inner', on='dw_id')

# plot
points =  [Point((x,y)) for x, y in zip(a.loc[:,'center_lon'],a.loc[:, 'center_lat'])]
df = {'geometry': points}
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
# Create a GeoDataFrame with the world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# plot all the points
ax1 = gdf.plot(ax=world.plot(figsize=(20, 12)), marker='o', color='red', markersize=2)
ax1.axis('off')