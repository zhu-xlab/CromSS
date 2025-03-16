# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 09:44:23 2023

@author: liu_ch
"""

import os
import shutil
import pandas as pd

MOVE=False

pdir = r'C:\liuchy\Research\Projects\20230412_NL_Pretrain\Codes\data_prepare\toy_data_SSL4EO\annotated\dw_train_osm'
tdir = r'C:\liuchy\Research\Projects\20230412_NL_Pretrain\Codes\data_prepare\data_check_SSL4EO\dw_GT\dw_train_osm'

subs1 = os.listdir(pdir)

fnew = []
for sub1 in subs1:
    pd1 = os.path.join(pdir, sub1)
    subs2 = os.listdir(pd1)
    for sub2 in subs2:
        pd2 = os.path.join(pdir, sub1, sub2)
        fnames = os.listdir(pd2)
        for f in fnames:
            fnew.append('dw_'+f[4:-4])
            if MOVE:
                file_path = os.path.join(pd2,f)
                destination_directory = os.path.join(tdir,sub1,sub2)
                shutil.move(file_path, destination_directory)
        
pd_new = pd.DataFrame(fnew,columns=['dw_id'])

pfail = r'C:\liuchy\Research\Projects\20230412_NL_Pretrain\Codes\data_prepare\toy_data_SSL4EO\annotated\empty_osm_files.xlsx'
pd_fail = pd.read_excel(pfail,names=['dw_id'],header=None)
# pd_emp = pd.merge(pd_fail,pd_new, how='outer',on='dw_id')

# Merge with indicator to get the difference
merged_df = pd.merge(pd_fail, pd_new, on=['dw_id'], how='outer', indicator=True)

# Filter rows where the '_merge' column is set to 'left_only' or 'right_only'
difference_df = merged_df[merged_df['_merge'].isin(['left_only'])]
# difference_df2 = merged_df[merged_df['_merge'].isin(['right_only'])]
pemp = r'C:\liuchy\Research\Projects\20230412_NL_Pretrain\Codes\data_prepare\toy_data_SSL4EO\annotated\empty_osm_files_new.xlsx'
difference_df.drop('_merge',axis=1,inplace=True)
difference_df.to_excel(pemp, index=False, header=False)


# # # # check number of files after moving
n_files = 0
for sub1 in subs1:
    pd1 = os.path.join(pdir, sub1)
    subs2 = os.listdir(pd1)
    for sub2 in subs2:
        pd2 = os.path.join(pdir, sub1, sub2)
        fnames = os.listdir(pd2)
        n_files += len(fnames)
print(f'Finally get {n_files} from osm database!')
        