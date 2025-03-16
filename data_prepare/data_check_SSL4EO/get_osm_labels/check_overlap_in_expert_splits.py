# -*- coding: utf-8 -*-
"""
Expert and Non-expert splits have some patches in common
This script is to check the overlap between the two splits
"""
import os
import pandas as pd


# # # # # # for data paired with an image # # # # # #
pdir = r'C:\liuchy\Research\Projects\20230412_NL_Pretrain\Codes\data_prepare\data_check_SSL4EO\dw_GT'
pexp = os.path.join(pdir, 'data_statistics_train_expert.xlsx')
pnexp = os.path.join(pdir, 'data_statistics_train_non_expert.xlsx')

df_exp = pd.read_excel(pexp, header=0).iloc[:,0]
df_nexp = pd.read_excel(pnexp, header=0).iloc[:,0]

set1 = set(df_exp)
set2 = set(df_nexp)

# Step 2: Find the common elements using set.intersection()
common_elements = set1.intersection(set2)



# # # # # # for all the data # # # # # #
pdir = r'C:\liuchy\Research\Projects\20230412_NL_Pretrain\Codes\data_prepare\toy_data_SSL4EO\annotated\meta\v1_dw_tile_metadata_for_public_release.xlsx'

df_exp = pd.read_excel(pdir, sheet_name='expert', header=0).iloc[:,0]
df_nexp = pd.read_excel(pdir, sheet_name='non_expert', header=0).iloc[:,0]

set1 = set(df_exp)
set2 = set(df_nexp)

# Step 2: Find the common elements using set.intersection()
common_elements_all = set1.intersection(set2)