# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:22:30 2023

@author: liu_ch
"""
import os
import pandas as pd

# Replace 'data_statistics.xlsx' with the path to your original Excel file
file_path = r'C:\liuchy\Research\Projects\20230412_NL_Pretrain\Codes\data_prepare\data_check_SSL4EO\dw_GT\data_statistics.xlsx'
sheet_name_train = 'train'

# Read the "train" sheet into a pandas DataFrame
df_train = pd.read_excel(file_path, sheet_name=sheet_name_train)

# Replace 'v1_dw_tile_metadata_for_public_release.xlsx' with the path to the file containing 'fnames'
fnames_file_path = r'C:\liuchy\Research\Projects\20230412_NL_Pretrain\Codes\data_prepare\toy_data_SSL4EO\annotated\meta\v1_dw_tile_metadata_for_public_release.xlsx'
sheet_name_expert = 'non_expert' #'expert' 

# Read the 'fnames' from the first column of the 'expert' sheet in 'fnames_file_path' Excel file (excluding the first row which is the column name)
fnames_df = pd.read_excel(fnames_file_path, sheet_name=sheet_name_expert, header=0)
fnames = fnames_df.iloc[:, 0].tolist()

# Create a new DataFrame containing rows where the first item is in fnames
filtered_df_train = df_train[df_train.iloc[:, 0].isin(fnames)]

# Filter the 'expert' DataFrame based on matching first item from 'train' DataFrame
filtered_df_expert = fnames_df[fnames_df.iloc[:, 0].isin(filtered_df_train.iloc[:, 0])]

# Replace 'data_statistics_expert.xlsx' with the path for the new Excel file
output_file_dir = r'C:\liuchy\Research\Projects\20230412_NL_Pretrain\Codes\data_prepare\data_check_SSL4EO\dw_GT'
output_file_path = os.path.join(output_file_dir, f'data_statistics_{sheet_name_train}_{sheet_name_expert}.xlsx')

# Save the filtered DataFrame to a new Excel file with the header
filtered_df_expert.to_excel(output_file_path, index=False, header=True)

print(f"Filtered data from '{sheet_name_train}' sheet saved to 'data_statistics_{sheet_name_train}_{sheet_name_expert}.xlsx'.")

