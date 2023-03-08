#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 09:46:22 2023

@author: jeremias
"""

import pandas as pd
import os


rootdir='/home/jeremias/Documentos/PruebaIndicadores_BOW/lsd_dataset/validation/'

first=True
for file in os.listdir(rootdir):
    if('LSD' in file):
        print(file)
        
        if first:
            df=pd.read_excel(rootdir+file, index_col=None)
            first=False
        else:
            new = pd.read_excel(rootdir+file, index_col=None)
            df =  pd.concat([df,new], axis = 0, ignore_index=True)
        
titulo_excel= rootdir + rootdir[60:-1] + '_data.xlsx'


writer = pd.ExcelWriter(titulo_excel, engine='xlsxwriter')# Create a Pandas Excel writer

# Write the dataframe data to XlsxWriter. Turn off the default header and
# index and skip one row to allow us to insert a user defined header.
df.to_excel(writer, sheet_name='Sheet1', startrow=1, header=False, index=False)


workbook = writer.book     # Get the xlsxwriter workbook and worksheet objects.
worksheet = writer.sheets['Sheet1']


(max_row, max_col) = df.shape   # Get the dimensions of the dataframe.


column_settings = [{'header': column} for column in df.columns]# Create a list of column headers


worksheet.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings})# Add the Excel table structure.


worksheet.set_column(0, max_col - 1, 12)# Make the columns wider.


writer.save()# Close the Pandas Excel writer and output the Excel file.

print('------------------------------------------------------')
print('--------------------- END ----------------------------')
print('------------------------------------------------------')