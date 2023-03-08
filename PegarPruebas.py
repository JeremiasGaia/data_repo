#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 17:31:35 2023

@author: jeremias


Previo a este se debe usar los archivos: ORB_Juntar_Archivos_Matlab_Python o bien LSD_Juntar_Archivos_Matlab_Python

"""
import pandas as pd
import os


rootdir='/home/jeremias/Matlab/Jere/Indicadores_Paper/RelacionIndicadoresDesempeno/PruebasDiciembre22/'

savepath='/home/jeremias/Documentos/PruebaIndicadores_BOW/LSD_dataset/sec_pegadas/'

seqname='LSD_KITTI_10'

laps=0
for file in os.listdir(rootdir):
    if (seqname in file):
        print(file)
        
        if laps==0:
            df=pd.read_excel(rootdir+file, index_col=0)
        else:
            new = pd.read_excel(rootdir+file, index_col=0)
            df =  pd.concat([df,new], axis = 0, ignore_index=True)
        laps +=1

titulo_excel= savepath + seqname +'_All_Experiments.xlsx'

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


