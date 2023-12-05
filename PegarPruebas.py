# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@QQ 
#             CÓDIGO ORIGINAL AL FINAL DE ESTE ARCHIVO
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@QQ

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 17:31:35 2023

@author: jeremias


Previo a este se debe usar los archivos: ORB_Juntar_Archivos_Matlab_Python o bien LSD_Juntar_Archivos_Matlab_Python

"""
import pandas as pd
import os
import numpy as np

rootdir='/home/jeremias/Matlab/Jere/Indicadores_Paper/RelacionIndicadoresDesempeno/Pruebas_Abril23/'

savepath='/home/jeremias/Documentos/PruebaIndicadores_BOW/lsd_dataset/sec_pegadas/'



"""
primero hago un vector con los nombres de las secuencias
"""
seqnames=[]
for filenames in np.sort(os.listdir(rootdir)):
    name = filenames[4:(filenames.find("_p"))]
    if name not in seqnames:
        seqnames.append(name)
        # print(filenames[4:(filenames.find("_p"))])



"""
Luego para cada elemento del vector pregunto para todos los files
"""


for seqname in seqnames:
    laps=0
    print("-------------------------------------------")
    print("Analizando: ", seqname)
    for file in np.sort(os.listdir(rootdir)):
        
        # Para evitar que los numeros cortos de seqname caigan en varias secuencias (error)
        # ej. 
        # TUM_sequence_1
        # LSD_TUM_sequence_10_p1.xlsx
        # LSD_TUM_sequence_11_p1.xlsx
        # hago una comparativa de largos de string en la linea 58 if (len(seqname) == len (file[4:(file.find("_p"))])):
        
        if (seqname in file):
            
            if (len(seqname) == len (file[4:(file.find("_p"))])):
                
                if laps==0:
                    df=pd.read_excel(rootdir+file, index_col=0)
                    print("Agregando: ", file)   
                else:
                    new = pd.read_excel(rootdir+file, index_col=0)
                    print("Agregando: ", file)   
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








# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Wed Feb  1 17:31:35 2023

# @author: jeremias


# Previo a este se debe usar los archivos: ORB_Juntar_Archivos_Matlab_Python o bien LSD_Juntar_Archivos_Matlab_Python

# """
# import pandas as pd
# import os


# rootdir='/home/jeremias/Matlab/Jere/Indicadores_Paper/RelacionIndicadoresDesempeno/PruebasDiciembre22/'

# savepath='/home/jeremias/Documentos/PruebaIndicadores_BOW/lsd_dataset/sec_pegadas/'

# seqname='LSD_KITTI_10'

# laps=0
# for file in os.listdir(rootdir):
    # if (seqname in file):
    #     print(file)
        
    #     if laps==0:
    #         df=pd.read_excel(rootdir+file, index_col=0)
    #     else:
    #         new = pd.read_excel(rootdir+file, index_col=0)
    #         df =  pd.concat([df,new], axis = 0, ignore_index=True)
    #     laps +=1

# titulo_excel= savepath + seqname +'_All_Experiments.xlsx'

# writer = pd.ExcelWriter(titulo_excel, engine='xlsxwriter')# Create a Pandas Excel writer

# # Write the dataframe data to XlsxWriter. Turn off the default header and
# # index and skip one row to allow us to insert a user defined header.
# df.to_excel(writer, sheet_name='Sheet1', startrow=1, header=False, index=False)


# workbook = writer.book     # Get the xlsxwriter workbook and worksheet objects.
# worksheet = writer.sheets['Sheet1']


# (max_row, max_col) = df.shape   # Get the dimensions of the dataframe.


# column_settings = [{'header': column} for column in df.columns]# Create a list of column headers


# worksheet.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings})# Add the Excel table structure.


# worksheet.set_column(0, max_col - 1, 12)# Make the columns wider.


# writer.save()# Close the Pandas Excel writer and output the Excel file.

# print('------------------------------------------------------')
# print('--------------------- END ----------------------------')
# print('------------------------------------------------------')




















