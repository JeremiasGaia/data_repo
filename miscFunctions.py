#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:09:32 2023

@author: jeremias
"""
import pandas as pd
import numpy as np
import os
import random

def splitDatasetCustom(rootdir,train_perc, valid_perc, test_perc):
    """
    

    Parameters
    ----------
    rootdir : recibe un directorio raíz y divide los archivos en  training, validacion y test
                de acuerdo a los porcentajes indicados
    train_perc :  train percentage
    valid_perc :  validation percentage
    test_perc :   test percentage
    Returns
    -------
    retorna las listas aleatoriamente ordenadas

    """
    
    
    filenames=os.listdir(rootdir) #get filenames



    # Create list filled with zeros
    train_files=c_list = [0] * len(filenames)
    valid_files=[0] * len(filenames)
    test_files=[0] * len(filenames)



    train_perc = 0.7 #
    valid_perc = 0.15 #
    test_perc  = 0.15 #
     


    train_count = int(np.round(train_perc * len(filenames)))
    valid_count = int(np.round(valid_perc * len(filenames)))
    test_count  = int(np.round(test_perc  * len(filenames)))

    # #shuffle list in place 
    # random.shuffle(filenames)
    # random.shuffle(filenames)


    # train_files=filenames[:train_idx]

    # valid_files=filenames[ (train_idx): (train_idx+valid_idx) ]

    # test_files=filenames[ (train_idx + valid_idx):]
        
    randIdx= random.sample(range(len(filenames)), len(filenames)) #generate random index numbers

    train_files=[ filenames[i] for i in randIdx[:train_count] ]

    valid_files=[ filenames[i] for i in randIdx[ (train_count): (train_count+valid_count) ] ]

    test_files=[  filenames[i] for i in randIdx[ (train_count + valid_count):] ]
    
    return [train_files, valid_files, test_files]


def joinAllFiles(filesList, src_dir, savePath, run_number):
    """
    Merge all files into a single one for Neural network train,validation or test and save it into a folder
    Parameters
    ----------
    filesList :  List of files to process
    src_dir   :  Source folder to read the selected files
    savePath  :  Destination folder

    Returns
    -------
    saves the resulting .xlsx file in the specified folder

    """
   
    i=run_number
    first=True
    for file in filesList:
    
        # print(file)
        
        if first:
            df=pd.read_excel(os.path.join(src_dir,file), index_col=None)
            first=False
        else:
            new = pd.read_excel(os.path.join(src_dir,file), index_col=None)
            df =  pd.concat([df,new], axis = 0, ignore_index=True)
            
    if 'train' in savePath:
        titulo_excel= os.path.join(savePath, 'train_data_run_'+str(i)+'.xlsx')
    if 'test' in savePath:
        titulo_excel= os.path.join(savePath, 'test_data_run_'+str(i)+'.xlsx')
    if 'validation' in savePath:
        titulo_excel= os.path.join(savePath, 'validation_data_run_'+str(i)+'.xlsx')


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
    return

def saveRunData(run, run_log_savePath, train_files, valid_files, test_files):
    
    textFilename= os.path.join(run_log_savePath,'run_info' ,'run_' + str(run) +'_info.txt')
    
    # fill with zeros to print the pandas dataframe
    valid_f = (valid_files + ['-']*len(train_files))[:len(train_files)]
    test_f  = (test_files  + ['-']*len(train_files))[:len(train_files)]

        
    df = pd.DataFrame({
        'Train':      train_files,
        'Validation': valid_f,
        'Test':       test_f })

    
 
    with open(textFilename, 'w') as f:
        dfasstring=df.to_string(index=False)
        f.write(dfasstring)
        
    return





































