# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:26:24 2023

@author: jereg
"""

import miscFunctions as mf
import netTraining as nt


#----- Linux Based
# train_data_path = "/home/jeremias/Documentos/PruebaIndicadores_BOW/lsd_dataset/train/"         #path to save training data
# valid_data_path = "/home/jeremias/Documentos/PruebaIndicadores_BOW/lsd_dataset/validation/"    #path to save validation data
# test_data_path = "/home/jeremias/Documentos/PruebaIndicadores_BOW/lsd_dataset/test/"           #path to save test data

# seq_data_path='/home/jeremias/Documentos/PruebaIndicadores_BOW/lsd_dataset/sec_pegadas/' #path containing fataset files

# run_log_path= 'run_log/' #path to save log information


#----- Windows Based
train_data_path =  r"C:\Users\jereg\OneDrive\Documentos\ProgramasPython\LSTM_Tesis_Doc\lsd_dataset\train"
valid_data_path =  r"C:\Users\jereg\OneDrive\Documentos\ProgramasPython\LSTM_Tesis_Doc\lsd_dataset\validation"    #path to save validation data
test_data_path  =  r"C:\Users\jereg\OneDrive\Documentos\ProgramasPython\LSTM_Tesis_Doc\lsd_dataset\test"           #path to save test data
seq_data_path   =  r"C:\Users\jereg\OneDrive\Documentos\ProgramasPython\LSTM_Tesis_Doc\lsd_dataset\sec_pegadas" #path containing fataset files
run_log_path    =  r"C:\Users\jereg\OneDrive\Documentos\ProgramasPython\LSTM_Tesis_Doc\lsd_dataset\run_log" #path to save log information


cant_entrenamientos=20
training_epochs=2

for i in range(cant_entrenamientos):
    
    [train_files, valid_files, test_files] = mf.splitDatasetCustom(seq_data_path,0.7, 0.15, 0.15)
    
    mf.saveRunData( i , run_log_path, train_files, valid_files, test_files) #guardamos cuales fueron las secuencias utilizadas
    
    
    
    #merge all train, validation and test files into a single one for Neural network train,validation or test and save it into a folder
    mf.joinAllFiles(filesList= train_files, src_dir= seq_data_path ,savePath= train_data_path, run_number= i)
    mf.joinAllFiles(filesList= valid_files, src_dir= seq_data_path ,savePath= valid_data_path, run_number= i)
    mf.joinAllFiles(filesList= test_files,  src_dir= seq_data_path ,savePath= test_data_path , run_number= i)
    
    
    nt.training(train_data_path, valid_data_path, test_data_path, i, run_log_path, training_epochs)







