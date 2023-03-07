#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:47:10 2023

@author: jeremias
"""
import pandas as pd
import numpy as np
import os
import random
from prettytable import PrettyTable as pt
import miscFunctions as mf




i=0
rootdir='/home/jeremias/Documentos/PruebaIndicadores_BOW/LSD_dataset/sec_pegadas/'
run_log_path= 'run_log/' 


[train_files, valid_files, test_files] = mf.splitDatasetCustom(rootdir,0.7, 0.15, 0.15)

mf.saveRunData( i , run_log_path, train_files, valid_files, test_files) #guardamos cuales fueron las secuencias utilizadas


#merge all files into a single one for Neural network train,validation or test and save it into a folder
mf.joinAllFiles(filesList= train_files, src_dir=rootdir ,savePath="/home/jeremias/Documentos/PruebaIndicadores_BOW/LSD_dataset/train/")
mf.joinAllFiles(filesList= valid_files, src_dir=rootdir ,savePath="/home/jeremias/Documentos/PruebaIndicadores_BOW/LSD_dataset/validation/")
mf.joinAllFiles(filesList= test_files,  src_dir=rootdir ,savePath="/home/jeremias/Documentos/PruebaIndicadores_BOW/LSD_dataset/test/")



    



    
    
    
    
    
    
    
    
    
    