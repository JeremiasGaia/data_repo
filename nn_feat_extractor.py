#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:46:09 2023

@author: jeremias
"""

import pandas as pd
import numpy as np
import os
import cv2

import matplotlib.pyplot as plt

import re
import glob

from scipy.io import savemat


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


rootdir=  '/media/jeremias/JereViejo/Dataset/KITTIDataset/gray/dataset/sequences/' #para KITTI
label_file_path= "/home/jeremias/Documentos/PruebaIndicadores_BOW/KITTI_Haralick_d6_GLRLM_"
x_train=[]
y_train=[]

for folder in os.listdir(rootdir):
    
    path = os.path.join(rootdir, folder,"undistorted","*.jpg")
    
    label_file = label_file_path+ folder + ".xlsx"
    
    df=pd.read_excel(label_file, index_col=0)
    
    filenames=sorted(glob.glob(path), key=numericalSort)
     
    
    
    i=0
    for img_file in filenames:
        
        print(img_file)
        
        img = cv2.imread(img_file,0)
        
        x_train.append(img)
        y_train.append(list(df.iloc[i,:-1]))
        
mdic = {"x_train": x_train, "y_train": y_train}
savemat("KITTI_data.mat", mdic)
                
            
            
            
            
            
            
            
            
            
            