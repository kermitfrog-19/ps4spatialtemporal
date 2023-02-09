#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:07:19 2023

@author: laurenparola
"""
import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import IMU_event_clean as tmp

def upload_sensor_txt(file_path):
    df_list = []
    print(os.path.join(file_path,'.txt'))
    with open(os.path.join(file_path,'.txt')) as f:
        data = f.read()
    split = [x for x in data.split()]
    
    temp_list = []
    name_list = []
    code_list = []
    for i in split:
        if 'B4' in i:
            name = ' '.join(temp_list)
            name_list.append(name)
            code_list.append(i)
            temp_list = []
        else:
            temp_list.append(i)
    
    #labels = pd.read_csv(os.path.join(file_path,'sensor_name.txt'), sep='\s+',names=['Label','ID'],dtype=str
    
    walking_path = os.path.join(file_path,'walking') 
    sensor_list = {}
    for sensor in os.listdir(walking_path):
        imu_pos = [name_list[loc] for loc in range(len(code_list)) if code_list[loc] in sensor][0]
        sensor_list[imu_pos] = pd.read_table(os.path.join(walking_path,sensor), sep="\t",  skiprows=4)
    return sensor_list

