#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 22:13:15 2023

@author: laurenparola
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
import IMU_event_clean as tmp
import Upload_XSENS_textfiles as pp

#file_path = '/Users/laurenparola/Library/CloudStorage/Box-Box/CMU_MBL/5_Datasets/Vu_Xsens_Test/Full_body_40Hz'
file_path = "/Users/Jose/Documents/School 22-23/wearable tech/test_walk"
sensor_list = pp.upload_sensor_txt(file_path)

dominant_limb = 'Right'

#Apply algorithm to shank sensor data
example_results = tmp.Sensor_Gait_Param_Analysis(sensor_list['Left shank']['Gyr_Y'], sensor_list['Right shank']['Gyr_Y'], sensor_list['Right shank']['PacketCounter'].values,dominant_limb, freq=40)
