#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 16:25:48 2022
"""
import numpy as np
import cv2 as cv
import pandas as pd



def openVideoFile(fname):
    video_cap = cv.VideoCapture(fname)
    if not video_cap.isOpened():
        print('Unable to open: ', fname)
        exit(0)
        
    return video_cap



def readVideoFrame(fcount, capture):
    ret, frame = capture.read()
    fcount += 1 

    if not ret:
        capture.release()
        print("End of video reached!")
        return -1, frame
    frame = frame[140:500,70:430,]
    return fcount, frame





#crop video data
video_name = "/Users/bhupendra/projects/cloud_motion/data/waggle-scott-tsi/sgptsimovieC1.a1.20170102.000000.mpg"

data_dir = "/Users/bhupendra/projects/camera_raindrops/data/train_data1/"

video_cap = openVideoFile(video_name)

file_list = []

fcount=0

while True:
    fcount, frame = readVideoFrame(fcount, video_cap)
    
    #Rain contamination period
    if fcount <= 360:
        img_path = data_dir + str(fcount) + ".jpg"
        cv.imwrite(img_path, frame)
        file_list.append(img_path)
    
    #clean mirror period
    if fcount > 540 and fcount <= 900:
        img_path = data_dir + str(fcount) + ".jpg"
        cv.imwrite(img_path, frame)
        file_list.append(img_path)
    if fcount >= 900:
        break





labels = [1 if x<360 else 0 for x in range(720)]
data = {'files':file_list, 'labels':labels}


df = pd.DataFrame(data=data)
df.to_csv(data_dir+"labels.csv")







