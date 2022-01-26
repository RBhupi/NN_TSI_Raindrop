#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 16:25:48 2022
"""
import numpy as np
import cv2 as cv
import pandas as pd
from os.path import basename


def openVideoFile(fname):
    video_cap = cv.VideoCapture(fname)
    if not video_cap.isOpened():
        print('Unable to open: ', fname)
        exit(0)
        
    return video_cap



def readVideoFrame(fcount, capture):
    ret, frame = capture.read()

    if not ret:
        capture.release()
        print("End of video reached!")
        return fcount, frame
    
    frame = frame[140:500,70:430,]
    fcount += 1 
    return fcount, frame





#crop video data
video_name = "/Users/bhupendra/projects/cloud_motion/data/waggle-scott-tsi/sgptsimovieC1.a1.20210109.000000.mpg"

data_dir = "/Users/bhupendra/projects/camera_raindrops/data/train_data5/"

video_cap = openVideoFile(video_name)

file_list = []

fcount=0
date_str = "2021-01-09_"

while True:
    fcount, frame = readVideoFrame(fcount, video_cap)
    
    
    #Rain contamination period
    if fcount <= 220:
        img_path = data_dir + date_str + str(fcount) + ".jpg"
        cv.imwrite(img_path, frame)
        file_list.append(basename(img_path))
    
    """#clean mirror period
    if fcount > 540 and fcount <= 900:
        img_path = data_dir + date_str + str(fcount) + ".jpg"
        cv.imwrite(img_path, frame)
        file_list.append(basename(img_path))"""
    if fcount >= 220:
        break


    #img_path = data_dir + date_str + str(fcount) + ".jpg"
    #cv.imwrite(img_path, frame)
    #file_list.append(basename(img_path))


#labels = [1 if x<360 else 0 for x in range(720)]
labels = [2 for x in range(220)]
data = {'files':file_list, 'labels':labels}
df = pd.DataFrame(data=data)
csv_name = data_dir + "labels.csv"

with open(csv_name, 'a') as f:
    df.to_csv(f, mode='a', header=f.tell()==0, index=False)









