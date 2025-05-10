#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 10:16:52 2022
"""
import numpy as np
import pandas as pd
from os.path import basename
import glob

data_dir = "/Users/bhupendra/projects/camera_raindrops/data/sage-training/"

file_list = glob.glob(data_dir+"clear_sky/*")

file_list = [basename(f) for f in file_list]

labels = [0 for x in range(len(file_list))]
data = {'files':file_list, 'labels':labels}
df = pd.DataFrame(data=data)
csv_name = data_dir + "labels.csv"

with open(csv_name, 'a') as f:
    df.to_csv(f, mode='a', header=f.tell()==0, index=False)









