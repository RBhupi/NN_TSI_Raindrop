#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 17:32:22 2022
"""
import pandas as pd
import cv2 as cv 
from os.path import join
import torch
from torch.utils.data import Dataset

#create data access class
class DropContaminationData(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """Initializes data loader class and reads labels and image paths from
        a .csv file. The 'init' method expects two column-csv with header. 
        The first column must be image filenames, the second column for labels."""
        
        csv_path = join(root_dir, csv_file)
        self.img_labels = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        """Returns image and labels with iterator."""
        
        img_path = join(self.root_dir, str(self.img_labels.iloc[index, 0]))
        image = cv.imread(img_path)
        labels = torch.tensor(int(float(self.img_labels.iloc[index, 1])))
        
        if self.transform:
            image = self.transform(image)
        
        return (image, labels)
    
    
    