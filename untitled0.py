#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Training and testing a Perceptron network to identify the state of the TSI
camera mirror dome. 
0:clear view of clouds, 
1:blocked by raindrops 
2:snow covered
3:clear mirror, no clouds

ToDo: add more layers and check the performance of the network.

Created on Tue Mar 22 14:29:11 2022
"""

from os.path import join

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

#Creeate CNN
class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 9, 
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels = 9, out_channels = 18, 
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(18*7*7)
        
        
        
        
        
        
        
        
        