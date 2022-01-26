#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Training and testing a Perceptron network to identify the state of the TSI
camera mirror dome. 
0:clear view of clouds, 
1:blocked by raindrops 
2:snow covered
3:clear mirror, no clouds

ToDo: add more layers and check the performance of the network.

Created on Tue Jan 18 14:19:41 2022
"""
from os.path import join

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader


from TSIData import DropContaminationData

class Perceptron(nn.Module):
    """Creates network"""
    
    def __init__(self, input_size, num_classes):
        super(Perceptron, self).__init__()
        self.input_layer = nn.Linear(input_size, 50)
        self.hidden_layer = nn.Linear(50, num_classes)

    def forward(self, tensor_in):
        tensor_act = F.relu(self.input_layer(tensor_in))
        tensor_out = self.hidden_layer(tensor_act)
        return tensor_out



def accuracy(loader, model):
    """Function to test acuracy in testing mode."""
    
    num_correct = 0 
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x=x.to(device=device)
            y=y.to(device=device)
            x=x.reshape(x.shape[0], -1)
            
            scores = model(x)
            
            _, preds = scores.max(1)
            num_correct += (preds==y).sum()
            num_samples += preds.size(0)
            
        acc = float(num_correct)/float(num_samples)*100
        
        return acc
        


# Load labeled dataset and split into traning and test set
root_dir = "/Users/bhupendra/projects/camera_raindrops/data/train_data/"

dataset = DropContaminationData(csv_file="labels.csv", 
                       root_dir=root_dir, transform=transforms.ToTensor())

data_len = len(dataset)
train_len = int(data_len/2)
test_len = int(data_len-train_len)
train_data, test_data = random_split(dataset, [train_len, test_len]) #720 data points


#Hyperparameters for Perceptraon
input_size = 360*360*3
num_classes = 4
learning_rate = 0.001
batch_size = 50
num_epochs = 2


train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


#Initialize the network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Perceptron(input_size, num_classes).to(device)

#Loss function and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


#Train the model
for epoch in range(num_epochs):
    for index, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        
        data = data.to(device=device)
        data = data.reshape(data.shape[0], -1)
        
        scores = model(data)
        loss = criterion(scores, targets)
        loss.backward()
        optimizer.step()
        


#Testing on both datasets
train_accu = accuracy(train_loader, model)
test_accu = accuracy(test_loader, model)

print(f'Training accuracy = {train_accu:.0f} % \n Testing accuracy = {test_accu:.0f} %')

save_dir= '/Users/bhupendra/projects/camera_raindrops/pt_models/'
path_nn = join(save_dir, "perceptron_4class.pt")
torch.save(model.state_dict(), path_nn)









