import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd 
import os
import sys

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size = 3 )
        self.conv2 = nn.Conv1d(64, 128, kernel_size = 3)
         
        self.dropout1 = nn.Dropout(0.01)
        self.dropout2 = nn.Dropout(0.01)

        self.fc1 = nn.Linear(49* 128, 184)
        self.fc2 = nn.Linear(184, 80)
        self.fc3 = nn.Linear(80,10)
        self.fc4 = nn.Linear(10,1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
    
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = x.view(x.size(0),-1)
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)

        x = self.dropout2(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        #output = F.log_softmax(x, dim=1)
        # This is regression, don't use the log_softmax at the beginning
        output = x


        return output