import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd 
import os
import sys

class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(84, 32, bias=True) 
        self.lin2 = nn.Linear(32, 8, bias=True)
        self.lin3 = nn.Linear(8, 1, bias=True)

    def forward(self, xb):
         
        x = torch.tanh(self.lin1(xb))
        x = torch.tanh(self.lin2(x))
        x = torch.tanh(self.lin3(x))
        x = x.view(-1,1)
        #x = F.softmax(x, dim =1)
        return x