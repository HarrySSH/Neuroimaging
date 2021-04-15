
import os
import sys
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np




class Cellmap_DataLoader(data.Dataset):
    def __init__(self,  inputs, targets, split):  # inputs and targets should be two list of array
        super(Cellmap_DataLoader, self).__init__()
        self.targets = list(targets)
        self.inputs  = list(inputs)
        

    def __len__(self):
        return len(self.targets)


    def __getitem__(self, index):

        sample = dict()
        sample["X"] = torch.from_numpy(self.inputs[index]).float()
        sample["y"] = torch.from_numpy(np.array(self.targets[index]).reshape(1)).float()
        sample["ID"] = index

        return sample






        
        
if __name__ == "__main__":
    
    dataset = ER_DataLoader(img_list='./data/Datasets/bsds&pascal_train.txt', transform=False, split="train", edge_guidance=False, is_edge=True, in_size=1024, out_size=256)
    data_loader = data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8)
    print(len(data_loader))
    for i, inputs in enumerate(data_loader):
        print(inputs['ID'])