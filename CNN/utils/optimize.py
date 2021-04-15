   
import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
   
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']  

def configure_optimizers(model, lr, weight_decay, gamma, lr_decay_every_x_epochs):
    optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_decay_every_x_epochs, gamma=gamma)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=lr_decay_every_x_epochs)
    return optimizer, scheduler

'''
def soft_iou_loss(pred, label):
    b = pred.size()[0]
    pred = pred.view(b, -1)
    label = label.view(b, -1)
    inter = torch.sum(torch.mul(pred, label), dim=-1, keepdim=False)
    unit = torch.sum(torch.mul(pred, pred) + label, dim=-1, keepdim=False) - inter
    return torch.mean(1 - inter / unit)
'''
'''
def weighted_edge_loss(prediction, label):
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask==1).float()).float()
    num_negative = torch.sum((mask==0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0

    cost = nn.functional.binary_cross_entropy(
        prediction.float(),label.float(), weight=mask, reduction='sum') / (num_negative + num_positive)
    
    return cost
'''
def R2_loss(pred, label):
    b = pred.size()[0]
    pred = pred.view(b, -1)
    label = label.view(b, -1)
    RSS = torch.sum(torch.square(label - pred), dim=-1, keepdim=False)
    TSS = torch.sum(torch.square(label - torch.mean(label)), dim=-1, keepdim=False)
    
    return torch.mean(RSS/TSS)


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice