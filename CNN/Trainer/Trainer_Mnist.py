import os
import sys
import time
import random
import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
#from visdom import Visdom
sys.path.append('..')

from utils.optimize import configure_optimizers, get_lr, R2_loss
from utils.utils import init_weights
from datasets.data_loader import Cellmap_DataLoader

from models import CNN_1D_v3
from models import MNist_NN

print("PyTorch Version: ", torch.__version__)

class trainer_regression(nn.Module):
    def __init__(self, params = None):
        super(trainer_regression, self).__init__()
        self.args = params
        self.global_step = 0
        self.current_step = 0
    
    def _dataloader(self, datalist, targets, split='train'):
        dataset = Cellmap_DataLoader(inputs = datalist, targets = targets, split=split)
        shuffle = True if split == 'train' else False
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=32, shuffle=shuffle)
        return dataloader

    
    def train_one_epoch(self, epoch, train_loader, model, optimizer, lr_scheduler):
        t0 = 0.0
        model.train()

        for Cell_map in train_loader:
            self.global_step += 1
            self.current_step +=1

            t1 = time.time()

            inputs, outputs = Cell_map['X'].cuda(), Cell_map['y'].cuda()
            inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1])
            #print(inputs.shape)
            optimizer.zero_grad()
            predicts = model(inputs)


            #total_loss = nn.SmoothL1Loss() 
            total_loss = nn.MSELoss()
            #total_loss = nn.MultiMarginLoss()
            loss = total_loss(predicts, outputs)
            #loss = R2_loss(predicts, outputs)
            
            loss.backward()
            optimizer.step()
            t0 += (time.time() - t1)


            if self.global_step % self.args.print_steps == 0:
                message = "Epoch: %d Step: %d LR: %.6f Total Loss: %.4f Runtime: %.2f s/%d iters." % (epoch+1, self.global_step, lr_scheduler.get_last_lr()[-1], loss, t0, self.current_step)
                print("==> %s" % (message))
                print("==> R square: " +str(r2_score(outputs.cpu().numpy().flatten(), predicts.cpu().detach().numpy().flatten())))
                self.current_step = 0
                t0 = 0.0
    
    def val_one_epoch(self, data_loader, model, epoch):
        with torch.no_grad():
            model.eval()

            for i, Cell_map in enumerate(data_loader):  # don't know what data_loader is
                inputs, outputs = Cell_map['X'].cuda(), Cell_map['y'].cuda()
                #inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1])
                predicts = model(inputs)
                
                if i == 0:
                    predicts_all = predicts
                    groundtruths  = outputs
                
                else:
                    predicts_all = torch.cat((predicts_all, predicts), dim = 0)
                    groundtruths = torch.cat((groundtruths, outputs), dim = 0)

            r2, RMSD = r2_score(groundtruths.cpu().numpy().flatten(),predicts_all.cpu().numpy().flatten()), mean_squared_error(groundtruths.cpu().numpy().flatten(),predicts_all.cpu().numpy().flatten(), squared = False)
        return r2


    def train(self):
        print('==> Create model')
        model = MNist_NN.Mnist_NN()
        model.cuda()
        model = nn.DataParallel(model, device_ids=self.args.gpu_list)

        print('==> List lreanable parameters')

        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t{}".format(name))     
        

        print("==> Load data.")

        train_data_loader = self._dataloader(self.args.train_data_list_X, self.args.train_data_list_y,split='train')
        val_data_loader = self._dataloader(self.args.val_data_list_X, self.args.val_data_list_y,split='val')

        print("==> Configure optimizer.")
        optimizer, lr_scheduler = configure_optimizers(model, self.args.init_lr, self.args.weight_decay,
                                                       self.args.gamma, self.args.lr_decay_every_x_epochs)  # didn't define this yet ...

        print("==> Start training")
        since = time.time()
        best_f1 = 0.0
        for epoch in range(self.args.epochs):

            self.train_one_epoch(epoch, train_data_loader, model, optimizer, lr_scheduler)
            epoch_f1 = self.val_one_epoch(val_data_loader, model, epoch)
            #print('Best R2 square: ' + str(epoch_f1))
            if epoch_f1 > best_f1:
                best_f1 = epoch_f1
                print('Best R2 square: ' + str(best_f1))
                torch.save({'epoch': epoch + 1,
                            'model_state_dict': model.module.state_dict()},
                           os.path.join(self.args.ckpt_dir, "checkpoints_best.pth"))

            if epoch % self.args.save_every_x_epochs == 0:
                torch.save({'epoch': epoch + 1,
                            'model_state_dict': model.module.state_dict()},
                           os.path.join(self.args.ckpt_dir, "checkpoints_epoch_" + str(epoch + 1) + ".pth"))
            lr_scheduler.step()

        print("==> Runtime: %.2f minutes." % ((time.time() - since) / 60.0))




