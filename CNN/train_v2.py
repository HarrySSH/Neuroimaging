
import os
import sys
import datetime
import argparse
import pandas as pd
import numpy as np
import subprocess
import scipy.io
import mat4py
import math
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from Trainer.Trainer import trainer_regression
from sklearn.datasets import make_regression
# A function which can convert a 2d list to a 1d list

def flatten_list(_2d_list):
    flat_list = []
    for element in _2d_list:
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list



if __name__ == "__main__":
    
    X , y= make_regression(n_samples=5700, n_features=84, n_informative= 10, bias=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.1,random_state =123)  
    

    parser = argparse.ArgumentParser()
 
    parser.add_argument('--norm', action='store_true', default=False)
    parser.add_argument('--init_lr', type=float, default=0.00001) 
    parser.add_argument('--weight_decay', type=float, default=0.0000005) 
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--print_steps', type=int, default=1)
    parser.add_argument('--save_every_x_epochs', type=int, default=1)
    parser.add_argument('--lr_decay_every_x_epochs', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.1) 
    parser.add_argument('--gpus', type=str, default="0,1")
    parser.add_argument('--show_results_every_x_steps', type=int, default=20)
    parser.add_argument('--train_data_list_X', type=str, default= X_train)
    parser.add_argument('--val_data_list_X', type=str, default= X_test)
    parser.add_argument('--train_data_list_y', type=str, default= y_train)
    parser.add_argument('--val_data_list_y', type=str, default= y_test)
    args = parser.parse_args()
    

    #args.pre_trained = os.path.join('data/train_log', args.pre_trained, 'checkpoints/checkpoints_50')
    
    # parse train log directory 
    hour = datetime.datetime.now().hour
    if hour < 10:
        hour = "0"+str(hour)
    else:
        hour = str(hour)
    minute = datetime.datetime.now().minute
    if minute < 10:
        minute = "0"+str(minute)
    else:
        minute = str(minute)     
           
    train_log_dir = "train_" \
         + str(datetime.datetime.now().year) \
         + str(datetime.datetime.now().month) \
         +str(datetime.datetime.now().day)\
         + "_" + hour + minute 
    args.train_log_dir = os.path.join('/mnt/data/PRAD/Shenghuan/train_log', train_log_dir)
    
    # parse checkpoints directory
    ckpt_dir = os.path.join(args.train_log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    args.ckpt_dir = ckpt_dir

    # parse code backup directory
    code_backup_dir = os.path.join(args.train_log_dir, 'codes')
    os.makedirs(code_backup_dir, exist_ok=True)
    subprocess.call('cp -r ./models ./Trainer ./datasets ./utils {}'.format(code_backup_dir), shell=True)
    subprocess.call('cp -r ./{} {}'.format(__file__, code_backup_dir), shell=True)

    # parse gpus 
    os.environ['CUDA_VISIBLE_DEVICE'] = args.gpus
    gpu_list = []
    for str_id in args.gpus.split(','):
        id = int(str_id)
        gpu_list.append(id)
    args.gpu_list = gpu_list
    
    # format printing configs
    print("*" * 50)
    table_key = []
    table_value = []
    for key, value in vars(args).items():
        table_key.append(key)
        table_value.append(str(value))
    

    # configure trainer and start training

        
    trainer = trainer_regression(args) 
    trainer.train()
    

    