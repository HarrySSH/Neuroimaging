
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
from Trainer.Trainer_Mnist import trainer_regression

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
    
    Connectomes = scipy.io.loadmat('../Cells2Connectomes/Connectomes.mat')
    Connectome_direct = Connectomes['C_dir']


    '''

    Region volumes, in a 424 vector, to get connectivity density, divide
    % each row in connectomes by each entry in the vector to get density. Units
    % are in 200 micron per vertex voxels.

    '''

    CellType_volumn = mat4py.loadmat('../Cells2Connectomes/Regional_Volumes.mat')
    CellType_volumn = CellType_volumn['region_vols']
    Celltype_volumn =np.array([np.array(xi) for xi in CellType_volumn])
    Celltype_volumn.shape

    # Nomarlize by the entry

    Connectome_direct_density = Connectome_direct/Celltype_volumn

    Cell_type = mat4py.loadmat('../Cells2Connectomes/CellType_Maps.mat')
    Cell_type = Cell_type['cellmaps']
    Celltype_mtx =np.array([np.array(xi) for xi in Cell_type])
    Celltype_mtx.shape

    # Important : normalizing via the columns

    Celltype_mtx_norm = (Celltype_mtx.max(axis=0)-Celltype_mtx) / (Celltype_mtx.max(axis=0) - Celltype_mtx.min(axis=0) )





    # distance 

    Distance = mat4py.loadmat('../Cells2Connectomes/Interregional_Distances.mat')
    Distance = Distance['D']
    Distance_mtx =np.array([np.array(xi) for xi in Distance])




    # region names
    Region_maps = mat4py.loadmat('../Cells2Connectomes/Region_Names.mat')
    Region_maps = Region_maps['region_names']
    Regionmaps_df = pd.DataFrame(Region_maps,columns = ['Anno1','Anno2','Anno3'])
    

    Region_maps_list = flatten_list(Region_maps)


    Region_maps_array = np.array(Region_maps_list)
    Region_maps_array_2d = np.reshape(Region_maps_array, (3,212))
    Regionmaps_df['Anno1'] = Region_maps_array_2d[0]
    Regionmaps_df['Anno2'] = Region_maps_array_2d[1]
    Regionmaps_df['Anno3'] = Region_maps_array_2d[2]



    #
    # 
    # index_l = Regionmaps_df[Regionmaps_df['Anno2'].isin(['Isocortex'])].index
    index_l = Regionmaps_df.index
    #index_l = Regionmaps_df.index
    index_r = [x+ 212 for x in index_l]
    index = list(index_l) + list(index_r)

    
    Distance_mtx_sub = Distance_mtx[index,:]
    Distance_mtx_sub = Distance_mtx_sub[:,index]
    # load data
    Target = []
    Dataset = []
    Celltype_mtx_norm_sub  = Celltype_mtx_norm[index]
    Connectome_direct_density_sub = Connectome_direct_density[index, :]
    Connectome_direct_density_sub = Connectome_direct_density_sub[:,index]
    for i in range(Celltype_mtx_norm_sub.shape[0]):
        for j in range(Celltype_mtx_norm_sub.shape[0]):
            #print(i)
            #print(j)
            if i == j:
                pass       
            else:
                Dataset.append(np.concatenate((Celltype_mtx_norm_sub[i,:],Celltype_mtx_norm_sub[j,:],Distance_mtx_sub[i,j].reshape(1),
                 Celltype_volumn[i].reshape(1),Celltype_volumn[j].reshape(1))))
                #Dataset.append(np.concatenate((Celltype_mtx_norm_sub[i,:],Celltype_mtx_norm_sub[j,:],
                #                            Celltype_mtx_norm_sub[i,:] - Celltype_mtx_norm_sub[j,:], 
                #                            np.log10((Celltype_mtx_norm_sub[i,:]+ 0.0001 )/ (Celltype_mtx_norm_sub[j,:]+0.0001)))))
                Target.append(Connectome_direct_density_sub[i,j])
                

    Dataset = np.stack(Dataset)
    capped_Target = [5 if x>5 else x for x in Target]
    capped_Target = [math.log2(x+1) for x in capped_Target]
    
    Target =np.array([np.array(xi) for xi in capped_Target])   
    #Target =np.array([np.array(xi) for xi in Target])   
    print(Dataset.shape)
    print(Target.shape)    
    X_train, X_test, y_train, y_test = train_test_split(Dataset, Target,test_size=.1,random_state =123)  


    parser = argparse.ArgumentParser()
 
    parser.add_argument('--norm', action='store_true', default=False)
    parser.add_argument('--init_lr', type=float, default=0.0001) 
    parser.add_argument('--weight_decay', type=float, default=0.000000005) 
    parser.add_argument('--batch_size', type=int, default=57000)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--print_steps', type=int, default=1)
    parser.add_argument('--save_every_x_epochs', type=int, default=10)
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
    

    