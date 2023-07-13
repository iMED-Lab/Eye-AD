#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :config.py
@Description :
@Time        :2022/08/16 16:26:32
@Author      :Jinkui Hao
@Version     :1.0
'''

import os
from datasets import datasetAD, datasetADMulti
import torch
from torchvision import transforms
#GPU
import torch.utils.data as data
from torch.utils.data import DataLoader
import random
import numpy as np


# set seed
GLOBAL_SEED = 1

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

GLOBAL_WORKER_ID = None

def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)


class Config(object):  
    gpu = '7' 
    # get attrbute
    def __getitem__(self, key):
        return self.__getattribute__(key)


class ConfigGraph(Config):

    fold = 1  #5-fold  1-5
    patchSize = 5

    root = '/01.data_process/data'
    
    dataName = 'combine'  
    saveName = 'EyeAD-' + dataName +'-BS16-F' + str(fold) +'-P' + str(patchSize)
    datapath = os.path.join(root, dataName)

    modals = ['浅层血管复合体', '深层血管复合体','脉络膜毛细血管层']

    resultPath = '...'
    savePath = os.path.join(resultPath, saveName)

    loadCNN = False
    loadGNN = False

    hnid_L1 = 128
    hidden = 512
    nclass = 2
    dropout = 0.05
    nb_heads = 2    #multi-head attention

    num_epochs_cnn_pre = 100
    num_epochs_gnn_pre = 100
    num_epochs = 200  #joint

    batchSize_cnn = 4
    batchSize_gnn = 4
    batchSize_joint = 4

    lr_cnn_pre =1e-4
    lr_gnn_pre = 1e-5
    base_lr = 5e-6
    base_lr_CNN =5e-5

    #For combine dataset in IMIG
    lossWeight1 = 0.5
    lossWeight2 = 1

    num_k1 = 1
    num_k2 = 4
    num_k3 = 1

    classLabel = [0,1]  

    model_flag = 'resnet18'
    pretrained = False

    in_channels = 1

    imgSize = 304
    batchSize_test = 1
    num_class = 2

    workers = 1

    dataset_train = datasetADMulti(datapath,modals,fold,imgSize,isTraining=True)
    dataset_test = datasetADMulti(datapath,modals,fold,imgSize,isTraining=False)

    dataloader_train_CNN = data.DataLoader(dataset=dataset_train,
                                batch_size=batchSize_cnn,
                                shuffle=True, num_workers=workers,worker_init_fn=worker_init_fn, drop_last=True)
    dataloader_train_GNN = data.DataLoader(dataset=dataset_train,
                                batch_size=batchSize_gnn,
                                shuffle=True, num_workers=workers,worker_init_fn=worker_init_fn, drop_last=True)
    dataloader_train_joint = data.DataLoader(dataset=dataset_train,
                                batch_size=batchSize_joint,
                                shuffle=True, num_workers=workers,worker_init_fn=worker_init_fn, drop_last=True)

    dataloader_test = data.DataLoader(dataset=dataset_test,
                                batch_size=batchSize_test,
                                shuffle=True,worker_init_fn=worker_init_fn)                        

    env = saveName
    weight_decay = 0.0005
    feat_in = 512  
    alpha = 0.2 #Alpha for the leaky_relu


# 环境关系

mapping = {
    # 'ADCNN': ConfigADCNN,
    # 'ADCnnMullti': ConfigADCnnMulti,
    'ADGraph': ConfigGraph
}

APP_ENV = os.environ.get('APP_ENV', 'ADGraph')
config = mapping[APP_ENV]() 
