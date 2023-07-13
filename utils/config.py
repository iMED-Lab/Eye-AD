#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :config.py
@Description :config
@Time        :2020/08/11 15:23:59
@Author      :Jinkui Hao
@Version     :1.0
'''

import os
class Config():

    #path
    datapath = '/media/hjk/10E3196B10E3196B/dataSets/胸腔积液' #数据和label路径
    resultPath = '/media/hjk/10E3196B10E3196B/dataSets/result/ChestCT' #模型保存路径
    featSavePath = '/media/hjk/10E3196B10E3196B/dataSets/胸腔积液/6.featForGraph-3'

    saveName = 'merge_SEResnet_2' #本次实验名称（保存路径）
    #v2：在每一个block后面加dropout
    savePath = os.path.join(resultPath, saveName)
    env = saveName

    batch_size = 8
    num_epochs = 100
    base_lr = 5e-5
    weight_decay = 0.0005

    eva_iter = 50

    #选择数据集
    isOri = True #是否将原图同时作为输入
    dataName = 'merged-2'  #'easy' or 'hard' or 'merged'

    #GPU
    gpu = '1'

class Config_graph():
    resultPath = '/media/hjk/10E3196B10E3196B/dataSets/result/ChestCT' #模型保存路径
    datapath = '/media/hjk/10E3196B10E3196B/dataSets/胸腔积液'

    saveName = 'graph_SE-252n' #本次实验名称（保存路径）

    savePath = os.path.join(resultPath, saveName)
    env = saveName

    num_epochs = 100
    base_lr = 1e-6
    weight_decay = 0.0005

    #选择数据集
    dataName = 'merged-2'  #'easy' or 'hard'

    #graph classification
    feat_in = 512  #节点特征维数
    hidden = 256
    nclass = 2
    dropout = 0.2
    nb_heads = 3 #multi-head attention
    alpha = 0.2 #Alpha for the leaky_relu

    imgNum = 40  #图像数量

    #GPU
    gpu = '1'
