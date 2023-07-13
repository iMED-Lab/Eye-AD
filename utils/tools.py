'''
Descripttion: 
version: 
Author: Kevin
Date: 2022-08-19 09:36:49
LastEditors: Kevin
LastEditTime: 2022-08-23 20:42:44
'''

import os
import numpy as np
from sklearn import metrics
import csv
from utils.matrix import claMetrixCsv
from utils.confuseMetrix import *
import torch
from models.model import threeBranchShareV2

def getResultFromFile(fileName):
    '''
    name: 
    test: test font
    msg: 存放5-fold的csv文件的文件夹
    return 合并的csv文件
    '''    
    allResult = []
    listFold = os.listdir(fileName)
    for fold in listFold:
        if '.csv' in fold:
            with open(os.path.join(fileName,fold)) as foldFile:
                reader = csv.reader(foldFile)
                for item in reader:
                    # allResult.append(np.array(item[1:]))
                    allResult.append((item))
    
    arrayRes = np.array(allResult)[:,1:].astype('float64')
    # print(arrayRes[:,2].astype('uint8'))
    Acc, AUC, RE, PRE, f1, kappa = claMetrixCsv(arrayRes[:,0],arrayRes[:,1].astype('uint8'),arrayRes[:,2].astype('uint8'),num_class=2)
    cmap = 'Blues'
    
    plot_confusion_matrix(os.path.join(fileName,fileName.split('/')[-1]+'.jpg'), arrayRes[:,2].astype('uint8'),arrayRes[:,1].astype('uint8'), classes=['Control','AD'], cmap=cmap)

    print('Acc:%04f, AUC:%04f, Recall:%04f, Precious:%04f, F1:%04f, Kappa:%04f'%(Acc, AUC, RE, PRE, f1, kappa))

    #将结果合并保存在csv文件
    with open(os.path.join(fileName,fileName.split('/')[-1]+'.csv'),'a+') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(allResult)

    return arrayRes


def generateAdj(patchNum):
    '''
    name: 
    msg: patch number 生成对应的邻接矩阵; patch只有水平和shuzhi的edge
        e.g.,如果是3*3,则patchNum = 3
    return {*}
    '''    
    adj = np.zeros((patchNum*patchNum,patchNum*patchNum))
    #horizontal edge
    for row in range(patchNum):
        for i in range(patchNum-1):
            adj[i+row*patchNum,row*patchNum+i+1] = 1
            adj[row*patchNum+i+1,i+row*patchNum] = 1

    for row in range(patchNum):
        for j in range(patchNum-1):
            adj[row+j*patchNum,row+j*patchNum+patchNum] = 1
            adj[row+j*patchNum+patchNum,row+j*patchNum] = 1
    # print(adj)
    return adj


def generateAdjL2(insNum = 3):
    '''
    name: 
    msg: 生成Level 2 的邻接矩阵,一个中心节点+instance node; 这里用的是有向图, 同UG-GAT
    return {*}
    '''    
    nodeNum = insNum + 1
    adj = np.zeros((nodeNum,nodeNum))

    # 首先将instance node和其他节点相连,单向
    for i in range(1,nodeNum):
        adj[0,i] = 1

    # Then, we add an edge between adjacent instance nodes,
    for j in range(1,nodeNum-1):
        adj[j, j+1] = 1
        adj[j+1, j] = 1

    return adj


if __name__ == '__main__':
    
    root = '/home/imed/personal/kevin/code/01.AD-graph/02.classification/code-graph-revised/result-csv/F2'
    result = getResultFromFile(root)
    
    # print(type(result), len(result), result[100])
    # print(result.shape)

    

    # adj = generateAdj(3)
    # adjMetrixL2 = np.load('/home/imed/personal/kevin/code/01.AD-graph/02.classification/code-graph-revised/utils/adjMetrix_level1.npy')
    # adj = generateAdjL2(5)
    # print(adj)
    print('done~')