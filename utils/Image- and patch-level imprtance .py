'''
Descripttion: generate the importance map of image- and patch-level
version: 
Author: Kevin
Date: 2022-08-19 09:36:49
LastEditors: JinkuiH jinkui7788@gmail.com
LastEditTime: 2023-07-13 15:20:13
'''


from cProfile import label
from html.entities import name2codepoint
from tkinter import image_names
from conf import config 
import torch
import os
import numpy as np
import scipy.sparse as sp
import csv
from utils.tools  import generateAdj
from utils.confuseMetrix import *
import random
import logging
from logging import handlers
from utils.matrix import claMetrix
from models.model import *
import cv2
import matplotlib.pyplot as plt
import math


def normalize(mx):
    """Row-normalize sparse matrix"""
    #对行求和，求倒数，去掉行为0的点，变成对角阵，点乘完成每行的归一化
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv) #2708*2708
    mx = r_mat_inv.dot(mx)
    return mx 


def generateAttMapPatch(path, savePath):
    vis = Visualizer(env='test', port=7788)
    test_loader = config.dataloader_test

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ignoreList = os.listdir('nameList')


    modelCNNs = []

    modelCNN = torch.load(path)
    modelCNN.eval()
    step = 0

    colormap =  'autumn' #'summer', 'autumn'
    with torch.no_grad(): 
        for item in test_loader:
            step += 1
            
            inputs, labels, name = item
            labels = labels.to(device)
            inputs = inputs.to(device)

            newName = name[0].split('/')[-1].replace(" ", "")

            if newName in ignoreList:
                continue

            outputsCNN, featureS, featureD, featureC, importanceT = modelCNN(inputs)
            importance = importanceT
            importanceRe = importance.reshape(config.batchSize_test,3,config.patchSize*config.patchSize)
            # print(inputs.shape)
            vis.img(name='images1', img_=inputs[0, 0, :, :])
            vis.img(name='images2', img_=inputs[0, 1, :, :])
            vis.img(name='images3', img_=inputs[0, 2, :, :])
            
            attentionMaps = torch.zeros((3,config.imgSize, config.imgSize))
            gridSize = math.ceil(config.imgSize / config.patchSize) #)每个网格的尺寸

            for i in range(attentionMaps.shape[0]):
                mapTmp = attentionMaps[i,:,:]
                attTmp = importance[0,i,:,:]
                attTmp = attTmp/torch.max(attTmp)
                for row in range(mapTmp.shape[0]):
                    for col in range(mapTmp.shape[1]):
                        mapTmp[row, col] = attTmp[row // gridSize, col // gridSize]
                imgMap = mapTmp.detach().cpu().numpy()

                image = inputs[0,i,:,:].detach().cpu().numpy()
                label = labels[0].detach().cpu().numpy()
                # heatmap = imgMap/np.max(imgMap)
                heatmap = np.uint8(255 * imgMap)
                plt.figure()
                plt.xticks([])
                plt.yticks([])

                
                plt.imshow(np.uint8(255 * image), cmap='gray')
                plt.savefig(savePath+name[0] + '_' + str(i) + '_' + str(label) +'_img.jpg')
                plt.close()

                plt.figure()
                plt.xticks([])
                plt.yticks([])
                plt.imshow(heatmap, alpha=1, cmap=colormap)
                plt.savefig(savePath+name[0] + '_' + str(i) + '_' + str(label) +'_map.jpg')
                plt.close()

                plt.figure()
                plt.xticks([])
                plt.yticks([])
                plt.imshow(np.uint8(255 * image), cmap='gray')
                plt.imshow(heatmap, alpha=0.4, cmap=colormap)
                plt.savefig(savePath+name[0] + '_' + str(i) + '_' + str(label) +'_cmb.jpg')
                plt.close()

            print(step)
    return 0


def generateAttMapAvg(paths, savePath):
    test_loader = config.dataloader_test

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelCNNs = []

    for path in paths:
        modelCNN = torch.load(path)
        modelCNN.eval()
        modelCNNs.append(modelCNN)
    step = 0

    colormap =  'autumn' #'summer', 'autumn'
    avgAttMap = torch.zeros((3, config.patchSize, config.patchSize))



    with torch.no_grad(): 
        itemNum = 0
        for item in test_loader:
            step += 1
            
            inputs, labels, name = item
            labels = labels.to(device)
            inputs = inputs.to(device)
            importance = torch.zeros((1, 3,7,7)).cuda()
            for i in range(1):
                outputsCNN, featureS, featureD, featureC, importanceT = modelCNNs[i](inputs)
                importance += importanceT

            importance = importance
            attentionMaps = torch.zeros((3,config.imgSize, config.imgSize))
            gridSize = math.ceil(config.imgSize / config.patchSize) #)每个网格的尺寸

            
            # if 'control' in name[0]:
            itemNum += 1
            if '_os' in name[0] or '_OS' in name[0]:
                importance = torch.flip(importance, [3])
            avgAttMap += importance.squeeze().cpu()

            print(step)
        print('itemNum:', itemNum)
        importance = avgAttMap/itemNum
        for i in range(attentionMaps.shape[0]):
            mapTmp = attentionMaps[i,:,:]
            attTmp = importance[i,:,:]
            attTmp = attTmp/torch.max(attTmp)
            for row in range(mapTmp.shape[0]):
                for col in range(mapTmp.shape[1]):
                    mapTmp[row, col] = attTmp[row // gridSize, col // gridSize]
            imgMap = mapTmp.detach().cpu().numpy()

                    
            heatmapAvg = np.uint8(255 * imgMap)

            plt.figure()
            plt.xticks([])
            plt.yticks([])
            plt.imshow(heatmapAvg, alpha=1, cmap=colormap)
            plt.savefig(savePath +'_' + str(i)+'_Avg_map.jpg')
            plt.cla()

 
    return 0


def generateInsImportance(modelCNN, modelGNN, savePath):
    #Save instance importance
    csvPath = os.path.join(savePath,'importanceIns.csv') 
    csvFile = open(csvPath, 'w')
    csvWriter = csv.writer(csvFile)

    adjMetrixL1 = generateAdj(config.patchSize)
    #adjMetrix = np.load('utils/adjMetrix_30.npy')
    adjMetrixL1 = torch.from_numpy(adjMetrixL1)
    adjMetrixL1 = sp.coo_matrix(adjMetrixL1)
    adjMetrixL1 = normalize(adjMetrixL1 + sp.eye(adjMetrixL1.shape[0])) 
    adjMetrixL1 = adjMetrixL1.todense()
    adjMetrixL1 = torch.from_numpy(adjMetrixL1)
    adjMetrixL1 = adjMetrixL1.float()
    adjMetrixL1 = adjMetrixL1.to(device)

    adjMetrixL2 = np.load('utils/adjMetrix_level2.npy')
    #adjMetrix = np.load('utils/adjMetrix_30.npy')
    adjMetrixL2 = torch.from_numpy(adjMetrixL2)
    adjMetrixL2 = sp.coo_matrix(adjMetrixL2)
    adjMetrixL2 = normalize(adjMetrixL2 + sp.eye(adjMetrixL2.shape[0])) 
    adjMetrixL2 = adjMetrixL2.todense()
    adjMetrixL2 = torch.from_numpy(adjMetrixL2)
    adjMetrixL2 = adjMetrixL2.float()
    adjMetrixL2 = adjMetrixL2.to(device)

    test_loader = config.dataloader_test
    weight = modelGNN.state_dict()['classfier.weight']
    
    step = 0
    with torch.no_grad(): 
        for item in test_loader:
            step += 1
            inputs, labels, name = item
            labels = labels.to(device)
            inputs = inputs.to(device)
            outputsCNN, featureS, featureD, featureC, importance = modelCNN(inputs) 
            importanceL1 = importance
            featureS, featureD, featureC, importance = featureS.squeeze(), featureD.squeeze(), featureC.squeeze(), importanceL1.squeeze()
            featureS = featureS.reshape(config.batchSize_test,512,config.patchSize*config.patchSize).transpose(1,2)
            featureD = featureD.reshape(config.batchSize_test,512,config.patchSize*config.patchSize).transpose(1,2)
            featureC = featureC.reshape(config.batchSize_test,512,config.patchSize*config.patchSize).transpose(1,2)

            importance = importance.reshape(config.batchSize_test,3,config.patchSize*config.patchSize)
            # output, _, _ = modelGNN(featureS, featureD, featureC, importance, adjL1, adjL2)
            output, _, _, _, _, feature = modelGNN(featureS, featureD, featureC, importance, adjMetrixL1, adjMetrixL2)
            
            #calculate the importance of instance
            S1 = torch.mm(weight[:,:512], feature[:,:512].T).cpu().numpy()
            S2 = torch.mm(weight[:,512:512*2], feature[:,512:512*2].T).cpu().numpy()
            S3 = torch.mm(weight[:,512*2:], feature[:,512*2:].T).cpu().numpy()

            a1 = np.sqrt(0.5*(np.power(S1[0]-0.5,2) + np.power(S1[1]-0.5,2)))
            a2 = np.sqrt(0.5*(np.power(S2[0]-0.5,2) + np.power(S2[1]-0.5,2)))
            a3 = np.sqrt(0.5*(np.power(S3[0]-0.5,2) + np.power(S3[1]-0.5,2)))

            #softmax
            a = [a1, a2, a3]
            importanceIns = np.exp(a)/np.sum(np.exp(a))
            # print(torch.argmax(output).cpu().numpy(),labels.cpu().numpy(), importanceIns)
            print(step)
            csvWriter.writerow([name[0], importanceIns[0][0], importanceIns[1][0], importanceIns[2][0], labels.cpu().numpy(), torch.argmax(output).cpu().numpy()])

    csvFile.close()


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CNNpath = ['pth/external/F9/cnn-part.pth', 'pth/external/P5/cnn-part.pth', 'pth/external/P7/cnn-part.pth']
    
    savePath = 'result-map/region'
   
    CNNpath = CNNpath[0]
    modelCNN = torch.load(CNNpath).to(device)
    modelCNN.eval()
    GNNpath = 'pth/external/F9/gnn-part.pth'
    modelGNN = torch.load(GNNpath).to(device)
    modelGNN.eval()

    generateInsImportance(modelCNN, modelGNN, savePath)

    