#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :dataset.py
@Description :
@Time        :2020/10/09 17:16:30
@Author      :Jinkui Hao
@Version     :1.0
'''

import torch.utils.data as data
import torch
import numpy as np
import os
from PIL import Image
import random
from torchvision import transforms
import torchvision.transforms.functional as TF
import cv2
from scipy import misc
import scipy.io as sio
import csv
import nibabel as nib
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader


def random_crop(data, label, crop_size):
    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(data, output_size=(crop_size, crop_size))
    data = TF.crop(data, i, j, h, w)
    label = TF.crop(label, i, j, h, w)
    return data, label

def img_transforms(img, label, crop_size):
    trans_pad = transforms.Pad(10)
    trans_tensor = transforms.ToTensor()

    img, label = trans_pad(img), trans_pad(label)
    img, label = random_crop(img, label, crop_size)
    img, label = trans_tensor(img), trans_tensor(label)

    return img, label


class datasetCT(data.Dataset):
    def __init__(self, root, isOri = False, isTraining = True, dataName = 'easy'):
        #dataPath = os.path.join(root,angel,illumination,struction)
        self.root = root
        self.isOri = isOri #是否同时将原图和肺部图叠加作为输入
        self.isTrain = isTraining
        self.dataName = dataName
        self.pathAndLabel = self.getPathAndLabel(root, isTraining)
        self.name = ''

    def __getitem__(self, index):
        imgPath, label = self.pathAndLabel[index]
        self.name = imgPath
    
        oriPath = os.path.join(self.root,'2.allJPG',imgPath)
        segPath = os.path.join(self.root,'3.allSeg',imgPath)

        oriImage = Image.open(oriPath)
        oriImage = oriImage.convert('L')

        segImage = Image.open(segPath)
        segImage = segImage.convert('L')

        
        
        imgTransform_test = transforms.Compose([
            # transforms.Pad(10),
            # transforms.RandomCrop(448),
            transforms.CenterCrop(448),
            transforms.ToTensor()
        ])

        if self.isTrain:
            rotate = 10
            angel = random.randint(-rotate, rotate)
            oriImage = oriImage.rotate(angel)
            segImage = segImage.rotate(angel)

            # gamma_v = round(np.random.uniform(0.7,1.9),2)
            # oriImage = TF.adjust_gamma(img=oriImage, gamma = gamma_v)
            # segImage = TF.adjust_gamma(img=segImage, gamma = gamma_v)
            oriImage, segImage = img_transforms(oriImage, segImage, 448)
        else:
            segImage = imgTransform_test(segImage)
            oriImage = imgTransform_test(oriImage)

        if self.isOri:
            image = torch.stack((oriImage,segImage),dim=0)
            image = torch.squeeze(image)
        else:
            #image = segImage
            image = oriImage

        return image, int(label)
        

    def __len__(self):
        return len(self.pathAndLabel)

    def getPathAndLabel(self,root, isTrain):
        
        if isTrain:
            #labelPath = os.path.join(root,'train.csv')
            labelPath = os.path.join(root,'label','slice',self.dataName,'train.csv')
        else:
            #labelPath = os.path.join(root,'test.csv')
            labelPath = os.path.join(root,'label','slice',self.dataName,'test.csv')

        items = []

        file = open(labelPath,'r')
        fileReader = csv.reader(file)
        for line in fileReader:

            pathRelative = os.path.join(line[0],line[1],line[2])
            items.append((pathRelative,line[3]))

        return items

    def getFileName(self):
        return self.name

class datasetCTall(data.Dataset):
    #处理所有文件，保存特征
    def __init__(self, root, isOri = False, isTraining = True, dataName = 'easy'):
        #dataPath = os.path.join(root,angel,illumination,struction)
        self.root = root
        self.isOri = isOri #是否同时将原图和肺部图叠加作为输入
        self.isTrain = isTraining
        self.dataName = dataName
        self.pathAndLabel = self.getPathAndLabel(root, isTraining)
        self.name = ''

    def __getitem__(self, index):
        imgPath, label = self.pathAndLabel[index]
        self.name = imgPath
    
        oriPath = os.path.join(self.root,'2.allJPG',imgPath)
        segPath = os.path.join(self.root,'4.allSegNoDiscard',imgPath)

        oriImage = Image.open(oriPath)
        oriImage = oriImage.convert('L')

        segImage = Image.open(segPath)
        segImage = segImage.convert('L')
        
        imgTransform_test = transforms.Compose([
            # transforms.Pad(10),
            # transforms.RandomCrop(448),
            transforms.CenterCrop(448),
            transforms.ToTensor()
        ])


        segImage = imgTransform_test(segImage)
        oriImage = imgTransform_test(oriImage)

        if self.isOri:
            image = torch.stack((oriImage,segImage),dim=0)
            image = torch.squeeze(image)
        else:
            image = segImage
            #image = oriImage

        return image, int(label)
        

    def __len__(self):
        return len(self.pathAndLabel)

    def getPathAndLabel(self,root, isTrain):
        

        labelPath = os.path.join(root,'label','graph',self.dataName,'test.csv')

        items = []

        file = open(labelPath,'r')
        fileReader = csv.reader(file)
        for line in fileReader:

            pathRelative = os.path.join(line[0],line[1],line[2])
            items.append((pathRelative,line[3]))

        return items

    def getFileName(self):
        return self.name

class datasetGraphCla(data.Dataset):
    #For graph classification
    def __init__(self, root, isTraining = True, imgNum = 20, dataName = 'easy'):
        '''
            @imgNum: graph节点的数量，中心点除外
        '''
        self.root = root
        self.isTrain = isTraining
        self.imgNum = imgNum
        self.dataName = dataName
        self.pathAndLabel = self.getPathAndLabel(root, isTraining)
        self.name = ''

    def __getitem__(self, index):
        pathList = self.pathAndLabel[index]
        label = pathList[-1]
        self.name = pathList[0].split('/')[-2]
    
        nodeFeature = np.zeros((self.imgNum+1,512),dtype=np.float32)
        uncertainty_all = np.zeros(self.imgNum+1,dtype=np.float32)
        for i in range(self.imgNum):
            allData = np.load(pathList[i])

            feat = allData['arr_0'].astype(np.float32)
            max_u = np.max(feat)
            min_u = np.min(feat)
            feat = (feat-min_u)/(max_u-min_u)
            nodeFeature[i+1,:] = feat

            uncertainty_one = allData['arr_1'].astype(np.float32)
            uncertainty_all[i+1] = uncertainty_one

        #nodeFeature[0,:] = np.mean(nodeFeature, axis=0)
        uncertainty_all[0] = 0

        #normalization and diagonalization of uncertainty
        max_u = np.max(uncertainty_all)
        min_u = np.min(uncertainty_all)
        uncertainty_all = (uncertainty_all-min_u)/(max_u-min_u)
        uncertainty_all = np.diag(uncertainty_all)
        
        featTransform = transforms.Compose([
            transforms.ToTensor()
        ])

        nodeFeature = featTransform(nodeFeature)
        uncertainty_all = featTransform(uncertainty_all)
            #image = oriImage
        return nodeFeature, uncertainty_all, int(label)
        
    def __len__(self):
        return len(self.pathAndLabel)

    def getPathAndLabel(self,root, isTrain):
        #
        if isTrain:
            #labelPath = os.path.join(root,'train.csv')
            labelPath = os.path.join(root,'label','graph',self.dataName,'train.csv')
        else:
            #labelPath = os.path.join(root,'test.csv')
            labelPath = os.path.join(root,'label','graph',self.dataName,'test.csv')

        items = []
        file = open(labelPath,'r')
        fileReader = csv.reader(file)
        currentName = ''
        for line in fileReader:
            graphPath = []
            if currentName == line[1]:
                continue
            currentName = line[1]
            pathRelative = os.path.join(self.root, '5.featForGraph','merged', line[0],line[1])
            featList = os.listdir(pathRelative)
            featList.sort()
            featNum = len(featList)

            if featNum < self.imgNum:
                #copy
                copyNum = self.imgNum-featNum
                lastName = ''
                for name in featList:
                    graphPath.append(os.path.join(self.root, '5.featForGraph','merged', line[0],line[1],name))
                    lastName = name

                for i in range(copyNum):
                    graphPath.append(os.path.join(self.root, '5.featForGraph', 'merged', line[0],line[1],lastName))
            else:
                startNum = int((featNum-self.imgNum)/2)
                for i in range(self.imgNum):
                    graphPath.append(os.path.join(self.root, '5.featForGraph','merged', line[0],line[1],featList[startNum+i]))

            graphPath.append(line[3])

            items.append(graphPath)

        return items

    def getFileName(self):
        return self.name


class dataset3dCNN(data.Dataset):
    #For 3D CNN classification
    def __init__(self, root, isTraining = True, imgNum = 30, size=128, dataName = 'merged'):
        '''
            @imgNum: volume中slice的数量
        '''
        self.root = root
        self.isTrain = isTraining
        self.imgNum = imgNum
        self.imgSize = size
        self.dataName = dataName
        self.pathAndLabel = self.getPathAndLabel(root, isTraining)
        self.name = ''

    def __getitem__(self, index):
        pathList = self.pathAndLabel[index]
        label = pathList[-1]
        self.name = pathList[0].split('/')[-2]

        volumeCT = np.zeros((self.imgSize, self.imgSize,self.imgNum),dtype=np.float32)
        rotate = 10
        angel = random.randint(-rotate, rotate)
        for i in range(self.imgNum):
            sliceCT = Image.open(pathList[i])
            sliceCT = sliceCT.resize((self.imgSize,self.imgSize))
            sliceCT = sliceCT.convert('L')

            if self.isTrain:
                #增强
                sliceCT = sliceCT.rotate(angel)
                gamma_v = round(np.random.uniform(0.7,1.9),2)
                sliceCT = TF.adjust_gamma(img=sliceCT, gamma = gamma_v)
            volumeCT[:,:,i] = sliceCT

        #nodeFeature[0,:] = np.mean(nodeFeature, axis=0)
        featTransform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        #volumeCT = np.unsqueeze(volumeCT,0)
        volumeCT = featTransform(volumeCT)
        volumeCT = volumeCT.unsqueeze(0)
            #image = oriImage
        return volumeCT, int(label)
        
    def __len__(self):
        return len(self.pathAndLabel)

    def getPathAndLabel(self,root, isTrain):
        #
        if isTrain:
            #labelPath = os.path.join(root,'train.csv')
            labelPath = os.path.join(root,'label','graph',self.dataName,'train.csv')
        else:
            #labelPath = os.path.join(root,'test.csv')
            labelPath = os.path.join(root,'label','graph',self.dataName,'test.csv')

        items = []
        file = open(labelPath,'r')
        fileReader = csv.reader(file)
        currentName = ''
        for line in fileReader:
            graphPath = []
            if currentName == line[1]:
                continue
            currentName = line[1]
            pathRelative = os.path.join(self.root, '2.allJPG', line[0],line[1])
            featList = os.listdir(pathRelative)
            featList.sort()
            featNum = len(featList)

            if featNum < self.imgNum:
                #copy
                copyNum = self.imgNum-featNum
                lastName = ''
                for name in featList:
                    graphPath.append(os.path.join(self.root, '2.allJPG', line[0],line[1],name))
                    lastName = name

                for i in range(copyNum):
                    graphPath.append(os.path.join(self.root, '2.allJPG',  line[0],line[1],lastName))
            else:
                startNum = int((featNum-self.imgNum)/2)
                for i in range(self.imgNum):
                    graphPath.append(os.path.join(self.root, '2.allJPG', line[0],line[1],featList[startNum]))

            graphPath.append(line[3])

            items.append(graphPath)

        return items

    def getFileName(self):
        return self.name

if __name__ == '__main__':
    path = '/media/hjk/10E3196B10E3196B/dataSets/胸腔积液'
    dataloader = DataLoader(dataset3dCNN(path), batch_size=1, num_workers=0, shuffle=True)
    batch_iterator = iter(dataloader)
    item, label = next(batch_iterator)