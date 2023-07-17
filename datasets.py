#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :datasets.py
@Description : dataset for fully-, semi-, supervised learning
@Time        :2022/08/27 16:16:03
@Author      :Jinkui Hao
@Version     :1.0
'''

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import csv
import os
from PIL import Image
import random
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from skimage.exposure import adjust_log
import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

import csv
import os
import torch.utils.data as data
import re
import torch


logger = logging.getLogger(__name__)

def rgb2ii(img, alpha):
    #color constancy
    """Convert RGB image to illumination invariant image."""
    ii_image = (0.5 + np.log(img[:, :, 1] / float(255)) -
                alpha * np.log(img[:, :, 2] / float(255)) -
                (1 - alpha) * np.log(img[:, :, 0] / float(255)))

    return ii_image


def make_one_hot(input, num_class):
    # return np.eye(num_class)[input.reshape(-1).T]
    return np.eye(num_class)[input].astype(np.int)


class datasetADMulti(Dataset):
    def __init__(self, root, modals, fold, imgSize=512, isTraining = True, isMask = False):
        super(datasetADMulti,self).__init__()
        self.root = root
        self.isTraining = isTraining
        self.modal = modals
        self.name = None
        self.imgsize = imgSize
        self.fold = fold
        self.allItems = self.getAllPath(root, isTraining)
        self.isMask = isMask

    def __getitem__(self,index):
        pathS, pathD, pathC, label = self.allItems[index]
       
        imgS = Image.open(pathS).convert('L')
        imgD = Image.open(pathD).convert('L')
        imgC = Image.open(pathC).convert('L')

        imgS = imgS.resize((self.imgsize,self.imgsize))
        imgD = imgD.resize((self.imgsize,self.imgsize))
        imgC = imgC.resize((self.imgsize,self.imgsize))

        name = pathS.split('/')[-3]
        # img = cv2.imread(pathS,1)

        imgS = np.asarray(imgS)
        imgD = np.asarray(imgD)
        imgC = np.asarray(imgC)

        if self.isMask:
            maskSize = 30
            maskArray = np.ones_like(imgS)
            for i in range(imgS.shape[0]):
                for j in range(imgS.shape[1]):
                    if i < maskSize or j < maskSize or imgS.shape[0]-i < maskSize or imgS.shape[1]-j < maskSize:
                        maskArray[i,j] = 0
            
            imgS = imgS*maskArray
            imgD = imgD*maskArray
            imgC = imgC*maskArray
        # cv2.imwrite('masked.jpg',maskArray*255)
        img = np.stack((imgS, imgD, imgC), axis=2)
        image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

        imgTransform= transforms.Compose([
            # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.Resize((self.imgsize,self.imgsize)),
            transforms.RandomCrop(size=self.imgsize, padding=10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(10),
            #transforms.RandomResizedCrop(size=self.imgsize, scale=(0.8, 1.2))
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.4168, 0.2911, 0.2598], std=[0.1882, 0.1983, 0.2151])
            # transforms.Normalize(mean=[0.3226], std=[0.2119])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        simpleTrans = transforms.Compose([
            transforms.Resize((self.imgsize,self.imgsize)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.4168, 0.2911, 0.2598], std=[0.1882, 0.1983, 0.2151])
            # transforms.Normalize(mean=[0.3226], std=[0.2119])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if self.isTraining:
            image = imgTransform(image)
            # imgD = imgTransform(imgD)
        else:
            image = simpleTrans(image)
            # imgD = simpleTrans(imgD)

        # img = torch.cat([imgS, imgD], 0)
        return image, int(label), name


    def __len__(self):
        return len(self.allItems)

    def getAllPath(self,root,isTraining):
        items = []
        if isTraining:
            filePath = os.path.join(root, 'fiveFold', str(self.fold), 'train.csv')            
        else:
            filePath = os.path.join(root, 'fiveFold', str(self.fold), 'test.csv')

        with open(filePath,'r') as csvFile:
            reader = csv.reader(csvFile)
            for item in reader:
                path = os.path.join(root, item[0], item[1], item[2])
                imgS = ''
                imgD = ''
                imgC = ''
                for imgName in os.listdir(path):
                    # print(imgName)
                    splitName = re.split('_|\s|\\.',imgName)
                    if splitName[-2] == self.modal[0]:
                        imgS = imgName
                    if splitName[-2] == self.modal[1]:
                        imgD = imgName
                    if splitName[-2] == self.modal[2]:
                        imgC = imgName

                label = item[-1]
                items.append([os.path.join(path,imgS), os.path.join(path,imgD),os.path.join(path,imgC), label])

        return items


if __name__ == '__main__':
    pass

    