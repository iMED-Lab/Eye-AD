'''
Descripttion: Visualization
version: 
Author: Kevin
Date: 2022-08-19 09:36:49
LastEditors: 
LastEditTime: 2023-06-01 15:16:23
'''

from email import header
from requests import head
from ast import Name
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad,LayerCAM,GradCAMElementWise
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from torchvision import models 
import torch.nn as nn
import torch
from conf import config 
import os
import numpy as np
import cv2
from models.model import IMIGCNN
from pack.utils_pack import KVLayersDataset, minEnclosingCircle, extract_maximum_connected_area, tensor2array, array2image, array2tensor, arrayHStack, FetchAllData
from pack.faz_seg_pack import FAZ_SEG
from pathlib import Path
from pack.utils_pack import Visualizer

def show_cam_on_image(imgori, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)
    cam = heatmap + np.float32(imgori)
    cam = cam / np.max(cam)
   
    return cam

def show_cam(mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    return heatmap

def moveFAZ(inputs, attMap):
    #affine the image to make the faz at the center
    imgS = inputs[:, 2, :, :].unsqueeze(1)
    imgS = imgS.repeat(1, 3, 1, 1)
    faz, _, ori = c.classify([imgS,imgS,imgS])
    try:
        faz_ext = extract_maximum_connected_area(extract_maximum_connected_area(tensor2array(faz), threshold=120), threshold=95)
    except: 
        return None
    ori =  tensor2array(ori)
    minEnclosingCircle_img, center, radius = minEnclosingCircle(faz_ext)
    delta_x = 152 - center[0]
    delta_y = 152 - center[1]
    
    M = np.float32([[1, 0, delta_x], [0, 1, delta_y]])

    attMap = cv2.cvtColor(attMap*255, cv2.COLOR_GRAY2RGB)
    dst = cv2.warpAffine(attMap, M, (config.imgSize, config.imgSize), borderValue=(0, 0, 0)) 
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    return dst
    
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ignoreList = os.listdir('nameList')

    #FAZ model
    vis = Visualizer(env="FAZ_seg_pack", port=2000)

    # c = FAZ_SEG("saved_models/state-382-FAZ-89.71524198321349.pth")
    c = FAZ_SEG("saved_models/state-epoch-251.pth")

    my_models = IMIGCNN(in_channels=config.in_channels, num_classes=config.num_class, patch_size = config.patchSize)

    

    CNNpath = ['pth/external/P5/cnn-part.pth', 'pth/external/P7/cnn-part.pth']

    path = CNNpath[0]

    my_models = torch.load(path)

    imgPath = 'result-map/cnnp5_GradCAM_new/C_42-15'
    if not os.path.exists(imgPath):
        os.makedirs(imgPath)
    #change here
    target_layers = [my_models.encoder1.layer4[-2]]

    cam = GradCAM(model=my_models, target_layers=target_layers, use_cuda=True)

    targets = [ClassifierOutputTarget(1)]
    step = 0
    itemNum = 0
    adNum = 0
    ctlNum = 0
    errorMap = 0
    test_loader = config.dataloader_test
    avgAttMap = np.zeros((config.imgSize, config.imgSize))
    avgAttMap_AD = np.zeros((config.imgSize, config.imgSize))
    avgAttMap_ctl = np.zeros((config.imgSize, config.imgSize))
    for item in test_loader:
        
        step += 1
        print(step)
        inputs, labels, name = item
        nameLabel = str(labels[0])

        newName = name[0].split('/')[-1].replace(" ", "")

        if newName in ignoreList:
            continue

        labels = labels.to(device)
        inputs = inputs.to(device)
        image = (inputs.cpu().numpy()*255).astype(np.uint8)
        target_index = None
        mask = cam(inputs, target_index)
        mask = mask.squeeze()

        camMap = show_cam(mask)
        newCam = moveFAZ(inputs, mask)

        

        image = image[-1,:].transpose((1,2,0))
        imgZ = np.zeros_like(image)
        for i in range(3):
            imgZ[:,:,i] = image[:,:,0]

        cv2.imwrite(os.path.join(imgPath,nameLabel+'_'+newName), 0.7*camMap+0.5*imgZ)

        if newCam is not None:
            cv2.imwrite(os.path.join(imgPath,nameLabel+'_new_'+newName), newCam)
            itemNum += 1
            if '_os' in name[0] or '_OS' in name[0]:
                newCam = np.flip(newCam,1)
            avgAttMap += newCam

            if int(labels[0].cpu().numpy()) == 1:
                adNum += 1
                if '_os' in name[0] or '_OS' in name[0]:
                    newCam = np.flip(newCam,1)
                avgAttMap_AD += newCam
            else:
                ctlNum += 1
                if '_os' in name[0] or '_OS' in name[0]:
                    newCam = np.flip(newCam,1)
                avgAttMap_ctl += newCam

    importance = avgAttMap/255/(itemNum)
    importance_ad = avgAttMap_AD/255/(adNum)
    importance_ctl = avgAttMap_ctl/255/(ctlNum)
    print('itemNum:', itemNum)
    name = os.path.join(imgPath,imgPath.split('/')[-1]+'_Avg_all.jpg')
    camMap = show_cam(importance)
    cv2.imwrite(name, camMap)

    name = os.path.join(imgPath,imgPath.split('/')[-1]+'_Avg_AD.jpg')
    camMap = show_cam(importance_ad)
    cv2.imwrite(name, camMap)

    name = os.path.join(imgPath,imgPath.split('/')[-1]+'_Avg_ctl.jpg')
    camMap = show_cam(importance_ctl)
    cv2.imwrite(name, camMap)
    print('done~')
