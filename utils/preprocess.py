#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :preprocess.py
@Description :将DCM格式转换为jpg
@Time        :2020/04/16 13:12:21
@Author      :Jinkui Hao
@Version     :1.0
'''

import SimpleITK as sitk
import pydicom
import os
import numpy as np
import dicom
import matplotlib.pyplot as plt
import cv2
import shutil
import csv
from PIL import Image



def dicom_ct_window(dat, offset, win_centre, win_length): 
    #lung window 
    dat = dat+offset 
    norm_dat = dat
    #% norm_dat(dat>0) = 0; 
    #% norm_dat(dat<-1200) = -1200; 

    max_val = win_centre+win_length/2
    min_val = win_centre-win_length/2 

    # min_val = -200.0
    # max_val = 300.0

    norm_dat = (norm_dat-min_val)/(max_val-min_val)
    norm_dat[dat>max_val] = 1.
    norm_dat[dat<min_val] = 0.  

    return norm_dat*255

def getInformation(isIndividul = True):
    #获取DICOM 文件的信息，写入csv文件

    root = '/media/hjk/10E3196B10E3196B/dataSets/胸腔积液/tempRes2/单纯性胸腔积液新'
    fileName = '/media/hjk/10E3196B10E3196B/dataSets/胸腔积液/infor_sim.csv'
    fw = open(fileName,'a+',encoding='utf-8-sig')
    rootList = os.listdir(root)
    for subfile in rootList:
        print(subfile)

        if subfile == '.DS_Store':
            continue
        subPath = os.path.join(root,subfile)
        #print(subPath)
        fileList = os.listdir(subPath)
        i = 1
        for name in fileList:##
            i = i+1
            if i < 10:
                continue
            
            fileName = os.path.join(subPath,name)
            img_tmp = np.zeros([512,512]) #初始化缓冲区域
            dcm = dicom.read_file(fileName)
            img = dcm.pixel_array
            img_array = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept

            try:
                dcm.SoftwareVersions
                dcm.SliceThickness
                dcm.WindowCenter
                dcm.WindowWidth
            except AttributeError:
                var_exists = False
            else:
                var_exists = True
            if var_exists:
                csv_writer = csv.writer(fw)
                csv_writer.writerow([subfile,dcm.SoftwareVersions,dcm.SliceThickness,dcm.WindowCenter,dcm.WindowWidth])
                break
            else:
                csv_writer = csv.writer(fw)
                csv_writer.writerow([subfile,'/'])
                break
    fw.close()

def moveFile(isIndividul = True):
    # root = '/home/hjk/COVID-19_classification/COVID-19（slice labeled）'
    # targetRoot = '/media/hjk/10E3196B10E3196B/dataSets/COVID19/newData/covid19'
    root = '/home/hjk/Desktop/image/细菌性肺炎（slice标注）'
    targetRoot = '/media/hjk/10E3196B10E3196B/dataSets/COVID19/newData/health'
    rootList = os.listdir(root)
    for subfile in rootList:
        subPath = os.path.join(root,subfile)
        #print(subPath)
        fileList = os.listdir(subPath)
        i = 1
        for name in fileList:##
            
            fileName = os.path.join(subPath,name)
            #info = loadFileInformation(fileName) #调用函数,获取图片的文件头说明信息,我这里没用

            image = cv2.imread(fileName)

            #保存路径
            savePath = os.path.join(targetRoot)
            mkdir(savePath)
            saveName = savePath + '/' + name[1:-4] + '.jpg'
            cv2.imwrite(saveName,image)

def generateSliceLabel():
    #生成label,包存volume名字，image名字，label
    #正常0 普通1 新冠2
    fileName = '/media/hjk/10E3196B10E3196B/dataSets/胸腔积液/NomalsliceLabel.csv'
    fw = open(fileName,'a+')
    csv_writer = csv.writer(fw)

    #读取文件
    root = '/media/hjk/10E3196B10E3196B/dataSets/胸腔积液/Normal-jpg'
    rootList = os.listdir(root)
    stainNum = 50
    for subfile in rootList:
        if subfile == '.DS_Store':
            continue
        subPath = os.path.join(root,subfile)
        #print(subPath)
        fileList = os.listdir(subPath)
        #每个文件夹选择30
        
        for people in fileList:
            peopleFile = os.path.join(subPath,people)
            #print(subPath)
            peopleList = os.listdir(peopleFile)
            peopleList.sort()
            fileNum = len(peopleList)
            startNum = int((fileNum-stainNum)/2)
            for i in range(stainNum):
                name = peopleList[startNum+i]
                label = 2 if 'Normal' in subfile else 3
                csv_writer.writerow([subfile,people,name,label])

    fw.close()
    print('done!')

def generateVolumeLabel():
    #将数据名称及label放到csv文件中，便于dataloder中读取
    #正常0 普通1 新冠2
    fileName = '/media/hjk/10E3196B10E3196B/dataSets/胸腔积液/selectRaw/2.newSimpleSeg/label1.csv'
    fw = open(fileName,'a+')
    csv_writer = csv.writer(fw)

    #读取文件
    root = '/media/hjk/10E3196B10E3196B/dataSets/胸腔积液/selectRaw/2.newSimpleSeg/复杂性胸腔积液X55'
    listPath = os.listdir(root)
    for name in listPath:
        csv_writer.writerow([name,1])

    fw.close()
    print('done!')

def DICM2JPG(isIndividul = True):
    '''
    胸腔积液CT前处理，转换JPG，并同时分成不同的文件存放
    '''
    root = '/media/hjk/10E3196B10E3196B/dataSets/胸腔积液/Normal-DCM'
    targetRoot = '/media/hjk/10E3196B10E3196B/dataSets/胸腔积液/Normal-jpg'

    rootList = os.listdir(root)
    for subfile in rootList:
        if subfile == '.DS_Store':
            continue
        subPath = os.path.join(root,subfile)
        #print(subPath)
        fileList = os.listdir(subPath)
        for people in fileList:
            if people == '.DS_Store':
                continue
            i = 1
            peopleFile = os.path.join(subPath,people)
            print(subPath)
            peopleList = os.listdir(peopleFile)
            for name in peopleList:##
                
                fileName = os.path.join(peopleFile,name)
                
                if '.DCM' in fileName:
                    ds = sitk.ReadImage(fileName) #用SimpleITK读取 dcm格式的文件
                    img_array = sitk.GetArrayFromImage(ds) #获取文件中图片的raw原始数据
                    img_array = img_array[0,:,:]
                else:
                    continue

                # if len(name.split('_'))<3:
                #     continue
                #infro = name.split('_')[2]

                dataReturn = dicom_ct_window(img_array,0,50,350)
                #dataReturn = dicom_ct_window(img_array,0,center,width)

                img_tmp = dataReturn.astype(np.uint8) #转换为0--256的灰度uint8类型

                #保存路径
                savePath = os.path.join(targetRoot,subfile,people)
                #savePath = os.path.join(targetRoot,subfile,people,infro+name.split('_')[-1].split('.')[0])
                if isIndividul is False:
                    savePath = os.path.join(targetRoot)
                #mkdir(savePath)
                if not os.path.isdir(savePath):
                    os.makedirs(savePath)
                saveName = savePath + '/' + name[:-4] + '.jpg'
                cv2.imwrite(saveName,img_tmp)
            i = i+1

def dataSelsect():
    #去掉前后15张图片，选择5mm厚度的文件夹、
    root = '/media/hjk/10E3196B10E3196B/dataSets/胸腔积液/tempRes'
    targetRoot = '/media/hjk/10E3196B10E3196B/dataSets/胸腔积液/tempRes2'

    rootList = os.listdir(root)
    for subfile in rootList:
        if subfile == '.DS_Store':
            continue
        subPath = os.path.join(root,subfile)
        #print(subPath)
        fileList = os.listdir(subPath)
        for people in fileList:
            if people == '.DS_Store':
                continue
            peopleFile = os.path.join(subPath,people)
            #print(subPath)
            peopleList = os.listdir(peopleFile)
            for WLfile in peopleList:
                #新路径
                newPath = os.path.join(targetRoot,subfile,people)
                
                allWLfile = os.path.join(subPath,people,WLfile)
                
                WLlist = os.listdir(allWLfile)
                WLlist.sort()
                num = len(WLlist)
                
                if num > 5 and num < 190 and int(WLfile.split('L')[-1])>0:
                    #去掉前后15张，并移动到新的文件夹
                    i = 0
                    for name in WLlist:
                        i = i + 1
                        if i < 20:
                            continue
                        if i > num - 15:
                            break
                        if not os.path.isdir(newPath):
                            os.makedirs(newPath)
                        shutil.copyfile(os.path.join(allWLfile,name),os.path.join(newPath,name))

if __name__ == '__main__':
    #convertJPG3(True)
    #moveFile()
    #convertJPG_MRI(True)
    #selectData()
    #generateLabel()
    #getInformation()
    generateSliceLabel()
    #readSimgleImg()
    #dataSelsect()
    #DICM2JPG()
    print('done')

    



  

    


   
