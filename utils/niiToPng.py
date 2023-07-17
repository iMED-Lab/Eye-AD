#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :niiToPng.py
@Description :将nii图像转换为png
@Time        :2020/04/17 14:56:23
@Author      :Jinkui Hao
@Version     :1.0
'''

import numpy as np
import os    #遍历文件夹
import nibabel as nib #nii格式一般都会用到这个包
import imageio   #转换成图像
import cv2
  
def nii_to_image(niifile):
 filenames = os.listdir(filepath) #读取nii文件夹
 slice_trans = []
  
 for f in filenames:
  #开始读取nii文件
  img_path = os.path.join(filepath, f)
  img = nib.load(img_path)    #读取nii
  img_fdata = img.get_fdata()
  fname = f.replace('.nii','')   #去掉nii的后缀名
  img_f_path = os.path.join(imgfile)
  #创建nii对应的图像的文件夹
  if not os.path.exists(img_f_path):
   os.mkdir(img_f_path)    #新建文件夹
  
  #开始转换为图像
  (x,y,z) = img.shape
  for i in range(z):      #z是图像的序列
   silce = img_fdata[:, :, i]   #选择哪个方向的切片都可以
   silce = cv2.flip(silce, 0)  #原型：cv2.flip(src, flipCode[, dst]) → dst  flipCode表示对称轴 0：x轴  1：y轴.  -1：both
   img_tmp = cv2.transpose(silce)
   MIN_BOUND = -1000.0
   MAX_BOUND = 400.0

   img_tmp = (img_tmp - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
   img_tmp[img_tmp>1] = 1.
   img_tmp[img_tmp<0] = 0.

   img_tmp = (img_tmp*255).astype(np.uint8) #转换为0--256的灰度uint8类型

   imageio.imwrite(os.path.join(img_f_path,fname+'_{}.png'.format(i)), img_tmp)
            #保存图像
  
if __name__ == '__main__':
 filepath = '/media/hjk/10E3196B10E3196B/dataSets/COVID19/segmentation/nii/rp_im_2'
 imgfile = '/media/hjk/10E3196B10E3196B/dataSets/COVID19/segmentation/images'
 nii_to_image(filepath)