import numpy as np
import os
import cv2
from scipy import misc
from PIL import Image
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import csv
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



def class1(csvFile):
    reader = csv.reader(csvFile)
    pred = []
    label = []
    pred1 = []
    for item in reader:
        pred1.append(int(item[2]))
        pred.append(float(item[1]))
        label.append(int(item[3]))

    csvFile.close()
    TN, FP, FN, TP = confusion_matrix(label,pred1).ravel()
   
    spe = TN/(TN+FP)
    acc =(TP+TN)/(TP+TN+FP+FN)
    sen = TP/(TP+FN)
    ppv = TP/(TP+FP)
    npv = TN/(TN+FN)
    F1 = (2*ppv*sen)/(ppv+sen)
    fpr, tpr, thresholds = metrics.roc_curve(label, pred)
    auc = roc_auc_score(label, pred)
    acc1 = (sen+spe)/2
    return fpr,tpr,auc

def class2(csvFile):
    reader = csv.reader(csvFile)
    pred = []
    label = []
    pred1 = []
    for item in reader:
        if int(item[2])==0:
            pred1.append(0)
            pred.append(1-float(item[1]))
            label.append(0)
        elif int(item[2])==1:
            if int(item[0])==1:
                pred1.append(1)
                pred.append(float(item[1]))
                label.append(1)
            else:
                pred1.append(0)
                pred.append(1 - float(item[1]))
                label.append(1)
        elif int(item[2])==2 :
            if int(item[0])==1:
                pred1.append(1)
                pred.append(float(item[1]))
                label.append(0)
            else:
                pred1.append(0)
                pred.append(1 - float(item[1]))
                label.append(0)


    csvFile.close()
    TN, FP, FN, TP = confusion_matrix(label,pred1).ravel()
    # TP = 0
    # FP = 0
    # TN = 0
    # FN = 0

    # for i in range(len(label)):
    #     if pred1[i] == label[i] == 1:
    #         TP += 1
    #     if pred1[i] == 1 and label[i] != pred1[i]:
    #         FP += 1
    #     if pred1[i] == label[i] == 0:
    #         TN += 1
    #     if pred1[i] == 0 and label[i] != pred1[i]:
    #         FN += 1
    spe = TN / (TN + FP)
    acc = (TP + TN) / (TP + TN + FP + FN)
    sen = TP / (TP + FN)
    ppv = TP / (TP + FP)
    npv = TN / (TN + FN)
    F1 = (2 * ppv * sen) / (ppv + sen)
    fpr, tpr, thresholds = metrics.roc_curve(label, pred)
    auc = roc_auc_score(label, pred)
    acc1 = (sen + spe) / 2
    return  fpr, tpr, auc

def class3(csvFile):
    reader = csv.reader(csvFile)
    pred = []
    label = []
    pred1 = []
    for item in reader:
        if int(item[3])==2:
            if int(item[1])==int(item[3]):
                pred1.append(1)
                pred.append(float(item[2]))
                label.append(1)
            else:
                pred1.append(0)
                pred.append(1-float(item[2]))
                label.append(1)
        else:
            if int(item[1])==int(item[3]):
                pred1.append(0)
                pred.append(1-float(item[2]))
                label.append(0)
            else:
                pred1.append(1)
                pred.append(float(item[2]))
                label.append(0)
    csvFile.close()
    TN, FP, FN, TP = confusion_matrix(label,pred1).ravel()
    # TP = 0
    # FP = 0
    # TN = 0
    # FN = 0

    # for i in range(len(label)):
    #     if pred1[i] == label[i] == 1:
    #         TP += 1
    #     if pred1[i] == 1 and label[i] != pred1[i]:
    #         FP += 1
    #     if pred1[i] == label[i] == 0:
    #         TN += 1
    #     if pred1[i] == 0 and label[i] != pred1[i]:
    #         FN += 1
    spe = TN / (TN + FP)
    acc = (TP + TN) / (TP + TN + FP + FN)
    sen = TP / (TP + FN)
    ppv = TP / (TP + FP)
    npv = TN / (TN + FN)
    F1 = (2 * ppv * sen) / (ppv + sen)
    fpr, tpr, thresholds = metrics.roc_curve(label, pred)
    auc = roc_auc_score(label, pred)
    acc1 = (sen + spe) / 2
    return fpr, tpr, auc

def class4(csvFile):
    reader = csv.reader(csvFile)
    pred1 = []
    label1 = []
    for item in reader:
            label1.append(int(item[2]))
            pred1.append(float(item[1]))

    csvFile.close()
    # print(pred1)
    # print(label1)

    fpr, tpr, thresholds = metrics.roc_curve(label1, pred1)
    auc = metrics.auc(fpr, tpr)
    return  fpr, tpr, auc

if __name__ == '__main__':
#   csvFile1 = open('result/VGG.csv', "r")
#   fpr, tpr, auc= class3(csvFile=csvFile1)
#   csvFile1 = open('result/Resnet.csv', "r")
#   fpr1, tpr1, auc1 = class3(csvFile=csvFile1)
#   csvFile1 = open('result/Xception.csv', "r")
#   fpr3, tpr3, auc3 = class3(csvFile=csvFile1)
#   csvFile1 = open('result/C3D.csv', "r")
#   fpr4, tpr4, auc4 = class3(csvFile=csvFile1)
#   csvFile1 = open('result/I3D.csv', "r")
#   fpr5, tpr5, auc5 = class3(csvFile=csvFile1)
#   csvFile1 = open('result/S3D.csv', "r")
#   fpr6, tpr6, auc6 = class3(csvFile=csvFile1)
#   csvFile1 = open('result/All.csv', "r")
#   fpr7, tpr7, auc7 = class3(csvFile=csvFile1)
#   csvFile1 = open('/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/2D_dataset/AUC/miccai_3D/msda_lstm_one_224.csv', "r")
#   fpr8, tpr8, auc8 = class2(csvFile=csvFile1)
#   csvFile1 = open('/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/2D_dataset/AUC/miccai_3D/xception_lstm_one_224.csv', "r")
#   fpr9, tpr9, auc9 = class2(csvFile=csvFile1)
#   csvFile1 = open('/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/2D_dataset/AUC/miccai_3D/our_method_one_224.csv', "r")
#   fpr10, tpr10, auc10 = class2(csvFile=csvFile1)
  # csvFile1 = open('/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/AUC/all_data_AUC/our_method_D_one_224.csv', "r")
  # fpr10, tpr10, auc10 = class3(csvFile=csvFile1)

  csvFile1 = open('AUC/1.csv', "r")
  fpr, tpr, auc= class1(csvFile=csvFile1)
  csvFile1 = open('AUC/2.csv', "r")
  fpr1, tpr1, auc1 = class1(csvFile=csvFile1)
  csvFile1 = open('AUC/3.csv', "r")
  fpr3, tpr3, auc3 = class1(csvFile=csvFile1)
  csvFile1 = open('AUC/4.csv', "r")
  fpr4, tpr4, auc4 = class1(csvFile=csvFile1)
  csvFile1 = open('AUC/5.csv', "r")
  fpr5, tpr5, auc5 = class1(csvFile=csvFile1)
  csvFile1 = open('AUC/6.csv', "r")
  fpr6, tpr6, auc6 = class1(csvFile=csvFile1)
  csvFile1 = open('AUC/7.csv', "r")
  fpr7, tpr7,auc7 = class1(csvFile=csvFile1)

  bwith = 0.4
  plt.rcParams['xtick.direction'] = 'in'
  plt.rcParams['ytick.direction'] = 'in'
  plt.figure(figsize=(6,5))
  ax = plt.gca()
  ax.spines['bottom'].set_linewidth(bwith)
  ax.spines['left'].set_linewidth(bwith)
  ax.spines['top'].set_linewidth(bwith)
  ax.spines['right'].set_linewidth(bwith)
  plt.grid(linestyle='-.',linewidth=0.05)

  plt.plot(fpr, tpr, label='Khawaldeh et al. (AUC={0:.4f})'.format(auc), color='seagreen', linestyle='-', linewidth=0.8)
  plt.plot(fpr1, tpr1, label='Chang et al. (AUC={0:.4f})'.format(auc1), color='royalblue', linestyle='-', linewidth=0.8)
  # plt.plot(fpr9, tpr9, label='Inception-V3 ACA(AUC={0:.4f})'.format(auc9), color='deepskyblue', linestyle='-.', linewidth=1)
  plt.plot(fpr3, tpr3, label='Zhou et al. (AUC={0:.4f})'.format(auc3), color='indigo', linestyle='-', linewidth=0.8)
  plt.plot(fpr4, tpr4, label='Liu et al. (AUC={0:.4f})'.format(auc4), color='deeppink', linestyle='-', linewidth=0.8)
  plt.plot(fpr5, tpr5, label='Zunair et al. (AUC={0:.4f})'.format(auc5), color='blue', linestyle='-', linewidth=0.8)
  plt.plot(fpr6, tpr6, label='Hao et al. (AUC={0:.4f})'.format(auc6), color='darkblue', linestyle='-', linewidth=0.8)
  plt.plot(fpr7, tpr7, label='Ours (AUC={0:.4f})'.format(auc7), color='red', linestyle='-', linewidth=0.5)
#   plt.plot(fpr8, tpr8, label='Xception+ConvLSTM (AUC={0:.4f})'.format(auc8), color='green', linestyle='-', linewidth=0.5)
#   plt.plot(fpr9, tpr9, label='MA-Net+ConvLSTM (AUC={0:.4f})'.format(auc9), color='pink', linestyle='-', linewidth=0.5)
#   plt.plot(fpr10, tpr10, label='Our method (AUC={0:.4f})'.format(auc10), color='red',
           #linestyle='-', linewidth=0.5)
  # plt.plot(fpr8, tpr8, label='S3DG (AUC={0:.4f})'.format(auc8), color='gray', linestyle='-',
  #          linewidth=0.5)
  # # plt.plot(fpr9, tpr9, label='Our method Loc (AUC={0:.4f})'.format(auc9), color='springgreen',
  # #          linestyle='-', linewidth=0.5)
  # plt.plot(fpr10, tpr10, label='Our method (AUC={0:.4f})'.format(auc10), color='red', linestyle='-',
  #          linewidth=0.5)
  # plt.plot(fpr5, tpr5, label='Scale1 + Scale2(auc={0:.4f})'.format(auc5), color='darkblue', linestyle='-',linewidth=1)
  # plt.plot(fpr55, tpr55, label='UNet(RW)-STARE(auc={0:.4f})'.format(auc55), color='lime', linestyle='-', linewidth=1)
  # plt.plot(fpr6, tpr6, label='Scale1 + Scale3(auc={0:.4f})'.format(auc6), color='chocolate', linestyle='-',
  #          linewidth=1)
  # # plt.plot(fpr66, tpr66, label='UNet(RW)-CHASEDB1(auc={0:.4f})'.format(auc66), color='deepskyblue', linestyle='-.', linewidth=1)
  #
  # # plt.plot(fpr33, tpr33, label='DDNet(RW)-CHASEDB1(auc={0:.4f})'.format(auc33), color='deepskyblue', linestyle='-.', linewidth=1)
  #
  # # plt.plot(fpr44, tpr44, label='UNet(RW)-DRIVE(auc={0:.4f})'.format(auc44), color='red', linestyle='-.', linewidth=1)
  # plt.plot(fpr7, tpr7, label='Scale2 + Scale3(auc={0:.4f})'.format(auc7), color='orange', linestyle='-', linewidth=1)
  # # plt.plot(fpr11, tpr11, label='DDNet(RW)-DRIVE(auc={0:.4f})'.format(auc11), color='red', linestyle='-.', linewidth=1)
  # plt.plot(fpr8, tpr8, label='Our MSRCNN(auc={0:.4f})'.format(auc8), color='red', linestyle='-', linewidth=1)
  font = {'family': 'Liberation Sans',
          'weight': 'normal',
           'size': 9,
           }
  font1 = {'family': 'Liberation Sans',
          'weight': 'normal',
          'size': 14,
          }
  plt.xlim(0, 1)
  plt.ylim(0, 1)
  plt.xticks(np.linspace(0, 1, 6))
  plt.yticks(np.linspace(0, 1, 11))
  plt.xlabel('1-Specificity',font1)
  plt.ylabel('Sensitivity',font1)
  plt.xticks(fontproperties='Liberation Sans', size=12, weight='normal')
  plt.yticks(fontproperties='Liberation Sans', size=12, weight='normal')

  plt.legend(loc='lower right',prop=font)
  plt.savefig('AUC/auc.eps')
  #plt.savefig('./figures/auc2.png')