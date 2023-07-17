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
from confuseMetrix import _test_data_class,plot_confusion_matrix


def class1(csvFile):
    reader = csv.reader(csvFile)
    pred = []
    label = []
    pred1 = []

    threshhold = 0.492

    for item in reader:
        # pred1.append(int(item[2]))
        # pred.append(float(item[1]))
        # label.append(int(item[3]))
        p = 0
        if float(item[1]) >= threshhold:
            p = 1
        else:
            p = 0
        pred1.append(p)
        pred.append(float(item[1]))
        label.append(int(item[3]))


    # threshhold = 0.5`
    # pre_01_all = pred
    # pre_01_all[pred >= threshhold] = 1
    # pre_01_all[pred < threshhold] = 0

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
    return auc, acc, acc1, ppv, npv, sen, spe, TP, TP + FP, TP + FN,F1

def class2(csvFile):
    reader = csv.reader(csvFile)
    pred = []
    label = []
    pred1 = []
    for item in reader:
        if int(item[3])==1:
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
    return auc, acc, acc1, ppv, npv, sen, spe, TP, TP + FP, TP + FN, F1

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
    return auc, acc, acc1, ppv, npv, sen, spe, TP, TP + FP, TP + FN, F1


def plotCM(savename,csvFile):
    reader = csv.reader(csvFile)
    lows = [low for low in reader]
    pred = [low[1] for low in lows]
    label = [low[3] for low in lows]

    #_test_data_class(savename,pred,label,columns=['Normal','BP','NCP'])
    #disp = plot_confusion_matrix(pred,label,labels=['Normal','BP','NCP'])
    plot_confusion_matrix(savename,pred,label,classes=['UPPE','CPPE'])

    return 1
    #plt.show()


if __name__ == '__main__':

  file_path = 'result-para'
  file_list = os.listdir(file_path)
  file_list.sort()
  item_name=[]
  AUC = []
  ACC = []
  BACC = []
  PPV = []
  NPV = []
  SEN = []
  SPE = []
  csvfile = open('tableTPara.csv', 'wt')
  writer = csv.writer(csvfile, delimiter=",")

  for item in file_list:
      #print(item)
      csvFile1 = open(os.path.join(file_path, item), "r")
      auc2, acc2, bacc2, ppv2, npv2, sen2, spe2, TP2, a2, b2, F12 = class1(csvFile=csvFile1)
      #print(round(auc2,4), round(acc2,4), round(bacc2,4), round(ppv2,4), round(npv2,4), round(sen2,4), round(spe2,4))
    #   print(item,  'sen:',round(sen2,4),'spe:',round(spe2,4), 'ACC:',round(acc2,4), 'AUC:',round(auc2,4),'F1:',round(F12,4))
      print(item, round(sen2,4),round(spe2,4),round(acc2,4),round(auc2,4),round(F12,4))
    #   auc3 = (auc + auc1 + auc2) / 3
    #   acc3 = (acc + acc1 + acc2) / 3
    #   bacc3 = (bacc + bacc1 + bacc2) / 3
    #   ppv3 = (ppv + ppv1 + ppv2) / 3
    #   npv3 = (npv + npv1 + npv2) / 3
    #   sen3 = (sen + sen1 + sen2) / 3
    #   spe3 = (spe + spe1 + spe2) / 3
    #   F13 = (F1+F11+F12)/3
    #   p0 = (TP + TP1 + TP2) / rowNum
    #   pe = (a * b + a1 * b1 + a2 * b2) / (rowNum * rowNum)
    #   kappa = (p0 - pe) / (1 - pe)
    #   # auc3 = (auc*0.318+auc1*0.468+auc2*0.214)
    #   # acc3 = (acc*0.318+acc1*0.468+acc2*0.214)
    #   # bacc3 = (bacc*0.318 + bacc1*0.468 + bacc2*0.214)
    #   # ppv3 = (ppv*0.318+ppv1*0.468+ppv2*0.214)
    #   # npv3 = (npv*0.318+npv1*0.468+npv2*0.214)
    #   # sen3 = (sen*0.318+sen1*0.468+sen2*0.214)
    #   # spe3 = (spe*0.318+spe1*0.468+spe2*0.214)
    #   AUC.append(auc3)
    #   ACC.append(acc3)
    #   BACC.append(bacc3)
    #   PPV.append(ppv3)
    #   NPV.append(npv3)
    #   SEN.append(sen3)
    #   SPE.append(spe3)
      writer.writerow([item,round(sen2,4),round(spe2,4),round(acc2,4),round(auc2,4),round(F12,4)])

      #print(item, round(kappa, 4), round(F13, 4),round(acc3, 4), round(bacc3, 4), round(sen3, 4), round(spe3, 4),auc2)








