import os
import csv
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import confusion_matrix, auc, roc_curve, f1_score,cohen_kappa_score
from sklearn.metrics import accuracy_score,precision_score,recall_score
import matplotlib.pyplot as plt
import numpy as np
# from matrix import claMetrix
from confuseMetrix import plot_confusion_matrix


if __name__ == '__main__':
    # csvPath = '/home/imed/personal/kevin/code/01.AD-graph/02.classification/code-graph/result-csv/All-GNN.csv'
    # csvPath1 = '/home/imed/personal/kevin/code/01.AD-graph/02.classification/code-graph/result-csv/All-CNN.csv'
    # csvPath = '/home/imed/personal/kevin/code/01.AD-graph/02.classification/code-graph/result-csv/CNN/fold-1-state-76-Result.csv'
    # csvPath = '/home/imed/personal/kevin/code/01.AD-graph/02.classification/code-graph/result-csv/CNN/fold-2-state-7-Result.csv'
    csvPath = '/home/imed/personal/kevin/code/01.AD-graph/02.classification/code-graph/result-csv/CNN/fold-3-state-6-Result.csv'
    # csvPath = '/home/imed/personal/kevin/code/01.AD-graph/02.classification/code-graph/result-csv/CNN/fold-4-state-116-Result.csv'
    # csvPath = '/home/imed/personal/kevin/code/01.AD-graph/02.classification/code-graph/result-csv/CNN/fold-5-state-73-Result.csv'
    
    # csvPath = '/home/imed/personal/kevin/code/01.AD-graph/02.classification/code-graph/result-csv/GNN/fold-1-state-51-Result.csv'
    pred = []
    label = []
    predMax = []
    with open(csvPath,'r') as csvFile:
        csvReader = csv.reader(csvFile)
        
        for item in csvReader:
            if item[2] == '0.0':
                item[2] = 0
            if item[2] == '1.0':
                item[2] = 1
            pred.append(float(item[1]))
            predMax.append(int(item[2]))
            label.append(int(item[3]))
    csvFile.close()

    TN, FP, FN, TP = confusion_matrix(label,predMax).ravel()
    F1 = f1_score(predMax,label,average='weighted')
    PRE = precision_score(predMax,label, average='weighted')
    sen1 = recall_score(predMax,label, average='weighted')
   
    spe = TN/(TN+FP)
    acc =(TP+TN)/(TP+TN+FP+FN)
    sen = TP/(TP+FN)
    ppv = TP/(TP+FP)
    npv = TN/(TN+FN)
    # F1 = (2*ppv*sen)/(ppv+sen)
    fpr, tpr, thresholds = metrics.roc_curve(label, pred)
    auc = roc_auc_score(label, pred)

    print('Acc:%04f, AUC:%04f, Sen:%04f, PRE:%04f, F1:%04f'%(acc, auc, sen1, PRE, F1))

    # pred1 = []
    # label1 = []
    # predMax1 = []
    # with open(csvPath1,'r') as csvFile1:
    #     csvReader = csv.reader(csvFile1)
        
    #     for item in csvReader:
    #         if item[2] == '0.0':
    #             item[2] = 0
    #         if item[2] == '1.0':
    #             item[2] = 1
    #         pred1.append(float(item[1]))
    #         predMax1.append(int(item[2]))
    #         label1.append(int(item[3]))
    # csvFile1.close()
    # fpr1, tpr1, thresholds1 = metrics.roc_curve(label1, pred1)
    # auc1 = roc_auc_score(label1, pred1)


    # bwith = 0.4
    # plt.rcParams['xtick.direction'] = 'in'
    # plt.rcParams['ytick.direction'] = 'in'
    # plt.figure(figsize=(6,5))
    # ax = plt.gca()
    # ax.spines['bottom'].set_linewidth(bwith)
    # ax.spines['left'].set_linewidth(bwith)
    # ax.spines['top'].set_linewidth(bwith)
    # ax.spines['right'].set_linewidth(bwith)
    # plt.grid(linestyle='-.',linewidth=0.05)

    # plt.plot(fpr, tpr, label='Ours. (AUC={0:.4f})'.format(auc), color='seagreen', linestyle='-', linewidth=0.8)
    # plt.plot(fpr1, tpr1, label='CNN-based. (AUC={0:.4f})'.format(auc1), color='royalblue', linestyle='-', linewidth=0.8)

    # font = {'family': 'Liberation Sans',
    #       'weight': 'normal',
    #        'size': 9,
    #        }
    # font1 = {'family': 'Liberation Sans',
    #         'weight': 'normal',
    #         'size': 14,
    #         }
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.xticks(np.linspace(0, 1, 6))
    # plt.yticks(np.linspace(0, 1, 11))
    # plt.xlabel('1-Specificity',font1)
    # plt.ylabel('Sensitivity',font1)
    # plt.xticks(fontproperties='Liberation Sans', size=12, weight='normal')
    # plt.yticks(fontproperties='Liberation Sans', size=12, weight='normal')

    # plt.legend(loc='lower right',prop=font)
    # plt.savefig('AUC/auc.eps')

    # conf = confusion_matrix(np.array(label),np.array(predMax),labels=np.array(range(2)))
    # cmap = 'Blues'
    # # save_path = os.path.join(save_dir, 'state-{}-{}.jpg'.format(epoch + 1,Acc))
    # save_path = os.path.join('AUC', 'CM.jpg')
    # # pretty_plot_confusion_matrix(save_path,conf_matrix, cmap=cmap)
    # plot_confusion_matrix(save_path, y_test=conf[0],predictions=conf[1], classes=['Control','AD'], cmap=cmap)
