'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-08-15 17:12:07
LastEditors: Kevin
LastEditTime: 2022-08-19 11:03:49
FilePath: /code-graph/utils/matrix.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :matrix.py
@Description :
@Time        :2021/08/29 12:19:50
@Author      :Jinkui Hao
@Version     :1.0
'''
from sklearn.metrics import confusion_matrix, auc, roc_curve, f1_score,cohen_kappa_score
from sklearn.metrics import accuracy_score,precision_score,recall_score
import numpy as np
import torch
from sklearn.metrics import roc_auc_score



    
def getAUC(y_true, y_score, num_class):
    '''AUC metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    '''
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if num_class == 2:
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = roc_auc_score(y_true, y_score)
    else:
        auc = 0
        for i in range(y_score.shape[1]):
            y_true_binary = (y_true == i).astype(float)
            y_score_binary = y_score[:, i]
            auc += roc_auc_score(y_true_binary, y_score_binary)
        ret = auc / y_score.shape[1]

    return ret


def claMetrix(predictions, gt, num_class=7):
   # Calculate metrics
   # prediction and gt are both one-hot 
   # Accuarcy
   epsilon = 1e-8
   # targets = gt.squeeze()
   targets = torch.nn.functional.one_hot(gt,num_class)
   predictions = predictions.cpu().numpy()
   targets = targets.cpu().numpy()
   acc = np.mean(np.equal(np.argmax(predictions,1),np.argmax(targets,1)))
   # Confusion matrix
   conf = confusion_matrix(np.argmax(targets,1),np.argmax(predictions,1),labels=np.array(range(num_class)))


   # Class weighted accuracy
   auc = getAUC(gt.cpu().numpy(), predictions, num_class)
   PRE = precision_score(np.argmax(targets,1),np.argmax(predictions,1), average='weighted')
   RE = recall_score(np.argmax(targets,1),np.argmax(predictions,1), average='weighted')
   f1 = f1_score(np.argmax(predictions,1),np.argmax(targets,1),average='weighted')
   kappa = cohen_kappa_score(np.argmax(predictions,1),np.argmax(targets,1))

   #plot
   
   return acc, auc, RE, PRE, f1, kappa,conf

def claMetrixCsv(probability, predictions, targets, num_class=2):
   # Calculate metrics
   # prediction and gt are both one-hot 
   # Accuarcy
   epsilon = 1e-8
   # targets = gt.squeeze()
#    targets = torch.nn.functional.one_hot(gt,num_class)
#    predictions = predictions.cpu().numpy()
#    targets = targets.cpu().numpy()
   acc = np.mean(np.equal(predictions,targets))
   # Confusion matrix
#    conf = confusion_matrix(targets,predictions,labels=np.array(range(num_class)))


   # Class weighted accuracy
#    auc = getAUC(gt.cpu().numpy(), predictions, num_class)
   auc =  roc_auc_score(targets, probability)
   PRE = precision_score(targets, predictions,average='weighted')
   RE = recall_score(targets, predictions,average='weighted')
   f1 = f1_score(targets, predictions,average='weighted')
   kappa = cohen_kappa_score(predictions,targets)

#    FP, FN, TP, TN = numeric_score(predictions, targets)
#    specificity = np.divide(TN, TN + FP)
#    recall = np.divide(TP, TP + FN)
#    precision = np.divide(TP, TP + FP)

#    print('specificity',specificity,'\n recall',recall, '\n precision', precision)


   #plot
   
   return acc, auc, RE, PRE, f1, kappa

