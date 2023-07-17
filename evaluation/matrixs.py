import torch
import numpy as np
from sklearn import metrics
from collections import defaultdict

from scipy import spatial
import numpy as np
import os
import cv2

def AUC_score(SR,GT,threshold=0.5):
    #SR = SR.cpu().numpy()
    #GT = GT.cpu().numpy()
    fpr, tpr, _ = metrics.roc_curve(GT, SR)
    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc

def confusion(output, target):
    # output = output.asty
    # target = target.double()
    output[output > 0.5] = 1
    output[output <= 0.5] = 0
    p = torch.sum(target == 1).item()
    n = torch.sum(target == 0).item()

    tp = (output * target).sum().item()
    tn = ((1 - output) * (1 - target)).sum().item()
    fp = ((1 - target) * output).sum().item()
    fn = ((1 - output) * target).sum().item()
    epslon = 0.000001
    res = {"P": p, "N": n, "TP": tp, "TN": tn, "FP": fp, "FN": fn, "TPR": (tp / (tp+fn+epslon)), "TNR": (tn /(tn+fp+epslon) ), "FPR": (fp / (n+epslon)),
           "FNR": (fn / (p+epslon)), "Accuracy": (tp + tn) / (tp+fn+tn+fp+epslon)}
    return res

def numeric_score(prediction, groundtruth):
    """Computation of statistical numerical scores:

    * FP = False Positives
    * FN = False Negatives
    * TP = True Positives
    * TN = True Negatives

    return: tuple (FP, FN, TP, TN)
    """
    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))
    return FP, FN, TP, TN

def recall_score(prediction, groundtruth):
    # TPR, sensitivity
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FN) <= 0.0:
        return 0.0
    TPR = np.divide(TP, TP + FN)
    return TPR * 100.0


def specificity_score(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TN + FP) <= 0.0:
        return 0.0
    TNR = np.divide(TN, TN + FP)
    return TNR * 100.0


def intersection_over_union(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP + FN) <= 0.0:
        return 0.0
    return TP / (TP + FP + FN) * 100.0


def accuracy_score(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy * 100.0