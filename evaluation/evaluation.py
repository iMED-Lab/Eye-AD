import torch
import numpy as np
from sklearn import metrics
from collections import defaultdict

from scipy import spatial
import numpy as np
import os
import cv2

#Evaluating the iris upper boundary of output image
def findUpBoundary(eng,edgeImg):
    '''
    :param eng: matlab engine
    :param edgeImg: segmentation result
    :return: upper boundary
    '''
    output = edgeImg.tolist()
    rest = eng.iris_seg_up(output)
    numpy_res = np.zeros((536, 536))
    numpy_res[:, :] = rest
    cv2.imwrite('a1.jpg',numpy_res*255)

    return numpy_res

def hausdorff_score(prediction, groundtruth):
    return spatial.distance.directed_hausdorff(prediction, groundtruth)[0]



#新的评价方式
class MetricManager(object):
    def __init__(self, metric_fns):
        self.metric_fns = metric_fns
        self.result_dict = defaultdict(float)
        self.num_samples = 0

    def __call__(self, prediction, ground_truth):
        self.num_samples += len(prediction)
        for metric_fn in self.metric_fns:
            for p, gt in zip(prediction, ground_truth):
                res = metric_fn(p, gt)
                dict_key = metric_fn.__name__
                self.result_dict[dict_key] += res

    def get_results(self):
        res_dict = {}
        for key, val in self.result_dict.items():
            res_dict[key] = val / self.num_samples
        return res_dict

    def reset(self):
        self.num_samples = 0
        self.result_dict = defaultdict(float)


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

def AUC_score(SR,GT,threshold=0.5):
    #SR = SR.numpy()
    GT = GT.cpu().numpy().ravel()  # we want to make them into vectors
    SR = SR.ravel()
    fpr, tpr, _ = metrics.roc_curve(GT, SR)
    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc

def dice_score(prediction, groundtruth):
    pflat = prediction.flatten()
    gflat = groundtruth.flatten()
    d = (1 - spatial.distance.dice(pflat, gflat)) * 100.0
    if np.isnan(d):
        return 0.0
    return d


def jaccard_score(prediction, groundtruth):
    pflat = prediction.flatten()
    gflat = groundtruth.flatten()
    return (1 - spatial.distance.jaccard(pflat, gflat)) * 100.0


def intersection_over_union(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP + FN) <= 0.0:
        return 0.0
    return TP / (TP + FP + FN) * 100.0


def precision_score(prediction, groundtruth):
    # PPV
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP) <= 0.0:
        return 0.0

    precision = np.divide(TP, TP + FP)
    return precision * 100.0


def recall_score(prediction, groundtruth):
    # TPR, sensitivity
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FN) <= 0.0:
        return 0.0
    TPR = np.divide(TP, TP + FN)
    return TPR * 100.0

def fdr_score(prediction, groundtruth):

    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    fdr = FP / (FP + TP)
    return  fdr

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


# 计算混淆矩阵，返回相应的dict
def confusion(output, target):
    output = output.double()
    target = target.double()
    output[output > 0.5] = 1
    output[output <= 0.5] = 0
    p = torch.sum(target == 1).item()
    n = torch.sum(target == 0).item()

    tp = (output * target).sum().item()
    tn = ((1 - output) * (1 - target)).sum().item()
    fp = ((1 - target) * output).sum().item()
    fn = ((1 - output) * target).sum().item()
    ebslon = 0.00001
    res = {"P": p, "N": n, "TP": tp, "TN": tn, "FP": fp, "FN": fn, "TPR": (tp / (tp+fn+ebslon)), "TNR": (tn /(tn+fp+ebslon) ), "FPR": (fp / (n+ebslon)),
           "FNR": (fn / (p+ebslon)), "Accuracy": (tp + tn) / (tp+fn+tn+fp+ebslon)}
    return res

# SR : Segmentation Result
# GT : Ground Truth
def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    if SR.size(0) != 1:
        tensor_size = SR.size(0) * SR.size(1)
    else:
        tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)

    acc = float(corr)/float(tensor_size)

    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = ((SR==1)+(GT==1))==2
    FN = ((SR==0)+(GT==1))==2

    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)     
    
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR==0)+(GT==0))==2
    FP = ((SR==1)+(GT==0))==2

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR==1)+(GT==1))==2
    FP = ((SR==1)+(GT==0))==2

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)
    
    Inter = torch.sum((SR+GT)==2)
    Union = torch.sum((SR+GT)>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.5):
    #DC : Dice Coefficient
    # print(SR.shape, GT.shape)
    # SR = SR > threshold
    # GT = GT == torch.max(GT)
    # Inter = torch.sum((SR+GT)==2)
    # DC = float(2*Inter) /(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    # return DC


    pflat = SR.flatten()
    gflat = GT.flatten()
    d = (1 - spatial.distance.dice(pflat, gflat)) * 100.0
    if np.isnan(d):
        return 0.0
    return d



def get_AUC(SR,GT,threshold=0.5):
    SR = SR.numpy().flatten()
    GT = GT.numpy().flatten().astype(int)

    #print("type sr:",GT,SR)
    fpr, tpr, thresholds = metrics.roc_curve(GT, SR, pos_label = 1)
    return metrics.auc(fpr, tpr)
