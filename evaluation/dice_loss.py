import torch
from torch.autograd import Function, Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



# def make_one_hot(input, num_classes):
#     """Convert class index tensor to one hot encoding tensor.
#     Args:
#          input: A tensor of shape [N, 1, *]
#          num_classes: An int of number of class
#     Returns:
#         A tensor of shape [N, num_classes, *]
#     """
#     shape = np.array(input.shape)
#     shape[1] = num_classes
#     shape = tuple(shape)
#     result = torch.zeros(shape)
#     result = result.scatter_(1, input.cpu(), 1)
#
#     return result
#
#
# class BinaryDiceLoss(nn.Module):
#     """Dice loss of binary class
#     Args:
#         smooth: A float number to smooth loss, and avoid NaN error, default: 1
#         p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
#         predict: A tensor of shape [N, *]
#         target: A tensor of shape same with predict
#     Returns:
#         Loss tensor according to arg reduction
#     Raise:
#         Exception if unexpected reduction
#     """
#     def __init__(self, smooth=1, p=2):
#         super(BinaryDiceLoss, self).__init__()
#         self.smooth = smooth
#         self.p = p
#
#     def forward(self, predict, target):
#         assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
#         predict = predict.contiguous().view(predict.shape[0], -1)
#         target = target.contiguous().view(target.shape[0], -1)
#
#         num = torch.sum(torch.mul(predict, target))*2 + self.smooth
#         den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth
#
#         dice = num / den
#         loss = 1 - dice
#         return loss
#
# class DiceLoss(nn.Module):
#     """Dice loss, need one hot encode input
#     Args:
#         weight: An array of shape [num_classes,]
#         ignore_index: class index to ignore
#         predict: A tensor of shape [N, C, *]
#         target: A tensor of same shape with predict
#         other args pass to BinaryDiceLoss
#     Return:
#         same as BinaryDiceLoss
#     """
#     def __init__(self, weight=None, ignore_index=None, **kwargs):
#         super(DiceLoss, self).__init__()
#         self.kwargs = kwargs
#         self.weight = weight
#         self.ignore_index = ignore_index
#
#     def forward(self, predict, target):
#         predict = make_one_hot(predict,2)
#         assert predict.shape == target.shape, 'predict & target shape do not match'
#         dice = BinaryDiceLoss(**self.kwargs)
#         total_loss = 0
#         predict = F.softmax(predict, dim=1)
#
#         for i in range(target.shape[1]):
#             if i != self.ignore_index:
#                 dice_loss = dice(predict[:, i], target[:, i])
#                 if self.weight is not None:
#                     assert self.weight.shape[0] == target.shape[1], \
#                         'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
#                     dice_loss *= self.weights[i]
#                 total_loss += dice_loss
#
#         return total_loss/target.shape[1]

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        #target = _make_one_hot(target, 2)
        self.save_for_backward(input, target)
        eps = 0.0001
        #dot是返回两个矩阵的点集
        #inter,uniun:两个值的大小分别是10506.6,164867.2
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
        #print("inter,uniun:",self.inter,self.union)

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None


        #这里没有打印出来，难道没有执行到这里吗
        #print("grad_input, grad_target:",grad_input, grad_target)

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    #print("size of input, target:", input.shape, target.shape)

    for i, c in enumerate(zip(input, target)):
        #c[0],c[1]的大小都是原图大小torch.Size([1, 576, 544])
        #print("size of c0 c1:", c[0].shape,c[1].shape)
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def dice_coeff_loss(input, target):

    return 1 - dice_coeff(input, target)