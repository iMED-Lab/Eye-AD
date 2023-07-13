#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :CNN model.py
@Description :
@Time        :2022/08/28 15:52:31
@Author      :Jinkui Hao
@Version     :1.0
'''
import torch.nn as nn
from torchvision import models 
import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import GraphAttentionLayerBS,UG_GraphAttentionLayerBS
import numpy as np
import random


class EyeADGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,L1_nhid=256, patchNum=3):
        """Batchsize version of GAT."""
        super(EyeADGNN, self).__init__()
        self.nhid = nhid
        self.dropout = dropout
        self.patchNum = patchNum

        self.attentions = [GraphAttentionLayerBS(nhid, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.GNNL1S = GNNL1V2ImpBS(nfeat, nhid, dropout, alpha, nheads, L1_nhid, patchNum)
        self.GNNL1D = GNNL1V2ImpBS(nfeat, nhid, dropout, alpha, nheads, L1_nhid, patchNum)
        self.GNNL1C = GNNL1V2ImpBS(nfeat, nhid, dropout, alpha, nheads, L1_nhid, patchNum)
                
        self.out_att = GraphAttentionLayerBS(nhid*nheads, nhid, dropout=dropout, alpha=alpha, concat=False)

        self.classfier = nn.Linear(nhid*3, nclass)

        self.const = nn.Linear(nhid*2, 128)

        self.classifier1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(128, nclass)
        )

        self.classifier2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(128, nclass)
        )
      

    def forward(self, featureS, featureD, featureC, importance, adjL1, adjL2):

        # importance = importance.reshape((3,9))
        importanceS = importance[:,0,:]
        importanceD = importance[:,1,:]
        importanceC = importance[:,2,:]

        importanceS = torch.cat([torch.diag(importanceS[i,:]).unsqueeze(0) for i in range(importanceS.shape[0])], dim=0)
        importanceD = torch.cat([torch.diag(importanceD[i,:]).unsqueeze(0) for i in range(importanceD.shape[0])], dim=0)
        importanceC = torch.cat([torch.diag(importanceC[i,:]).unsqueeze(0) for i in range(importanceC.shape[0])], dim=0)

        outS = self.GNNL1S(featureS,adjL1,importanceS)
        outD = self.GNNL1D(featureD,adjL1,importanceD)
        outC = self.GNNL1C(featureC,adjL1,importanceC)

        x = torch.cat([outS.unsqueeze(1), outD.unsqueeze(1), outC.unsqueeze(1)], 1)

        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adjL2) for att in self.attentions], dim=2)

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adjL2))
        outT = torch.cat([x[:,i,:] for i in range(x.shape[1])],1)
        # x = x.unsqueeze(0)
        out = self.classfier(outT)

        randNum1 = random.randint(0,2)
        randNum2 = random.randint(0,2)
        while randNum2 == randNum1:
            randNum2 = random.randint(0,2)
        const1 = self.const(torch.cat([x[:,randNum1,:],x[:,randNum1,:]], dim=1))
        const2 = self.const(torch.cat([x[:,randNum2,:],x[:,randNum2,:]], dim=1))

        out1 = self.classifier1(const1)
        out2 = self.classifier2(const2)

        return out, const1, const2, out1, out2, outT



class EyeADCNN(nn.Module):
    '''
    基于V2改进, 主要修改ICM
    '''
    def __init__(self, in_channels, num_classes, patch_size = 3):
        super(EyeADCNN, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)

        self.conv1_2 = nn.Conv2d(in_channels, 64, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.conv1_3 = nn.Conv2d(in_channels, 64, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(64)

        self.encoder1 = ExtractorShareV3(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes, patch_size = patch_size)
        self.encoder2 = ExtractorShareV3(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes, patch_size = patch_size)
        self.encoder3 = ExtractorShareV3(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes, patch_size = patch_size)
       
        self.linear1 = nn.Linear(512 * 3, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU(512)
        self.classifier = nn.Linear(512, num_classes)

        self.ICM = ICM(patch_size)


    def forward(self,input):

        input1 = F.relu(self.bn1_1(self.conv1_1(input[:, 0, :, :].unsqueeze(1))))
        input2 = F.relu(self.bn1_2(self.conv1_2(input[:, 1, :, :].unsqueeze(1))))
        input3 = F.relu(self.bn1_3(self.conv1_3(input[:, 2, :, :].unsqueeze(1))))

        out1_l1, out1_l2, out1_l3, out1_l4, out1 = self.encoder1(input1)
        out2_l1, out2_l2, out2_l3, out2_l4, out2 = self.encoder2(input2)
        out3_l1, out3_l2, out3_l3, out3_l4, out3 = self.encoder3(input3)

        gate = self.ICM((torch.cat([out1_l1,out2_l1,out3_l1],dim=1), 
                        torch.cat([out1_l2,out2_l2,out3_l2],dim=1),
                        torch.cat([out1_l3,out2_l3,out3_l3],dim=1),
                        torch.cat([out1_l4,out2_l4,out3_l4],dim=1))
                        )
        
        outS = F.adaptive_avg_pool2d(gate[:,0,:,:].unsqueeze(1)*out1 , (1,1)) 
        outD = F.adaptive_avg_pool2d(gate[:,1,:,:].unsqueeze(1)*out2 , (1,1)) 
        outC = F.adaptive_avg_pool2d(gate[:,2,:,:].unsqueeze(1)*out3 , (1,1)) 

        outS = outS.view(outS.size(0), -1)
        outD = outD.view(outD.size(0), -1)
        outC = outC.view(outC.size(0), -1)
        
        feature = torch.cat([outS, outD, outC],1)
        feature = self.linear1(feature)
        output = self.bn1(feature)
        output = self.relu1(output)
        output = self.classifier(output)

        return output, out1, out2, out3, gate
        
class GNNL1V2ImpBS(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads,L1_nhid = 256,patchNum = 3):
        """Dense version of GAT."""
        super(GNNL1V2ImpBS, self).__init__()
        self.dropout = dropout

        # self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        self.attentions = [UG_GraphAttentionLayerBS(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayerBS(nhid * nheads, L1_nhid, dropout=dropout, alpha=alpha, concat=False)

        self.linear = nn.Linear(L1_nhid * patchNum*patchNum, nhid)

    def forward(self, x, adj, importance):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj,importance) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        x = torch.cat([x[:,i,:] for i in range(x.shape[1])],1)
        # x = x.unsqueeze(0)
        x = self.linear(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ExtractorShareV3(nn.Module):

    def __init__(self, block, num_blocks, in_channels=1, num_classes=2, patch_size = 3):
        super(ExtractorShareV3, self).__init__()
        self.in_planes = 64

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((patch_size, patch_size))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = self.avgpool(out4)

        return out1, out2, out3, out4, out

class threeBranch(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(threeBranch, self).__init__()

        self.encoder1 = Extractor(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes)
        self.encoder2 = Extractor(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes)
        self.encoder3 = Extractor(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes)

        self.linear1 = nn.Linear(512 * 3, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU(512)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self,input):
        out1 = self.encoder1(input[:, 0, :, :].unsqueeze(1))
        out2 = self.encoder2(input[:, 1, :, :].unsqueeze(1))
        out3 = self.encoder3(input[:, 2, :, :].unsqueeze(1))

        feature = torch.cat([out1, out2, out3],1)
        feature = self.linear1(feature)
        output = self.bn1(feature)
        output = self.relu1(output)
        output = self.classifier(output)

        return output
        
     
class ICM(nn.Module):
    def __init__(self, patchSze):
        super(ICM, self).__init__()

        self.layer1 = nn.Conv2d(64*3, 256, 1, 1, 0)
        self.layer2 = nn.Conv2d(128*3, 256, 1, 1, 0)
        self.layer3 = nn.Conv2d(256*3, 256, 1, 1, 0)
        self.layer4 = nn.Conv2d(512*3, 256, 1, 1, 0)

        self.conv_instance = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.AdaptiveAvgPool2d((patchSze, patchSze)),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            # nn.Softmax(dim=1)
            nn.Sigmoid()
        )

        # self.conv_region = nn.Sequential(
        #     nn.Conv2d(256, 64, 3, padding=1),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
        #     nn.Softmax()
        #     # nn.Sigmoid()
        # )
    
    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W)) + y

    def forward(self, inputs):
        input1, input2, input3, input4 = inputs
        p1 = self.layer1(input1)
        p2 = self._upsample_add(p1, self.layer2(input2))
        p3 = self._upsample_add(p2, self.layer3(input3))
        p4 = self._upsample_add(p3, self.layer4(input4))

        region = self.conv_instance(p4)
        return region


