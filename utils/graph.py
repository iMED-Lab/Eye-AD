#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :graph.py
@Description : 用于图数据处理的相关函数
@Time        :2020/10/12 10:36:45
@Author      :Jinkui Hao
@Version     :1.0
'''
import csv
import numpy as np

def generateAdj():
    #根据excel生成邻接矩阵,无向图
    pointNum = 3
    Triangle_path = 'utils/triangle_level2.csv'
    f = open(Triangle_path,'r')
    csv_reader = csv.reader(f)
    adjMetrix = np.zeros([pointNum,pointNum])
    i = 0
    for triangle in csv_reader:
        #每个三角3个
        v1 = int(triangle[0])
        v2 = int(triangle[1])
        v3 = int(triangle[2])
        adjMetrix[v1,v2] = 1
        adjMetrix[v1,v3] = 1
        adjMetrix[v2,v3] = 1
        adjMetrix[v2,v1] = 1
        adjMetrix[v3,v1] = 1
        adjMetrix[v3,v2] = 1
        i = i+1
        if i == pointNum:
            break
    print(adjMetrix)
    np.save("utils/adjMetrix_level2.npy", adjMetrix)
    print('done~')

def generateAdj_direct():
    #根据excel生成邻接矩阵，有向图
    pointNum = 40
    Triangle_path = 'utils/triangle_40.csv'
    f = open(Triangle_path,'r')
    csv_reader = csv.reader(f)
    adjMetrix = np.zeros([pointNum+1,pointNum+1])
    i = 0
    for triangle in csv_reader:
        #每个三角3个
        v1 = int(triangle[0])
        v2 = int(triangle[1])
        v3 = int(triangle[2])
        adjMetrix[v1,v2] = 1
        adjMetrix[v1,v3] = 1
        adjMetrix[v2,v3] = 1
        # adjMetrix[v2,v1] = 1
        # adjMetrix[v3,v1] = 1
        adjMetrix[v3,v2] = 1
        i = i+1
        if i == pointNum:
            break
    np.save("utils/adjMetrix_direct_40.npy", adjMetrix)
    print('done~')

if __name__ == '__main__':
    generateAdj()