'''
Descripttion: 
version: 
Author: Kevin
Date: 2023-06-01 10:24:51
LastEditors: Kevin
LastEditTime: 2023-06-01 10:24:51
'''
import os
import torch
from PIL import Image
from torch import Tensor
import torch.nn as nn
from models.ourModel import VAFFNet
from torchvision import transforms


# FAZ分割接口封装
class FAZ_SEG():

    def __init__(self, parameter_path: str):
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # if isinstance(self.model, torch.nn.DataParallel):
        #     self.model = self.model.module

        self.model: nn.Module = VAFFNet()
        # self.model = torch.nn.DataParallel(self.model).cuda()
        # # self.model = self.model.module
        # # p = torch.load(parameter_path)
        self.model.load_state_dict(torch.load(parameter_path))

        # self.model = torch.load(parameter_path, map_location="cpu")

        self.model = self.model.to(self.device)
        # self.model.eval()

    def classify(self, images: list):
        image0 = images[0].to(self.device)
        image1 = images[1].to(self.device)
        image2 = images[2].to(self.device)
        pred_FAZ, pred_Vess, _, _ = self.model(image0, image1, image2)
        return pred_FAZ.squeeze(0), pred_Vess.squeeze(0), image0.squeeze(0)

    def prepare(self, image_li: list):
        imgTransform = transforms.Compose([
            transforms.Resize((304, 304)),
            transforms.ToTensor(),
        ])

        images = [imgTransform(x).unsqueeze(0) for x in image_li]
        return images

    def do(self, input: list):
        inputs = [Image.open(x).convert("RGB") for x in input]
        inputs = self.prepare(inputs)

        return self.classify(inputs)
