import visdom
from torchvision import transforms
import time
import cv2
import numpy as np
from PIL import Image
import torch
import glob
from pathlib import Path
import os
import random
import torch.utils.data as data
import pandas as pd


def minEnclosingCircle(img_src):
    contours, _ = cv2.findContours(img_src, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    center, radius = cv2.minEnclosingCircle(contours[-1])
    center = np.int0(center)

    # 5.绘制最小外接圆
    img_result = img_src.copy()
    cv2.circle(img_result, tuple(center), int(radius), (255, 255, 255), 2)

    return img_result, center, int(radius)


def minEnclosingCircle(img_src):
    contours, _ = cv2.findContours(img_src, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    center, radius = cv2.minEnclosingCircle(contours[len(contours) - 1])
    center = np.int0(center)

    # 5.绘制最小外接圆
    img_result = img_src.copy()
    cv2.circle(img_result, tuple(center), int(radius), (255, 255, 255), 2)

    return img_result, center, int(radius)


# 提取最大联通区域
def extract_maximum_connected_area(mat, threshold: int = 110):
    contours, _ = cv2.findContours(mat, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))

    max_idx = np.argmax(area)
    for k in range(len(contours)):
        if k != max_idx:
            cv2.fillPoly(mat, [contours[k]], 0)

    _, mat = cv2.threshold(mat, threshold, 255, cv2.THRESH_BINARY)
    return mat


def tensor2array(tensor):
    array1 = tensor.cpu().detach().numpy()  #将tensor数据转为numpy数据
    maxValue = array1.max()
    array1 = array1 * 255 / maxValue  #normalize，将图像数据扩展到[0,255]
    mat = np.uint8(array1)  #float32-->uint8
    mat = mat.transpose(1, 2, 0)
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    return mat


def array2image(array, mode: str = "L"):
    return Image.fromarray(array, mode=mode)


def array2tensor(array):
    return torch.Tensor(array)


# 横向拼接array
def arrayHStack(array_li: list):
    return np.concatenate(array_li, axis=1)


# 纵向拼接array
def arrayVStack(array_li: list):
    return np.concatenate(array_li, axis=0)


class Visualizer(object):
    """
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    或者`self.function`调用原生的visdom接口
    比如
    self.text('hello visdom')
    self.histogram(t.randn(1000))
    self.line(t.arange(0, 10),t.arange(1, 11))
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        # 画的第几个数，相当于横坐标
        # 比如（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name, value) i.e. ('loss', 0.11)
        """
        for k, v in d.iteritems():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        # self.plot('loss', 1.00)

        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]), win=name, opts=dict(title=name), update=None if x == 0 else 'append', **kwargs)
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img', t.Tensor(64, 64))
        self.img('input_imgs', t.Tensor(3, 64, 64))
        self.img('input_imgs', t.Tensor(100, 1, 64, 64))
        self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)
        """
        self.vis.images(img_, win=name, opts=dict(title=name), **kwargs)

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1, 'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        """
        self.function 等价于self.vis.function
        自定义的plot,image,log,plot_many等除外
        """
        return getattr(self.vis, name)


class FetchAllData():

    def __init__(self, root_path: str):

        self.root_path = root_path

        self.NameAndLabel = self.make_dataset()

    def make_dataset(self):
        tmp = [str(Path(x).parent) for x in glob.glob(self.root_path + "/**/*.png", recursive=True) if "3x3" in x or "3_3" in x]

        tmp = list(set(tmp))

        ret_li = []
        for t in tmp:
            svc_li = [x for x in glob.glob(t + "/**/*.*", recursive=True) if "浅层血管复合体" in x]
            dvc_li = [x for x in glob.glob(t + "/**/*.*", recursive=True) if "深层血管复合体" in x]
            chc_li = [x for x in glob.glob(t + "/**/*.*", recursive=True) if "脉络膜毛细血管层" in x]

            if len(svc_li) == 0 and len(dvc_li) == 0:
                continue
            svc_ = svc_li[0] if len(svc_li) >= 1 else dvc_li[0]
            dvc_ = dvc_li[0] if len(dvc_li) >= 1 else svc_li[0]
            chc_ = chc_li[0] if len(chc_li) >= 1 else svc_li[0]

            ret_li.append([svc_, dvc_, chc_])

        return ret_li

    def fetch(self):
        return self.NameAndLabel


class KVLayersDataset(data.Dataset):

    def __init__(self, data_path, csv_path, layers, resize_to, is_training=True, is_cat=False):
        self.layers = layers
        self.is_training = is_training
        self.csv_path = csv_path
        self.data_path = data_path
        self.NameAndLabel = self.make_dataset()
        # self.imgTransform = transforms.Compose([
        #     transforms.Resize((self.resize_to, self.resize_to)),
        #     transforms.ToTensor(),
        # ])

    # 读取数据集
    def make_dataset(self):
        res = []
        csv_path = os.path.join(self.csv_path, "train.csv") if self.is_training else os.path.join(self.csv_path, "test.csv")
        data = pd.read_csv(csv_path, encoding='utf-8', header=None).values
        for i in data:
            path = os.path.join(self.data_path, i[0], i[1], i[2])
            aim = [x for x in glob.glob(path + "/*.*", recursive=False)]
            aim.sort(reverse=True)
            tmp = []
            for l in self.layers:
                for t in aim:
                    if l in t:
                        tmp.append(t)
                        break
            if len(tmp) == len(self.layers):
                tmp.append(i[3])
                res.append(tmp)

        print(len(res))
        random.shuffle(res)
        ct = ad = 0
        for i in res:
            if i[3] == 1:
                ad += 1
            else:
                ct += 1

        print(ad, ct)
        return res
