import os
import random
import time
import numpy as np
import torch
from torch import nn
from pathlib import Path
from glob import glob
import warnings
from torchvision import transforms
from PIL import Image
from pathlib import Path
from GAT.models import *
from models.model import threeBranchShareV2
import scipy.sparse as sp
from utils.tools  import generateAdj
from conf import config 
from utils.confuseMetrix import *
from utils.matrix import claMetrix


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv) #2708*2708
    mx = r_mat_inv.dot(mx)
    return mx 



class Test(object):

    def __init__(self, data_path: str):
        self.resize_to = 384
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnnpath = '.pth'
        gnnpath = '.pth'

        self.model_CNN = torch.load(cnnpath)
        self.model_GNN = torch.load(gnnpath)
        self.file_paths = self.__prepare1(data_path)

        self.imgTransform = transforms.Compose([
            transforms.Resize((self.resize_to, self.resize_to)),
            transforms.ToTensor(),
        ])
        self.pad_trans = transforms.Pad(35)


    def test_IMIG(self, model_path: str):

            # adjMetrixL1 = np.load('utils/adjMetrix_level1.npy')
        adjMetrixL1 = generateAdj(config.patchSize)
        #adjMetrix = np.load('utils/adjMetrix_30.npy')
        adjMetrixL1 = torch.from_numpy(adjMetrixL1)
        adjMetrixL1 = sp.coo_matrix(adjMetrixL1)
        adjMetrixL1 = normalize(adjMetrixL1 + sp.eye(adjMetrixL1.shape[0])) 
        adjMetrixL1 = adjMetrixL1.todense()
        adjMetrixL1 = torch.from_numpy(adjMetrixL1)
        adjMetrixL1 = adjMetrixL1.float()
        adjMetrixL1 = adjMetrixL1.cuda()

        adjMetrixL2 = np.load('utils/adjMetrix_level2.npy')
        #adjMetrix = np.load('utils/adjMetrix_30.npy')
        adjMetrixL2 = torch.from_numpy(adjMetrixL2)
        adjMetrixL2 = sp.coo_matrix(adjMetrixL2)
        adjMetrixL2 = normalize(adjMetrixL2 + sp.eye(adjMetrixL2.shape[0])) 
        adjMetrixL2 = adjMetrixL2.todense()
        adjMetrixL2 = torch.from_numpy(adjMetrixL2)
        adjMetrixL2 = adjMetrixL2.float()
        adjMetrixL2 = adjMetrixL2.cuda()

        modelCNN: nn.Module = self.model_CNN
        modelGNN: nn.Module = self.model_GNN

        modelCNN.cuda()
        modelCNN.eval()

        modelGNN.cuda()
        modelGNN.eval()

        step = 0
        for x in self.file_paths:
            step += 1
            input_1, input_2, input_3, label, name = self.__read(x)
            input_1, input_2, input_3 = input_1.cuda().unsqueeze(0), input_2.cuda().unsqueeze(0), input_3.cuda().unsqueeze(0)
            label = torch.tensor(label, dtype=torch.int64).unsqueeze(0)
            with torch.no_grad():
                # output = model(input_1, input_2, input_3)
                outputsCNN, featureS, featureD, featureC, importance = modelCNN(torch.cat([input_1, input_2, input_3], dim=1)) 
                featureS, featureD, featureC, importance = featureS.squeeze(), featureD.squeeze(), featureC.squeeze(), importance.squeeze()
                featureS = featureS.reshape(config.batchSize_test,512,config.patchSize*config.patchSize).transpose(1,2)
                featureD = featureD.reshape(config.batchSize_test,512,config.patchSize*config.patchSize).transpose(1,2)
                featureC = featureC.reshape(config.batchSize_test,512,config.patchSize*config.patchSize).transpose(1,2)

                importance = importance.reshape(config.batchSize_test,3,config.patchSize*config.patchSize)

                output, _, _, _, _= modelGNN(featureS, featureD, featureC, importance, adjMetrixL1, adjMetrixL2)

                output = nn.Softmax(dim=1)(output)

            if step == 1:
                predictionAll = output
                GTAll = label
            else:
                predictionAll = torch.cat([predictionAll, output], dim=0)
                GTAll = torch.cat([GTAll, label], dim=0)

        return claMetrix(predictionAll, GTAll, num_class=2)


    def __prepare1(self, data_path: str):
        all_png = [x for x in glob(data_path + "/**/*.*", recursive=True) if "png" in x and "3_3" in x]
        all_parent = [str(Path(x).parent) for x in all_png]
        all_parent = list(set(all_parent))

        res = []
        for x in all_parent:
            label = 0 if "control" in x else 1
            files = [x for x in glob(x + "/*.png") if "enface_304x304.png" in x or "deep.png" in x or ("choriocapillaris.png" in x or "choroid.png" in x)]
            if len(files) == 3:
                tmp = []
                tmp.append(self.__filter(files, "enface_304x304.png"))
                tmp.append(self.__filter(files, "deep.png"))
                tmp.append(self.__filter(files, "chor"))
                tmp.append(label)
                res.append(tmp)

        # random.shuffle(res)
        return res

    def __prepare(self, data_path: str):
        all_png = [x for x in glob(data_path + "/**/*.*", recursive=True) if "png" in x and "3x3" in x]
        all_parent = [str(Path(x).parent) for x in all_png]
        all_parent = list(set(all_parent))

        res = []
        for x in all_parent:
            label = 0 if "control" in x else 1
            files = [x for x in glob(x + "/*.png") if "浅层血管复合体" in x or "深层血管复合体" in x or  "脉络膜毛细血管层" in x]
            if len(files) == 3:
                tmp = []
                tmp.append(self.__filter(files, "浅层血管复合体"))
                tmp.append(self.__filter(files, "深层血管复合体"))
                tmp.append(self.__filter(files, "脉络膜毛细血管层"))
                tmp.append(label)
                res.append(tmp)

        random.shuffle(res)
        return res

    def __filter(self, file_li: list, filter: str):
        for i in file_li:
            if filter in i:
                return i

    def __read(self, x: str):
        images = [self.pad_trans(Image.open(v).convert("L")) for v in x[:-1]]
        tmages = [self.imgTransform(x) for x in images]
        label = x[-1]
        name = x[0].split('/')[-3]

        return *tmages, int(label), name


warnings.filterwarnings("ignore")

if __name__ == '__main__':
    data_path = "..."
    t = Test(data_path=data_path)

    ACC, AUC, RE, PRE, F1, kappa, confMtrix = t.test_IMIG("save/LateFusion_32_['浅层血管复合体', '深层血管复合体', '脉络膜毛细血管层']/Model-fold-2-First-acc.pth")
    print("最优模型 ACC={} AUC={} Kappa={}".format(ACC, AUC, kappa))
    # confuseMtrix("IMIG", confMtrix, num_class=2)
    plot_confusion_matrix('IMIG.jpg', y_test=confMtrix[0],predictions=confMtrix[1], classes=['Control','AD'], cmap='autumn_r')
