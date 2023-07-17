from email import header

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :
@Description : Eye-AD triaing and testing 
@Time        :2022/08/16 13:57:35
@Author      :Jinkui Hao
@Version     :1.0
'''

from conf import config 
import torch
from utils.WarmUpLR import WarmupLR
from utils.Visualizer import Visualizer
import os
import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader
from torch import optim, nn
import csv
from evaluation.matrixs import *
from utils.tools  import generateAdj, generateAdjL2
from utils.confuseMetrix import *
import random
import logging
from logging import handlers
from utils.matrix import claMetrix
from models.model import EyeADCNN,EyeADGNN
import collections

# set seed
GLOBAL_SEED = 1

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

GLOBAL_WORKER_ID = None

def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv) #2708*2708
    mx = r_mat_inv.dot(mx)
    return mx 

def sve_as_fig(hist,name, show=False, path='Train_hist.png'):
    plt.cla()
    x = range(len(hist))

    plt.plot(x, hist, label=name)

    plt.xlabel('Iter')
    plt.ylabel('Value')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(path)


class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }     

    def __init__(self,filename,level='info',backCount=10,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)                 
        self.logger.setLevel(self.level_relations.get(level))
        
        sh = logging.StreamHandler()  
        sh.setFormatter(format_str)   
        self.logger.addHandler(sh)    
        
        fh = handlers.RotatingFileHandler(filename=filename,maxBytes=10485760,backupCount=backCount)   
        fh.setLevel(self.level_relations.get(level))
        fh.setFormatter(format_str)   
        self.logger.addHandler(fh)


def get_lr(optier):
    for param_group in optier.param_groups:
        return param_group['lr']

def test(modelGNN, modelCNN, data_loaderTest, adjL1, adjL2, epoch, preTrain = False):
    
    print('Testing...')
    modelGNN.eval()
    modelCNN.eval()
    test_loss = 0
    correct = 0
    predictionAll = []
    GTAll = []
    step = 0
    if preTrain == True:
        csvPath = os.path.join(config.savePath, 'GNNPre-fold-{}-state-{}-Result.csv'.format(config.fold,epoch+1))
    else:
        csvPath = os.path.join(config.savePath, 'Joint-fold-{}-state-{}-Result.csv'.format(config.fold,epoch+1))
    csvFile = open(csvPath, "w")
    with torch.no_grad(): 
        
        for item in data_loaderTest:
            step += 1
            
            inputs, labels, name = item
            labels = labels.to(device)
            inputs = inputs.to(device)
            outputsCNN, featureS, featureD, featureC, importance = modelCNN(inputs) 
            importanceL1 = importance
            featureS, featureD, featureC, importanceL1 = featureS.squeeze(), featureD.squeeze(), featureC.squeeze(), importanceL1.squeeze()#, importanceL2.squeeze()
            featureS = featureS.reshape(config.batchSize_test,512,config.patchSize*config.patchSize).transpose(1,2)
            featureD = featureD.reshape(config.batchSize_test,512,config.patchSize*config.patchSize).transpose(1,2)
            featureC = featureC.reshape(config.batchSize_test,512,config.patchSize*config.patchSize).transpose(1,2)

            importanceL1 = importanceL1.reshape(config.batchSize_test,3,config.patchSize*config.patchSize)

            output, _, _, _, _,_ = modelGNN(featureS, featureD, featureC, importanceL1, adjL1, adjL2)

            output = nn.Softmax(dim=1)(output)
            
            if step == 1:
                predictionAll = output
                GTAll = labels
            else:
                predictionAll = torch.cat([predictionAll,output], dim=0)
                GTAll = torch.cat([GTAll, labels], dim=0)

            value = output[:,1]
            threshhold = 0.5
            #大于 threshhold
            zero = torch.zeros_like(value)
            one = torch.ones_like(value)
            pred = torch.where(value > threshhold, one, zero)

            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # target = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(labels.view_as(pred)).sum().item()
            print(config.saveName,"%d/%d" % (step, (len(test_loader.dataset) - 1) // test_loader.batch_size + 1))

            writer = csv.writer(csvFile)
            value, predicted = value.cpu().detach().numpy(), pred.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            data = [name[0], value[0], predicted[0], labels[0]]
            writer.writerow(data)

    test_loss /= step
    Acc, AUC, RE, PRE, f1, kappa, confMtrix = claMetrix(predictionAll,GTAll,num_class=2)

    print('Acc:%04f, AUC:%04f, Recall:%04f, Precious:%04f, F1:%04f, Kappa:%04f'%(Acc, AUC, RE, PRE, f1, kappa))
    #print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
        #100. * correct / len(test_loader.dataset)))
    modelGNN.train()
    modelCNN.train()
    csvFile.close()

    return Acc, AUC, RE, PRE, f1, kappa, (GTAll.cpu().numpy(), np.argmax(predictionAll.cpu().numpy(),1))


def CNN_pretrain(vis, model,dataloader_train,dataloader_test,save_dir):
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr_cnn_pre, momentum=0.99)
    criterion = nn.CrossEntropyLoss()
    models = []

    scheduler_steplr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.num_epochs_cnn_pre)
    schedulers = WarmupLR(scheduler_steplr, init_lr=1e-7, num_warmup=5, warmup_strategy='cos')

    best_metrix = 0.0
    best_auc = 0
    best_model = model

    figACC = []
    figAUC = []
    figKappa = []

    csvPath = os.path.join(config.savePath, 'CNNPre-fold-{}-ACC.csv'.format(config.fold))
    csvFile = open(csvPath, "w")

    for epoch in range(config.num_epochs_cnn_pre):
        schedulers.step(epoch)
        epoch_loss = 0
        step = 0

        for item in dataloader_train:
            step += 1
            optimizer.zero_grad()

            inputs, labels, _ = item
            labels = labels.to(device)
            inputs = inputs.to(device)
            outputs, outS, outD, outC, gate = model(inputs)   

            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if  (step-1) % int(len(dataloader_train)/8)  == 0:
                print('-' * 10)
                print(config.saveName,"%d/%d,train_loss:%0.4f" % (step, (len(dataloader_train.dataset) - 1) // dataloader_train.batch_size + 1, epoch_loss/(step+1)))
                
        print('Epoch %d/%d' % (epoch+1, config.num_epochs_cnn_pre))
        current_lr = get_lr(optimizer)

        if epoch > config.num_epochs_cnn_pre*0.75:  # early stop
            break

        # if epoch > epoch*0.5 and epoch%5 == 0:
        # if epoch > epoch*0.5 and epoch%5 == 0:
        models.append(model)

    worker_state_dict=[x.state_dict() for x in models]
    weight_keys=list(worker_state_dict[0].keys())
    fed_state_dict=collections.OrderedDict()
    for key in weight_keys:
        key_sum=0
        for i in range(len(models)):
            key_sum=key_sum+worker_state_dict[i][key]
        fed_state_dict[key]=key_sum/len(models)
    #### update fed weights to fl model
    best_model.load_state_dict(fed_state_dict)
    # best_model = model  
    save_path = os.path.join(save_dir, 'CNNLast-{:.4f}-{:.4f}-{:.4f}.pth'.format(Acc,AUC,kappa))
    torch.save(best_model, save_path)
    save_path = os.path.join(save_dir, 'Acc_cnn.png')
    sve_as_fig(figACC, 'Acc', path=save_path)

    save_path = os.path.join(save_dir, 'AUC_cnn.png')
    sve_as_fig(figAUC, 'AUC', path=save_path)

    save_path = os.path.join(save_dir, 'kappa_cnn.png')
    sve_as_fig(figKappa, 'kappa', path=save_path)
    csvFile.close()

    return best_model


def GNN_pretrain(modelGNN, modelCNN, data_loaderTrain,dataloader_test, adjL1, adjL2):
    # Create Optimizer
    optimizer = optim.Adam(modelGNN.parameters(), lr = config.lr_gnn_pre, weight_decay=config.weight_decay)

    criterion = nn.CrossEntropyLoss().to(device)

    # criterion = torch.nn.CrossEntropyLoss()
    criterion_const = torch.nn.MSELoss()

    scheduler_steplr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.num_epochs_gnn_pre)
    schedulers = WarmupLR(scheduler_steplr, init_lr=1e-7, num_warmup=5, warmup_strategy='cos')

    # meters to record stats on learning
    #train_loss = AverageValueMeter()
    best_metrix = 0.0
    best_auc = 0
    best_model = modelGNN


    figACC = []
    figAUC = []
    figKappa = []
    # Train model on the dataset
    for epoch in range(config.num_epochs_gnn_pre):
        schedulers.step(epoch)
        print('Epoch %d/%d' % (epoch, config.num_epochs_gnn_pre - 1))
        print('-' * 10)
        runing_loss = 0.0
        modelGNN.train(mode=True)

        for i, data in enumerate(data_loaderTrain, 0):
            inputs, labels, _ = data
            labels = labels.to(device)
            inputs = inputs.to(device)
            with torch.no_grad():
                outputsCNN, featureS, featureD, featureC, importance = modelCNN(inputs) 

            featureS, featureD, featureC, importance = featureS.squeeze(), featureD.squeeze(), featureC.squeeze(), importance.squeeze()
            featureS = featureS.reshape(config.batchSize_joint,512,config.patchSize*config.patchSize).transpose(1,2)
            featureD = featureD.reshape(config.batchSize_joint,512,config.patchSize*config.patchSize).transpose(1,2)
            featureC = featureC.reshape(config.batchSize_joint,512,config.patchSize*config.patchSize).transpose(1,2)

            importance = importance.reshape(config.batchSize_joint,3,config.patchSize*config.patchSize)

            optimizer.zero_grad()

            outputs, const1, const2, out1, out2, _ = modelGNN(featureS, featureD, featureC, importance, adjL1, adjL2)
            
            loss = criterion(outputs, labels) + config.lossWeight1*criterion_const(const1, const2) + config.lossWeight2*(criterion(out2, labels) + criterion(out1, labels))

            loss.backward()
            optimizer.step()
            runing_loss += loss.item()
            if  (i-1) % int(len(data_loaderTrain)/8)  == 0:
                print('-' * 10)
                print(config.saveName,"%d/%d,train_loss:%0.4f" % (i, (len(data_loaderTrain.dataset) - 1) // data_loaderTrain.batch_size + 1, runing_loss/(i+1)))

        print('Epoch %d/%d' % (epoch+1, config.num_epochs_gnn_pre))
        current_lr = get_lr(optimizer)

        Acc, AUC, RE, PRE, f1, kappa,conf = test(modelGNN, modelCNN, dataloader_test, adjL1, adjL2, epoch, preTrain=True)

        figACC.append(Acc)
        figAUC.append(AUC)
        figKappa.append(kappa)


        if (Acc > best_metrix or AUC > best_auc) and epoch > 2:
            # conf_matrix = pd.DataFrame(conf, index=range(config.num_class), columns=range(config.num_class))
            conf_matrix = conf
            cmap = 'Blues'
            save_path = os.path.join(save_dir, 'GNNPre-{}-{:.4f}-{:.4f}-{:.4f}.jpg'.format(epoch + 1,Acc,AUC,kappa))
            plot_confusion_matrix(save_path, y_test=conf_matrix[0],predictions=conf_matrix[1], classes=['Control','AD'], cmap=cmap)

            save_path = os.path.join(save_dir, 'GNNPre-{}-{}.pth'.format(epoch + 1,Acc))
            torch.save(modelGNN, save_path)
            if Acc > best_metrix:
                best_metrix = Acc
                
            else:
                best_auc = AUC
            # best_model = model
        if epoch > 140:
            break # early stop

    best_model = modelGNN
    save_path = os.path.join(save_dir, 'Acc_gnn.png')
    sve_as_fig(figACC, 'Acc', path=save_path)

    save_path = os.path.join(save_dir, 'AUC_gnn.png')
    sve_as_fig(figAUC, 'AUC', path=save_path)

    save_path = os.path.join(save_dir, 'kappa_gnn.png')
    sve_as_fig(figKappa, 'kappa', path=save_path)

    return best_model



def jointTrain(modelGNN, modelCNN, data_loaderTrain,dataloader_test, adjL1, adjL2):
    lrate = config.base_lr
    optimizer = optim.Adam(modelGNN.parameters(), lr = lrate)

    criterion = torch.nn.CrossEntropyLoss()
    criterion_const = torch.nn.MSELoss()

    schedulers = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.num_epochs)

    name_list = ['linear1', 'bn1','relu1','classifier']
    for name, value in modelCNN.named_parameters():
        if name in name_list:
            value.requires_grad = False
    parems = filter(lambda p:p.requires_grad, modelCNN.parameters())

    optimizer_CNN = torch.optim.SGD(parems, lr=config.base_lr_CNN, momentum=0.9)
    schedulers_CNN = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_CNN,T_max=config.num_epochs)

    # meters to record stats on learning 
    #train_loss = AverageValueMeter()
    best_metrix = 0.0
    best_auc = 0

    figACC = []
    figAUC = []
    figF1 = []
    figKappa = []
    # Train model on the dataset
    csvPath = os.path.join(config.savePath, 'Joint-fold-{}-ACC.csv'.format(config.fold))
    csvFile = open(csvPath, "w")

    for epoch in range(config.num_epochs):
        schedulers.step(epoch)
        schedulers_CNN.step(epoch)
        print('Epoch %d/%d' % (epoch, config.num_epochs - 1))
        print('-' * 10)
        runing_loss = 0.0
        modelGNN.train(mode=True)
        modelCNN.train(mode=True)

        data_iter = iter(data_loaderTrain)
        # for i, data in enumerate(data_loaderTrain, 0):
        for i in range(int(len(data_loaderTrain)/config.num_k2)):            
            #train CNN
            for i1 in range(config.num_k1):
                try:
                    data = data_iter.__next__()
                except StopIteration:
                    data_iter = iter(data_loaderTrain)
                    data = data_iter.__next__()

                inputs, labels, _ = data
                labels = labels.to(device)
                inputs = inputs.to(device)
                optimizer_CNN.zero_grad()
                optimizer.zero_grad()

                modelCNN.train()
                outputsCNN, featureS, featureD, featureC, importance = modelCNN(inputs)
                loss = criterion(outputsCNN, labels)
                loss.backward()
                optimizer_CNN.step()
                optimizer_CNN.zero_grad()


            #train GNN
            for i2 in range(config.num_k2):
                try:
                    data = data_iter.__next__()
                except StopIteration:
                    data_iter = iter(data_loaderTrain)
                    data = data_iter.__next__()

                inputs, labels, _ = data
                labels = labels.to(device)
                inputs = inputs.to(device)

                with torch.no_grad():
                    modelCNN.eval()
                    outputsCNN, featureS, featureD, featureC, importanceL1 = modelCNN(inputs) 

                featureS, featureD, featureC, importanceL1 = featureS.squeeze(), featureD.squeeze(), featureC.squeeze(), importanceL1.squeeze()#, importanceL2.squeeze()
                featureS = featureS.reshape(config.batchSize_joint,512,config.patchSize*config.patchSize).transpose(1,2)
                featureD = featureD.reshape(config.batchSize_joint,512,config.patchSize*config.patchSize).transpose(1,2)
                featureC = featureC.reshape(config.batchSize_joint,512,config.patchSize*config.patchSize).transpose(1,2)

                importanceL1 = importanceL1.reshape(config.batchSize_joint,3,config.patchSize*config.patchSize)

            
                outputs, const1, const2, out1, out2, _ = modelGNN(featureS, featureD, featureC, importanceL1, adjL1, adjL2)
                loss = criterion(outputs, labels) + config.lossWeight1*criterion_const(const1, const2) + criterion(out2, labels) + criterion(out1, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                runing_loss += loss.item()
                if  (i-1) % int(len(data_loaderTrain)/6)  == 0:
                    print('-' * 10)
                    print(config.saveName,"%d/%d,train_loss:%0.4f" % (i, (len(data_loaderTrain.dataset) - 1) // data_loaderTrain.batch_size + 1, runing_loss/(i+1)))

            #train GNN
            for i2 in range(config.num_k3):
                try:
                    data = data_iter.__next__()
                except StopIteration:
                    data_iter = iter(data_loaderTrain)
                    data = data_iter.__next__()

                inputs, labels, _ = data
                labels = labels.to(device)
                inputs = inputs.to(device)

                with torch.no_grad():
                    modelCNN.eval()
                    outputsCNN, featureS, featureD, featureC, importance = modelCNN(inputs) 

                importanceL1 = importance

                featureS, featureD, featureC, importanceL1 = featureS.squeeze(), featureD.squeeze(), featureC.squeeze(), importanceL1.squeeze()
                featureS = featureS.reshape(config.batchSize_joint,512,config.patchSize*config.patchSize).transpose(1,2)
                featureD = featureD.reshape(config.batchSize_joint,512,config.patchSize*config.patchSize).transpose(1,2)
                featureC = featureC.reshape(config.batchSize_joint,512,config.patchSize*config.patchSize).transpose(1,2)

                importanceL1 = importanceL1.reshape(config.batchSize_joint,3,config.patchSize*config.patchSize)

            
                outputs, const1, const2, out1, out2, _ = modelGNN(featureS, featureD, featureC, importanceL1, adjL1, adjL2)
                loss = criterion(outputs, labels) + config.lossWeight1*criterion_const(const1, const2) + criterion(out2, labels) + criterion(out1, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                runing_loss += loss.item()

        print('Epoch %d/%d' % (epoch, config.num_epochs - 1))
        current_lr = get_lr(optimizer)

        Acc, AUC, RE, PRE, f1, kappa,conf = test(modelGNN, modelCNN, dataloader_test, adjL1, adjL2, epoch)

        figACC.append(Acc)
        figAUC.append(AUC)
        figF1.append(f1)
        figKappa.append(kappa)

        writer = csv.writer(csvFile)
        data = [Acc, AUC, f1, kappa]
        writer.writerow(data)

        vis.plot('Joint Test ACC', Acc)
        vis.plot('Joint Test AUC', AUC)
        vis.plot('Joint Test Kappa', kappa)

        if (Acc > best_metrix or AUC > best_auc) :
            # conf_matrix = pd.DataFrame(conf, index=range(config.num_class), columns=range(config.num_class))
            conf_matrix = conf
            cmap = 'Blues'
            # save_path = os.path.join(save_dir, 'state-{}-{}.jpg'.format(epoch + 1,Acc))
            save_path = os.path.join(save_dir, 'Joint-state-{}-{:.4f}-{:.4f}-{:.4f}.jpg'.format(epoch + 1,Acc,AUC,kappa))
            # pretty_plot_confusion_matrix(save_path,conf_matrix, cmap=cmap)
            plot_confusion_matrix(save_path, y_test=conf_matrix[0],predictions=conf_matrix[1], classes=['Control','AD'], cmap=cmap)

            save_path_gnn = os.path.join(save_dir, 'JointGNN-state-{}-{}.pth'.format(epoch + 1,Acc))
            save_path_cnn = os.path.join(save_dir, 'JointCNN-state-{}-{}.pth'.format(epoch + 1,Acc))

            torch.save(modelGNN, save_path_gnn)
            torch.save(modelCNN, save_path_cnn)

            if Acc > best_metrix:
                best_metrix = Acc
            else:
                best_auc = AUC

    save_path = os.path.join(save_dir, 'Acc_joint.png')
    sve_as_fig(figACC, 'Acc', path=save_path)

    save_path = os.path.join(save_dir, 'AUC_joint.png')
    sve_as_fig(figAUC, 'AUC', path=save_path)

    save_path = os.path.join(save_dir, 'F1_joint.png')
    sve_as_fig(figF1, 'F1', path=save_path)

    save_path = os.path.join(save_dir, 'kappa_joint.png')
    sve_as_fig(figKappa, 'kappa', path=save_path)

    csvFile.close()

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    vis = Visualizer(env=config.saveName, port=7788)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = config.savePath
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    log = Logger(save_dir+'/log_file.log',level='debug')
    log.logger.info('imgage size:'+ str(config.imgSize))
    log.logger.info('batch size:'+ str(config.batchSize_joint))
    log.logger.info('learning rate:'+ str(config.base_lr))
    log.logger.info('Fold:')
    log.logger.info(config.fold)
    log.logger.info(config.modals)

    adjMetrixL1 = generateAdj(config.patchSize)
    adjMetrixL1 = torch.from_numpy(adjMetrixL1)
    adjMetrixL1 = sp.coo_matrix(adjMetrixL1)
    adjMetrixL1 = normalize(adjMetrixL1 + sp.eye(adjMetrixL1.shape[0])) 
    adjMetrixL1 = adjMetrixL1.todense()
    adjMetrixL1 = torch.from_numpy(adjMetrixL1)
    adjMetrixL1 = adjMetrixL1.float()
    adjMetrixL1 = adjMetrixL1.to(device)

    adjMetrixL2 = np.load('utils/adjMetrix_level2.npy')
    adjMetrixL2 = torch.from_numpy(adjMetrixL2)
    adjMetrixL2 = sp.coo_matrix(adjMetrixL2)
    adjMetrixL2 = normalize(adjMetrixL2 + sp.eye(adjMetrixL2.shape[0])) 
    adjMetrixL2 = adjMetrixL2.todense()
    adjMetrixL2 = torch.from_numpy(adjMetrixL2)
    adjMetrixL2 = adjMetrixL2.float()
    adjMetrixL2 = adjMetrixL2.to(device)


    print('==> Preparing data...')
    train_loader_cnn = config.dataloader_train_CNN
    train_loader_gnn = config.dataloader_train_GNN
    train_loader_joint = config.dataloader_train_joint
    test_loader = config.dataloader_test


    print('==> Pre-train CNN ...')
    CNN = EyeADCNN(in_channels=config.in_channels, 
                    num_classes=config.num_class, 
                    patch_size = config.patchSize
                    )
    CNN = CNN.to(device)
    model_CNN = CNN_pretrain(vis, CNN,train_loader_cnn,test_loader,save_dir)
    model_CNN.eval()

    print('==> Pre-train GNN ...')
    GNN = EyeADGNN(nfeat=config.feat_in, 
                    nhid=config.hidden, 
                    nclass=config.nclass, 
                    dropout=config.dropout, 
                    nheads=config.nb_heads, 
                    alpha=config.alpha,
                    L1_nhid = config.hnid_L1,
                    patchNum = config.patchSize
                    )

    model_GNN = GNN.to(device)
    model_GNN = GNN_pretrain(GNN, model_CNN, train_loader_gnn, test_loader, adjMetrixL1, adjMetrixL2)
    model_CNN.train()
    model_GNN.train()

    print('==> Joint training ...')
    jointTrain(model_GNN, model_CNN, train_loader_joint, test_loader, adjMetrixL1, adjMetrixL2)
