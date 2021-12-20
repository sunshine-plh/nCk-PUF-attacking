#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.special import comb
import itertools
import os
import torch
import random
from time import *
import pandas as pd

import scipy.io as scio
import torch.nn as nn
import itertools as it

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device.".format(device))




# In[ ]:


class MyDataset(Dataset):
    def __init__(self, root, c_name, r_name, train=True):
        self.train = train
        self.c = torch.from_numpy(np.load(root + '/' + c_name))
        self.r = torch.from_numpy(np.load(root + '/' + r_name))

    def __getitem__(self, index):
        return self.c[index, :], self.r[index]

    def __len__(self):
        return self.c.shape[0]


# %%

class Es_Xor_Classifier(nn.Module):
    def __init__(self, c_dim, xor_num):
        super().__init__()
        self.fc1 = nn.Linear(c_dim, xor_num, bias=False)
        nn.init.normal_(self.fc1.weight, 100, 1)
        self.fc2 = nn.Linear(xor_num, xor_num, bias=True)
        self.fc3 = nn.Linear(xor_num, 1, bias=True)
        self.bn1 = nn.BatchNorm1d(xor_num)
        self.bn2 = nn.BatchNorm1d(xor_num)

    def forward(self, x):
        r_hat = torch.sigmoid(self.bn1(self.fc1(x)))
        r_hat = torch.relu(self.bn2(self.fc2(r_hat)))
        r_hat = torch.sigmoid(self.fc3(r_hat))
        return r_hat


# %%

class Es_Seq3Xor_Classifier(nn.Module):
    def __init__(self, c_dim, xor_num):
        super().__init__()
        self.es1 = nn.Linear(c_dim, 2, bias=False)
        nn.init.normal_(self.es1.weight, 100, 1)
        self.es2 = nn.Linear(c_dim, 1, bias=False)
        nn.init.normal_(self.es2.weight, 100, 1)
        self.h1 = nn.Linear(2, 2, bias=True)
        self.xor1 = nn.Linear(2, 1, bias=True)
        self.h2 = nn.Linear(2, 2, bias=True)
        self.xor2 = nn.Linear(2, 1, bias=True)
        self.es_bn1 = nn.BatchNorm1d(2)
        self.es_bn2 = nn.BatchNorm1d(1)
        self.h_bn1 = nn.BatchNorm1d(2)
        self.h_bn2 = nn.BatchNorm1d(2)
        self.xor_bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        o1 = torch.sigmoid(self.es_bn1(self.es1(x)))
        o2 = torch.sigmoid(self.es_bn2(self.es2(x)))
        h1 = torch.relu(self.h_bn1(self.h1(o1)))
        xor1 = torch.sigmoid(self.xor_bn1(self.xor1(h1)))
        o2 = torch.cat((o2, xor1), 1)
        h2 = torch.relu(self.h_bn2(self.h2(o2)))
        r_hat = torch.sigmoid(self.xor2(h2))
        return r_hat


# In[ ]:


def get_parameters(model):
    parm = {}
    for name, parameters in model.named_parameters():
        parm[name] = parameters.cpu().detach().numpy()
    return parm


# In[ ]:


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    parm = get_parameters(model)
    with torch.no_grad():
        for c, r in loader:
            c = c.to(device=device, dtype=dtype)
            r = r.to(device=device, dtype=dtype)
            p = model(c)
            preds = (p >= 0.5).reshape(-1)
            num_correct += (r == preds).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        # print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc


# In[ ]:


class EarlyStopping:
    
    def __init__(self, patience=30, verbose=False, delta=0.0001):
       
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_acc,epochs, model):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc,epochs,model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc,epochs, model)
            self.counter = 0

    def save_checkpoint(self, val_acc,epochs, model):

        if self.verbose:
            print('Epoch is {},Current val_acc is {}'.format(epochs,val_acc))            



# In[ ]:


class PUF_Attack():
    mem_num=0
    sel_num=0
    xor_num=0
    data_num=0
    times=0
    batch_size=0
    def __init__(self,mem_num,sel_num,xor_num,train_num,batch_size,times):
        self.mem_num=mem_num
        self.sel_num=sel_num
        self.xor_num=xor_num
        self.train_num=train_num
        self.times=times
        self.batch_size=int(batch_size)

    def dataset(self):
        ROOT = "/home/lihui/nCkPUF_MMA_test/data_v2" + "/" + "{}_sel_{}".format(self.mem_num, self.sel_num) + "/" + "xor_{}".format(self.xor_num)
        slide=self.times-1
            
        c_train_name = 'crp_c_train.npy'
        r_train_name = 'crp_r_train_pycharm.npy'
        c_test_name = 'crp_c_test.npy'
        r_test_name = 'crp_r_test_pycharm.npy'
        
        ratio=0.25
        chip1=int(self.train_num*ratio)
        max_onetest=2000000
        
        train_data = MyDataset(ROOT, c_train_name, r_train_name, train=True)
        val_data = MyDataset(ROOT, c_train_name, r_train_name, train=True)
        test_data=MyDataset(ROOT, c_test_name, r_test_name, train=True)
        N = len(train_data)
        N0 =self.train_num
     
        loader_train = DataLoader(train_data, batch_size=self.batch_size,
                          sampler=sampler.SubsetRandomSampler(range(int(N0 * 0.8+max_onetest*slide))))
        loader_val = DataLoader(val_data, batch_size=self.batch_size,
                         sampler=sampler.SubsetRandomSampler(range(int(N - N0 *ratio), N)))
        loader_test=DataLoader(test_data, batch_size=self.batch_size,
                          sampler=sampler.SubsetRandomSampler(range(int(chip1+max_onetest*slide*ratio))))
             
        return loader_train,loader_val,loader_test
                               
    def train(self,epochs=500): 
        model = Es_Seq3Xor_Classifier(self.mem_num, self.xor_num)
        model.to(device=device, dtype=dtype)

        patience = 30
        early_stopping = EarlyStopping(patience, verbose=None)
                               
        learning_rate = 1e-2
        decayRate = 0.98

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
        criterion = torch.nn.BCELoss()
                       
        loader_train,loader_val,loader_test = self.dataset()
        train_acc = [] 
        val_acc = []
                               
        begin_time = time()
        for e in range(epochs):
            model.train()
            for t, (c, r) in enumerate(loader_train):
                c = c.to(device=device, dtype=dtype)
                r = r.to(device=device, dtype=dtype)

                p = model(c).reshape(-1)
                loss = criterion(p, r)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            lr_scheduler.step()
            train_acc.append(check_accuracy(loader_train, model))
            
            model.eval()
            for t, (c, r) in enumerate(loader_val):
                c = c.to(device=device, dtype=dtype)
                r = r.to(device=device, dtype=dtype)

                p = model(c).reshape(-1)
                val_loss = criterion(p, r)

                optimizer.zero_grad()
                val_loss.backward()
                optimizer.step()
            lr_scheduler.step()
            val_acc.append(check_accuracy(loader_val,model))
            #设置早停起始准确率
            if val_acc[e]>0.90:
                early_stopping(val_acc[e],e, model)
                if early_stopping.early_stop:
                    #print("Train_Finish")
                    break
        iterations = len(train_acc)
                               
        end_time=time()
        run_time=end_time-begin_time
                               
        test_acc= round(check_accuracy(loader_test, model),4)                    
        attack_time=round(run_time,4)
        train_acc=round(train_acc[-1],4)
        val_acc=round(val_acc[-1],5)   
                               
        return train_acc,val_acc,test_acc,attack_time,iterations



def collect_result(path,times):
    para_table=pd.read_excel(path,engine='openpyxl')
    
    train_acc=np.zeros((para_table.shape[0],1))
    val_acc=np.zeros((para_table.shape[0],1))
    test_acc=np.zeros((para_table.shape[0],1))
    attack_time=np.zeros((para_table.shape[0],1))
    iterations= np.zeros((para_table.shape[0],1))                    

    for i in range(para_table.shape[0]):
        mem_num=int(para_table['mem_num'].values[i])
        sel_num=int(para_table['sel_num'].values[i])
        xor_num=int(para_table['xor_num'].values[i])
        train_num=int(para_table['train_num'].values[i])
        batch_size=int(para_table['batch_size'].values[i])
        my_puf_attack=PUF_Attack(mem_num,sel_num,xor_num,train_num,batch_size,times)
        train_acc[i],val_acc[i],test_acc[i],attack_time[i],iterations[i]=my_puf_attack.train()
        print('train_acc=%.4f,val_acc=%.5f,test_acc=%.4f,train_time=%.4f,iteration=%.4f,' % (train_acc[i],val_acc[i],test_acc[i],attack_time[i],iterations[i]))
       # print('train_time:',attack_time[i])
        #print('iteration:',iterations[i])
    
    #增添5列
    para_table.insert(4,'train_acc',train_acc)
    para_table.insert(5,'val_acc',val_acc)
    para_table.insert(6,'test_acc',test_acc)
    para_table.insert(7,'time',attack_time) 
    para_table.insert(8,'iterations',iterations) 
    para_table.to_excel("/home/lihui/nCkPUF_MMA_test/pycharm_test/result_pycharm_123xor_patience30_Es_Seq3Xor_No.{}.xlsx".format(times))
  


# In[ ]:

#实例化


#实例化
path1="E:/para_table_pycharm/para_36_sel_6.xlsx"
path2="/home/lihui/nCkPUF_MMA_test/pycharm_test/pycharm_para_Es_Seq3Xor_Classifier.xlsx"
path=[path2]
times=3

for k in range(times):
    for i in range(len(path)):
        collect_result(path[i],k+1)

