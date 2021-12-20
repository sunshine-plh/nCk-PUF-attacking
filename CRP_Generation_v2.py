#!/usr/bin/env python
# coding: utf-8

# In[4]:


'''step1:产生CRP数据并保存为.npy格式'''

import tensorflow as tf
import numpy as np
from scipy.special import comb
import random
import itertools
from sklearn.model_selection import train_test_split
import os

class n_sel_k_sum_XPUF():
    mem_num=0
    sel_num=0
    xor_num=0#类变量
    times=0
    
    def __init__(self,mem_num,sel_num,xor_num,times):
        self.mem_num=mem_num#实例变量
        self.sel_num=sel_num
        self.xor_num=xor_num
        self.times=times
        
    def mem_generation(self):
        mean=100
        std=1
        mem_res=np.random.normal(mean,std,(self.xor_num,self.mem_num))
        return mem_res

    def Addr_Gene_small(self):
        N=int(comb(self.mem_num,self.sel_num)*comb(self.sel_num,self.sel_num/2)/2)
        N1=int(comb(self.mem_num,self.sel_num))
        N2=int(comb(self.sel_num,self.sel_num/2))
        
        Address=np.zeros((N,self.sel_num))
        Address_index=(np.linspace(0,N-1,N)).astype(int)
        random.shuffle(Address_index)
        
        Temp_A=(np.linspace(0,self.mem_num-1,self.mem_num)).astype(int)
        C_1=list(itertools.combinations(Temp_A,self.sel_num))
        
        count=0
        for i in range(N1):
            Temp_B=C_1[i]
            C_2=list(itertools.combinations(Temp_B,int(self.sel_num/2)))
            C_2_np=np.array(C_2)
            
            for j in range(int(N2/2)):
                Index=Address_index[count]
                Address[Index,:]=(np.concatenate(( C_2_np[j], C_2_np[-(j+1)]),axis=0))
                count+=1
        return (Address).astype(int)
    
    def Addr_Gene_large(self):
        crp_needmax=20000000
        Temp_A=(np.linspace(0,self.mem_num-1,self.mem_num)).astype(int)
        Address=np.zeros((crp_needmax,self.sel_num))
        
        for i in range(crp_needmax):
            random.shuffle(Temp_A)  
            Temp_B=Temp_A[:self.sel_num]
            Address[i,:]=(np.sort(Temp_B)).astype(int)
            
        Address_del=np.array(list(set([tuple(t) for t in Address])))#去除冗余***考虑有没有其他更高效代码
        return (Address_del).astype(int)
    
    def crp_cal(self):
        mem_res=self.mem_generation()
        if self.mem_num<=36:
            Address=self.Addr_Gene_small()
            #print(Address)
        else:
            Address=self.Addr_Gene_large()
        
        crp_c=(np.zeros((np.shape(Address)[0],self.mem_num))).astype(int)
        crp_middle=np.zeros((np.shape(Address)[0],self.xor_num))               
        crp_r=np.zeros((np.shape(Address)[0],1))
        
        for m in range(np.shape(Address)[0]):
            for j in range(self.xor_num):
                Group1=0
                Group2=0
                for i in range(self.sel_num):
                    if i<(int(self.sel_num/2)):
                        Group1=Group1+mem_res[j,Address[m,i]]
                        crp_c[m,Address[m,i]]=-1
                    else:
                        Group2=Group2+mem_res[j,Address[m,i]]
                        crp_c[m,Address[m,i]]=1
                        
                if Group1>=Group2:
                    crp_middle[m,j]=1
                else:
                    crp_middle[m,j]=0       
            crp_r[m]=crp_middle[m,0]          
            for s in range(1,self.xor_num): 
                crp_r[m]=(crp_r[m]).astype(int)^ (crp_middle[m,s]).astype(int)
        return crp_c,crp_r.astype(int)
    
    def separate(self):
        count=0
        crp_c,crp_r=self.crp_cal()
        
        for i in (crp_r):
            if(i==1):
                count+=1
        uniformity=count/len(crp_r)
        
        positive=[]
        negative=[]
        count1=0
        count2=0
        for j in range(len(crp_r)):
            if crp_r[j]==1:
                count1+=1
                positive.append(j)
            else:
                count2+=1
                negative.append(j)
                
        CRP_C=np.zeros(np.shape(crp_c))
        CRP_R=np.zeros(np.shape(crp_r))
        
        for k in range(min(count1,count2)):
            CRP_C[2*k+1]=crp_c[positive[k]]
            CRP_R[2*k+1]=crp_r[positive[k]]
            CRP_C[2*k]=crp_c[negative[k]]
            CRP_R[2*k]=crp_r[negative[k]]
        #print(count1,count2)
        for k in range(min(count1,count2),max(count1,count2)):
            if count1>count2:
                CRP_C[k+min(count1,count2)]=crp_c[positive[k]]
                CRP_R[k+min(count1,count2)]=crp_r[positive[k]]
            else:
                CRP_C[k+min(count1,count2)]=crp_c[negative[k]]
                CRP_R[k+min(count1,count2)]=crp_r[negative[k]]
        #print(count1,count2)
        chip=int(np.shape(CRP_C)[0]*0.8)
        
        CRP_C_Train= CRP_C[:chip]
        CRP_R_Train=CRP_R[:chip]
        CRP_C_Test=CRP_C[chip:np.shape(CRP_C)[0]]
        CRP_R_Test=CRP_R[chip:np.shape(CRP_R)[0]]
        return CRP_C_Train,CRP_R_Train,CRP_C_Test,CRP_R_Test
    
    def crp_save(self):
        if self.mem_num<=16:
            save_path="D:/python/mrsPUF_attack/data_v2"+"/"+"{}_sel_{}_No.{}".format(self.mem_num, self.sel_num,self.times)+"/"+"xor_{}".format(self.xor_num)
        else:
            save_path="D:/python/mrsPUF_attack/data_v2"+"/"+"{}_sel_{}".format(self.mem_num, self.sel_num)+"/"+"xor_{}".format(self.xor_num)
               
        if not os.path.exists(save_path):
            os.makedirs(save_path)
                
        crp_c_train,crp_r_train,crp_c_test,crp_r_test=self.separate()
        
        crp_r_train_nps=np.transpose(crp_r_train)
        crp_r_train_pycharm=( crp_r_train_nps.astype(int)).reshape(np.shape(crp_r_train_nps)[1])
        
        crp_r_test_nps=np.transpose(crp_r_test)
        crp_r_test_pycharm=( crp_r_test_nps.astype(int)).reshape(np.shape(crp_r_test_nps)[1])
        
        #print(crp_r_test_pycharm[10])
        #print(np.shape(crp_r_test_pycharm))
        
        np.save(save_path+"/"+"crp_c_train",crp_c_train)
        np.save(save_path+"/"+"crp_r_train_tensor",crp_r_train)
        np.save(save_path+"/"+"crp_c_test",crp_c_test)
        np.save(save_path+"/"+"crp_r_test_tensor",crp_r_test)
        np.save(save_path+"/"+"crp_r_train_pycharm",crp_r_train_pycharm)
        np.save(save_path+"/"+"crp_r_test_pycharm",crp_r_test_pycharm)
        #print(crp_r_test)
        #print(np.transpose(crp_r_test))


# In[5]:


#实例化(利用列表遍历)
#mem_num=[36,64,100]
#sel_num=[6,8,10]
#xor_num=[1,2,3,4]

mem_num=[36]
sel_num=[6]
xor_num=[1]

times=1
#for i in range(1):   ##产生三组16选4的CRP数据
for i in range(len(mem_num)):
    if mem_num[i]<=16:
        for k in range(times):
            for j in xor_num:
                N_SEL_K_SUM_XPUF=n_sel_k_sum_XPUF(mem_num[i],sel_num[i],j,k+1)
                N_SEL_K_SUM_XPUF.crp_save() 
            print("{}_sel_{} _No{} DONE!".format(mem_num[i], sel_num[i],k+1))
    else:
        for j in xor_num:
            N_SEL_K_SUM_XPUF=n_sel_k_sum_XPUF(mem_num[i],sel_num[i],j,times)
            N_SEL_K_SUM_XPUF.crp_save() 
            print("{}_sel_{}_xor_{} DONE!".format(mem_num[i], sel_num[i], j))


# In[3]:


'''
#单次数据产生
N_SEL_K_SUM_XPUF=n_sel_k_sum_XPUF(16,4,1,1)
crp_c_train,crp_r_train,crp_c_test,crp_r_test=N_SEL_K_SUM_XPUF.separate()
print(np.shape(crp_c_train))
print(np.shape(crp_c_test))
'''

