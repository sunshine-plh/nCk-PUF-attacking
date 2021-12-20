
'''step1:装载CRP数据并划分数据集'''
import tensorflow as tf
import numpy as np
import random
from time import *
import pandas as pd
import matplotlib.pyplot as plt
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class PUF_Attack():
    mem_num=0
    sel_num=0
    xor_num=0
    data_num=0
    times=0
    batch_size=0
    def __init__(self,mem_num,sel_num,xor_num,train_num,times,batch_size):
        self.mem_num=mem_num
        self.sel_num=sel_num
        self.xor_num=xor_num
        self.train_num=train_num
        self.times=times
        self.batch_size=batch_size
        
    def preprocess(self):
        ratio=0.25
        chip1=int(self.train_num*ratio)
        max_onetest=9500000
        
        if self.mem_num<=16:
            get_path="/home/lihui/nCkPUF_MMA_test/data_v2"+"/"+"{}_sel_{}_No.{}".format(self.mem_num,self.sel_num,self.times)+"/"+"xor_{}".format(self.xor_num)
            slide=0
        else:
            #get_path="G:/pyt"+"/"+"{}_sel_{}".format(self.mem_num,self.sel_num)+"/"+"xor_{}".format(self.xor_num)
            get_path="/home/lihui/nCkPUF_MMA_test/data_v2"+"/"+"{}_sel_{}".format(self.mem_num,self.sel_num)+"/"+"xor_{}".format(self.xor_num)
            slide=self.times-1
            
        crp_c_train=np.load(get_path+"/"+"crp_c_train.npy")
        crp_r_train=np.load(get_path+"/"+"crp_r_train_tensor.npy")
        crp_c_test=np.load(get_path+"/"+"crp_c_test.npy")
        crp_r_test=np.load(get_path+"/"+"crp_r_test_tensor.npy")
        
        x_train=crp_c_train[int(slide*max_onetest*0.8):(self.train_num+int(slide*max_onetest*0.8))]
        y_train=crp_r_train[int(slide*max_onetest*0.8):(self.train_num+int(slide*max_onetest*0.8))]
        
        x_test=crp_c_test[int(slide*max_onetest*0.2):(chip1+int(slide*max_onetest*0.2))]
        y_test=crp_r_test[int(slide*max_onetest*0.2):(chip1+int(slide*max_onetest*0.2))] 
        
        
        return x_train,y_train,x_test,y_test
    
    def ann_net(self):
        x_train,y_train,x_test,y_test=self.preprocess()
        
        model=tf.keras.models.Sequential()
        
        #输入层：
        model.add(tf.keras.layers.Dense(units=256,
                                        input_dim=self.mem_num,
                                        use_bias=True, 
                                        kernel_initializer='uniform',
                                        #kernel_initializer=tf.keras.initializers.RandomNormal(mean=100, stddev=1, seed=None),
                                        bias_initializer='zeros',
                                        activation="relu"))
        model.add(tf.keras.layers.Dropout(rate=0.01))
        #第一层：
        model.add(tf.keras.layers.Dense(units=512,activation="relu"))
        model.add(tf.keras.layers.Dropout(rate=0.01))
        #第二层：
        model.add(tf.keras.layers.Dense(units=256, activation="relu"))
        model.add(tf.keras.layers.Dropout(rate=0.01))
        #第三层：
        #model.add(tf.keras.layers.Dense(units=8, activation="relu"))
        #model.add(tf.keras.layers.Dropout(rate=0.01))
        #输出层：
        model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
        #model.summary()

        #模型设置
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])

        #早停处理
        my_callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0,patience=30,mode='max')
        
       #model.fit模式训练
    
        begin_time=time()
        
        train_history=model.fit(x=x_train,
                                y=y_train,
                                validation_split=0.2,
                                epochs=1000,
                                batch_size=self.batch_size,
                               callbacks=[my_callbacks],
                                shuffle=True,
                                verbose=None)
        
        end_time=time()
        run_time=end_time-begin_time
        #print('运行时间为:',run_time)

        #模型评估：
        evaluate_result=model.evaluate(x=x_test,y=y_test)
        #print(evaluate_result)
        #model.metrics_names
        
        test_acc=round(evaluate_result[1],4)
        attack_time=round(run_time,4)
        train_acc=round(train_history.history['accuracy'][-1],4)
        val_acc=round(train_history.history['val_accuracy'][-1],4)
        patience=10
        iterations=len(train_history.history['accuracy'])-patience
        
        return train_acc,val_acc,test_acc,attack_time,iterations


# In[ ]:


def collect_result(path,times):
    #para_table=pd.read_excel(path,engine='openpyxl')
    para_table = pd.read_excel(path)
    train_acc=np.zeros((para_table.shape[0],1))
    val_acc=np.zeros((para_table.shape[0],1))
    test_acc=np.zeros((para_table.shape[0],1))
    attack_time=np.zeros((para_table.shape[0],1))
    iterations=np.zeros((para_table.shape[0],1))
                         
    #利用excel表格遍历

    for i in range(para_table.shape[0]):
        mem_num=para_table['mem_num'].values[i]
        sel_num=para_table['sel_num'].values[i]
        xor_num=para_table['xor_num'].values[i]
        train_num=para_table['train_num'].values[i]
        batch_size=para_table['batch_size'].values[i]
    
        my_puf_attack=PUF_Attack(mem_num,sel_num,xor_num,train_num,times,batch_size)
        train_acc[i],val_acc[i],test_acc[i],attack_time[i],iterations[i]=my_puf_attack.ann_net()

    #增添四列
    para_table.insert(4,'train_acc',train_acc)
    para_table.insert(5,'val_acc',val_acc)
    para_table.insert(6,'test_acc',test_acc)
    para_table.insert(7,'time',attack_time) 
    para_table.insert(8,'iterations',iterations) 
    
    #print(para_table)
    #para_table.to_excel("F:/attack_result/result_7_16_times_{}_v1.xlsx".format(times))
    para_table.to_excel("/home/lihui/nCkPUF_MMA_test/round3(earlystopping30)/result_9xor_patience30_256_512_256_No{}.xls".format(times))
    #para_table.to_excel("F:/attack_result/result_{}_sel_{} No.{}_v1.xlsx".format(mem_num,sel_num,times))
    print("{}_sel_{} No.{} DONE!".format(mem_num,sel_num,times))


# In[ ]:


#实例化
#path1="G:/para_table1/para_16_sel_4.xlsx"
#path1="/home/lihui/nCkPUF_MMA_test/data_v2/para_table_tens/5_xor/para_36_sel_6.xls"
#path2="/home/lihui/nCkPUF_MMA_test/data_v2/para_table_tens/5_xor/para_64_sel_8.xls"
#path3="/home/lihui/nCkPUF_MMA_test/data_v2/para_table_tens/5_xor/para_100_sel_10.xls"
#path1="G:/para_table/para_36_sel_6.xlsx"
#path2="G:/para_table/para_64_sel_8.xlsx"
#path3="G:/para_table/para_100_sel_10.xlsx"
#path=[path1,path2,path3,path4]
#path1="G:/para_table/para_7_16.xlsx"


path1="/home/lihui/nCkPUF_MMA_test/round3(earlystopping30)/para_1xor.xls"
path2="/home/lihui/nCkPUF_MMA_test/round3(earlystopping30)/para_2xor.xls"
path3="/home/lihui/nCkPUF_MMA_test/round3(earlystopping30)/para_3xor.xls"
path4="/home/lihui/nCkPUF_MMA_test/round3(earlystopping30)/para_4xor.xls"
path5="/home/lihui/nCkPUF_MMA_test/round3(earlystopping30)/para_5xor.xls"
path6="/home/lihui/nCkPUF_MMA_test/round3(earlystopping30)/para_9xor_256_512_256.xls"
#path=[path1,path2,path3,path4,path5,path6]
path=[path6]
#path=[path3]
times=2
#3*4个结果表
for k in range(times):
    for P_i in range(len(path)):
        collect_result(path[P_i],k+1)

