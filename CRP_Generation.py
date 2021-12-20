# %%

import tensorflow as tf
from time import *
import numpy as np
import random
from scipy.special import comb
import math
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# Parameters of normal distribution(正态分布参数)
# mean = 100
# stdev =1
# alpha = 0   # 1/80 1/20 1/10(加噪声)

# %%

def CRP_Generation_function_2(Memristors_Res, Address, Mem_Num, Sel_Mem_Num, Xor_Num):
    CRP_C = np.zeros(Mem_Num)
    CRP_R = 0
    CRP_R_Xor_Num = np.zeros((1, Xor_Num))
    # Generate CRP_R for each PUF, CRP_R_Xor_Num===

    for j in range(Xor_Num):
        Group1 = 0;
        Group2 = 0;
        for i in range(0, Sel_Mem_Num):
            if i < (Sel_Mem_Num / 2):
                Group1 = Group1 + Memristors_Res[j, Address[i]]
                CRP_C[Address[i]] = 1
            else:
                Group2 = Group2 + Memristors_Res[j, Address[i]]
                CRP_C[Address[i]] = 2
        if Group1 >= Group2:
            CRP_R_Xor_Num[0, j] = 1
        else:
            CRP_R_Xor_Num[0, j] = 0

    # Generate CRP_R===
    CRP_R = CRP_R_Xor_Num[0, 0];
    # print(CRP_R)

    for j in range(1, Xor_Num):
        CRP_R = ((CRP_R).astype(int)) ^ ((CRP_R_Xor_Num[0, j]).astype(int))
    return CRP_C, CRP_R


# %%

def CRP_Generation_function_3(Mem_Num, Sel_Mem_Num, Xor_Num, N, M, Num_tr, Num_tst):
    # Address adjust
    if Mem_Num <= 36:

        CRP_C_train = np.zeros((N, Mem_Num))  # Parameter vector
        CRP_R_train = np.zeros((N, 1))  # Response

        CRP_C_test = np.zeros((M, Mem_Num))
        CRP_R_test = np.zeros((M, 1))

        Address = []
        A = (np.linspace(1, Mem_Num, Mem_Num)).astype(int) - 1
        C = list(itertools.combinations(A, Sel_Mem_Num))
        D = (comb(Mem_Num, Sel_Mem_Num)).astype(int)
        rowrank = (np.linspace(1, D, D)).astype(int) - 1
        random.shuffle(rowrank)

        for x in rowrank:
            Address.append(C[x])  # 索引问题
        Address = np.array(Address)
        # print(np.shape(Address))
        # print(Address)
    else:
        train_num = Num_tr  # set experiment real train_num
        test_num = Num_tst  # set experiment real test_num
        num = train_num + test_num
        N = train_num
        M = test_num
        CRP_C_train = np.zeros((N, Mem_Num))  # Parameter vector
        CRP_R_train = np.zeros((N, 1))  # Response

        CRP_C_test = np.zeros((M, Mem_Num))
        CRP_R_test = np.zeros((M, 1))

        Address = np.zeros((num, Sel_Mem_Num))
        A = (np.linspace(1, Mem_Num, Mem_Num)).astype(int) - 1
        for i in range(num):
            random.shuffle(A)
            K = A[:Sel_Mem_Num]
            Address[i, :] = np.sort(K)
        Address = Address.astype(int)
        # print(np.shape(Address))
        # print(Address)

    # Parameters of normal distribution(正态分布参数)
    mean = 100
    stdev = 1
    alpha = 0  # 1/80 1/20 1/10(加噪声)
    Memristors_Res = np.random.normal(mean, stdev, (Xor_Num, Mem_Num))
    # print(Memristors_Res)
    # Memristors_Res=np.random.normal(mean, stdev, (Xor_Num,Mem_Num))
    # plt.plot(Memristors_Res)
    # plt.show()
    # print(Memristors_Res)
    # Generate CRP
    for i in range(N):
        Chall, Resp = CRP_Generation_function_2(Memristors_Res, Address[i], Mem_Num, Sel_Mem_Num, Xor_Num)
        CRP_C_train[i, :] = Chall
        CRP_R_train[i] = Resp

    for j in range(N, N + M):
        Chall, Resp = CRP_Generation_function_2(Memristors_Res, Address[j], Mem_Num, Sel_Mem_Num, Xor_Num)
        CRP_C_test[j - N, :] = Chall
        CRP_R_test[j - N] = Resp

    return CRP_C_train, CRP_R_train, CRP_C_test, CRP_R_test


# %%

# 标签编码处理
def onehot(values):
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded
    # print(onehot_encoded)


# invert first example
# inverted= label_encoder.inverse_transform([tf.argmax(onehot_encoded[0, :])])
# print(inverted)

# %%

# 数据预处理
def preprocess(Mem_Num, Sel_Mem_Num, Xor_Num, N1, M1, Num_tr, Num_tst, num_train):
    CRP_C_train, CRP_R_train, CRP_C_test, CRP_R_test = CRP_Generation_function_3(Mem_Num, Sel_Mem_Num, Xor_Num, N1, M1,
                                                                                 Num_tr, Num_tst)
    CRP_C_train = CRP_C_train.astype(int)
    CRP_R_train = CRP_R_train.astype(int)
    CRP_C_test = CRP_C_test.astype(int)
    CRP_R_test = CRP_R_test.astype(int)

    sample = int(num_train / 2)
    count1 = 0
    count2 = 0
    count = len(CRP_R_train) + len(CRP_R_test)
    uniformity = 0
    positive_train = []
    negative_train = []
    for j in range(len(CRP_R_train)):
        if CRP_R_train[j] == 1:
            count1 = count1 + 1
            positive_train.append(j)
        else:
            count2 = count2 + 1
            negative_train.append(j)

    if min(count1, count2) >= sample:
        addr1 = positive_train[:sample]
        addr2 = negative_train[:sample]
        addr = addr1 + addr2
        random.shuffle(addr)
        CRP_C_train1 = CRP_C_train[addr]
        CRP_R_train1 = CRP_R_train[addr]
        print(len(CRP_R_train1))
    else:
        print("训练集正样本为：", count1)
        print("训练集负样本为：", count2)
        print("采样值为：", sample)
        print("需要重新调整sample值")

    chip = int(num_train * 0.2)
    m = CRP_C_train1
    m1 = CRP_R_train1
    t = CRP_C_test[:chip]
    t1 = CRP_R_test[:chip]

    m_np = np.array(m)
    m1_np = np.array(m1)
    t_np = np.array(t)
    t1_np = np.array(t1)
    # print('111')
    return m_np, m1_np, t_np, t1_np

# %%

def generation(N, K, XOR, Num_tr, Num_tst, num_train):
    # define variables
    Mem_Num = N  # 16 32 64 100
    Sel_Mem_Num = K  # 4 6 8 10

    All_condi = comb(Mem_Num, Sel_Mem_Num)
    Xor_Num = XOR  # 1 2 3 4 5 6

    N1 = math.ceil(All_condi * 0.8)
    M1 = math.floor(All_condi * 0.2)
    print(N1)
    print(M1)
    print(N1 + M1)

    x_train, y_train, x_test, y_test = preprocess(Mem_Num, Sel_Mem_Num, Xor_Num, N1, M1, Num_tr, Num_tst, num_train)
    return x_train, y_train, x_test, y_test

# %%

# x_train,y_train,x_test,y_test=generation(16,4,1,1000,200,100)
# print(x_train[10])
