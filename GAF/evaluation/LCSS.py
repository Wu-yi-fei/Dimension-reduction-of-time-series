#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
# from method.LTTB import ReadLTTBData, LargestTriangleThreeBuckets, LttbException
from pyts.approximation.paa import PiecewiseAggregateApproximation

li =[]

def lcs(a, b):
    """
    改进后的最优子结构LCSS算法：将原来的强约束条件减弱

    原LCSS递归原理
    设输入序列是X [0 .. m-1] 和 Y [0 .. n-1]，长度分别为 m 和 n。和设序列 L(X [0 .. m-1]，Y[0 .. n-1])
    是这两个序列的 LCS 的长度，以下为 L(X [0 .. M-1]，Y [0 .. N-1]) 的递归定义：
    1）如果两个序列的最后一个元素匹配（即X [M-1] == Y [N-1]）
    则：L（X [0 .. M-1]，Y [0 .. N-1]）= 1 + L（X [0 .. M-2]，Y [0 .. N-1]）
    2）如果两个序列的最后字符不匹配（即X [M-1] != Y [N-1]）
　　则：L(X [0 .. M-1]，Y [0 .. N-1]) = MAX(L(X [0 .. M-2]，Y [0 .. N-1])，L(X [0 .. M-1]，Y [0 .. N-2]))

    :param a: 时间序列a
    :param b: 时间序列b
    :return: 指导阵flag
    """
    lena = len(a)
    lenb = len(b)
    c = [[0 for i in range(lenb + 1)] for j in range(lena + 1)]
    flag = [[0 for i in range(lenb + 1)] for j in range(lena + 1)]
    for i in range(lena):
        for j in range(lenb):
            if a[i] != 0:
                if a[i] <= b[j]+0.1 and a[i] >= b[j]-0.1:
                    c[i + 1][j + 1] = c[i][j] + 1
                    flag[i + 1][j + 1] = 'ok'
                elif c[i + 1][j] > c[i][j + 1]:
                    c[i + 1][j + 1] = c[i + 1][j]
                    flag[i + 1][j + 1] = 'left'
                else:
                    c[i + 1][j + 1] = c[i][j + 1]
                    flag[i + 1][j + 1] = 'up'

    return  flag


def printLcs(flag, a, i, j):
    if i == 0 or j == 0:
        return
    if flag[i][j] == 'ok':
        printLcs(flag, a, i - 1, j - 1)
        # print a[i - 1]
        li.append(a[i-1])
    elif flag[i][j] == 'left':
        printLcs(flag, a, i, j - 1)
    else:
        printLcs(flag, a, i - 1, j)
    return len(li)/36




# his_data = np.loadtxt('1.csv', delimiter=",", skiprows=0)
# his_data1 = his_data[0:576,4]
# image = ReadLTTBData()
# image = np.array(image)
# image = image[:,1]
# # c, flag = lcs(his_data1, image)
# # printLcs(flag, his_data1, len(his_data1), len(image))
# # print(li)
# # print(len(li))
#
#
# image2 = his_data1.reshape(1,-1)
# paa = PiecewiseAggregateApproximation(
#     window_size=None, output_size=36
# )
# X_paa = paa.fit_transform(image2)
# X_paa = X_paa.reshape(-1)
# flag = lcs(his_data1, X_paa)
#
# printLcs(flag, his_data1, len(his_data1), len(X_paa))
# print(li)
# print(len(li)/len(X_paa)) #归一化结果