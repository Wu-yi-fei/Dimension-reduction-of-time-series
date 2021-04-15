#!/usr/bin/env python3



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from utils.GAF import GAF
import seaborn as sns
from numpy.linalg import eig
from method.LTTB import ReadLTTBData1
from method.LTTB_Pikaz import ReadLTTBData2
from method.LTTB_WA import ReadLTTBData3
from sklearn.decomposition import PCA
import csv
import cv2 as cv
import math




# def PCADataReduction(image):
#     X = image
#     print(X)
#     pca = PCA(n_components=20)  # 降到20维
#     pca.fit(X)  # 训练
#     newX = pca.fit_transform(X)  # 降维后的数据
#     # PCA(copy=True, n_components=2, whiten=False)
#     print(pca.explained_variance_ratio_)  # 输出贡献率
#     with open('GAF1.csv', 'w') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerows(newX)
#     return newX.T


def calculate_error(st, seq_range):
    x = np.arange(seq_range[0], seq_range[1] + 1)
    y = np.array(st[seq_range[0]:seq_range[1] + 1])
    A = np.ones((len(x), 2), float)
    A[:, 0] = x
    # 返回回归系数、残差平方和、自变量X的秩、X的奇异值
    (p, residuals, ranks, s) = np.linalg.lstsq(A, y, rcond=None)#最小二乘法回归
    try:
        error = residuals[0]
    except IndexError:
        error = 0.0
    return error


def SelectImportantPoints(T, R):
    """
    重要点采样技术
    :param T: 原始序列
    :param R: 采样参数
    :return: 采样点集合
    """
    X = []
    for i in range(0, len(T)):
        X.append([i, T[i]])
    vital_point = []
    vital_point.insert(0, X[0])
    index = 0
    for i in range(1, len(T) - 1):
        if T[i] >= T[i - 1] and T[i] > T[i + 1]:
            if T[i] / (T[index]+0.005) > T[i]:
                index += 1
                vital_point.insert(index, X[i])
        elif T[i] <= T[i - 1] and T[i] < T[i + 1]:
            if T[i] == 0 or T[index] / (T[i]+0.005) > R:
                index += 1
                vital_point.insert(index, X[i])
        elif T[i] > T[i - 1] and T[i] >= T[i + 1]:
            if T[i] / (T[index]+0.005) > T[i]:
                index += 1
                vital_point.insert(index, X[i])
        elif T[i] < T[i - 1] and T[i] <= T[i + 1]:
            if T[i] == 0 or T[index] / (T[i]+0.005) > R:
                index += 1
                vital_point.insert(index, X[i])

    index += 1
    vital_point.insert(index, X[len(T) - 1])
    return vital_point

def PLR_EFP():
    pass

def SAX():
    pass




if __name__ == '__main__':
    his_data = np.loadtxt('1.csv', delimiter=",", skiprows=0)
    his_data = his_data[0:576,6]
    data = SelectImportantPoints(his_data, 0.1)
    data = np.array(data)
    print(data)





    image_old = GAF(data,39,39)
    hm = sns.heatmap(image_old[0], cmap='Greens', cbar=True, annot=None, square=True, annot_kws={"size": 10})
    plt.ylabel("time", fontsize=15)
    plt.xlabel("time", fontsize=15)
    plt.show()

    # image_old1 = GAF(his_data, 576, 36)
    # hm = sns.heatmap(image_old1[0], cmap='Greens', cbar=True, annot=None, square=True, annot_kws={"size": 10})
    # plt.ylabel("time", fontsize=15)
    # plt.xlabel("time", fontsize=15)
    # plt.show()

    # image = PCADataReduction(his_data)
    # image_new = GAF(image)
    # sns.set(font_scale=1.25)
    # hm = sns.heatmap(image_new[0], cmap='Greens', cbar=True, annot=None, square=True, annot_kws={"size": 10})
    # plt.ylabel("time", fontsize=15)
    # plt.xlabel("time", fontsize=15)
    # plt.show()

    # image = ReadLTTBData3()
    # image = np.array(image)
    # print(image)
    # data1 = []
    # print(image[1,0])
    #
    #
    # image_new = GAF(image,36,36)
    # sns.set(font_scale=1.25)
    # hm = sns.heatmap(image_new[0], cmap='Greens', cbar=True, annot=None, square=True, annot_kws={"size": 10})
    # plt.ylabel("time", fontsize=15)
    # plt.xlabel("time", fontsize=15)
    # plt.show()
    # exit()













