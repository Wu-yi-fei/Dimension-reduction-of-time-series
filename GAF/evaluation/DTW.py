#!/usr/bin/env python3

import numpy as np
# from method.LTTB import LargestTriangleThreeBuckets, LttbException, ReadLTTBData1
import seaborn as sns
import matplotlib.pyplot as plt
from pyts.approximation.paa import PiecewiseAggregateApproximation
import pandas as pd

def dtw_distance(ts_a, ts_b, d=lambda x, y: abs(x - y), mww=10000):
    """计算两个时间序列之间的dtw距离

    Args:
        ts_a: 时间序列a
        ts_b: 时间序列b
        d: 距离函数

    Returns:
        dtw distance
    """

    # 创建cost矩阵
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = np.ones((M, N))

    # 初始化
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

    # 在窗口内填充剩余的成本矩阵
    for i in range(1, M):
        for j in range(max(1, i - mww), min(N, i + mww)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    # 返回给定窗口的DTW距离
    return cost[-1, -1]

# if __name__ == '__main__':
#     his_data = np.loadtxt('1.csv', delimiter=",", skiprows=0)
#     image1 = his_data[0:576, 4]
#     plt.title("curve")
#     plt.plot(his_data[0:576,0], image1)
#     plt.show()
#
#     image2 = image1.reshape(1,-1)
#     paa = PiecewiseAggregateApproximation(
#         window_size=None, output_size=36
#     )
#     X_paa = paa.fit_transform(image2)
#     X_paa=X_paa.reshape(-1, 1)
#     plt.plot(his_data[:36, 0], X_paa)
#     plt.show()
#     plt.plot(his_data[:36, 0],X_paa, linestyle='none', marker='o')
#     plt.show()
#
#     print("PAA:",dtw_distance(image1,X_paa))
#
#
#     image = ReadLTTBData()
#     image = np.array(image)
#     print(image)
#     plt.title("curve")
#     plt.plot(image[:,0], image[:,1])
#     plt.show()
#     print("LTTB:",dtw_distance(image1,image[:,1]))
#     plt.plot(his_data[:36, 0], image[:,1], linestyle='none', marker='o')
#     plt.show()
#
#     exit()