#!/usr/bin/env python3



import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
# from method.LTTB import ReadLTTBData, LargestTriangleThreeBuckets, LttbException
from pyts.approximation.paa import PiecewiseAggregateApproximation

def KL(x,y):
    """

    :param x: 规约后时间序列
    :param y: 规约前时间序列
    :return: KL距离均值
    """
    KL_distance = 0

    every = len(y) / len(x)
    for j in range(int(every)):
        z=[]
        for i in range(len(x)):
            z.append(y[int(np.float(i*every+j))])
        y1 = z
        s1 = x  / np.max(x)
        s2 = y1 / np.max(y1)
        for i in range(len(x)):
            if s1[i] !=0 and s2[i] !=0:
                KL_distance += s1[i] * np.log(s1[i] / s2[i])

    return KL_distance/every


# his_data = np.loadtxt('1.csv', delimiter=",", skiprows=0)
# his_data1 = his_data[0:576,4]
# image = ReadLTTBData()
# image = np.array(image)
# image = image[:,1]
#
# image2 = his_data1.reshape(1, -1)
# paa = PiecewiseAggregateApproximation(
#     window_size=None, output_size=36
# )
# X_paa = paa.fit_transform(image2)
# X_paa = X_paa.reshape(-1)
# print("PAA:", KL(X_paa, his_data1))
