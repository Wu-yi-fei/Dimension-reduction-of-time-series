#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
# from method.LTTB import ReadLTTBData1, LargestTriangleThreeBuckets, LttbException
from pyts.approximation.paa import PiecewiseAggregateApproximation

def EDis(x,y):
    """
    一种经过修正的欧氏距离测法，可以用来检验两个不等长序列的欧氏距离
    :param x:规约后数据
    :param y:规约前数据
    :return: 修正后的distance
    """
    distance = 0

    every = len(y) / len(x)

    for j in range(int(every)):
        z=[]
        for i in range(len(x)):
            z.append(y[int(np.float(i*every+j))])
        y1 = z
        s1 = (x - np.mean(x)) / np.std(x)
        s2 = (y1 - np.mean(y1)) / np.std(y1)
        distance1 = np.sqrt(np.sum(np.square(s1 - s2)))

        ASD = abs(np.sum(s1 - s2))
        SAD = np.sum(abs(s1 - s2))
        if SAD == 0:
            return 0
        else:
            distance += distance1 * (2 - ASD / SAD)

    return distance/every






his_data = np.loadtxt('1.csv', delimiter=",", skiprows=0)
his_data1 = his_data[0:576,6]
# image = ReadLTTBData1()
#image = np.array(image)
#image = image[:,1]

#print(EDis(image,his_data1))

#plt.title("curve")
#plt.plot(his_data[:36, 0], image)
#plt.show()

image2 = his_data1.reshape(1,-1)
paa = PiecewiseAggregateApproximation(
    window_size=None, output_size=36
)
X_paa = paa.fit_transform(image2)
X_paa = X_paa.reshape(-1)
