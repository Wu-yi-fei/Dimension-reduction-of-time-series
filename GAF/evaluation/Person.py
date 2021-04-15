#!/usr/bin/env python3



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
#from method.LTTB import ReadLTTBData, LargestTriangleThreeBuckets, LttbException
from pyts.approximation.paa import PiecewiseAggregateApproximation

from math import sqrt

def multipl(a,b):
    sumofab=0.0
    for i in range(len(a)):
        temp=a[i]*b[i]
        sumofab+=temp
    return sumofab

def corrcoef(x,y):
    z=[]
    every = len(y) / len(x)
    for i in range(len(x)):
        z.append(y[int(np.float(i * every))])
    y = z
    n=len(x)
    sum1=sum(x)
    sum2=sum(y)
    sumofxy=multipl(x,y)
    sumofx2 = sum([pow(i,2) for i in x])
    sumofy2 = sum([pow(j,2) for j in y])
    num=sumofxy-(float(sum1)*float(sum2)/n)

    den=sqrt((sumofx2-float(sum1**2)/n)*(sumofy2-float(sum2**2)/n))
    return num/den

def Person_corr(x,y):
    z = []
    every = len(y) / len(x)
    for i in range(len(x)):
        z.append(y[int(np.float(i * every))])
    y = z
    return np.corrcoef(x,y)

# his_data = np.loadtxt('1.csv', delimiter=",", skiprows=0)
# his_data1 = his_data[0:576,4]
# image = ReadLTTBData()
# image = np.array(image)
# image = image[:,1]
# print(image)
# print( corrcoef(image,his_data1))
# plt.title("curve")
# plt.plot(his_data[:36, 0], image)
# plt.show()
#
# image2 = his_data1.reshape(1,-1)
# paa = PiecewiseAggregateApproximation(
#     window_size=None, output_size=36
# )
# X_paa = paa.fit_transform(image2)
# X_paa = X_paa.reshape(-1)
# print( corrcoef(X_paa,his_data1))
# print(Person_corr(X_paa,his_data1))