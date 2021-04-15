#!/usr/bin/env python3


import numpy as np
import csv
import math
import matplotlib.pyplot as plt
from evaluation.DTW import dtw_distance
from pyts.approximation.paa import PiecewiseAggregateApproximation
from evaluation.ED import EDis
from evaluation.KL_Distance import KL
from evaluation.Person import Person_corr
from evaluation.LCSS import lcs,printLcs


class LttbException(Exception):
    pass


def LargestTriangleThreeBuckets(data, threshold):
    """
    LTTB的原始版本
    返回一个数据规约后的时序数据
    Parameters
    ----------
    data: list of lists/tuples
        数据必须以这种方式格式化: [[x,y], [x,y], [x,y], ...]
                                    or: [(x,y), (x,y), (x,y), ...]
    threshold: int
        threshold  >= 2 and <=  the len of data
    Returns
    -------
    data, 下采样基于 threshold
    """

    # 检查数据和阈值是否有效
    if not isinstance(data, list):
        raise LttbException("data is not a list")
    if not isinstance(threshold, int) or threshold <= 2 or threshold >= len(data):
        raise LttbException("threshold not well defined")
    for i in data:
        if not isinstance(i, (list, tuple)) or len(i) != 2:
            raise LttbException("datapoints are not lists or tuples")

    # Bucket尺寸. 为开始和结束数据点留出空间
    every = (len(data) - 2) / (threshold - 2)
    a = 0  # 最初的a是第一个点
    next_a = 0
    max_area_point = (0, 0)
    sampled = [data[0]]  # 一定加上第一个点

    for i in range(0, threshold - 2):
        # 计算下一个bucket的平均点 (containing c)
        avg_x = 0
        avg_y = 0
        avg_range_start = int(math.floor((i + 1) * every) + 1)
        avg_range_end = int(math.floor((i + 2) * every) + 1)
        avg_rang_end = avg_range_end if avg_range_end < len(data) else len(data)

        avg_range_length = avg_rang_end - avg_range_start

        while avg_range_start < avg_rang_end:
            avg_x += data[avg_range_start][0]
            avg_y += data[avg_range_start][1]
            avg_range_start += 1

        avg_x /= avg_range_length
        avg_y /= avg_range_length

        # 得到每个bucket的上下限
        range_offs = int(math.floor((i + 0) * every) + 1)
        range_to = int(math.floor((i + 1) * every) + 1)

        # Point a
        point_ax = data[a][0]
        point_ay = data[a][1]

        max_area = -1

        while range_offs < range_to:
            # 超过3buckets计算三角形面积
            area = math.fabs(
                (point_ax - avg_x)
                * (data[range_offs][1] - point_ay)
                - (point_ax - data[range_offs][0])
                * (avg_y - point_ay)
            ) * 0.5

            if area > max_area:
                max_area = area
                max_area_point = data[range_offs]
                next_a = range_offs  # Next a 就是 range_offs
            range_offs += 1

        sampled.append(max_area_point)  # 将满足条件的点从bucket取出加入新数据中
        a = next_a  # 迭代

    sampled.append(data[len(data) - 1])  # 将data时列中的最后一个point加入规约后的数据中

    return sampled


def ReadLTTBData1():
    with open('1.csv', 'r') as f:
        c = 0
        data = []
        csvf = csv.reader(f, delimiter=',')
        for row in csvf:
            if c == 0:
                c += 1
                continue
            data.append([int(row[0]), float(row[6])])
            c += 1
        data = data[0:576]
        sampled = LargestTriangleThreeBuckets(data, 36)
        with open('sampled.csv', 'w') as f2:
            csvf2 = csv.writer(f2, delimiter=',')
            for row in sampled:
                csvf2.writerow(row)
        return sampled


if __name__ == '__main__':
    his_data = np.loadtxt('1.csv', delimiter=",", skiprows=0)
    image1 = his_data[0:576, 6]
    plt.title("original data")
    plt.plot(his_data[0:576,0], image1)
    plt.show()

    image2 = image1.reshape(1,-1)
    paa = PiecewiseAggregateApproximation(
        window_size=None, output_size=36
    )
    X_paa = paa.fit_transform(image2)
    X_paa=X_paa.reshape(-1, 1)
    plt.title("PAA_36")
    plt.plot(his_data[:36, 0], X_paa)
    plt.show()
    plt.plot(his_data[:36, 0],X_paa, linestyle='none', marker='o')
    plt.show()

    print("PAA_DTW:",dtw_distance(image1,X_paa))
    print("paa_ED:", EDis(X_paa, image1))
    print("PAA_KL:",KL(X_paa, image1))
    X_paa = X_paa.reshape(-1)
    print("PAA_Pe:",Person_corr(X_paa,image1))
    flag = lcs(image1, X_paa)

    print("PAA_LCSS:",printLcs(flag, image1, len(image1), len(X_paa)))


    image = ReadLTTBData1()
    image = np.array(image)

    plt.title("LTTB_36")
    plt.plot(image[:,0], image[:,1])
    plt.show()
    print("LTTB_DTW:",dtw_distance(image1,image[:,1]))
    print("LTTB_ED:", EDis(image[:,1], image1))
    print("LTTB_KL:", KL(image[:,1], image1))

    print("LTTB_Pe:", Person_corr(image[:,1], image1))
    flag = lcs(image1, image[:,1])

    print("PAA_LCSS:", printLcs(flag, image1, len(image1), len(image[:,1])))
    plt.plot(his_data[:36, 0], image[:,1], linestyle='none', marker='o')
    plt.show()

    exit()
