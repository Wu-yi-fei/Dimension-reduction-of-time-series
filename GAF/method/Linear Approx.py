#!/usr/bin/env python3

"""
采用分段线性分割PLR方法对时间序列信号进行分割
包括了自下而上、自上而下

"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as pyplot
from evaluation.DTW import dtw_distance
import matplotlib.pyplot as plt
from evaluation.ED import EDis
from utils.segMean import SegmentationMean
from evaluation.KL_Distance import KL
from evaluation.Person import Person_corr
from evaluation.LCSS import lcs,printLcs
from utils.GAF import GAF

def CalculateError(T_seq, method='regression', column=1):
    """ 给定时间序列序列，计算线性分割近似误差，第 0 列默认为 x
    Args:
      T_seq: numpy时间序列序列数组，两列，一列表示时间，一列表示值
      method: 指定近似方法类型的字符串，即“回归”或“插值”
      column: 输入序列中用作y的列数
    Return:
      approximation error: 平方根误差
    """

    global error
    if method == 'regression':
        A = np.vstack([T_seq[:, 0], np.ones(len(T_seq[:, 0]))]).T
        # print T_seq[:,0]
        A = np.array(A, dtype='float')
        y = T_seq[:, column]
        slope, intercept = np.linalg.lstsq(A, y)[0]
        error = np.sqrt(np.sum(((A[:, 0] * slope + intercept - y) ** 2)))
    elif method == 'interpolation':
        return 0
    return error


def sliding_window_segment(seq, max_error, step_length=1, method='regression', column=1, online=False):
    """给定时间序列序列，应用sliding_window算法对序列进行分割。分割的标准是分段线性回归的近似误差
    Args:
      seq: 时间序列
      max_error: max_error决定接受或拒绝段，<error拒绝并生成新的分段，>error接受并添加到当前段
      step_length: 滑动步长点
      method: string to decide whether to use 'regression' or 'interpolation' to
      approximate error
      column: 列作为y来计算误差
      online: if True, return the first segment at the time get it
    Returns:
      起始和结束分段点
    """

    anchor = 0
    # seg_arr = []
    start_anchors = []
    end_anchors = []
    # anchor_arr = []
    while anchor < np.shape(seq)[0]:
        right_end = 2
        start_anchors.append(anchor)
        start_anchor = anchor
        while anchor + right_end <= np.shape(seq)[0] and CalculateError(seq[anchor:anchor + right_end, :],
                                                                          method=method, column=column) < max_error:
            right_end += step_length
        end_anchor = anchor + right_end - 1
        if online:
            return start_anchor, end_anchor
        else:
            # seg_arr.append(seq[anchor:anchor+right_end-1,:])
            anchor += right_end - 1
            end_anchors.append(anchor)
    # anchor_arr = np.array([start_anchors, end_anchors])
    return start_anchors, end_anchors


def bottom_up_merge(seq, max_error, method='regression', column=1):
    """给定一个时间序列序列，将其分成n/2个片段，并将其合并成尽可能大的片段，使所有片段的近似误差小于最大误差
    Args:
      seq: 时间序列
      max_error: 最大误差
    Return:
      段头和段尾
    """

    seg_arr = []
    cost_arr = []
    start_anchors = [0]
    end_anchors = []
    for i in range(0, np.shape(seq)[0] - 1, 2):
        seg_arr.append(seq[i:i + 2, :])
    for i in range(0, len(seg_arr) - 1):
        cost_arr.append(CalculateError(np.vstack((seg_arr[i], seg_arr[i + 1])), method=method, column=column))
    while len(seg_arr) > 1 and np.amin(cost_arr) < max_error:
        index = cost_arr.index(np.amin(cost_arr))
        seg_arr[index] = np.vstack((seg_arr[index], seg_arr[index + 1]))
        del seg_arr[index + 1]
        del cost_arr[index]
        if index + 1 < len(seg_arr):
            cost_arr[index] = CalculateError(np.vstack((seg_arr[index], seg_arr[index + 1])), method=method,
                                               column=column)
        if index > 0:
            cost_arr[index - 1] = CalculateError(np.vstack((seg_arr[index - 1], seg_arr[index])), method=method,
                                                   column=column)
    for seg in seg_arr:
        start_anchors.append(start_anchors[-1] + len(seg))
        end_anchors.append(start_anchors[-1])
    start_anchors = start_anchors[:-1]
    # anchor_arr = np.array([start_anchors[:-1], end_anchors])
    return start_anchors, end_anchors


def SWAB_segment(seq, max_error, buffer_size, method='regression', column=1):
    """给定一个时间序列序列，用抽签算法对序列进行分割
    SWAB算法
    Args:
      seq: 输入要分割的序列
      max_error: 最大误差
      buffer_size: 每个循环的缓冲区大小
      column: 用来计算误差的列数
    Return:
      开始段点和结束段点
    """
    buffer_seq = seq[0:buffer_size, :]
    # seg_arr = []
    start_anchors = [0]
    end_anchors = []
    while len(buffer_seq) > 1 and start_anchors[-1] < np.shape(seq)[0]:
        st, et = bottom_up_merge(buffer_seq, max_error=max_error, method=method, column=column)
        # seg_arr.append(buffer_seg_arr[0])
        start_anchors.append(start_anchors[-1] + et[0] - st[0])
        end_anchors.append(start_anchors[-1])
        if len(seq[end_anchors[-1]:, :]) != 0:
            online_st, online_et = sliding_window_segment(seq[end_anchors[-1]:, :], max_error=max_error, column=column,
                                                          online=True)
            print(online_st,online_st)
            new_seq = seq[end_anchors[-1]:end_anchors[-1] + online_et, :]
            buffer_seq = new_seq[0:buffer_size, :]
    # anchor_arr = np.array([start_anchors[:-1], end_anchors])
    start_anchors = start_anchors[:-1]
    return start_anchors, end_anchors


def unit_tests():
    # 测试
    seq = np.loadtxt('2.csv', delimiter=",", skiprows=0)
    seq = seq[0:576, 0:2]

    print(seq)
    err = CalculateError(seq, method='regression')
    print("test _calcuate_error")
    print("test passed:", err < 10 ** -10)
    # test SlidingWindow
    input_seq = seq
    print(input_seq)
    pyplot.plot(input_seq[:, 0], input_seq[:, 1])
    start_anchors, end_anchors = sliding_window_segment(input_seq, max_error=1)
    print("=============Test sliding_window_segmentation ============" )
    print("=============Start Anchors======================")
    print(start_anchors)
    print("=============End Anchors======================")
    print(end_anchors)
    print(len(start_anchors))
    pyplot.vlines(start_anchors, 2, 3, color='red')

    bu = SegmentationMean(start_anchors, end_anchors, seq)

    print(len(bu))
    print("BU:", dtw_distance(bu, seq[:576, 1]))
    print("SWAB_DTW:", dtw_distance(bu, seq[:576, 1]))
    print("SWAB_ED:", EDis(bu, seq[:576, 1]))
    print("LTTB_KL:", KL(bu, seq[:576, 1]))
    print("LTTB_Pe:", Person_corr(bu, seq[:576, 1]))
    flag = lcs(seq[:576, 1], bu)
    print("PAA_LCSS:", printLcs(flag, seq[:576, 1], len(seq[:576, 1]), len(bu)))
    image_old = GAF(bu, len(bu))
    hm = sns.heatmap(image_old[0], cmap='Greens', cbar=True, annot=None, square=True, annot_kws={"size": 10})
    plt.ylabel("time", fontsize=15)
    plt.xlabel("time", fontsize=15)
    plt.show()

    plt.title("curve")
    plt.plot(seq[:len(bu), 0], bu)
    plt.show()

    # test BU
    pyplot.figure()
    pyplot.plot(input_seq[:, 0], input_seq[:, 1])
    start_anchors, end_anchors = bottom_up_merge(input_seq, max_error=1)
    print("=============Test bottom_up_merge ============")
    print("=============Start Anchors======================")
    print(start_anchors)
    print("=============End Anchors======================" )
    print(end_anchors)
    print(len(start_anchors))
    pyplot.vlines(start_anchors, 2, 3, color='red')

    # test SWAB segment algorithm
    pyplot.figure()
    pyplot.plot(input_seq[:, 0], input_seq[:, 1])
    start_anchors, end_anchors = SWAB_segment(input_seq, max_error=3, buffer_size=17)
    print("=============Test SWAB segmentation ============")
    print("=============Start Anchors======================")
    print(start_anchors)
    print("=============End Anchors======================")
    print(end_anchors)
    print(len(start_anchors))
    swab = SegmentationMean(start_anchors,end_anchors,seq)
    print(len(swab))
    # print("SWAB_DTW:", dtw_distance(swab, seq[:576, 1]))
    # print("SWAB_ED:",EDis(swab,seq[:576,1]))
    # print("LTTB_KL:", KL(swab, seq[:576,1]))
    # print("LTTB_Pe:", Person_corr(swab, seq[:576,1]))
    # flag = lcs(seq[:576,1], swab)
    # print("PAA_LCSS:", printLcs(flag, seq[:576,1],len(seq[:576,1]), len(swab)))
    # image_old = GAF(swab,37)
    # hm = sns.heatmap(image_old[0], cmap='Greens', cbar=True, annot=None, square=True, annot_kws={"size": 10})
    # plt.ylabel("time", fontsize=15)
    # plt.xlabel("time", fontsize=15)
    # plt.show()

    pyplot.figure()
    pyplot.plot(input_seq[:, 0], input_seq[:, 1])
    pyplot.vlines(start_anchors, 2, 3, color='red')
    pyplot.show()

    plt.title("curve")
    plt.plot(seq[:37, 0], swab)
    plt.show()
    # sys.exit(1)



if __name__ == "__main__":
    import cProfile

    pr = cProfile.Profile()
    pr.enable()
    unit_tests()
    pr.disable()
    pr.create_stats()