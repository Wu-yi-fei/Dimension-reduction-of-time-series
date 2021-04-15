import numpy as np

def SegmentationMean(start,end,data):
    data_reduce = []
    for i in range(0,len(start)):
        value = np.mean(data[start[i]:end[i],1])
        data_reduce.append(value)
    return data_reduce