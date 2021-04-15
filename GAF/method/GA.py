#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 576            # DNA 长度
POP_SIZE = 20            # population 尺寸
CROSS_RATE = 0.6
MUTATION_RATE = 0.01
N_GENERATIONS = 200
X_BOUND = [0, 13]


def F(x): pass    # 找到函数最大值


class MGA(object):
    """
    基于CRO算法做时间序列降维论文的思想做
    """
    def __init__(self, DNA_size, DNA_bound, cross_rate, mutation_rate, pop_size):
        pass


    def translateDNA(self, pop):
       pass

    def get_fitness(self, product):
        pass


    def crossover(self, loser_winner):      # 交叉配对
        pass

    def mutate(self, loser_winner):         # 突变
        pass

    def evolve(self, n):    # 基于适应度函数的自然选择
        pass


