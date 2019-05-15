# -*- coding: utf-8 -*-

"""
note something here
user based CF
计算余弦相似度
取最近邻的用户进行打分，生成结果矩阵
"""

__author__ = 'Wang Chen'
__time__ = '2019/4/15'

import numpy as np
import os
import time

from baselines.static import *


def cosSimilarity(a, b):
    return np.vdot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def generateMatrix(rate_matrix, similiarity_matrix, pos_list, neightbor_n):
    """
    根据k近邻生成结果矩阵。计算公式：weight_j* rate_j
    :param rate_matrix:  打分矩阵
    :param similiarity_matrix:  相似性矩阵
    :param pos_list:  求k近邻打分的电影的pos列表
    :param neightbor_n:  n近邻
    :return:
    """
    argsort_matrix = np.argsort(-similiarity_matrix, axis=1)
    result_matrix = np.zeros(rate_matrix.shape, dtype=np.float32)
    for target_pos in pos_list:
        # 最相近的一定是自己，权重为1。因此取前[1,n+1)的元素
        most_similiar_movie_list = argsort_matrix[target_pos, 1:neightbor_n + 1]

        weight_list = similiarity_matrix[most_similiar_movie_list, target_pos]
        neighbor_rate_matrix = rate_matrix[most_similiar_movie_list,].T
        weight_matrix = np.multiply(weight_list, neighbor_rate_matrix).T
        result_row = np.sum(weight_matrix, axis=0)
        result_matrix[target_pos] = result_row

        '''
        result = [0 for i in range(rate_matrix.shape[1])]
        for i, weight in enumerate(weight_list):
            neighbor_movie = most_similiar_movie_list[i]
            result += weight*rate_matrix[neighbor_movie]
            '''
    return result_matrix


'''
if __name__ == '__main__':
    rate_matrix = np.array([[1, 2, 3, 3],
                            [4, 5, 6, 6],
                            [7, 8, 9, 9]])
    x = np.array([[1, 0.2, 0.3],
                  [0.2, 1, 0.6],
                  [0.3, 0.6, 1]])
    result = generateMatrix(rate_matrix, x, pos_list=np.arange(3), neightbor_n=2)
    print(result)
    pass
'''


if __name__ == '__main__':
    k = 5
    rate_matrix = np.load((os.path.join(tensorflow_data_3_dir, 'movie_matrix.npy')))
    movie_num, tag_num = rate_matrix.shape
    similarity_matrix = np.eye(movie_num)
    start = time.time()
    print('start ...')
    for i in range(movie_num):
        for j in range(i+1, movie_num, 1):
            if i == j:
                pass
            similarity_ij = cosSimilarity(rate_matrix[i], rate_matrix[j])
            similarity_matrix[i][j] = similarity_matrix[j][i] = similarity_ij
    # 应保存similarity matrix
    np.save('ucf_simi.npy', similarity_matrix)
    print('calculate time {}'.format(time.time() - start))
    check = np.argsort(-similarity_matrix, axis=1)
    print(check[:, 0])
    result_matrix = generateMatrix(rate_matrix, similarity_matrix, np.arange(movie_num), k)
    print('total time {}'.format(time.time() - start))
    np.save()
