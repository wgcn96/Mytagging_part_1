#

import os
import json
import numpy as np
import datetime
from sklearn.model_selection import train_test_split

from script.ori_data import get_60_spilit_matrix
from static import *

from moudle.ndcg.ndcg import *


def fillDir(basedir, filename, n):
    full_path = os.path.join(basedir, str(n), filename)
    # print(full_path)
    return full_path


def getSample(vector, n, condition):
    """
    采样函数
    :param vector: 行向量
    :param n: 采样本数量
    :param condition:  采样条件
    :return: 采样list转化为np array
    """
    candidate_list = []
    result_list = []
    for pos, item in enumerate(vector):
        if item == condition:
            candidate_list.append(pos)
    length = len(candidate_list)
    if length < n:
        print("采样数量大于最大采样值")
        return -1

    permatation = np.random.permutation(length)
    result_list = np.asarray(candidate_list)[permatation[:n]]
    # print(result_list)
    return result_list


def check(vector_a, vector_b):
    result = np.zeros(vector_a.shape)
    for pos, item in enumerate(vector_a):
        if item in vector_b:
            result[pos] = 1
    return result


def loadMatrix(dir, k, fileName=None):
    if fileName is None:
        fileName = "resultMatrix_k={}.npy".format(k)
    filePath = os.path.join(dir, fileName)
    matrix = np.load(filePath)
    return matrix


def tag_evaluate(matrix, top_h=10, n=10):
    matrix_path = fillDir(tensorflow_data_3_dir, "comprehensive_index_matrix_sample.npy", n)
    comprehensive_index_matrix = np.load(matrix_path)
    shape = comprehensive_index_matrix.shape
    # print("comprehensive_index_matrix shape: ", shape)

    index_matrix = np.argsort(-matrix, axis=1)
    sort_index_matrix = index_matrix[:, :]  # 每部电影取前top_h个标签

    # pos_list = np.load(os.path.join(tensorflow_data_3_dir, 'abundant_movie_pos.npy'))       # 20
    # pos_list = np.load(os.path.join(tensorflow_data_3_dir, '{}movie_pos.npy'.format(30)))     # 30
    # pos_list = np.load(os.path.join(tensorflow_data_3_dir, '{}movie_pos.npy'.format(40)))     # 40
    pos_list = np.load(os.path.join(tensorflow_data_3_dir, 'top250_movie_pos.npy'))     # 60


    result_hr = []
    result_ndcg = []
    for movie_pos in pos_list:
        # print('current {}'.format(movie_pos))
        count = 0
        vector = comprehensive_index_matrix[movie_pos]
        sample = getSample(vector, n, -1)

        top_h_vec = np.zeros((top_h,))
        for j in range(shape[1]):
            if count < top_h:
                cur_tag_pos = sort_index_matrix[movie_pos][j]
                if comprehensive_index_matrix[movie_pos][cur_tag_pos] != 1:
                    top_h_vec[count] = cur_tag_pos
                    count += 1
            else:
                break
        result = check(top_h_vec, sample)

        # hr, ndcg
        hr = hit_ratio_cal(result, top_h, n)
        ndcg = ndcg_cal(result, top_h, n)
        result_hr.append(hr)
        result_ndcg.append(ndcg)
    mean_hr = np.array(result_hr).mean()
    mean_ndcg = np.array(result_ndcg).mean()
    mean_hr = round(mean_hr, 3)
    mean_ndcg = round(mean_ndcg, 3)
    return mean_hr, mean_ndcg


if __name__ == '__main__':
    import pandas as pd
    k = 30
    # n = 5
    # top_h = 10
    '''
    for n in range(5, 25, 5):
        result_list = []
        for top_h in range(1, 21):
            current = []
            matrix = loadMatrix(server3_dir, k, "resultMatrix_k={}_sample_{}.npy".format(k, n))
            mean_hr, mean_ndcg = tag_evaluate(matrix, top_h, n)
            print(mean_hr, mean_ndcg)
            current.append(mean_hr)
            current.append(mean_ndcg)
            result_list.append(current)
        result_list = np.array(result_list)

        df = pd.DataFrame(result_list, index=[str(i) for i in range(1, 21)], columns=['hr', 'ndcg'])
        df = df.T
        df.to_csv('n{}_own.csv'.format(n))
    '''

    for n in range(1,3,1):
        result_list = []
        for top_h in range(1, 21):
            current = []
            matrix = loadMatrix(server3_dir, k, "resultMatrix_k={}_sample_{}.npy".format(k, n))
            mean_hr, mean_ndcg = tag_evaluate(matrix, top_h, n=5)
            print(mean_hr, mean_ndcg)
            current.append(mean_hr)
            current.append(mean_ndcg)
            result_list.append(current)
        result_list = np.array(result_list)

        df = pd.DataFrame(result_list, index=[str(i) for i in range(1, 21)], columns=['hr', 'ndcg'])
        df = df.T
        df.to_csv('n{}_own_review.csv'.format(n))
