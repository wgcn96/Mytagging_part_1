# -*- coding: utf-8 -*-

"""
note something here
将矩阵转秩为u-i，进行user-cf运算
"""

__author__ = 'Wang Chen'
__time__ = '2019/4/15'


import numpy as np

from baselines.static import *
from baselines.myuserCF import *

if __name__ == '__main__':
    k = 100
    rate_matrix = np.load((os.path.join(tensorflow_data_3_dir, 'movie_matrix.npy')))
    movie_num, tag_num = rate_matrix.shape
    similarity_matrix = np.eye(tag_num)
    rate_matrix = rate_matrix.T
    start = time.time()
    print('start ...')
    for i in range(tag_num):
        for j in range(i+1, tag_num, 1):
            similarity_ij = cosSimilarity(rate_matrix[i], rate_matrix[j])
            similarity_matrix[i][j] = similarity_matrix[j][i] = similarity_ij
    # 应保存similarity matrix
    np.save('icf_simi.npy', similarity_matrix)
    print('calculate time {}'.format(time.time() - start))
    check = np.argsort(-similarity_matrix, axis=1)
    print(check[:, 0])
    result_matrix = generateMatrix(rate_matrix, similarity_matrix, np.arange(tag_num), 10)
    result_matrix = result_matrix.T
    print(result_matrix.shape)
    print('total time {}'.format(time.time() - start))
    np.save('item_CF.npy', result_matrix)
