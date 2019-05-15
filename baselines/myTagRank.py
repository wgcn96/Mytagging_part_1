# -*- coding: utf-8 -*-

"""
note something here

"""

import numpy as np
import os
import time

from baselines.static import *


def cosSimilarity(a, b):
    return np.vdot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


__author__ = 'Wang Chen'
__time__ = '2019/5/3'


if __name__ == '__main__':
    print("Loading rating matrix")
    rate_matrix = np.load((os.path.join(tensorflow_data_3_dir, 'movie_matrix.npy')))
    movie_num, tag_num = rate_matrix.shape
    similarity_matrix = np.eye(movie_num)
    start = time.time()
    print("Caculating Similar_matrix")

    for i in range(movie_num):
        print(i)
        for j in range(i + 1, movie_num):
            result = cosSimilarity(rate_matrix[i], rate_matrix[j])
            if result > 0.3:
                similarity_matrix[i][j] = result
                similarity_matrix[j][i] = result

    np.save('rank_simi.npy', similarity_matrix)

    print('calculate time {}'.format(time.time() - start))
    check = np.argsort(-similarity_matrix, axis=1)
    print(check[:, 0])
    iteration = 1

    for n in range(iteration):
        for j in range(rate_matrix.shape[1]):
            print(j)
            rate_matrix[:, j] = np.dot(similarity_matrix, rate_matrix[:, j].reshape(-1, 1)).reshape(-1)
    print('total time {}'.format(time.time() - start))
    np.save('TagRank.npy', rate_matrix)
