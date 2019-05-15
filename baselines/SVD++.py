# -*- coding: utf-8 -*-

"""
note something here

"""

__author__ = 'Wang Chen'
__time__ = '2019/5/12'

import os
import time
import datetime

import numpy as np

from surprise import SVD, SVDpp
from surprise import Dataset, dataset, Reader, Trainset
# from surprise.model_selection import cross_validate

from baselines.static import *


if __name__ == '__main__':

    # Load the movielens-100k dataset (download it if needed).
    reader = Reader(line_format='user item rating', sep='\t', rating_scale=(0, 1))
    file_path = os.path.join(baseline_dir, 'data', 'mt_ori.train.rating')
    data = Dataset.load_from_file(file_path, reader=reader)

    trainset = data.build_full_trainset()

    # Use the famous SVD algorithm.
    algo = SVDpp()

    print("begin to fit the model")
    start = time.time()
    algo.fit(trainset)
    print(time.time() - start)

    print("begin to predict")
    matrix = np.load(os.path.join(tensorflow_data_3_dir, 'movie_matrix.npy'))   # 这里应该需要转化成test_data的格式
    (movie_count, tag_count) = matrix.shape
    result = []
    for i in range(movie_count):
        all_rate_list = []
        for j in range(tag_count):
            all_rate_list.append((str(i), str(j), matrix[i][j]))
        predictions = algo.test(all_rate_list)
        for uid, iid, true_r, est, _ in predictions:
            result.append(est)

    result = np.array(result)
    result = np.reshape(result, matrix.shape)
    np.save('svdpp_ori.npy', result)
    print(time.time() - start)


'''
if __name__ == '__main__':
    matrix = np.load('svdpp_ori.npy')
'''