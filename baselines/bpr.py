# -*-encoding:utf-8 -*-

"""

"""

import numpy as np
from scipy.sparse import coo_matrix
import datetime
import os

# from lightfm.datasets import fetch_movielens
from lightfm import LightFM

from static import *

# fetch data and format it
#data = fetch_movielens(min_rating=4.0)  # only collect the movies with a rating of 4 or higher



if __name__ == "__main__":
    missing_n = 100
    epoch = 5
    data = np.load(os.path.join(tensorflow_data_3_dir, str(missing_n), 'ori_matrix_sample_{}.npy'.format(missing_n)))
    a = np.where(data == -1)
    data[a[0], a[1]] = 0
    print(np.sum(data))
    data = coo_matrix(data)

    # print(data.toarray())
    '''repr()函数将对象转化为供解释器读取的形式'''
    result = np.zeros(data.shape)
    # create model
    model = LightFM(no_components=30, loss='bpr')  # warp = weighted approximate-rank pairwise

    print(datetime.datetime.now())
    model.fit(data, epochs=epoch, num_threads=2, verbose=True)
    print(datetime.datetime.now())

    n_users, n_items = data.shape

    for i in range(n_users):
        scores = model.predict(i, np.arange(n_items))
        result[i] = scores

    np.save(os.path.join(baseline_output_dir, 'bpr_{}_wtreview.npy'.format(missing_n)), result)
