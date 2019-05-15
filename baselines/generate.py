import numpy as np
import random
import os

from static import *

'''
if __name__ == '__main__':
    for n in range(5, 25, 5):
        matrix = np.load(os.path.join(data_dir, str(n), 'ori_matrix_sample_{}.npy'.format(n)))
        # ground_truth = np.load('extreme_ground_truth.npy')

        with open('matrix/mt{}_ori.train.rating'.format(n), 'w') as f:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    result = ""
                    if matrix[i][j] == 1:
                        result += str(i)
                        result += '\t'
                        result += str(j)
                        result += '\t'
                        result += '1'
                        result += '\n'
                    f.write(result)

    n = 100
    matrix = np.load(os.path.join(data_dir, str(n), 'ori_matrix_sample_{}.npy'.format(n)))
    # ground_truth = np.load('extreme_ground_truth.npy')

    with open('matrix/mt100_ori.train.rating', 'w') as f:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                result = ""
                if matrix[i][j] == 1:
                    result += str(i)
                    result += '\t'
                    result += str(j)
                    result += '\t'
                    result += '1'
                    result += '\n'
                f.write(result)


'''


if __name__ == '__main__':
    matrix = np.load(os.path.join(tensorflow_data_3_dir, 'movie_matrix.npy'))
    (movie_count, tag_count) = matrix.shape
    # ground_truth = np.load('extreme_ground_truth.npy')
    neg_sample = 4
    k = 0
    with open('data/mt_ori.train.rating', 'w') as f:
        '''
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):

                if matrix[i][j] == 1:
                    k += 1
                    result = ""
                    result += str(i)
                    result += '\t'
                    result += str(j)
                    result += '\t'
                    result += '1'
                    result += '\n'
                    f.write(result)
                    '''
        # begin possample
        pos_sample = np.where(matrix == 1)
        x_data = np.array(pos_sample).T  # 转化为480000*2的二维
        k = x_data.shape[0]
        for item in x_data:
            result = ""
            result += str(item[0])
            result += '\t'
            result += str(item[1])
            result += '\t'
            result += '1'
            result += '\n'
            f.write(result)

        # begin negsample
        count = 0
        while count < neg_sample*k:
            movie_index = np.random.randint(movie_count)
            tag_index = np.random.randint(tag_count)
            if matrix[movie_index][tag_index] != 1:
                count += 1
                result = ""
                result += str(movie_index)
                result += '\t'
                result += str(tag_index)
                result += '\t'
                result += '0'
                result += '\n'
                f.write(result)
