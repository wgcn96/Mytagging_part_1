"""
读电影详细属性文件，划分出最热门的1000部电影，并准备libsvm数据格式
"""
import numpy as np

import os


from static import *
from movie_redundancy.prepare_data import array2file


def get_60_split(X_path, y_path):

    matrix = np.load(X_path)
    print(np.sum(matrix))
    abundant_movie_matrix_path = os.path.join(tensorflow_data_dir, 'abundant_movie_matrix.npy')
    abundant_movie_matrix = np.load(abundant_movie_matrix_path)
    print(np.sum(abundant_movie_matrix))
    matrix = np.where(matrix == 1, matrix, abundant_movie_matrix)
    print(np.sum(matrix))
    y = np.load(y_path)
    _60_movie_pos_file_path = os.path.join(tensorflow_data_dir, '60_movie_pos.txt')
    _60_movie_id_file = open(_60_movie_pos_file_path, encoding='utf-8')
    id_list = []
    while True:
        line = _60_movie_id_file.readline()
        line = line.strip()

        if line == "" or line is None:
            break

        id_list.append(int(line))
    _60_movie_id_file.close()
    print(len(id_list))

    X_train = X[id_list,]
    y_train = y[id_list,]
    X_test = np.delete(X, id_list, 0)
    y_test = np.delete(y, id_list, 0)
    print(X_train.shape, y_train.shape)
    return X_train, y_train, X_test, y_test


def get_60_spilit_matrix(X,y):
    _60_movie_pos_file_path = os.path.join(tensorflow_data_dir, '60_movie_pos.txt')
    _60_movie_id_file = open(_60_movie_pos_file_path, encoding='utf-8')
    id_list = []
    while True:
        line = _60_movie_id_file.readline()
        line = line.strip()

        if line == "" or line is None:
            break

        id_list.append(int(line))
    _60_movie_id_file.close()
    print(len(id_list))

    X_train = X[id_list,]
    y_train = y[id_list,]
    X_test = np.delete(X, id_list, 0)
    y_test = np.delete(y, id_list, 0)
    print(X_train.shape, y_train.shape)
    return X_train, y_train, X_test, y_test


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


if __name__ == "__main__":
    # print(1)
    # X_train, X_test, y_train, y_test = prepare_data(m1matrix_2)
    # X_train, y_train = shuffle_in_unison(X_train, y_train)
    # array2file(X_train, y_train, redundancy_output_dir + '\\svm\\m1_train_2.data')
    # array2file(X_test, y_test, redundancy_output_dir + '\\svm\\m1_test_2.data')

    print("prepare r1matrix data")
    abundant_movie_matrix = os.path.join(redundancy2_output_dir, 'r1matrix_2.npy')
    abundant_movie_rates = os.path.join(tensorflow_data_dir, 'abundant_movie_rates.npy')
    X_train, y_train, X_test, y_test = get_60_split(abundant_movie_matrix, abundant_movie_rates)
    array2file(X_train, y_train, redundancy2_output_dir + '\\svm\\m1_train_addi.data')
    array2file(X_test, y_test, redundancy2_output_dir + '\\svm\\m1_test_addi.data')

