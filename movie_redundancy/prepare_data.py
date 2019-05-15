"""
划分libsvm数据格式

"""

from sklearn.model_selection import train_test_split

import numpy as np
import os

from static import *


def get_data(X_path, y_path):
    X = np.load(X_path)
    y = np.load(y_path)
    return train_test_split(X, y, random_state=1)


def array2file(ndarray, y, file):
    f = open(file, 'w')
    shape = ndarray.shape
    for i in range(shape[0]):
        vec = ndarray[i]
        line = ''
        line += str(y[i][0])
        line += ' '
        pos = 0
        for item in vec:
            pos += 1
            if item:
                line += str(pos)
                line += ':'
                line += str(item)
                line += ' '
        line += '\n'
        f.write(line)
    f.close()


if __name__ == "__main__":
    '''
    m1_matrix_path = os.path.join(redundancy_output_dir, 'm1_matrix.npy')
    m2_matrix_path = os.path.join(redundancy_output_dir, 'm2_matrix.npy')
    m4_matrix_path = os.path.join(redundancy_output_dir, 'm4_matrix.npy')
    m8_matrix_path = os.path.join(redundancy_output_dir, 'm8_matrix.npy')
    redundancy_y_path = os.path.join(redundancy_output_dir, 'movie_redundancy_y.npy')

    print(1)
    X_train, X_test, y_train, y_test = get_data(m1_matrix_path, redundancy_y_path)
    array2file(X_train, y_train, redundancy_output_dir + '\\svm\\m1_train.data')
    array2file(X_test, y_test, redundancy_output_dir + '\\svm\\m1_test.data')

    print(2)
    X_train, X_test, y_train, y_test = get_data(m2_matrix_path, redundancy_y_path)
    array2file(X_train, y_train, redundancy_output_dir + '\\svm\\m2_train.data')
    array2file(X_test, y_test, redundancy_output_dir + '\\svm\\m2_test.data')

    print(3)
    X_train, X_test, y_train, y_test = get_data(m4_matrix_path, redundancy_y_path)
    array2file(X_train, y_train, redundancy_output_dir + '\\svm\\m4_train.data')
    array2file(X_test, y_test, redundancy_output_dir + '\\svm\\m4_test.data')

    print(4)
    X_train, X_test, y_train, y_test = get_data(m8_matrix_path, redundancy_y_path)
    array2file(X_train, y_train, redundancy_output_dir + '\\svm\\m8_train.data')
    array2file(X_test, y_test, redundancy_output_dir + '\\svm\\m8_test.data')
    '''

    m1_matrix_path = os.path.join(redundancy2_output_dir, 'r1matrix_2.npy')
    m2_matrix_path = os.path.join(redundancy2_output_dir, 'r2matrix_2.npy')
    m4_matrix_path = os.path.join(redundancy2_output_dir, 'r4matrix_2.npy')
    m8_matrix_path = os.path.join(redundancy2_output_dir, 'r8matrix_2.npy')
    redundancy_y_path = os.path.join(tensorflow_data_dir, 'abundant_movie_rates.npy')

    print(1)
    X_train, X_test, y_train, y_test = get_data(m1_matrix_path, redundancy_y_path)
    array2file(X_train, y_train, redundancy2_output_dir + '\\svm\\m1_train.data')
    array2file(X_test, y_test, redundancy2_output_dir + '\\svm\\m1_test.data')

    print(2)
    X_train, X_test, y_train, y_test = get_data(m2_matrix_path, redundancy_y_path)
    array2file(X_train, y_train, redundancy2_output_dir + '\\svm\\m2_train.data')
    array2file(X_test, y_test, redundancy2_output_dir + '\\svm\\m2_test.data')

    print(3)
    X_train, X_test, y_train, y_test = get_data(m4_matrix_path, redundancy_y_path)
    array2file(X_train, y_train, redundancy2_output_dir + '\\svm\\m4_train.data')
    array2file(X_test, y_test, redundancy2_output_dir + '\\svm\\m4_test.data')

    print(4)
    X_train, X_test, y_train, y_test = get_data(m8_matrix_path, redundancy_y_path)
    array2file(X_train, y_train, redundancy2_output_dir + '\\svm\\m8_train.data')
    array2file(X_test, y_test, redundancy2_output_dir + '\\svm\\m8_test.data')

    '''
        print(np.sum(X))
    ori_data = np.load('D:\\workProject\\pythonProject\\Mytagging_part_2\\data_2\\abundant_movie_matrix.npy')
    print(ori_data.shape)
    print(np.sum(ori_data))
    X = np.where(ori_data == 1, ori_data, X)

    print(np.sum(X))
    '''