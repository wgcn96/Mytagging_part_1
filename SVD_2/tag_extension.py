
import os
import json
import numpy as np
import datetime
from sklearn.model_selection import train_test_split

from script.ori_data import get_60_spilit_matrix
from static import *
from movie_redundancy.prepare_data import array2file


# 取原来的矩阵，在原来矩阵的基础上，根据结果矩阵对movie进行n个tag扩充
def tag_extension(matrix, n=10):
    matrix_path = os.path.join(tensorflow_data_dir, "comprehensive_index_matrix.npy")
    result_matrix = np.load(matrix_path)
    shape = result_matrix.shape
    print("result_matrix shape: ", shape)

    index_matrix = np.argsort(-matrix, axis=1)
    sort_index_matrix = index_matrix[:, :]     # 每部电影取前n个标签

    # print(sort_index_matrix.shape)        # 应该是 59551 * n

    extension_matrix = np.zeros(shape)      # 补标签矩阵


    total_count = 0
    for i in range(shape[0]):
        count = 0
        for j in range(shape[1]):

            if count < n:
                cur_tag_pos = sort_index_matrix[i][j]
                if result_matrix[i][cur_tag_pos] != 1:
                    result_matrix[i][cur_tag_pos] = 1
                    extension_matrix[i][cur_tag_pos] = 1        # 每个电影补的标签
                    count += 1
            else:
                break
        total_count += count

    print("total extends tags: {}".format(total_count))
    print("extension_matrix shape", extension_matrix.shape)
    return result_matrix, extension_matrix, sort_index_matrix


# 取原来的矩阵，根据结果矩阵对movie取排序最高的前n个tag扩充（只取前n，原来没有就打上）
def tag_extension_2(matrix, n=10):
    matrix_path = os.path.join(tensorflow_data_dir, "comprehensive_index_matrix.npy")
    result_matrix = np.load(matrix_path)
    shape = result_matrix.shape
    print("result_matrix shape: ", shape)

    index_matrix = np.argsort(-matrix, axis=1)
    sort_index_matrix = index_matrix[:, :]     # 每部电影取前n个标签

    # print(sort_index_matrix.shape)        # 应该是 59551 * n

    extension_matrix = np.zeros(shape)      # 补标签矩阵

    total_count = 0
    for i in range(shape[0]):
        count = 0
        for j in range(n):

            cur_tag_pos = sort_index_matrix[i][j]
            if result_matrix[i][cur_tag_pos] != 1:
                result_matrix[i][cur_tag_pos] = 1
                extension_matrix[i][cur_tag_pos] = 1        # 每个电影补的标签
                count += 1

        total_count += count

    print("total extends tags: {}".format(total_count))
    print("extension_matrix shape", extension_matrix.shape)
    return result_matrix, extension_matrix, sort_index_matrix


def loadMatrix(dir, k):
    fileName = "resultMatrix_k={}.npy".format(k)
    filePath = os.path.join(dir, fileName)
    matrix = np.load(filePath)
    return matrix


def loadTagList(tagFile):
    tagList = []
    f = open(tagFile, 'r', encoding='utf-8')
    while True:
        line = f.readline()
        line = line.strip()

        if line == "" or line == None:
            break

        tag = line.split(" ")[0]
        tagList.append(tag)
    f.close()
    print(" tag List len: {}".format(len(tagList)))

    return tagList


def loadmovieList(movieFile):
    movieList = []
    f = open(movieFile, 'r', encoding='utf-8')
    while True:
        line = f.readline()
        line = line.strip()

        if line == "" or line == None:
            break

        movie = line.split(" ")[0]
        movieList.append(movie)
    f.close()
    print(" movie List len: {}".format(len(movieList)))

    return movieList


def matrix2json(matrix, movie_list, tag_list, filePath):
    f = open(filePath, encoding='utf-8', mode='w')

    movie_tag_dict = {}

    for pos, movie in enumerate(movie_list):
        cur_tags = matrix[pos]
        movie_tag_dict[movie] = []
        tag_indexes = cur_tags.tolist()
        for tag_pos, indicator in enumerate(tag_indexes):
            if indicator:
                movie_tag_dict[movie].append(tag_list[tag_pos])
            else:
                pass

    json.dump(movie_tag_dict, f, indent=4, sort_keys=True, ensure_ascii=False)
    f.close()


if __name__ == "__main__":
    '''
    matrix = np.load(os.path.join(tensorflow_data_dir, "abundant_movie_matrix.npy"))
    movie_rates_y = np.load(os.path.join(tensorflow_data_dir, "abundant_movie_rates.npy"))

    # X_train, X_test, y_train, y_test = train_test_split(matrix, movie_rates_y, random_state=1)
    X_train, y_train, X_test, y_test = get_60_spilit_matrix(matrix, movie_rates_y)
    array2file(X_train, y_train, SVD2_output_dir + '\\svm\\ori_train.data')
    array2file(X_test, y_test, SVD2_output_dir + '\\svm\\ori_test.data')

    print(datetime.datetime.now())
    '''

    '''
    # global variables

    k = 30

    matrix = loadMatrix(tensorflow_data_dir, k)
    # matrix = np.load(os.path.join(tensorflow_data_dir, 'result_matrix_2.npy'))

    loadmatrix = matrix
    matrix, extension_matrix, sort_index_matrix = tag_extension_2(matrix)
    movie_rates_y = np.load(os.path.join(tensorflow_data_dir, "abundant_movie_rates.npy"))

    # X_train, X_test, y_train, y_test = train_test_split(matrix, movie_rates_y, random_state=1)
    X_train, y_train, X_test, y_test = get_60_spilit_matrix(matrix, movie_rates_y)
    array2file(X_train, y_train, SVD2_output_dir + '\\svm\\k{}_train.data'.format(k))
    array2file(X_test, y_test, SVD2_output_dir + '\\svm\\k{}_test.data'.format(k))

    tagFile = os.path.join(tensorflow_data_dir, 'abundant_tag.txt')
    tag_list = loadTagList(tagFile)
    movieFile = os.path.join(tensorflow_data_dir, 'abundant_movie.txt')
    movie_list = loadmovieList(movieFile)
    SVD_movie_result = os.path.join(SVD2_output_dir, "SVD_movie_result_{}.json".format(k))
    matrix2json(extension_matrix, movie_list, tag_list, SVD_movie_result)

    print(datetime.datetime.now())
    '''


    # global variables
    k = 30
    n = 25

    matrix = loadMatrix(own_dir, k)
    matrix, extension_matrix, sort_index_matrix = tag_extension_2(matrix, n)
    movie_rates_y = np.load(os.path.join(tensorflow_data_dir, "abundant_movie_rates.npy"))

    # X_train, X_test, y_train, y_test = train_test_split(matrix, movie_rates_y, random_state=1)
    X_train, y_train, X_test, y_test = get_60_spilit_matrix(matrix, movie_rates_y)
    array2file(X_train, y_train, SVD2_output_dir + '\\svm\\own_k{}_train.data'.format(k))
    array2file(X_test, y_test, SVD2_output_dir + '\\svm\\own_k{}_test.data'.format(k))

    tagFile = os.path.join(tensorflow_data_dir, 'abundant_tag.txt')
    tag_list = loadTagList(tagFile)
    movieFile = os.path.join(tensorflow_data_dir, 'abundant_movie.txt')
    movie_list = loadmovieList(movieFile)
    SVD_movie_result = os.path.join(SVD2_own_output_dir, "own_result_{}.json".format(k))
    SVD_movie_result_ex = os.path.join(SVD2_own_output_dir, "own_result_{}_ex.json".format(k))
    matrix2json(matrix, movie_list, tag_list, SVD_movie_result)
    matrix2json(extension_matrix, movie_list, tag_list, SVD_movie_result_ex)

    print(datetime.datetime.now())

