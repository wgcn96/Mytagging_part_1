# -*- coding: utf-8 -*-
# python3
#

"""
main 函数
标签扩展
"""

__author__ = 'Wang Chen'

import sys
sys.path.extend(['D:\\workProject\\pythonProject\\Mytagging\\count',\
                'D:\\workProject\\pythonProject\\Mytagging'])

import os
import json
import numpy as np
import datetime

from genre_prediction.static import *
from genre_prediction.util import *


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
    matrix_path = os.path.join(tensorflow_data_3_dir, "comprehensive_index_matrix.npy")
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


def train_test_split(matrix, genre, all_genres_dict, file_path):
    """
    生成某个genre的训练测试数据，并写入file路径文件
    :param matrix:
    :param genre:
    :param all_genres_dict:
    :return: 四个矩阵
    """

    train_pos_list = []
    test_pos_list = []
    all_test_pos_list = np.load(os.path.join(tensorflow_data_3_dir, 'top250_movie_pos.npy'))
    y_all = np.zeros((19004), dtype=np.int32)
    movie_list, _ = loadmovieList(os.path.join(tensorflow_data_3_dir, 'movie.txt'))
    for pos, movie in enumerate(movie_list):
        if movie not in all_genres_dict:
            continue
        else:

            genres_list = all_genres_dict[movie]
            if genre in genres_list:
                y_all[pos] = 1

            if pos in all_test_pos_list:
                test_pos_list.append(pos)
            else:
                train_pos_list.append(pos)

    y_train = y_all[train_pos_list]
    y_test = y_all[test_pos_list]
    X_train = matrix[train_pos_list, ]
    X_test = matrix[test_pos_list, ]

    array2file(X_train, y_train, file=file_path + '.train')
    array2file(X_test, y_test, file=file_path + '.test')
    return X_train, X_test, y_train, y_test


def four_folder_split(matrix, genre, all_genres_dict, file_path, file_name):
    """
    生成某个genre的训练测试数据，并写入file路径文件
    :param matrix: ndarray 扩展后的电影和标签矩阵
    :param genre: str 进行二分类的电影类型
    :param all_genres_dict: dict 所有电影和其类型的字典
    :param file_path: str 根路径
    :param file_name: str 文件名
    :return: 切分后的四个矩阵
    """

    y_all = np.zeros((19004), dtype=np.int32)
    pos_list = []
    movie_list, _ = loadmovieList(os.path.join(tensorflow_data_3_dir, 'movie.txt'))
    for pos, movie in enumerate(movie_list):
        if movie not in all_genres_dict:
            continue
        else:
            pos_list.append(pos)
            genres_list = all_genres_dict[movie]
            if genre in genres_list:
                y_all[pos] = 1

    # pos_list = np.array(pos_list, dtype=np.int32)
    print(len(pos_list))
    print(len(y_all))
    split_length = int(len(pos_list)/4)

    for i in range(4):
        root_path = os.path.join(file_path, 'folder_{}'.format(i+1))
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        head = i*split_length
        tail = (i+1)*split_length
        test_index = pos_list[head:tail]
        train_index = pos_list[:head] + pos_list[tail:]
        y_train = y_all[train_index]
        y_test = y_all[test_index]
        X_train = matrix[train_index, ]
        X_test = matrix[test_index, ]

        array2file(X_train, y_train, file=os.path.join(root_path, file_name+'.train'))
        array2file(X_test, y_test, file=os.path.join(root_path, file_name+'.test'))
    return X_train, X_test, y_train, y_test


def remove_tag_in_genre(matrix, tags_dict, genres_list):
    """
    将电影的标签中包含其类型的列移除
    :param matrix: ndarray 电影标签矩阵
    :param tags_dict: dict 标签对应matrix列号的dict
    :param genres_list: list 电影类型列表
    :return: matrix， revised后的matrix
    """
    remove_pos_list = []
    for genre in genres_list:
        pos = tags_dict.get(genre, -1)
        if pos > -1:
            remove_pos_list.append(pos)
    matrix = np.delete(matrix, remove_pos_list, axis=1)
    print("total remove length: {}".format(len(remove_pos_list)))
    print("matrix shape: ", matrix.shape)
    return matrix


if __name__ == "__main__":
    k = 30
    n = 10
    filePath = 'all_genres_dict.json'
    f = open(filePath, 'r', encoding='utf-8')
    all_genres_dict = json.load(f)

    attribute = 'ori'
    matrix = np.load(os.path.join(tensorflow_data_3_dir, 'movie_matrix.npy'))
    all_genres_list, _ = loadmovieList('genres.txt')
    for pos, genre in enumerate(all_genres_list):
        train_test_split(matrix, genre, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute, '{}'.format(pos)))
        # four_folder_split(matrix, genre, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute), '{}'.format(pos))

    attribute = 'own'
    # matrix = loadMatrix(server3_dir, k)
    matrix = np.load('C:\\Users\\netlab\\Desktop\\criteria\\ori\\unfixed_user_wtbatch5_200.npy')
    matrix, _, _ = tag_extension_2(matrix, n)
    _, tags_dict = loadmovieList(os.path.join(tensorflow_data_3_dir, 'tag.txt'))
    all_genres_list, _ = loadmovieList('genres.txt')
    matrix = remove_tag_in_genre(matrix, tags_dict, all_genres_list)
    for pos, genre in enumerate(all_genres_list[:14]):
        train_test_split(matrix, genre, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute, 'top{}_genre{}'.format(n, pos)))
        # four_folder_split(matrix, genre, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute), 'top{}_genre{}'.format(n, pos))

    attribute = 'bpr'
    matrix = np.load('C:\\Users\\netlab\\Desktop\\criteria\\ori\\ori_bpr.npy')
    matrix, _, _ = tag_extension_2(matrix, n)
    _, tags_dict = loadmovieList(os.path.join(tensorflow_data_3_dir, 'tag.txt'))
    all_genres_list, _ = loadmovieList('genres.txt')
    matrix = remove_tag_in_genre(matrix, tags_dict, all_genres_list)
    for pos, genre in enumerate(all_genres_list):
        train_test_split(matrix, genre, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute, 'top{}_genre{}'.format(n, pos)))
        # four_folder_split(matrix, genre, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute), 'top{}_genre{}'.format(n, pos))

    attribute = 'svd'
    matrix = np.load('C:\\Users\\netlab\\Desktop\\criteria\\ori\\ori_svd_k30.npy')
    matrix, _, _ = tag_extension_2(matrix, n)
    _, tags_dict = loadmovieList(os.path.join(tensorflow_data_3_dir, 'tag.txt'))
    all_genres_list, _ = loadmovieList('genres.txt')
    matrix = remove_tag_in_genre(matrix, tags_dict, all_genres_list)
    for pos, genre in enumerate(all_genres_list):
        train_test_split(matrix, genre, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute, 'top{}_genre{}'.format(n, pos)))
        # four_folder_split(matrix, genre, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute), 'top{}_genre{}'.format(n, pos))
    print('extension finish')

