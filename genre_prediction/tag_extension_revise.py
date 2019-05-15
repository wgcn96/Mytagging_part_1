# -*- coding: utf-8 -*-
# python3
#

"""
main 函数
标签扩展
"""

__author__ = 'Wang Chen'

import sys

sys.path.extend(['D:\\workProject\\pythonProject\\Mytagging\\count', \
                 'D:\\workProject\\pythonProject\\Mytagging'])

import os
import json
import numpy as np

np.random.seed(0)
import datetime

from genre_prediction.static import *
from genre_prediction.util import *


# 取原来的矩阵，在原来矩阵的基础上，根据结果矩阵对movie进行n个tag扩充
def tag_extension(matrix, n=10):
    matrix_path = os.path.join(tensorflow_data_3_dir, "movie_matrix.npy")
    result_matrix = np.load(matrix_path)
    shape = result_matrix.shape
    print("result_matrix shape: ", shape)

    index_matrix = np.argsort(-matrix, axis=1)
    sort_index_matrix = index_matrix[:, :]  # 每部电影取前n个标签

    # print(sort_index_matrix.shape)        # 应该是 59551 * n

    extension_matrix = np.zeros(shape)  # 补标签矩阵

    total_count = 0
    for i in range(shape[0]):
        count = 0
        for j in range(shape[1]):

            if count < n:
                cur_tag_pos = sort_index_matrix[i][j]
                if result_matrix[i][cur_tag_pos] != 1:
                    result_matrix[i][cur_tag_pos] = 1
                    extension_matrix[i][cur_tag_pos] = 1  # 每个电影补的标签
                    count += 1
            else:
                break
        total_count += count

    print("total extends tags: {}".format(total_count))
    print("extension_matrix shape", extension_matrix.shape)
    return result_matrix, extension_matrix, sort_index_matrix


# 取原来的矩阵，根据结果矩阵对movie取排序最高的前n个tag扩充（只取前n，原来没有就打上）
def tag_extension_2(matrix, n=10):
    matrix_path = os.path.join(tensorflow_data_3_dir, "movie_matrix.npy")
    result_matrix = np.load(matrix_path)
    shape = result_matrix.shape
    print("result_matrix shape: ", shape)

    index_matrix = np.argsort(-matrix, axis=1)
    sort_index_matrix = index_matrix[:, :]  # 每部电影取前n个标签

    # print(sort_index_matrix.shape)        # 应该是 59551 * n

    extension_matrix = np.zeros(shape)  # 补标签矩阵

    total_count = 0
    for i in range(shape[0]):
        count = 0
        for j in range(n):

            cur_tag_pos = sort_index_matrix[i][j]
            if result_matrix[i][cur_tag_pos] != 1:
                result_matrix[i][cur_tag_pos] = 1
                extension_matrix[i][cur_tag_pos] = 1  # 每个电影补的标签
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


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def train_test_split(matrix, genre, all_genres_dict, file_path, file_name):
    """
    生成某个genre的训练测试数据，并写入file路径文件
    :param matrix: ndarray 扩展后的电影和标签矩阵
    :param genre: str 进行二分类的电影类型
    :param all_genres_dict: dict 所有电影和其类型的字典
    :param file_path: str 根路径
    :param file_name: str 文件名
    :return: 切分后的四个矩阵
    """

    neg_pos_list = []  # 这个电影类型的负样本列表
    pos_pos_list = []  # 这个电影类型的正样本列表
    count = 0  # 这个电影类型的正样本数量
    y_all = np.zeros((19004), dtype=np.int32)
    movie_list, _ = loadmovieList(os.path.join(tensorflow_data_3_dir, 'movie.txt'))
    for pos, movie in enumerate(movie_list):
        if movie in all_genres_dict:
            genres_list = all_genres_dict[movie]
            if genre in genres_list:
                pos_pos_list.append(pos)
                y_all[pos] = 1
            else:
                neg_pos_list.append(pos)
    count = len(pos_pos_list)
    split_length = int(count / 4)
    print(count)

    if count > len(neg_pos_list):
        pos_pos_list = np.random.choice(pos_pos_list, len(neg_pos_list))
        pos_pos_list = pos_pos_list.tolist()
    else:
        neg_pos_list = np.random.choice(neg_pos_list, count, )
        neg_pos_list = neg_pos_list.tolist()

    test_index = neg_pos_list[:split_length] + pos_pos_list[:split_length]
    train_index = neg_pos_list[split_length:] + pos_pos_list[split_length:]
    y_train = y_all[train_index]
    y_test = y_all[test_index]
    X_train = matrix[train_index,]
    X_test = matrix[test_index,]

    X_train, y_train = shuffle_in_unison(X_train, y_train)
    X_test, y_test = shuffle_in_unison(X_test, y_test)

    array2file(X_train, y_train, file=os.path.join(file_path, file_name + '.train'))
    array2file(X_test, y_test, file=os.path.join(file_path, file_name + '.test'))
    return X_train, X_test, y_train, y_test


def train_test_split_revise(matrix, genre, tags_dict, all_genres_dict, file_path, file_name):
    """
    生成某个genre的训练测试数据，并写入file路径文件
    :param matrix: ndarray 扩展后的电影和标签矩阵
    :param genre: str 进行二分类的电影类型
    :param all_genres_dict: dict 所有电影和其类型的字典
    :param file_path: str 根路径
    :param file_name: str 文件名
    :return: 切分后的四个矩阵
    """

    if genre in tags_dict:
        pos = tags_dict.get(genre)
        matrix = np.delete(matrix, pos, axis=1)

    neg_pos_list = []  # 这个电影类型的负样本列表
    pos_pos_list = []  # 这个电影类型的正样本列表
    count = 0  # 这个电影类型的正样本数量
    y_all = np.zeros((19004), dtype=np.int32)
    movie_list, _ = loadmovieList(os.path.join(tensorflow_data_3_dir, 'movie.txt'))
    for pos, movie in enumerate(movie_list):
        if movie in all_genres_dict:
            genres_list = all_genres_dict[movie]
            if genre in genres_list:
                pos_pos_list.append(pos)
                y_all[pos] = 1
            else:
                neg_pos_list.append(pos)
    count = len(pos_pos_list)
    split_length = int(count / 4)
    print(count)

    if count > len(neg_pos_list):
        pos_pos_list = np.random.choice(pos_pos_list, len(neg_pos_list))
        pos_pos_list = pos_pos_list.tolist()
    else:
        neg_pos_list = np.random.choice(neg_pos_list, count, )
        neg_pos_list = neg_pos_list.tolist()

    test_index = neg_pos_list[:split_length] + pos_pos_list[:split_length]
    train_index = neg_pos_list[split_length:] + pos_pos_list[split_length:]
    y_train = y_all[train_index]
    y_test = y_all[test_index]
    X_train = matrix[train_index,]
    X_test = matrix[test_index,]

    X_train, y_train = shuffle_in_unison(X_train, y_train)
    X_test, y_test = shuffle_in_unison(X_test, y_test)

    array2file(X_train, y_train, file=os.path.join(file_path, file_name + '.train'))
    array2file(X_test, y_test, file=os.path.join(file_path, file_name + '.test'))
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


'''
if __name__ == "__main__":

    # global variables
    filePath = 'all_genres_dict.json'
    f = open(filePath, 'r', encoding='utf-8')
    all_genres_dict = json.load(f)

    k = 30
    n = 10

    
    attribute = 'own'
    # matrix = loadMatrix(server3_dir, k)
    matrix = np.load('C:\\Users\\netlab\\Desktop\\criteria\\ori\\unfixed_user_wtbatch0_20.npy')
    matrix, _, _ = tag_extension(matrix, n)
    _, tags_dict = loadmovieList(os.path.join(tensorflow_data_3_dir, 'tag.txt'))
    all_genres_list, _ = loadmovieList('genres_eight.txt')
    matrix = remove_tag_in_genre(matrix, tags_dict, all_genres_list)
    for pos, genre in enumerate(all_genres_list):
        train_test_split(matrix, genre, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute),'top{}_genre{}'.format(n, pos))
    

    
    attribute = 'ori'
    matrix = np.load(os.path.join(tensorflow_data_3_dir, 'movie_matrix.npy'))
    _, tags_dict = loadmovieList(os.path.join(tensorflow_data_3_dir, 'tag.txt'))
    all_genres_list, _ = loadmovieList('genres_eight.txt')
    matrix = remove_tag_in_genre(matrix, tags_dict, all_genres_list)
    for pos, genre in enumerate(all_genres_list):
        train_test_split(matrix, genre, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute), 'top{}_genre{}'.format(n, pos))
    

    
    attribute = 'bpr'
    matrix = np.load('C:\\Users\\netlab\\Desktop\\criteria\\ori\\ori_bpr.npy')
    matrix, _, _ = tag_extension(matrix, n)
    _, tags_dict = loadmovieList(os.path.join(tensorflow_data_3_dir, 'tag.txt'))
    all_genres_list, _ = loadmovieList('genres_eight.txt')
    matrix = remove_tag_in_genre(matrix, tags_dict, all_genres_list)
    for pos, genre in enumerate(all_genres_list):
        train_test_split(matrix, genre, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute), 'top{}_genre{}'.format(n, pos))
    

    
    attribute = 'NeuMF'
    matrix = np.load('C:\\Users\\netlab\\Desktop\\criteria\\ori\\NeuMF_result_mt_ori.npy')
    matrix, _, _ = tag_extension(matrix, n)
    _, tags_dict = loadmovieList(os.path.join(tensorflow_data_3_dir, 'tag.txt'))
    all_genres_list, _ = loadmovieList('genres_eight.txt')
    matrix = remove_tag_in_genre(matrix, tags_dict, all_genres_list)
    for pos, genre in enumerate(all_genres_list):
        train_test_split(matrix, genre, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute), 'top{}_genre{}'.format(n, pos))
    

    attribute = 'MLP'
    matrix = np.load('C:\\Users\\netlab\\Desktop\\criteria\\ori\\MLP_result_mt_ori.npy')
    matrix, _, _ = tag_extension(matrix, n)
    _, tags_dict = loadmovieList(os.path.join(tensorflow_data_3_dir, 'tag.txt'))
    all_genres_list, _ = loadmovieList('genres_eight.txt')
    matrix = remove_tag_in_genre(matrix, tags_dict, all_genres_list)
    for pos, genre in enumerate(all_genres_list):
        train_test_split(matrix, genre, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute), 'top{}_genre{}'.format(n, pos))


    
    attribute = 'item_CF'
    matrix = np.load('C:\\Users\\netlab\\Desktop\\criteria\\ori\\item_CF_ori.npy')
    matrix, _, _ = tag_extension(matrix, n)
    _, tags_dict = loadmovieList(os.path.join(tensorflow_data_3_dir, 'tag.txt'))
    all_genres_list, _ = loadmovieList('genres_eight.txt')
    matrix = remove_tag_in_genre(matrix, tags_dict, all_genres_list)
    for pos, genre in enumerate(all_genres_list):
        train_test_split(matrix, genre, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute), 'top{}_genre{}'.format(n, pos))


    attribute = 'TagRank'
    matrix = np.load('C:\\Users\\netlab\\Desktop\\criteria\\ori\\TagRank_ori.npy')
    matrix, _, _ = tag_extension(matrix, n)
    _, tags_dict = loadmovieList(os.path.join(tensorflow_data_3_dir, 'tag.txt'))
    all_genres_list, _ = loadmovieList('genres_eight.txt')
    matrix = remove_tag_in_genre(matrix, tags_dict, all_genres_list)
    for pos, genre in enumerate(all_genres_list):
        train_test_split(matrix, genre, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute), 'top{}_genre{}'.format(n, pos))
    

    
    attribute = 'svd'
    matrix = np.load('C:\\Users\\netlab\\Desktop\\criteria\\ori\\ori_svd_k30.npy')
    matrix, _, _ = tag_extension(matrix, n)
    _, tags_dict = loadmovieList(os.path.join(tensorflow_data_3_dir, 'tag.txt'))
    all_genres_list, _ = loadmovieList('genres_eight.txt')
    matrix = remove_tag_in_genre(matrix, tags_dict, all_genres_list)
    for pos, genre in enumerate(all_genres_list):
        train_test_split(matrix, genre, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute), 'top{}_genre{}'.format(n, pos))
    
    
    attribute = 'review'
    # matrix = loadMatrix(server3_dir, k)
    matrix = np.load('D:\\workProject\\pythonProject\\Mytagging_part_2\\data_3\\comprehensive_index_matrix.npy')
    # matrix, _, _ = tag_extension(matrix, n)
    _, tags_dict = loadmovieList(os.path.join(tensorflow_data_3_dir, 'tag.txt'))
    all_genres_list, _ = loadmovieList('genres_eight.txt')
    matrix = remove_tag_in_genre(matrix, tags_dict, all_genres_list)
    for pos, genre in enumerate(all_genres_list):
        train_test_split(matrix, genre, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute),'top{}_genre{}'.format(n, pos))
    
    print('extension finish')
'''

if __name__ == "__main__":

    # global variables
    filePath = 'all_genres_dict.json'
    f = open(filePath, 'r', encoding='utf-8')
    all_genres_dict = json.load(f)

    k = 30
    n = 10

    attribute = 'own'
    # matrix = loadMatrix(server3_dir, k)
    matrix = np.load('C:\\Users\\netlab\\Desktop\\criteria\\ori\\unfixed_user_wtbatch0_20.npy')
    matrix, _, _ = tag_extension(matrix, n)
    _, tags_dict = loadmovieList(os.path.join(tensorflow_data_3_dir, 'tag.txt'))
    all_genres_list, _ = loadmovieList('genres_eight.txt')
    # matrix = remove_tag_in_genre(matrix, tags_dict, all_genres_list)
    for pos, genre in enumerate(all_genres_list):
        train_test_split_revise(matrix, genre, tags_dict, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute),
                                'top{}_genre{}'.format(n, pos))

    attribute = 'ori'
    matrix = np.load(os.path.join(tensorflow_data_3_dir, 'movie_matrix.npy'))
    _, tags_dict = loadmovieList(os.path.join(tensorflow_data_3_dir, 'tag.txt'))
    all_genres_list, _ = loadmovieList('genres_eight.txt')
    # matrix = remove_tag_in_genre(matrix, tags_dict, all_genres_list)
    for pos, genre in enumerate(all_genres_list):
        train_test_split_revise(matrix, genre, tags_dict, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute),
                                'top{}_genre{}'.format(n, pos))

    attribute = 'bpr'
    matrix = np.load('C:\\Users\\netlab\\Desktop\\criteria\\ori\\ori_bpr.npy')
    matrix, _, _ = tag_extension(matrix, n)
    _, tags_dict = loadmovieList(os.path.join(tensorflow_data_3_dir, 'tag.txt'))
    all_genres_list, _ = loadmovieList('genres_eight.txt')
    # matrix = remove_tag_in_genre(matrix, tags_dict, all_genres_list)
    for pos, genre in enumerate(all_genres_list):
        train_test_split_revise(matrix, genre, tags_dict, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute),
                                'top{}_genre{}'.format(n, pos))

    attribute = 'NeuMF'
    matrix = np.load('C:\\Users\\netlab\\Desktop\\criteria\\ori\\NeuMF_result_mt_ori.npy')
    matrix, _, _ = tag_extension(matrix, n)
    _, tags_dict = loadmovieList(os.path.join(tensorflow_data_3_dir, 'tag.txt'))
    all_genres_list, _ = loadmovieList('genres_eight.txt')
    # matrix = remove_tag_in_genre(matrix, tags_dict, all_genres_list)
    for pos, genre in enumerate(all_genres_list):
        train_test_split_revise(matrix, genre, tags_dict, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute),
                                'top{}_genre{}'.format(n, pos))

    attribute = 'MLP'
    matrix = np.load('C:\\Users\\netlab\\Desktop\\criteria\\ori\\MLP_result_mt_ori.npy')
    matrix, _, _ = tag_extension(matrix, n)
    _, tags_dict = loadmovieList(os.path.join(tensorflow_data_3_dir, 'tag.txt'))
    all_genres_list, _ = loadmovieList('genres_eight.txt')
    # matrix = remove_tag_in_genre(matrix, tags_dict, all_genres_list)
    for pos, genre in enumerate(all_genres_list):
        train_test_split_revise(matrix, genre, tags_dict, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute),
                                'top{}_genre{}'.format(n, pos))

    attribute = 'item_CF'
    matrix = np.load('C:\\Users\\netlab\\Desktop\\criteria\\ori\\item_CF_ori.npy')
    matrix, _, _ = tag_extension(matrix, n)
    _, tags_dict = loadmovieList(os.path.join(tensorflow_data_3_dir, 'tag.txt'))
    all_genres_list, _ = loadmovieList('genres_eight.txt')
    # matrix = remove_tag_in_genre(matrix, tags_dict, all_genres_list)
    for pos, genre in enumerate(all_genres_list):
        train_test_split_revise(matrix, genre, tags_dict, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute),
                                'top{}_genre{}'.format(n, pos))

    attribute = 'TagRank'
    matrix = np.load('C:\\Users\\netlab\\Desktop\\criteria\\ori\\TagRank_ori.npy')
    matrix, _, _ = tag_extension(matrix, n)
    _, tags_dict = loadmovieList(os.path.join(tensorflow_data_3_dir, 'tag.txt'))
    all_genres_list, _ = loadmovieList('genres_eight.txt')
    # matrix = remove_tag_in_genre(matrix, tags_dict, all_genres_list)
    for pos, genre in enumerate(all_genres_list):
        train_test_split_revise(matrix, genre, tags_dict, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute),
                                'top{}_genre{}'.format(n, pos))

    attribute = 'svd'
    matrix = np.load('C:\\Users\\netlab\\Desktop\\criteria\\ori\\ori_svd_k30.npy')
    matrix, _, _ = tag_extension(matrix, n)
    _, tags_dict = loadmovieList(os.path.join(tensorflow_data_3_dir, 'tag.txt'))
    all_genres_list, _ = loadmovieList('genres_eight.txt')
    # matrix = remove_tag_in_genre(matrix, tags_dict, all_genres_list)
    for pos, genre in enumerate(all_genres_list):
        train_test_split_revise(matrix, genre, tags_dict, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute),
                                'top{}_genre{}'.format(n, pos))
    '''
    attribute = 'review'
    # matrix = loadMatrix(server3_dir, k)
    matrix = np.load('D:\\workProject\\pythonProject\\Mytagging_part_2\\data_3\\comprehensive_index_matrix.npy')
    # matrix, _, _ = tag_extension(matrix, n)
    _, tags_dict = loadmovieList(os.path.join(tensorflow_data_3_dir, 'tag.txt'))
    all_genres_list, _ = loadmovieList('genres_eight.txt')
    matrix = remove_tag_in_genre(matrix, tags_dict, all_genres_list)
    for pos, genre in enumerate(all_genres_list):
        train_test_split_revise(matrix, genre, all_genres_dict, os.path.join(os.getcwd(), 'output', attribute),
                                'top{}_genre{}'.format(n, pos))
    '''
    print('extension finish')
