# -*- coding: utf-8 -*-

"""
note something here
标签可视化，用于写例子
"""

__author__ = 'Wang Chen'
__time__ = '2019/4/29'

import os
import json
import numpy as np
import datetime

from static import *

# 取原来的矩阵，在原来矩阵的基础上，根据结果矩阵对movie进行n个tag扩充
def tag_extension(matrix, n=10):
    matrix_path = os.path.join(tensorflow_data_3_dir, "movie_matrix.npy")
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
    matrix_path = os.path.join(tensorflow_data_3_dir, "movie_matrix.npy")
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


def loadTagList(tagFile):
    tag_dict = {}       # 反取，取下标
    tagList = []        # 正取，取tag名称
    f = open(tagFile, 'r', encoding='utf-8')
    count = -1
    while True:
        count += 1
        line = f.readline()
        line = line.strip()

        if line == "" or line is None:
            break

        tag = line.split(" ")[0]
        tagList.append(tag)
        tag_dict[tag] = count
    f.close()
    print(" tag List len: {}".format(len(tagList)))

    return tagList, tag_dict


def loadmovieList(movieFile):
    movie_dict = {}       # 反取，取下标
    movieList = []        # 正取，取movie名称
    f = open(movieFile, 'r', encoding='utf-8')
    count = -1
    while True:
        count += 1
        line = f.readline()
        line = line.strip()

        if line == "" or line is None:
            break

        movie = line.split(" ")[0]
        movieList.append(movie)
        movie_dict[movie] = count
    f.close()
    print(" movie List len: {}".format(len(movieList)))

    return movieList, movie_dict


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


def cooccurrence(tag, c_matrix, tag_dict):
    """
    指定一个标签，从标准输入循环读标签，看均值和当前值
    :param tag:
    :param c_matrix:
    :param tag_dict:
    :return:
    """
    tag_pos = tag_dict[tag]
    cur_row = c_matrix[tag_pos]
    print("当前行的均值 {}".format(np.average(cur_row)))
    while True:
        check_tag = input("input a check tag")

        if check_tag == 'quit':
            break

        if check_tag not in tag_dict:
            print(" not include")
            continue

        check_pos = tag_dict[check_tag]
        print("当前c值 {}".format(cur_row[check_pos]))


def coappearance(tag1, tag2, review_matrix, tag_dict):
    """
    看两个tag再影评中是否共现
    :param tag1:
    :param tag2:
    :param review_matrix:
    :param tag_dict:
    :return:
    """
    pos1 = tag_dict[tag1]
    pos2 = tag_dict[tag2]
    print(review_matrix[pos1][pos2])


if __name__ == "__main__":

    matrix = np.load(os.path.join(tensorflow_data_3_dir, 'svd_own_0210', 'unfixed_user_wtbatch5_20.npy'))
    matrix, extension_matrix, sort_index_matrix = tag_extension(matrix)

    tagFile = os.path.join(tensorflow_data_3_dir, 'tag.txt')
    tag_list, tag_dict = loadTagList(tagFile)
    movieFile = os.path.join(tensorflow_data_3_dir, 'movie.txt')
    movie_list, movie_dict = loadmovieList(movieFile)
    # movie_output_file = os.path.join(tensorflow_data_3_dir, 'svd_own_0210', "movie_tag_result.json")
    # matrix2json(extension_matrix, movie_list, tag_list, movie_output_file)

    '''
    # wrong
    top250_list = np.load(os.path.join(tensorflow_data_3_dir, 'top250_movie_pos.npy'))
    top250_movie_list = [movie_list[item] for item in top250_list]
    movie_output_file = os.path.join(tensorflow_data_3_dir, 'svd_own_0210', "top250_movie_tag_result.json")
    matrix2json(extension_matrix, top250_movie_list, tag_list, movie_output_file)

    '''

    c_matrix = np.loadtxt(os.path.join(tensorflow_data_3_dir, 'svd_own_0210', "c_matrix.npy"))
    review_matrix = np.load(os.path.join(tensorflow_data_3_dir, 'review_implicit_matrix.npy'))

