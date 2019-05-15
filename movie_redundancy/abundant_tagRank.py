# -*- coding: utf-8 -*-

"""
复现movie_redundancy实验
"""
import time
import datetime
import json
import os

import numpy as np

from movie_redundancy.static import *

shape = (4311, 1506)


# 读入相似性矩阵W
def load_redundancy_matrix():

    matrix = np.zeros((59370, 59370), dtype=np.float32)  # Hard code here
    with open(redundancy_matrix) as f:
        index = 0
        while True:
            # print(index)
            line = f.readline()
            if not line:
                break
            temp_list = line.split()
            for i, value in enumerate(temp_list):
                matrix[index][i] = value
            index += 1
    f.close()

    return matrix


# 读文件，获取冗余行，得到 movie -- pos 字典
def get_redundancy_pos_dict():
    tmp_redundancy_movie_dict = {}
    pos = -1
    redundancy_file = open(redundancymovie_id, encoding='utf-8')
    while True:
        line = redundancy_file.readline().strip()
        if line != "" and line is not None:
            pos += 1
            tmp_redundancy_movie_dict[line] = pos
        else:
            break
    print('total movie id', pos)
    return tmp_redundancy_movie_dict


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


def get_abundant_matrix(ori_matrix, ori_pos_dict):
    movieFile = os.path.join(tensorflow_data_dir, "abundant_movie.txt")
    movieList, movie_dict = loadmovieList(movieFile)
    line_pos_list = []
    for pos, item in enumerate(movieList):
        try:
            cur_pos = ori_pos_dict[item]
            line_pos_list.append(cur_pos)
        except:
            print(pos, item)
    print('from ori matrix get pos', len(line_pos_list))

    result_matrix = ori_matrix[line_pos_list, :]
    result_matrix = result_matrix[:, line_pos_list]
    print('result matrix shape', result_matrix.shape)
    return line_pos_list, result_matrix


# 初始化指示向量
def load_indicator_matrix():
    file_path = os.path.join(tensorflow_data_dir, 'abundant_movie_matrix.npy')
    indicator_matrix = np.load(file_path)
    return indicator_matrix


def calculateMatrix(matrix, tmp):

    for i in range(8):
        tmp = np.dot(matrix, tmp)
        norm_matrix = np.linalg.norm(tmp, axis=1, ord=1, keepdims=True)
        shape = tmp.shape
        for j in range(shape[0]):
            tmp[j, :] = tmp[j, :] / norm_matrix[j, :]

        if i == 0:
            m1_result = tmp

        if i == 1:
            m2_result = tmp

        if i == 3:
            m4_result = tmp

        if i == 7:
            m8_result = tmp

    return m1_result, m2_result, m4_result, m8_result


# 从redundancy矩阵相乘的结果中，每个movie取前n个tag
def movie_tag_extension_3(orimatrix, n):
    count = 0
    # resultmatrix = np.zeros(shape, dtype=np.float32)
    resultmatrix = np.zeros((4311, 1506), dtype=np.float32)
    shape = orimatrix.shape     # 59334 * 2015
    index_matrix = np.argsort(-orimatrix, axis=1)
    sort_index_matrix = index_matrix[:, :n]     # 每部电影取前n个标签
    print(sort_index_matrix.shape)        # 应该是 59334 * n

    for row_pos, row in enumerate(sort_index_matrix):       # 取行
        for item in row:                # 取列
            if resultmatrix[row_pos, item] != 1:
                resultmatrix[row_pos, item] = 1
                count += 1
    print(count)

    return resultmatrix


def matrix2json(matrix, filePath):
    f = open(filePath, encoding='utf-8', mode='w')

    movieFile = os.path.join(tensorflow_data_dir, "abundant_movie.txt")
    movieList, movie_dict = loadmovieList(movieFile)

    tagFile = os.path.join(tensorflow_data_dir, "abundant_tag.txt")
    tagList, tag_dict = loadTagList(tagFile)

    movie_tag_dict = {}

    for moviepos, movie in enumerate(movieList):
        cur_tags = matrix[moviepos]
        movie_tag_dict[movie] = []
        tag_indexes = cur_tags.tolist()
        for tagpos, indicator in enumerate(tag_indexes):
            if indicator:
                movie_tag_dict[movie].append(tagList[tagpos])
            else:
                pass

    json.dump(movie_tag_dict, f, indent=4, sort_keys=True, ensure_ascii=False)
    f.close()


# ==================================================================
# 补标签
if __name__ == '__main__':
    k = 10
    '''
    print(datetime.datetime.now(), "begining load matrix...")
    start = time.time()
    matrix = load_redundancy_matrix()
    print(datetime.datetime.now(), "load matrix finish")
    print("total seconds, ", time.time() - start)

    ori_pos_dict = get_redundancy_pos_dict()
    abundant_matrix = get_abundant_matrix(matrix, ori_pos_dict)
    pos_list, abundant_matrix = abundant_matrix
    path = os.path.join(tensorflow_data_dir, 'redundancy_matrix.npy')
    np.save(path, abundant_matrix)
    '''
    path = os.path.join(tensorflow_data_dir, 'redundancy_matrix_2.npy')

    abundant_matrix = np.load(path)
    indicator_matrix = load_indicator_matrix()

    
    # np.linalg.norm
    print(datetime.datetime.now(), "begining calculating matrix...")
    start = time.time()
    m1_result, m2_result, m4_result, m8_result = calculateMatrix(abundant_matrix, indicator_matrix)
    print("calculating matrix using {:.2f} seconds".format(time.time() - start))
    
    print(datetime.datetime.now(), "begin extending tags...")
    start = time.time()
    m1matrix_2 = movie_tag_extension_3(m1_result, k)
    m2matrix_2 = movie_tag_extension_3(m2_result, k)
    m4matrix_2 = movie_tag_extension_3(m4_result, k)
    m8matrix_2 = movie_tag_extension_3(m8_result, k)
    print(datetime.datetime.now(), "extending tags finish!")
    print("total seconds {:.2f}".format(time.time() - start))

    np.save(os.path.join(redundancy2_output_dir, "r1matrix_2.npy"), m1matrix_2)
    np.save(os.path.join(redundancy2_output_dir, "r2matrix_2.npy"), m2matrix_2)
    np.save(os.path.join(redundancy2_output_dir, "r4matrix_2.npy"), m4matrix_2)
    np.save(os.path.join(redundancy2_output_dir, "r8matrix_2.npy"), m8matrix_2)

    matrix2json(m1matrix_2, os.path.join(redundancy2_output_dir, "r1matrix_2.json"))
    matrix2json(m2matrix_2, os.path.join(redundancy2_output_dir, "r2matrix_2.json"))
    matrix2json(m4matrix_2, os.path.join(redundancy2_output_dir, "r4matrix_2.json"))
    matrix2json(m8matrix_2, os.path.join(redundancy2_output_dir, "r8matrix_2.json"))

