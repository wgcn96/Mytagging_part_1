#

import os
import json
import numpy as np
import datetime
from sklearn.model_selection import train_test_split

from script.ori_data import get_60_spilit_matrix
from static import *

from moudle.ndcg.ndcg import *
from SVD_2.tag_extension import loadmovieList, loadTagList


def getSample(vector, n, condition):
    """
    采样函数
    :param vector: 行向量
    :param n: 采样本数量
    :param condition:  采样条件
    :return: 采样list转化为np array
    """
    candidate_list = []
    result_list = []
    for pos, item in enumerate(vector):
        if item == condition:
            candidate_list.append(pos)
    length = len(candidate_list)
    if length < n:
        print("采样数量大于最大采样值")
        return -1

    permatation = np.random.permutation(length)
    result_list = np.asarray(candidate_list)[permatation[:n]]
    # print(result_list)
    return result_list


def check(vector_a, vector_b):
    result = np.zeros(vector_a.shape)
    for pos, item in enumerate(vector_a):
        if item in vector_b:
            result[pos] = 1
    return result


def loadMatrix(dir, k):
    fileName = "resultMatrix_k={}.npy".format(k)
    filePath = os.path.join(dir, fileName)
    matrix = np.load(filePath)
    return matrix


def tag_evaluate(matrix, top_h=10, n=10):
    matrix_path = os.path.join(tensorflow_data_dir, "comprehensive_index_matrix_sample.npy")
    comprehensive_index_matrix = np.load(matrix_path)
    shape = comprehensive_index_matrix.shape
    print("comprehensive_index_matrix shape: ", shape)

    index_matrix = np.argsort(-matrix, axis=1)
    sort_index_matrix = index_matrix[:, :]     # 每部电影取前top_h个标签

    _60_movie_pos_file_path = os.path.join(tensorflow_data_dir, '60_movie_pos.txt')
    _60_movie_pos_file = open(_60_movie_pos_file_path, encoding='utf-8')
    pos_list = []
    while True:
        line = _60_movie_pos_file.readline()
        line = line.strip()

        if line == "" or line is None:
            break

        pos_list.append(int(line))
    _60_movie_pos_file.close()
    result_hr = []
    result_ndcg = []
    for movie_pos in pos_list:
        print('current {}'.format(movie_pos))
        count = 0
        vector = comprehensive_index_matrix[movie_pos]
        sample = getSample(vector, n, -1)

        top_h_vec = np.zeros((top_h,))
        for j in range(shape[1]):
            if count < top_h:
                cur_tag_pos = sort_index_matrix[movie_pos][j]
                if comprehensive_index_matrix[movie_pos][cur_tag_pos] != 1:
                    top_h_vec[count] = cur_tag_pos
                    count += 1
            else:
                break
        result = check(top_h_vec, sample)

        # hr, ndcg
        hr = hit_ratio_cal(result, top_h, n)
        ndcg = ndcg_cal(result, top_h, n)
        result_hr.append(hr)
        result_ndcg.append(ndcg)
    mean_hr = np.array(result_hr).mean()
    mean_ndcg = np.array(result_ndcg).mean()
    return mean_hr, mean_ndcg


def tag_extension(matrix, n=10):
    matrix_path = os.path.join(tensorflow_data_dir, "comprehensive_index_matrix_sample.npy")
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


if __name__ == '__main__':
    k = 30
    n = 10
    top_h = 10

    matrix = loadMatrix(own_dir, k)

    mean_hr, mean_ndcg = tag_evaluate(matrix, top_h, n)
    print(mean_hr, mean_ndcg)

    '''
    matrix, extension_matrix, sort_index_matrix = tag_extension(matrix, n)
    tagFile = os.path.join(tensorflow_data_dir, 'abundant_tag.txt')
    tag_list = loadTagList(tagFile)
    movieFile = os.path.join(tensorflow_data_dir, 'abundant_movie.txt')
    movie_list = loadmovieList(movieFile)
    SVD_movie_result = os.path.join(SVD2_own_output_dir, "own_sample_result_{}.json".format(k))     # 最终的结果
    SVD_movie_matrix = os.path.join(SVD2_own_output_dir, "own_sample_matrix_{}.json".format(k))     # 降序排列的标签
    SVD_movie_result_ex = os.path.join(SVD2_own_output_dir, "own_sample_result_{}_ex.json".format(k))   # 扩展的标签
    matrix2json(matrix, movie_list, tag_list, SVD_movie_result)
    matrix2json(sort_index_matrix, movie_list, tag_list, SVD_movie_matrix)
    matrix2json(extension_matrix, movie_list, tag_list, SVD_movie_result_ex)
    '''
