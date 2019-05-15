
import os
import json
import numpy as np
import datetime

from static import *
from movie_redundancy.prepare_data import array2file
from sklearn.model_selection import train_test_split


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
    # global variables

    k = 20

    matrix = loadMatrix(tensorflow_data_dir, k)
    matrix, extension_matrix, sort_index_matrix = tag_extension_2(matrix)
    movie_rates_y = np.load(os.path.join(tensorflow_data_dir, "abundant_movie_rates.npy"))

    X_train, X_test, y_train, y_test = train_test_split(matrix, movie_rates_y, random_state=1)
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

    '''
    # ori
    matrix = np.load(os.path.join(tensorflow_data_3_dir, "movie_matrix.npy"))
    movie_rates_y = np.load(os.path.join(tensorflow_data_3_dir, "movie_rates.npy"))
    abundant_id_list = np.load(os.path.join(tensorflow_data_3_dir, 'top250_movie_pos.npy'))
    X_test = matrix[abundant_id_list,]
    y_test = movie_rates_y[abundant_id_list,]
    X_train = np.delete(matrix, abundant_id_list, 0)
    y_train = np.delete(movie_rates_y, abundant_id_list, 0)
    print(X_train.shape, y_train.shape)

    array2file(X_train, y_train, SVD3_output_dir + '\\svm\\ori_train.data')
    array2file(X_test, y_test, SVD3_output_dir + '\\svm\\ori_test.data')

    print('ori finish')
    '''

    '''
    # own
    for n in range(1, 41, 1):
        k = 30
        # n = 35
        print(n)

        # matrix = loadMatrix(server3_dir, k)
        matrix = np.load('C:\\Users\\netlab\\Desktop\\criteria\\5\\unfixed_user_0_8.npy')
        matrix, extension_matrix, sort_index_matrix = tag_extension_2(matrix, n)
        movie_rates_y = np.load(os.path.join(tensorflow_data_3_dir, "movie_rates.npy"))

        abundant_id_list = np.load(os.path.join(tensorflow_data_3_dir, 'top250_movie_pos.npy'))
        X_test = matrix[abundant_id_list,]
        y_test = movie_rates_y[abundant_id_list,]
        X_train = np.delete(matrix, abundant_id_list, 0)
        y_train = np.delete(movie_rates_y, abundant_id_list, 0)
        # print(X_train.shape, y_train.shape)

        array2file(X_train, y_train, SVD3_output_dir + '\\svm0408\\own_k{}_train_{}.data'.format(k, n))
        array2file(X_test, y_test, SVD3_output_dir + '\\svm0408\\own_k{}_test_{}.data'.format(k, n))

    print('ourown extension finish')
    '''

    '''
    # svd
    k = 30
    n = 35

    matrix = loadMatrix(tensorflow_data_3_dir, k)
    matrix, extension_matrix, sort_index_matrix = tag_extension_2(matrix, n)
    movie_rates_y = np.load(os.path.join(tensorflow_data_3_dir, "movie_rates.npy"))

    abundant_id_list = np.load(os.path.join(tensorflow_data_3_dir, 'top250_movie_pos.npy'))
    X_test = matrix[abundant_id_list,]
    y_test = movie_rates_y[abundant_id_list,]
    X_train = np.delete(matrix, abundant_id_list, 0)
    y_train = np.delete(movie_rates_y, abundant_id_list, 0)
    print(X_train.shape, y_train.shape)

    array2file(X_train, y_train, SVD3_output_dir + '\\svm\\k{}_train_{}.data'.format(k, n))
    array2file(X_test, y_test, SVD3_output_dir + '\\svm\\k{}_test_{}.data'.format(k, n))

    print('svd extension finish')
    '''

    '''
    # NeuMF
    matrix = np.load(os.path.join(workdir, 'SVD_3', 'baselines', 'NeuMF_result_mt_com.npy'))
    movie_rates_y = np.load(os.path.join(tensorflow_data_3_dir, "movie_rates.npy"))

    for n in range(5, 25, 5):
        result_matrix, extension_matrix, sort_index_matrix = tag_extension_2(matrix, n)
        abundant_id_list = np.load(os.path.join(tensorflow_data_3_dir, 'top250_movie_pos.npy'))
        X_test = result_matrix[abundant_id_list,]
        y_test = movie_rates_y[abundant_id_list,]
        X_train = np.delete(result_matrix, abundant_id_list, 0)
        y_train = np.delete(movie_rates_y, abundant_id_list, 0)
        print(X_train.shape, y_train.shape)

        array2file(X_train, y_train, SVD3_output_dir + '\\NeuMF\\NeuMF_train_{}.data'.format(n))
        array2file(X_test, y_test, SVD3_output_dir + '\\NeuMF\\NeuMF_test_{}.data'.format(n))

    print('NeuMF extension finish')
    '''

    # MLP
    matrix = np.load(os.path.join(workdir, 'SVD_3', 'baselines', 'MLP_result_mt_ori.npy'))
    movie_rates_y = np.load(os.path.join(tensorflow_data_3_dir, "movie_rates.npy"))

    for n in range(5, 25, 5):
        result_matrix, extension_matrix, sort_index_matrix = tag_extension_2(matrix, n)
        abundant_id_list = np.load(os.path.join(tensorflow_data_3_dir, 'top250_movie_pos.npy'))
        X_test = result_matrix[abundant_id_list,]
        y_test = movie_rates_y[abundant_id_list,]
        X_train = np.delete(result_matrix, abundant_id_list, 0)
        y_train = np.delete(movie_rates_y, abundant_id_list, 0)
        print(X_train.shape, y_train.shape)

        array2file(X_train, y_train, SVD3_output_dir + '\\MLP\\MLP_train_{}.data'.format(n))
        array2file(X_test, y_test, SVD3_output_dir + '\\MLP\\MLP_test_{}.data'.format(n))

    print('MLP extension finish')
