
import os
import json
import numpy as np


from static import *
from movie_redundancy.prepare_data import array2file


# 取原来的矩阵，在原来矩阵的基础上，根据结果矩阵对movie进行k个tag扩充
def tag_extension(matrix, n=10):
    matrix_path = os.path.join(tensorflow_data_dir, "movie_rates_matrix.npy")
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
                    pass
            else:
                break
        total_count += count

    print("total extends tags: {}".format(total_count))
    print("extension_matrix shape", extension_matrix.shape)
    return result_matrix, extension_matrix


def loadMatrix(dir, k):
    fileName = "resultMatrix_k={}.npy".format(k)
    filePath = os.path.join(dir, fileName)
    matrix = np.load(filePath)
    return matrix


def prepare_data(round_matrix, round_y, most_popular_movie_list, all_movie_pos_dict):
    matrix_path = os.path.join(tensorflow_data_dir, "movie_rates_matrix.npy")
    result_matrix = np.load(matrix_path)

    test_matrix = np.zeros((1000, 2015), dtype=np.int32)
    test_y = np.zeros((1000, 1), dtype=np.float32)

    remove_pos_list = []

    count = 0
    for item in most_popular_movie_list:

        if count >= 1000:
            break

        try:
            cur_row_pos = all_movie_pos_dict.get(item, -1)
            if cur_row_pos > -1:
                cur_row = result_matrix[cur_row_pos]
                test_matrix[count] = cur_row
                test_y[count] = round_y[cur_row_pos]

                remove_pos_list.append(cur_row_pos)
                count += 1
        except Exception:
            print(item)

    train_matrix = np.delete(round_matrix, remove_pos_list, 0)
    train_y = np.delete(round_y, remove_pos_list, 0)

    return train_matrix, test_matrix, train_y, test_y


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


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


def matrix2json(matrix, all_movie_pos_dict, tag_list, filePath):
    f = open(filePath, encoding='utf-8', mode='w')

    movie_tag_dict = {}
    for movie, pos in all_movie_pos_dict.items():
        cur_tags = matrix[pos]
        movie_tag_dict[movie] = []
        tag_indexes = cur_tags.tolist()
        for pos, indicator in enumerate(tag_indexes):
            if indicator:
                movie_tag_dict[movie].append(tag_list[pos])
            else:
                pass

    json.dump(movie_tag_dict, f, indent=4, sort_keys=True, ensure_ascii=False)
    f.close()


if __name__ == "__main__":

    # global variables
    k = 10

    matrix = loadMatrix(tensorflow_data_dir, k)
    matrix, extension_matrix = tag_extension(matrix)
    movie_rates_y = np.load(os.path.join(redundancy_output_dir, "movie_rates_y.npy"))
    all_movie_pos_file_Path = os.path.join(workdir, "all_movie_matrix_dict.json")
    all_movie_pos_file = open(all_movie_pos_file_Path, encoding='utf-8', mode='r')
    all_movie_pos_dict = json.load(all_movie_pos_file)

    most_popular_file_Path = os.path.join(redundancy_output_dir, "most_popular_movie.json")
    most_popular_file = open(most_popular_file_Path, encoding='utf-8', mode='r')
    most_popular_movie_dict = json.load(most_popular_file)
    most_popular_movie_list = most_popular_movie_dict.keys()

    X_train, X_test, y_train, y_test = prepare_data(matrix, movie_rates_y, most_popular_movie_list, all_movie_pos_dict)
    X_train, y_train = shuffle_in_unison(X_train, y_train)
    array2file(X_train, y_train, SVD_output_dir + '\\svm\\k{}_train.data'.format(k))
    array2file(X_test, y_test, SVD_output_dir + '\\svm\\k{}_test.data'.format(k))

    tagFile = os.path.join(redundancy_output_dir, 'tag.txt')
    tag_list = loadTagList(tagFile)
    SVD_movie_result = os.path.join(SVD_output_dir, "SVD_movie_result_{}.json".format(k))
    matrix2json(extension_matrix, all_movie_pos_dict, tag_list, SVD_movie_result)

