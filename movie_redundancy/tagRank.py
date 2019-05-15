"""
复现movie_redundancy实验
"""
import time
import datetime
import json

import numpy as np

from movie_redundancy.static import *



# 读入相似性矩阵W
def load_redundancy_matrix():
    print(datetime.datetime.now(), "begining load matrix...")
    start = time.time()
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
    print(datetime.datetime.now(), "load matrix finish")
    print("total seconds, ", time.time() - start)
    return matrix


matrix = load_redundancy_matrix()


# 加载电影id序号，并删除多余元素
def load_dict(matrix):
    redundancy_movie_dict = {}
    redundancy_movie_list = []
    redundancy_pos_list = []
    pos = 0
    redundancy_file = open(redundancymovie_id, encoding='utf-8')
    while True:
        line = redundancy_file.readline().strip()
        if line != "" and line is not None:
            check_pos = redundancy_movie_dict.get(line, -1)
            if check_pos != -1:
                redundancy_pos_list.append(pos)
            else:
                redundancy_movie_list.append(line)
                redundancy_movie_dict[line] = pos
                pos += 1
        else:
            break
    redundancy_file.close()
    refined_matrix = np.delete(matrix, redundancy_pos_list, 0)
    refined_matrix = np.delete(refined_matrix, redundancy_pos_list, 1)
    print(redundancy_pos_list.__len__())
    print(refined_matrix.shape)
    return refined_matrix, redundancy_movie_dict, redundancy_movie_list, pos


# 读文件，获取冗余行
def get_redundancy_pos_list():
    tmp_redundancy_movie_dict = {}
    redundancy_pos_list = []
    pos = 0
    redundancy_file = open(redundancymovie_id, encoding='utf-8')
    while True:
        line = redundancy_file.readline().strip()
        if line != "" and line is not None:
            check_pos = tmp_redundancy_movie_dict.get(line, -1)
            if check_pos != -1:
                redundancy_pos_list.append(pos)
            else:
                tmp_redundancy_movie_dict[line] = pos
                pos += 1

        else:
            break
    redundancy_file.close()
    print(redundancy_pos_list.__len__(), tmp_redundancy_movie_dict.__len__(), pos)
    return redundancy_pos_list


matrix, redundancy_movie_dict, redundancy_movie_list, redundancy_movie_number = load_dict(matrix)


# 初始化指示向量
def load_indicator_matrix():
    indicator_matrix = []

    for tag, pos in tag_order_dict.items():
        indicator_vec = np.zeros((redundancy_movie_number, 1), dtype=np.float32)
        movie_tag_vec = user_item_matrix[:, pos]
        pos_list = np.where(movie_tag_vec == 1)[0]
        for j in pos_list:
            cur_movie = movie_list[j]
            cur_movie_pos = redundancy_movie_dict.get(cur_movie, -1)
            if cur_movie_pos > -1:
                indicator_vec[cur_movie_pos, 0] = 1

        indicator_matrix.append(indicator_vec)

    indicator_matrix = np.asarray(indicator_matrix)
    indicator_matrix = indicator_matrix.reshape(indicator_matrix.shape[1], indicator_matrix.shape[0])       # 59334 * 2015
    return indicator_matrix


indicator_matrix = load_indicator_matrix()


def calculateMatrix(tmp):

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


# np.linalg.norm
print(datetime.datetime.now(), "begining calculating matrix...")
start = time.time()
m1_result, m2_result, m4_result, m8_result = calculateMatrix(indicator_matrix)
print("calculating one matrix using {} seconds", time.time() - start)

# ============================================================================
# 保存读入矩阵
# np.save(os.path.join(redundancydir, "similiarity_matrix.npy"), matrix)
#
# np.save(os.path.join(redundancydir, "m2_matrix.npy"), m2_result)
# np.save(os.path.join(redundancydir, "m4_matrix.npy"), m4_result)
# np.save(os.path.join(redundancydir, "m8_matrix.npy"), m8_result)


# 取原来的矩阵，行复制，再从redundancy矩阵相乘的结果中，每个tag取前n个结果
def movie_tag_extension(orimatrix, n):
    count = 0
    resultmatrix = np.zeros((59370, 2015), dtype=np.float32)
    # resultmatrix = np.zeros((59334, 2015), dtype=np.float32)
    for i in range(len(movie_list)):
        tag_movie = movie_list[i]
        pos = redundancy_movie_dict.get(tag_movie, -1)
        if pos > -1:
            resultmatrix[pos] = user_item_matrix[i]
            count += 1
        else:
            pass
    print(count)        # 行赋值数

    count = 0
    shape = orimatrix.shape     # 59370 * 2015
    index_matrix = np.argsort(-orimatrix, axis=0)
    sort_matrix = index_matrix[:n, :]
    print(sort_matrix.shape)        # 应该是 n * 2015

    for j in range(shape[1]):       # 0 - 2014
        cur_column = sort_matrix[:, j]      # 取一列
        for i in cur_column:            # 列赋值
            cur_movie = redundancy_movie_list[i]
            cur_tag_pos = j
            if resultmatrix[i, j] != 1:
                resultmatrix[i, j] = 1
                count += 1
    print(count)        # 补标签数
    return resultmatrix


# 取原来的矩阵，行复制，再从redundancy矩阵相乘的结果中，每个movie取前n个tag
def movie_tag_extension_2(orimatrix, n):
    count = 0
    # resultmatrix = np.zeros((59370, 2015), dtype=np.float32)
    resultmatrix = np.zeros((59334, 2015), dtype=np.float32)
    for i in range(len(movie_list)):
        tag_movie = movie_list[i]
        pos = redundancy_movie_dict.get(tag_movie, -1)
        if pos > -1:
            resultmatrix[pos] = user_item_matrix[i]
            count += 1
        else:
            pass
    print(count)        # 行赋值数

    count = 0
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


# 从redundancy矩阵相乘的结果中，每个movie取前n个tag
def movie_tag_extension_3(orimatrix, n):
    count = 0
    # resultmatrix = np.zeros((59370, 2015), dtype=np.float32)
    resultmatrix = np.zeros((59334, 2015), dtype=np.float32)
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


def tag2ListNFile(filePath):
    f = open(filePath, 'w', encoding='utf-8')
    tag_list = []
    for tag, pos in tag_order_dict.items():
        tag_list.append(tag)
        line = tag + ' ' + str(pos) + '\n'
        f.write(line)
    f.close()
    print(tag_list.__len__())
    return tag_list


def matrix2json(matrix, filePath):
    f = open(filePath, encoding='utf-8', mode='w')

    movie_tag_dict = {}
    for movie, pos in redundancy_movie_dict.items():
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


tag_file = os.path.join(redundancy_output_dir, 'tag.txt')
tag_list = tag2ListNFile(tag_file)


# ==================================================================
# 补标签
print()
print(datetime.datetime.now(), "begin extending tags...")
start = time.time()
m1matrix_2 = movie_tag_extension_3(m1_result, 10)
m2matrix_2 = movie_tag_extension_3(m2_result, 10)
m4matrix_2 = movie_tag_extension_3(m4_result, 10)
m8matrix_2 = movie_tag_extension_3(m8_result, 10)
print(datetime.datetime.now(), "extending tags finish!")
print("total seconds ", time.time() - start)

np.save(os.path.join(redundancy_output_dir, "r1matrix_2.npy"), m1matrix_2)
np.save(os.path.join(redundancy_output_dir, "r2matrix_2.npy"), m2matrix_2)
np.save(os.path.join(redundancy_output_dir, "r4matrix_2.npy"), m4matrix_2)
np.save(os.path.join(redundancy_output_dir, "r8matrix_2.npy"), m8matrix_2)

matrix2json(m1matrix_2, os.path.join(redundancy_output_dir, "r1matrix_2.json"))
matrix2json(m2matrix_2, os.path.join(redundancy_output_dir, "r2matrix_2.json"))
matrix2json(m4matrix_2, os.path.join(redundancy_output_dir, "r4matrix_2.json"))
matrix2json(m8matrix_2, os.path.join(redundancy_output_dir, "r8matrix_2.json"))


def generate_y():
    y = np.zeros((59334, 1), dtype=np.float32)
    for pos, movie in enumerate(redundancy_movie_list):
        rate = movie_rate_dict.get(movie, -1)
        if rate > -1:
            y[pos] = rate
        else:
            pass
    return y


movie_redundancy_y = generate_y()
np.save(os.path.join(redundancy_output_dir, 'movie_redundancy_y.npy'), movie_redundancy_y)
