# -*- coding: utf-8 -*-
# python3
#
"""
_author: Wang Chen
根据word2vec的向量结果，计算tag之间的余弦相似度，聚类成簇
"""

import json
import os
import time
import datetime

import matplotlib.pyplot as plt
import numpy as np

from tag_community.static import *
# from moudle.myCos import cos_sim
from moudle.fcm.fcm import *
from moudle.fcm.fcm_se import *


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


def get_file_content(file_path, tagFile):
    file = open(file_path, 'r', encoding='utf-8')
    content = json.load(file)

    tag_list = loadTagList(tagFile)

    matrix = []
    for item in tag_list:
        if item in content.keys():
            matrix.append(content[item])
        else:
            # zeros = [0.01] * 100
            zeros = np.random.randn(100)
            matrix.append(zeros)
            print(item)

    matrix = np.array(matrix)
    return matrix


def get_file_order():
    file_path = os.path.join(relative_data_dir, 'full_tag_vector.json')
    file = open(file_path, 'r', encoding='utf-8')
    content = json.load(file)

    full_tag_dict = {}
    order = -1
    for key, value in content.items():
        order += 1
        full_tag_dict[key] = order
    print(order)
    file_path = os.path.join(relative_data_dir, 'full_tag_order.json')
    json.dump(full_tag_dict, open(file_path, 'w', encoding='utf-8'), indent=4, sort_keys=True, ensure_ascii=False)


def get_full_matrix():
    file_path = os.path.join(relative_data_dir, 'full_tag_vector.json')
    file = open(file_path, 'r', encoding='utf-8')
    content = json.load(file)

    matrix = []
    for key, value in content.items():
        matrix.append(value)

    matrix = np.array(matrix)
    print(matrix.shape)
    return matrix


def get_part_matrix(order_file_path, tag_file_path, shape):
    file = open(order_file_path, 'r', encoding='utf-8')
    content = json.load(file)

    tag_list = loadTagList(tag_file_path)

    ori_final_location = np.load(os.path.join(relative_data_dir, 'full_final_U.npy'))
    ori_fuzzy_U = np.load(os.path.join(relative_data_dir, 'full_fuzzy_U.npy'))
    fuzzy_U = np.zeros(shape)
    final_location = np.zeros(shape)
    for pos, item in enumerate(tag_list):
        if item in content.keys():
            final_location[pos] = ori_final_location[content[item]]
            fuzzy_U[pos] = ori_fuzzy_U[content[item]]
    print(pos)

    return fuzzy_U, final_location


def reuslt2file(fuzzy_U, final_location, n, tagFile):

    output_file_path = os.path.join(relative_data_dir, 'result', 'fuzzy_U.txt')
    np.savetxt(output_file_path, fuzzy_U)

    tag_list = loadTagList(tagFile)

    for i in range(n):
        file_path = os.path.join(relative_data_dir,  'result', '{}.txt'.format(i))
        # if not os.path.exists(check_dir):
        #     os.path.mkdirs(check_dir)
        file = open(file_path, 'w', encoding='utf-8')
        pos = np.where(final_location[:, i] == 1)[0]
        print(pos)
        for item in pos:
            file.write(tag_list[item] + '\n')
        file.close()


'''
# 1500 tag
if __name__ == '__main__':
    file_path = os.path.join(relative_data_dir, 'tag_vector.json')
    tagFile = os.path.join(tensorflow_data_dir, 'abundant_tag.txt')

    n = 50
    matrix = get_file_content(file_path, tagFile)[:, :]
    fuzzy_U, final_location = fuzzy(matrix, n, 1.1)
    
    reuslt2file(fuzzy_U, final_location, n)
    output_file_path = os.path.join(relative_data_dir, 'result', 'fuzzy_U.npy')
    np.save(output_file_path, np.asarray(fuzzy_U).astype(np.float32))
    output_file_path = os.path.join(relative_data_dir, 'result', 'final_U.npy')
    np.save(output_file_path, np.array(final_location).astype(np.float32))
'''


# full tag
if __name__ == '__main__':
    '''
    matrix = get_full_matrix()
    n = 50

    print(datetime.datetime.now())
    fuzzy_U, final_location = fuzzy(matrix, n, 1.1)
    print(datetime.datetime.now())
    '''

    '''
    output_file_path = os.path.join(relative_data_dir, 'full_fuzzy_U.npy')
    np.save(output_file_path, np.asarray(fuzzy_U).astype(np.float32))
    output_file_path = os.path.join(relative_data_dir, 'full_final_U.npy')
    np.save(output_file_path, np.array(final_location).astype(np.float32))

    file_path = os.path.join(relative_data_dir, 'full_tag_order.json')
    tag_list = json.load(open(file_path, 'r', encoding='utf-8'), ).keys()
    tag_list = list(tag_list)

    for i in range(n):
        file_path = os.path.join(relative_data_dir,  'result', '{}.txt'.format(i))
        # if not os.path.exists(check_dir):
        #     os.path.mkdirs(check_dir)
        file = open(file_path, 'w', encoding='utf-8')
        pos = np.where(final_location[:, i] == 1)[0]
        print(pos)
        for item in pos:
            file.write(tag_list[item] + '\n')
        file.close()
    '''
    order_file_path = os.path.join(relative_data_dir, 'full_tag_order.json')
    tag_file_path = os.path.join(tensorflow_data_3_dir, 'tag.txt')

    part_fuzzy_U, part_final_location = get_part_matrix(order_file_path, tag_file_path, shape=(1896, 50))
    np.save(os.path.join(tensorflow_data_3_dir, 'fuzzy_U.npy'), part_fuzzy_U)
    np.save(os.path.join(tensorflow_data_3_dir, 'final_U.npy'), part_final_location)

'''
# 1900 tag
if __name__ == '__main__':
    file_path = os.path.join(relative_data_dir, 'tag_vector.json')
    tagFile = os.path.join(tensorflow_data_3_dir, 'tag.txt')

    n = 50
    matrix = get_file_content(file_path, tagFile)[:, :]
    fuzzy_U, final_location = fuzzy(matrix, n, 1.1)

    reuslt2file(fuzzy_U, final_location, n, tagFile)
    output_file_path = os.path.join(relative_data_dir, 'result', 'fuzzy_U.npy')
    np.save(output_file_path, np.array(fuzzy_U).astype(np.float32))
    output_file_path = os.path.join(relative_data_dir, 'result', 'final_U.npy')
    np.save(output_file_path, np.array(final_location).astype(np.float32))
'''

'''
plt.style.use('ggplot')

plt.figure()
y = np.where(final_location[:, 0] == 1)[0]
plt.plot(matrix[:, 0][y], matrix[:, 1][y], 'bs')
y = np.where(final_location[:, 1] == 1)[0]
plt.plot(matrix[:, 0][y], matrix[:, 1][y], 'kx')
y = np.where(final_location[:, 2] == 1)[0]
plt.plot(matrix[:, 0][y], matrix[:, 1][y], 'ro')

y = np.where(final_location[:, 3] == 1)[0]
plt.plot(matrix[:, 0][y], matrix[:, 1][y], 'gs')
y = np.where(final_location[:, 4] == 1)[0]
plt.plot(matrix[:, 0][y], matrix[:, 1][y], 'yx')
y = np.where(final_location[:, 5] == 1)[0]
plt.plot(matrix[:, 0][y], matrix[:, 1][y], 'o')

y = np.where(final_location[:, 6] == 1)[0]
plt.plot(matrix[:, 0][y], matrix[:, 1][y], 's')
y = np.where(final_location[:, 7] == 1)[0]
plt.plot(matrix[:, 0][y], matrix[:, 1][y], 'x')
y = np.where(final_location[:, 8] == 1)[0]
plt.plot(matrix[:, 0][y], matrix[:, 1][y], 'o')
y = np.where(final_location[:, 9] == 1)[0]
plt.plot(matrix[:, 0][y], matrix[:, 1][y], 's')

plt.show()
'''