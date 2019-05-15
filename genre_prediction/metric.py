# -*- coding: utf-8 -*-

"""
note something here
准确率，精确率和召回率
"""

__author__ = 'Wang Chen'
__time__ = '2019/4/10'


import numpy as np


def get_accuracy(pre, groundtruth):

    pre = np.array(pre, dtype=np.int32)
    groundtruth = np.array(groundtruth, dtype=np.int32)

    # 计算同或
    hit = np.sum(np.logical_not(np.logical_xor(pre, groundtruth)))
    return round(hit/len(groundtruth), 3)


def get_precision():
    pass


def get_recall():
    pass


def ave_accuracy(file_prefix_path):
    accu = []
    for genre_order in range(8):
        tmp_pre = []
        tmp_ground = []
        current_output_file_name = file_prefix_path + "{}.txt".format(genre_order)
        current_groundtruth_file_name = file_prefix_path + "{}.test".format(genre_order)
        with open(current_output_file_name, 'r', encoding='utf-8') as output_file:
            for line in output_file.readlines():
                tmp_pre.append(int(line[0]))
        with open(current_groundtruth_file_name, 'r', encoding='utf-8') as groundtruth_file:
            for line in groundtruth_file.readlines():
                tmp_ground.append(int(line[0]))
        current_accu = get_accuracy(tmp_pre, tmp_ground)
        print(current_accu)
        accu.append(current_accu)
    accu = np.array(accu)
    return round(np.mean(accu), 3)


def all_accuracy(file_prefix_path):
    pre = []
    groundtruth = []
    for genre_order in range(8):
        tmp_pre = []
        tmp_ground = []
        current_output_file_name = file_prefix_path + "{}.txt".format(genre_order)
        current_groundtruth_file_name = file_prefix_path + "{}.test".format(genre_order)
        with open(current_output_file_name, 'r', encoding='utf-8') as output_file:
            for line in output_file.readlines():
                pre.append(int(line[0]))
                tmp_pre.append(int(line[0]))
        with open(current_groundtruth_file_name, 'r', encoding='utf-8') as groundtruth_file:
            for line in groundtruth_file.readlines():
                groundtruth.append(int(line[0]))
                tmp_ground.append(int(line[0]))
        print(get_accuracy(tmp_pre, tmp_ground))
    pre = np.array(pre, dtype=np.int32)
    groundtruth = np.array(groundtruth, dtype=np.int32)

    # 计算同或
    hit = np.sum(np.logical_not(np.logical_xor(pre, groundtruth)))
    return round(hit/len(groundtruth), 3)


def all_precision(file_prefix_path):
    pre = []
    groundtruth = []
    for genre_order in range(30):
        current_output_file_name = file_prefix_path + "{}.txt".format(genre_order)
        current_groundtruth_file_name = file_prefix_path + "{}.test".format(genre_order)
        with open(current_output_file_name, 'r', encoding='utf-8') as output_file:
            for line in output_file.readlines():
                pre.append(int(line[0]))
        with open(current_groundtruth_file_name, 'r', encoding='utf-8') as groundtruth_file:
            for line in groundtruth_file.readlines():
                groundtruth.append(int(line[0]))

    pre = np.array(pre, dtype=np.int32)
    groundtruth = np.array(groundtruth, dtype=np.int32)

    # 计算精确率
    # 预测对的/预测为1的
    hit_array = np.where(pre == 1, groundtruth, np.zeros(len(groundtruth), dtype=np.int32))
    return np.sum(hit_array)/np.sum(pre)


def all_recall(file_prefix_path):
    pre = []
    groundtruth = []
    for genre_order in range(30):
        current_output_file_name = file_prefix_path + "{}.txt".format(genre_order)
        current_groundtruth_file_name = file_prefix_path + "{}.test".format(genre_order)
        with open(current_output_file_name, 'r', encoding='utf-8') as output_file:
            for line in output_file.readlines():
                pre.append(int(line[0]))
        with open(current_groundtruth_file_name, 'r', encoding='utf-8') as groundtruth_file:
            for line in groundtruth_file.readlines():
                groundtruth.append(int(line[0]))

    pre = np.array(pre, dtype=np.int32)
    groundtruth = np.array(groundtruth, dtype=np.int32)

    # 计算召回率
    # 预测对的/groundtruth
    hit_array = np.where(pre == 1, groundtruth, np.zeros(len(groundtruth), dtype=np.int32))
    return np.sum(hit_array)/np.sum(groundtruth)


if __name__ == '__main__':
    # file_prefix_path = 'D:\\workProject\\pythonProject\\Mytagging\\genre_prediction\\output\\ori\\'
    file_prefix_path = 'D:\\workProject\\pythonProject\\Mytagging\\genre_prediction\\output\\ori\\top10_genre'
    ori = ave_accuracy(file_prefix_path)

    file_prefix_path = 'D:\\workProject\\pythonProject\\Mytagging\\genre_prediction\\output\\svd\\top10_genre'
    svd = ave_accuracy(file_prefix_path)

    file_prefix_path = 'D:\\workProject\\pythonProject\\Mytagging\\genre_prediction\\output\\review\\top10_genre'
    review = ave_accuracy(file_prefix_path)

    file_prefix_path = 'D:\\workProject\\pythonProject\\Mytagging\\genre_prediction\\output\\bpr\\top10_genre'
    bpr = ave_accuracy(file_prefix_path)

    file_prefix_path = 'D:\\workProject\\pythonProject\\Mytagging\\genre_prediction\\output\\NeuMF\\top10_genre'
    NeuMF = ave_accuracy(file_prefix_path)

    file_prefix_path = 'D:\\workProject\\pythonProject\\Mytagging\\genre_prediction\\output\\MLP\\top10_genre'
    MLP = ave_accuracy(file_prefix_path)

    file_prefix_path = 'D:\\workProject\\pythonProject\\Mytagging\\genre_prediction\\output\\item_CF\\top10_genre'
    item_CF = ave_accuracy(file_prefix_path)

    file_prefix_path = 'D:\\workProject\\pythonProject\\Mytagging\\genre_prediction\\output\\TagRank\\top10_genre'
    TagRank = ave_accuracy(file_prefix_path)

    file_prefix_path = 'D:\\workProject\\pythonProject\\Mytagging\\genre_prediction\\output\\own\\top10_genre'
    own = ave_accuracy(file_prefix_path)

    print(ori, review, svd, bpr,item_CF, MLP, NeuMF, TagRank, own)
