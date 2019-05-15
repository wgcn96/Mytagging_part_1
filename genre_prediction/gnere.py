# -*- coding: utf-8 -*-
# python3
#

"""
根据电影详细属性文件生成电影类型
"""

__author__ = 'Wang Chen'


import sys
sys.path.extend(['D:\\workProject\\pythonProject\\Mytagging\\count',\
                'D:\\workProject\\pythonProject\\Mytagging'])

import json
import os
import time
import collections

import numpy as np


from count.static import *


def get_files(file_dir):
    for root, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        # print(files)  # 当前路径下所有非目录子文件
        files
    return files


def get_gerne(file_path):
    """
    获取一个json文件中电影的id和它的类型
    :param file:绝对路径
    :return: 一个json文件的字典
    """
    file = open(file_path, encoding='utf-8')
    content = json.load(file)
    data = content["data"]
    result = {}
    for item in data:
        genres = item.get('genres', None)
        id = item.get('id', None)
        if len(genres) == 0 or id is None:
            continue
        result[id] = genres
    return result


def get_all_genres():
    """
    获取所有电影的类型
    :return: 所有电影的字典
    """
    files = get_files(movie_detail_dir)
    result = {}
    for one_file in files:
        file_path = os.path.join(movie_detail_dir, one_file)
        one_dict = get_gerne(file_path)
        result.update(one_dict)
    return result


def statistic(all_genres_dict):
    """
    统计函数
    :param all_genres_dict:
    :return:
    """
    print(len(all_genres_dict))
    genres_set = set()
    for key, item in all_genres_dict.items():
        genres_set.update(item)
    '''
    file = open('genres.txt', 'w', encoding='utf-8')
    for item in genres_set:
        file.write(item + '\n')
    file.close()
    '''
    return genres_set


def restatistic(all_genres_dict):
    """
    将每个类型多少电影统计出来，并写入文件中
    :param all_genres_dict:
    :return: genre --> count
    """
    genres_dict = {}
    for key, item in all_genres_dict.items():
        for item_genre in item:
            if genres_dict.get(item_genre, None) is None:
                genres_dict[item_genre] = 1
            else:
                genres_dict[item_genre] += 1

    file = open('genres.txt', 'w', encoding='utf-8')
    for key, item in genres_dict.items():
        file.write(key + ' ' + str(item) + '\n')
    file.close()
    return genres_dict


def revise_dict(all_genres_dict):
    """
    定义规则
    删除
    替换
    :param all_genres_dict:
    :return: revised all_genres_dict
    """
    remove_list = ['Game-Show', 'News', '荒诞', '悬念', '鬼怪', '同性', '儿童', '舞台艺术', '运动']
    replace_dict = {'動作 Action': '动作', 'Adult': '成人', 'Reality-TV': '真人秀', '驚悚 Thriller': '惊悚',\
                    '紀錄片 Documentary': '纪录片', '愛情 Romance': '爱情', '音樂 Music': '音乐', 'Comedy': '喜剧',\
                    '動畫 Animation': '动作', '记录': '纪录片', 'Talk-Show': '脱口秀', '惊栗': '惊悚'}
    for key, item in all_genres_dict.items():
        for item_genre in item:
            if item_genre in remove_list:
                while item_genre in item:
                    item.remove(item_genre)
            if item_genre in replace_dict.keys():
                goal = replace_dict[item_genre]
                while item_genre in item:
                    item.remove(item_genre)
                if goal not in item:
                    item.append(goal)

        for remove_item in remove_list:
            if remove_item in item:
                item.remove(remove_item)

    return all_genres_dict


if __name__ == '__main__':
    all_genres_dict = get_all_genres()
    all_genres_dict = revise_dict(all_genres_dict)
    # restatistic(all_genres_dict)
    filePath = 'all_genres_dict.json'
    f = open(filePath, encoding='utf-8', mode='w')
    json.dump(all_genres_dict, f, indent=4, sort_keys=True, ensure_ascii=False)
    f.close()

    '''
    f = open(filePath, 'r', encoding='utf-8')
    all_genres_dict = json.load(f)
    '''