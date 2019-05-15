# -*- coding: utf-8 -*-
"""
libSVM工具函数，加载文件，改写成训练格式
"""

__author__ = 'Wang Chen'
__time__ = '2019/4/10'


import json


def array2file(ndarray, y, file):
    f = open(file, 'w')
    shape = ndarray.shape
    for i in range(shape[0]):
        vec = ndarray[i]
        line = ''
        line += str(y[i])
        line += ' '
        pos = 0
        for item in vec:
            pos += 1
            if item:
                line += str(pos)
                line += ':'
                line += str(item)
                line += ' '
        line += '\n'
        f.write(line)
    f.close()


def loadmovieList(movie_file):
    movie_dict = {}       # 反取，取下标
    movie_list = []        # 正取，取movie名称
    f = open(movie_file, 'r', encoding='utf-8')
    count = -1
    while True:
        count += 1
        line = f.readline()
        line = line.strip()

        if line == "" or line is None:
            break

        movie = line.split(" ")[0]
        movie_list.append(movie)
        movie_dict[movie] = count
    f.close()
    print("movie list len: {}".format(len(movie_list)))

    return movie_list, movie_dict


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