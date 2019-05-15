# -*- coding: utf-8 -*-

"""
note something here

"""

__author__ = 'Wang Chen'
__time__ = '2019/4/21'

if __name__ == '__main__':
    '''
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(9.5, 7))
    name_list = ['ORI', 'Bias-SVD', 'BPR-MF', 'NeuMF', 'TagRec']
    performance_list = [74.2, 77.7, 79.2, 78.3, 80]
    rects = plt.bar(range(len(performance_list)), performance_list, color='rgbcy')
    # X轴标题
    index = [0, 1, 2, 3, 4]
    index = [float(c) for c in index]
    plt.ylim(ymax=88, ymin=60)
    plt.yticks(np.arange(60, 90, 5), fontsize=16)
    plt.xticks(index, name_list, fontsize=16)
    plt.ylabel("average arrucay(%)", fontsize=16)  # y轴标签
    plt.xlabel("different algorithms", fontsize=16)  # y轴标签
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height) + '%', ha='center', va='bottom', fontsize=16)
    plt.savefig('genre_prediction')
    plt.show()
    '''

    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(9.5, 7))
    name_list = ['ORI', 'Bias-SVD', 'BPR-MF', 'MLP', 'NeuMF', 'TagRec']
    performance_list = [77.0, 78.1, 80.3, 77.5, 78.5, 81.1]
    rects = plt.bar(range(len(performance_list)), performance_list, color='ygmbcr')
    # X轴标题
    index = [0, 1, 2, 3, 4, 5]
    index = [float(c) for c in index]
    plt.ylim(ymax=88, ymin=60)
    plt.yticks(np.arange(60, 90, 5), fontsize=16)
    plt.xticks(index, name_list, fontsize=16)
    plt.ylabel("average arrucay(%)", fontsize=16)  # y轴标签
    # plt.xlabel("different algorithms", fontsize=16)  # y轴标签
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height) + '%', ha='center', va='bottom', fontsize=16)
    plt.savefig('genre_prediction')
    plt.show()