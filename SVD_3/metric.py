# -*-encoding:utf-8-*-
import sys
import math

import numpy as np
import pandas as pd


def getMAE(list1, list2):
    total_error = 0
    if len(list1) != len(list2):
        raise Exception("维度不同，输入有误")
    elif len(list1) == 0:
        raise Exception("维度为0，输入有误")
    for i in range(len(list1)):
        total_error += abs(list1[i] - list2[i])
    return total_error / len(list1)


def getRMAE(list1, list2):
    total_error = 0
    if len(list1) != len(list2):
        raise Exception("维度不同，输入有误")
    elif len(list1) == 0:
        raise Exception("维度为0，输入有误")
    for i in range(len(list1)):
        total_error += (list1[i] - list2[i]) * (list1[i] - list2[i])
    return total_error / len(list1)


def getSVMresult(n):
    file_path_one = 'D:\workProject\githubProject\libsvm-master\mydata2\k30_output_{}.txt'.format(n)
    file_path_two = 'D:\workProject\githubProject\libsvm-master\mydata2\k30_test_{}.data'.format(n)

    score_list1 = []
    score_list2 = []
    with open(file_path_one) as f:
        for line in f.readlines():
            score_list1.append(float(line.split(' ')[0]))

    with open(file_path_two) as f:
        for line in f.readlines():
            score_list2.append(float(line.split(' ')[0]))

    mae = round(getMAE(score_list1, score_list2), 3)
    mse = round(getRMAE(score_list1, score_list2), 3)
    return mae, mse


def getOWNresult(n):
    file_path_one = 'D:\workProject\githubProject\libsvm-master\mydata0408\own_k30_output_{}.txt'.format(n)
    file_path_two = 'D:\workProject\githubProject\libsvm-master\mydata0408\own_k30_test_{}.data'.format(n)

    score_list1 = []
    score_list2 = []
    with open(file_path_one) as f:
        for line in f.readlines():
            score_list1.append(float(line.split(' ')[0]))

    with open(file_path_two) as f:
        for line in f.readlines():
            score_list2.append(float(line.split(' ')[0]))

    # print("MAE: " + str(getMAE(score_list1, score_list2)))
    # print("MSE: " + str(getRMAE(score_list1, score_list2)))
    # print("RMSE: " + str(math.sqrt(getRMAE(score_list1, score_list2))))
    mae = round(getMAE(score_list1, score_list2), 3)
    mse = round(getRMAE(score_list1, score_list2), 3)
    return mae, mse


def getNeuMFresult(n):
    file_path_one = 'D:\\workProject\\githubProject\\libsvm-master\\mydata2\\NeuMF_output_{}.txt'.format(n)
    file_path_two = 'D:\\workProject\\githubProject\\libsvm-master\\mydata2\\NeuMF_test_{}.data'.format(n)

    score_list1 = []
    score_list2 = []
    with open(file_path_one) as f:
        for line in f.readlines():
            score_list1.append(float(line.split(' ')[0]))

    with open(file_path_two) as f:
        for line in f.readlines():
            score_list2.append(float(line.split(' ')[0]))

    # print("MAE: " + str(getMAE(score_list1, score_list2)))
    # print("MSE: " + str(getRMAE(score_list1, score_list2)))
    # print("RMSE: " + str(math.sqrt(getRMAE(score_list1, score_list2))))
    mae = round(getMAE(score_list1, score_list2), 3)
    mse = round(getRMAE(score_list1, score_list2), 3)
    return mae, mse

'''
if __name__ == '__main__':
    result_list = []
    for n in range(1, 41, 1):
        current = []
        print(n)
        # getSVMresult(n=n)
        mae, mse = getOWNresult(n=n)
        current.append(mae)
        current.append(mse)
        result_list.append(current)
    result_list = np.array(result_list)

    df = pd.DataFrame(result_list, index=[str(i) for i in range(1, 41, 1)], columns=['mae', 'mse'])
    df = df.T
    df.to_csv('own_predict_result_0408.csv'.format(n))
'''

if __name__ == '__main__':
    result_list = []
    for n in range(5, 25, 5):
        current = []
        print(n)
        # getSVMresult(n=n)
        mae, mse = getNeuMFresult(n=n)
        current.append(mae)
        current.append(mse)
        result_list.append(current)
    result_list = np.array(result_list)

    df = pd.DataFrame(result_list, index=[str(i) for i in range(5, 25, 5)], columns=['mae', 'mse'])
    df = df.T
    df.to_csv('NeuMFresult_0505.csv')

    print(result_list)
