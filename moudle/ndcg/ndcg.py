import math


def dcg_cal(dk):
    dcg_value = 0.
    log_ki = []

    for ki in range(len(dk)):
        # log_ki.append(math.log(ki + 1, 2) if math.log(ki + 1, 2) != 0. else 1.)
        log_ki.append(math.log(ki + 2, 2))
        dcg_value += dk[ki] / log_ki[ki]

    return dcg_value


def ndcg_cal(predict, top_h, n):
    """
    :param predict:  预测的列表
    :param top_h:  矩阵打分，选取top-h个
    :param n:  留一法留下的正样本的数量，top_h > n

    :return:  float


    eg : 留下一个正样本，选top3，正样本的打分排在第二位
    predict = [0, 1, 0]
    top_h = 3
    n = 1

    """
    ground_truth = [1]*n + [0]*(top_h-n)
    idcg = dcg_cal(ground_truth)

    dcg = dcg_cal(predict[:top_h])

    print(idcg, dcg)

    result = dcg/idcg
    return result


def hit_ratio_cal(predict, top_h, n):
    """

    :param predict:
    :param top_h:
    :param n:
    :return: float
    """
    return float(sum(predict[:top_h]))/n


if __name__ == '__main__':

    a0 = [1, 1, 0, 0]
    a1 = [1, 0, 1, 0, 1]
    idcg = ndcg_cal(a1,top_h=5,n=3)
    print(idcg)
    # hr = hit_ratio_cal(a1,top_h=4,n=2)
    # print(idcg, hr)