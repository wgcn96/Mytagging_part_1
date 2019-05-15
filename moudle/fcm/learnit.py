from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

from moudle.fcm.fcm import *


def info(data):

    print(dir(data))  # 查看data所具有的属性或方法
    print(data.DESCR)  # 查看数据集的简介

    plt.style.use('ggplot')

    X = data.data  # 只包括样本的特征，150x4
    y = data.target  # 样本的类型，[0, 1, 2]
    features = data.feature_names  # 4个特征的名称
    targets = data.target_names  # 3类鸢尾花的名称，跟y中的3个数字对应

    plt.figure(figsize=(10, 4))
    plt.plot(X[:, 2][y == 0], X[:, 3][y == 0], 'bs', label=targets[0])
    plt.plot(X[:, 2][y == 1], X[:, 3][y == 1], 'kx', label=targets[1])
    plt.plot(X[:, 2][y == 2], X[:, 3][y == 2], 'ro', label=targets[2])
    plt.xlabel(features[2])
    plt.ylabel(features[3])
    plt.title('Iris Data Set')
    plt.legend()
    plt.savefig('Iris Data Set.png', dpi=200)
    plt.show()

    return data


if __name__ == '__main__':
    iris_info = load_iris()
    # info(iris_info)
    data = iris_info.data[:, :4]

    # 随机化数据
    # data, order = randomise_data(data)

    start = time.time()
    # 现在我们有一个名为data的列表，它只是数字
    # 我们还有另一个名为cluster_location的列表，它给出了正确的聚类结果位置
    # 调用模糊C均值函数
    fuzzy_U, final_location = fuzzy(data, 3, 2)

    # 还原数据
    # final_location = de_randomise_data(final_location, order)

    plt.style.use('ggplot')

    plt.figure()
    y = np.where(final_location[:, 0] == 1)[0]
    plt.plot(data[:, 0][y], data[:, 1][y], 'bs')
    y = np.where(final_location[:, 1] == 1)[0]
    plt.plot(data[:, 0][y], data[:, 1][y], 'kx')
    y = np.where(final_location[:, 2] == 1)[0]
    plt.plot(data[:, 0][y], data[:, 1][y], 'ro')
    plt.show()
    # 准确度分析
    print("用时：{0}".format(time.time() - start))
