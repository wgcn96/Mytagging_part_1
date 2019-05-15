# 基础的协同过滤算法
import numpy as np
import math
from cofiCostFunc import *
from scipy.io import loadmat
import scipy.optimize as op
import math


def test():
    Y=np.mat([[1,2,0],[3,4,4]]) # matrix definition
    R= np.mat([[1,1,0], [1,1,1]])
    X=np.mat([[1,2,1,1],[1,2,1,0],[0,1,2,0]])
    Theta=np.mat([[1,1,3,1],[2,1,0,1]])
    return Y,R,X,Theta

def loadData(num_features):
    ratingdata=loadmat('movielens100k_movies.mat') # 评分数据, Y, R
    paramsdata=loadmat('movielens100k_movieParams.mat') # 初始化X,Theta
    Y=np.mat(ratingdata['Y'],dtype=np.float)
    R=np.mat(ratingdata['R'],dtype=np.float)
    X=np.mat(paramsdata['X'],dtype=np.float) # 电影特征
    Theta=np.mat(paramsdata['Theta'],dtype=np.float) # 用户特征
    if num_features < 10:
        g1_X = np.mat(X[:, 0:num_features])  # 电影特征 (1682,7)
        g1_Theta = np.mat(Theta[:, 0:num_features])  # 用户特征 (943,7)
        return Y, R, g1_X, g1_Theta
    if num_features > 10:
        g2_X = np.mat(X[:, :])  # 电影特征
        g2_X = np.column_stack((g2_X, X[:, 0:num_features - 10]))  # (841,12)
        g2_Theta = np.mat(Theta[:, :])  # 用户特征
        g2_Theta = np.column_stack((g2_Theta, Theta[:, 0:num_features - 10]))  # (943, 12)
        return Y, R, g2_X, g2_Theta

    return Y,R,X,Theta

def MF(Y,R,X,Theta,num_features):
    # 载入数据
    num_users=Y.shape[1]
    num_movies=Y.shape[0]
    lamda=10
    params=unfold(X,Theta)
    # TODO
    # data split
    '''
    ratio = 0.75
    [train_R, train_idx, test_idx] = dataSplit(R, ratio)
    '''
    ######### optimize传入的必须是一维数组!
    result = op.minimize(fun=cofiCostFunc, x0=params, args=(Y, R, num_users, num_movies, num_features, lamda),method='TNC', jac=gradient) # 优化函数
    print(result)
    print('params:')
    params=result['x']
    print(params)
    x_part=params[0:num_movies*num_features]
    X=np.mat(x_part).reshape(num_movies,num_features)
    theta_part=params[num_movies*num_features:]
    Theta=np.mat(theta_part).reshape(num_users,num_features)
    print(X*Theta.T)
    return X,Theta
    # 出现的问题：点乘的时候溢出？？# 改进：使用minimize函数

# 比较误差RMSE(先和原始评分比较,只进行有评分的误差分析）
# 输入：计算出的X,Theta,测试集的Y,R
def RMSE(result,Y,R):
    print("Calculating RMSE...")
    error = 0
    test_num = 0
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if R[i,j] == 1:
                test_num = test_num + 1
                error = error + (result[i,j] - Y[i,j])**2
    RMSE = math.sqrt(error / test_num)
    return RMSE

# 基础的协同过滤方法
def basicCF():
    num_features = 8
    [Y, R, X, Theta]=loadData(num_features)
    [final_X,final_Theta]=MF(Y, R, X, Theta,num_features)
    result = final_X*final_Theta.T
    rmse = RMSE(result,Y,R)
    print('RMSE:')
    print(rmse)

if __name__ == '__main__':
    basicCF()

# 结果记录
# 将总体的数据作为训练集与测试集时候的RMSE：0.7997124954884923