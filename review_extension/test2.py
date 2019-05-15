"""
svm rbf 回归

"""

from sklearn.cross_validation import train_test_split
from sklearn.svm import SVR
from sklearn import metrics

import numpy as np
import os

from static import *

X_path = 'before_review_tags.npy'
X_path = os.path.join(workdir, X_path)
y_path = 'before_review_y.npy'
y_path = os.path.join(workdir, y_path)

X = np.load(X_path)
y = np.load(y_path)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
# svr_lin = SVR(kernel='linear', C=1e3)
# svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
# y_lin = svr_lin.fit(X_train, y_train).predict(X_test)
# y_poly = svr_poly.fit(X_train, y_train).predict(X_test)

print("MSE:", metrics.mean_squared_error(y_test, y_rbf))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_rbf)))


'''
运行结果：
MSE: 1.1845668754293937
RMSE: 1.0883780939679895
'''