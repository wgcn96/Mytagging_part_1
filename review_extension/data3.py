"""
划分libsvm数据格式

"""

from sklearn.cross_validation import train_test_split

import numpy as np
import os

from static import *

X_path = 'after_review_tags.npy'
X_path = os.path.join(workdir, X_path)
y_path = 'after_review_y.npy'
y_path = os.path.join(workdir, y_path)

X = np.load(X_path)
y = np.load(y_path)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


def array2file(ndarray, y, file):
    f = open(file, 'w')
    shape = ndarray.shape
    for i in range(shape[0]):
        vec = ndarray[i]
        line = ''
        line += str(y[i][0])
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


array2file(X_train, y_train, 'complement_train.data')
array2file(X_test, y_test, 'complement_test.data')
