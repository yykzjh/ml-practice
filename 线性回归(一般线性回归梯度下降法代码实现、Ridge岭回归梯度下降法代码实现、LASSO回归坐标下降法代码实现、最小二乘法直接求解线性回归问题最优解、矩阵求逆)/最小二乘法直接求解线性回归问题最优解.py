# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/25 17:17
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import numpy as np
import matplotlib.pyplot as plt



def load_data():
    data = np.loadtxt(r"./data/data1.txt", delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]
    return X, y



def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    X = (X - mu) / sigma
    return X



def predict(X, theta):
    # 先标准化
    X = feature_normalize(X)

    # 定义一维全为1的特征列
    c = np.ones(X.shape[0])
    # 将c插入到数据集X中，这样可以直接与theta相乘
    X = np.insert(X, 0, values=c, axis=1)

    return X @ theta



def main():
    # 加载数据
    X_origin, y = load_data()
    # 特征标准化
    X_origin = feature_normalize(X_origin)
    # 扩充一列全是1的维度
    X = np.insert(X_origin, 0, values=1, axis=1)

    # 最小二乘法直接求最优解
    theta = np.linalg.inv(X.T @ X) @ X.T @ y

    # 画数据散点图和求得的回归线
    plt.scatter(X_origin, y)
    y_hat = X_origin @ theta[1:] + theta[0]
    plt.plot(X_origin, y_hat)
    plt.show()







if __name__ == '__main__':
    main()

















