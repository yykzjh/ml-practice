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


def compute_loss(X, y, theta, lambd=0.2):
    # 获取数据维度
    m, n = X.shape
    # 计算损失值
    loss = np.sum(np.power(X @ theta - y, 2)) + lambd * np.sum(np.abs(theta))
    return loss



def coordinate_descent_LASSO_training(X, y, max_iter=500, lambd=0.2):
    # 获取数据维度
    m, n = X.shape

    # 定义一维全为1的特征列
    c = np.ones(m)
    # 将c插入到数据集X中，这样可以把b归入theta中一起训练
    X = np.insert(X, 0, values=c, axis=1)

    # 修改维度信息
    n += 1
    # 初始化参数权重
    theta = np.random.normal(loc=0.0, scale=1.0, size=(n,))

    # 定义一个结构存储每次迭代输出的损失值
    costs = []
    # 迭代训练
    for epoch in range(max_iter):
        # 计算zk
        z = np.sum(X**2, axis=0)
        # 循环更新每一个theta
        for k in range(n):
            # 将第k列特征切分出来
            X_slice = np.concatenate([X[:, :k], X[:, k+1:]], axis=1)
            # 第k个theta切分出来
            theta_slice = np.concatenate([theta[:k], theta[k+1:]], axis=0)
            # 计算pk
            pk = (y - X_slice @ theta_slice) @ X[:, k]
            # 分类讨论，调整theta
            if pk < -lambd / 2:
                theta[k] = (pk + lambd / 2) / z[k]
            elif pk > lambd / 2:
                theta[k] = (pk - lambd / 2) / z[k]
            else:
                theta[k] = 0
        # 计算更新后的损失值
        costs.append(compute_loss(X, y, theta, lambd=lambd))
    return theta, costs



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
    X, y = load_data()
    # 特征标准化
    X = feature_normalize(X)

    # 训练
    theta, costs = coordinate_descent_LASSO_training(X, y, max_iter=500, lambd=0.2)

    # 画出损失值学习曲线
    x_axis = np.linspace(1, len(costs), len(costs))
    plt.plot(x_axis, costs)
    plt.show()

    # 画数据散点图和求得的回归线
    plt.scatter(X, y)
    y_hat = X @ theta[1:] + theta[0]
    plt.plot(X, y_hat)
    plt.show()







if __name__ == '__main__':
    main()

























