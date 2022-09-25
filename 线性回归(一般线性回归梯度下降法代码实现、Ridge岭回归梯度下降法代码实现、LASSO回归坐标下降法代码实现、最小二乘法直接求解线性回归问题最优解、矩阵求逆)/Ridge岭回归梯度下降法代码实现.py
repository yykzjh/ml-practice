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


def compute_loss(X, y, theta, lambd=0.02):
    # 获取数据维度
    m, n = X.shape
    # 计算损失值
    loss = np.sum(np.power(X @ theta - y, 2)) / (2 * m) + lambd * np.sum(theta**2)
    return loss



def gradient_descent_ridge_training(X, y, max_iter=500, lr=0.01, lambd=0.02):
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
        theta += lr * (((y - X @ theta).reshape((1, -1)) @ X).reshape((-1,)) / m - 2 * lambd * theta)
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
    theta, costs = gradient_descent_ridge_training(X, y, max_iter=500, lr=0.01, lambd=0.02)

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

















