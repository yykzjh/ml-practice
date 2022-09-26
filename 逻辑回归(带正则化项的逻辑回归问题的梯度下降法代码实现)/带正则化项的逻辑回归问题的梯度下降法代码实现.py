# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/26 1:04
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import matplotlib.markers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import accuracy_score



def load_data():
    data = np.loadtxt(r"./data/data1.txt", delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]
    return X, y



def plot_scatter(X, y):
    cm_dark = mpl.cm.colors.ListedColormap(['r', 'b'])
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', cmap=cm_dark)
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.show()



def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    X = (X - mu) / sigma
    return X



# sigmoid激活函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))



# 实现假设函数
def hypothesis(X, theta):
    z = X @ theta
    return sigmoid(z)



# 计算损失值
def compute_loss(X, y, theta, lambd):
    # 获取数据维度
    m, n = X.shape
    # 计算预测值
    y_hat = hypothesis(X, theta)
    # 计算对数损失函数
    log_loss = np.sum(-1 * y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)) / m
    # 计算带正则项的损失函数
    loss = log_loss + (lambd / (2 * m)) * np.sum(theta ** 2)
    return loss



def gradient_descent_logistic_training(X, y, max_iter=500, lr=0.01, lambd=0.02):
    # 获取数据维度
    m, n = X.shape

    # 在X前面插入全为1的列
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # 修改维度信息
    n += 1
    # 初始化参数权重
    theta = np.random.normal(loc=0.0, scale=1.0, size=(n,))

    # 定义一个结构存储每次迭代输出的损失值
    costs = []
    # 迭代训练
    for epoch in range(max_iter):
        theta -= (lr / m) * (X.T @ (hypothesis(X, theta) - y) + lambd * theta)
        costs.append(compute_loss(X, y, theta, lambd=lambd))
    return theta, costs



def predict(X, theta):
    # 先标准化
    X = feature_normalize(X)

    # 定义一维全为1的特征列
    c = np.ones(X.shape[0])
    # 将c插入到数据集X中，这样可以直接与theta相乘
    X = np.insert(X, 0, values=c, axis=1)

    # 求解预测概率值
    y_hat = hypothesis(X, theta)
    # 根据预测概率值求解类别
    y_hat[y_hat >= 0.5] = 1
    y_hat[y_hat < 0.5] = 0

    return y_hat



# 画决策边界
def plot_decision_boundary(X, y, theta):
    # 画散点图
    cm_dark = mpl.cm.colors.ListedColormap(['r', 'b'])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', cmap=cm_dark, s=30)

    # 画分类决策面 theta0 + theta1 * x1 + theta2 * x2 = 0, x2 = -theta0 / theta2 - theta1 * x1 / theta2
    x1 = np.arange(min(X[:, 0]), max(X[:, 1]), 0.1)
    x2 = -(theta[1] * x1 + theta[0]) / theta[2]
    # 画分界线
    plt.plot(x1, x2)

    plt.show()



def main():
    # 加载数据
    X, y = load_data()
    # 特征标准化
    X = feature_normalize(X)
    # 显示散点图
    plot_scatter(X, y)

    # 训练
    theta, costs = gradient_descent_logistic_training(X, y, max_iter=250000, lr=0.008, lambd=0.01)

    # 画出损失值学习曲线
    x_axis = np.linspace(1, len(costs), len(costs))
    plt.plot(x_axis, costs)
    plt.show()

    # 画数据散点图和求得的回归线
    plot_decision_boundary(X, y, theta)

    # 计算训练集上的准确率
    y_hat = predict(X, theta)
    print("My Train Accuracy=", np.mean(y_hat == y))
    print("Sklearn Train Accuracy=", accuracy_score(y, y_hat))





if __name__ == '__main__':
    main()












