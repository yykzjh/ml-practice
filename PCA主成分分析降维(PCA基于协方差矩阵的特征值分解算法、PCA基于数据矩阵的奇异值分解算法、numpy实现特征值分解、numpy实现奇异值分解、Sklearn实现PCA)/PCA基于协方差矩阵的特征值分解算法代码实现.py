# -*- encoding: utf-8 -*-
'''
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/21 23:06
@Version  :   1.0
@License  :   (C)Copyright 2022-
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



def load_data():
    data = np.loadtxt(r"./data/pca_data.csv", delimiter=',')
    return data


def feature_normalize(X):
    # 求每个特征的均值
    mu = np.mean(X, axis=0).reshape((1, -1))
    # 求每个特征的标准差
    sigma = np.std(X, axis=0, ddof=1).reshape((1, -1))
    # 计算数据样本标准化后的值
    X = (X - mu) / sigma
    return X, mu, sigma


# 基于协方差矩阵的特征值分解算法的PCA主成分分析
def pca(X, k):
    # 获取数据维度信息
    m, n = X.shape
    # 计算协方差矩阵
    sigma = np.dot(X.T, X) / (m - 1)

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(sigma)

    # 按照特征值从大到小排序，获得特征值排序后每一位数在原序列中的索引
    sorted_index = np.argsort(-eigenvalues)
    # 获得根据特征值从大到小排序后的特征向量序列
    eigenvectors = eigenvectors[:, sorted_index]

    # 去除特征向量的前K列作为变换矩阵
    W = eigenvectors[:, 0:k]

    # 返回降维后的数据和所有特征向量
    return np.dot(X, W), eigenvectors



# 将降维数据还原到原始数据
def recover_data(Z, eigenvectors, k):
    # 获取降维时用到的变换矩阵
    W = eigenvectors[:, 0:k]
    # 降维后的数据乘上变换矩阵的转置
    X = np.dot(Z, np.transpose(W))  # X @ W @ W.T
    return X



# 画图
def plot_data(X_origin, X_rec):
    plt.scatter(X_origin[:, 0], X_origin[:, 1])
    plt.scatter(X_rec[:, 0], X_rec[:, 1], c="red")
    plt.show()




def main():
    # 读取数据
    X = load_data()

    # 数据标准化
    X, _, _ = feature_normalize(X)

    # pca降维
    Z, eigenvectors = pca(X, k=1)
    # print(Z)

    # 升维
    X_rec = recover_data(Z, eigenvectors, k=1)

    # 画图比较
    plot_data(X, X_rec)







if __name__ == '__main__':
    main()





















