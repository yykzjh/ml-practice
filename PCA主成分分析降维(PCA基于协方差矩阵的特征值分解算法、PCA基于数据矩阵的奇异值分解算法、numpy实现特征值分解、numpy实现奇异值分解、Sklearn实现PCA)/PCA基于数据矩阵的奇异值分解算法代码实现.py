# -*- encoding: utf-8 -*-
'''
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/21 23:07
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



# 基于数据矩阵的奇异值分解算法的PCA主成分分析
def pca(X, k):
    # 获取数据维度信息
    m, n = X.shape

    # 对数据矩阵进行SVD分解
    u, s, vT = np.linalg.svd(X, full_matrices=0)

    # 获得变换矩阵
    W = vT.T[:, 0:k]

    # 返回降维后的数据和u,s,vT
    return X.dot(W), u, s, vT



# 将降维数据还原到原始数据
def recover_data(Z, vT, k):
    # 获取降维时用到的变换矩阵
    W = vT.T[:, 0:k]
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
    Z, u, s, vT = pca(X, k=1)
    print(Z)

    # 升维
    X_rec = recover_data(Z, vT, k=1)

    # 画图比较
    plot_data(X, X_rec)







if __name__ == '__main__':
    main()












