# -*- encoding: utf-8 -*-
'''
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/20 13:21
@Version  :   1.0
@License  :   (C)Copyright 2022-
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io



def load_data():
    data = np.loadtxt(r"./data/cluster_data.csv", delimiter=',')
    plt.scatter(data[:, 0], data[:, 1], s=20)
    plt.show()

    return data


# 随机初始化k个质心
def random_init_k_centroids(X, k):
    # 从X中随机选择k个样本作为质心
    indexes = np.random.randint(0, len(X), k)
    return X[indexes]



# 计算每个样本离得最近的质心，并将其归类
def find_closest_centroid(X, centroids):
    # 初始化分类数组
    idx = np.zeros((X.shape[0], ))
    # 遍历每个样本
    for i in range(X.shape[0]):
        # 初始化最近的距离
        min_dist = float('inf')
        # 初始化最近距离的质心索引
        index = -1
        # 遍历所有的质心
        for k in range(centroids.shape[0]):
            # 计算当前样本到质心的距离（欧氏距离）
            distance = np.linalg.norm(X[i] - centroids[k])
            # 判断是否更新
            if distance < min_dist:
                min_dist = distance
                index = k
        # 记录当前样本最近的质心索引
        idx[i] = index
    return idx



# 重新计算质心
def update_centroids(X, idx):
    # 确定所有质心的索引集合
    indexes = set(idx)
    indexes = list(indexes)
    # 初始化新的质心
    centroids = np.zeros((len(indexes), X.shape[1]))
    # 遍历所有的质心索引
    for i, k in enumerate(indexes):
        # 获得属于当前质心的样本
        samples = X[np.where(idx == k)]
        # 计算样本的均值
        centroids[i] = np.mean(samples, axis=0)
    return centroids




# k_means算法
def k_means(X, k, max_iter):
    # 初始化质心
    centroids = random_init_k_centroids(X, k)
    # 迭代
    for i in range(max_iter):
        # 计算样本到质心距离，并返回每个样本所属的质心
        idx = find_closest_centroid(X, centroids)
        # 更新质心
        centroids = update_centroids(X, idx)

    return idx, centroids




def visualize(X, idx, centroids):
    # 定义类别颜色
    cm_dark = mpl.colors.ListedColormap(["g", "r", "b"])
    # 画样本点
    plt.scatter(X[:, 0], X[:, 1], c=idx, cmap=cm_dark, s=20)
    # 画质心
    plt.scatter(centroids[:, 0], centroids[:, 1], c=np.arange(len(centroids)), cmap=cm_dark, marker="*", s=500)
    plt.show()



def main():
    # 加载数据
    X = load_data()

    # 执行k-means算法
    idx, centroids = k_means(X, 3, 100)

    # 画图
    visualize(X, idx, centroids)






if __name__ == '__main__':
    main()














