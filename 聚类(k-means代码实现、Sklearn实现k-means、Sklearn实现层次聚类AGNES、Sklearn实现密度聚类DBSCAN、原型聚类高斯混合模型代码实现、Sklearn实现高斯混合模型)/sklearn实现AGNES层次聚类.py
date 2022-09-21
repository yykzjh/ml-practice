# -*- encoding: utf-8 -*-
'''
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/20 16:43
@Version  :   1.0
@License  :   (C)Copyright 2022-
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io

from sklearn.cluster import AgglomerativeClustering


def load_data():
    data = np.loadtxt(r"./data/cluster_data.csv", delimiter=',')
    plt.scatter(data[:, 0], data[:, 1], s=20)
    plt.show()

    return data



def visualize(X, idx):
    # 定义类别颜色
    cm_dark = mpl.colors.ListedColormap(["g", "r", "b"])
    # 画样本点
    plt.scatter(X[:, 0], X[:, 1], c=idx, cmap=cm_dark, s=20)
    plt.show()




def main():
    # 加载数据
    X = load_data()

    # 定义模型
    """
    linkage可取值：
        - ward: 最小方差
        - complete: 最大距离
        - average: 平均距离
        - single: 最小距离
    """
    model = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="complete")
    # 训练
    model.fit(X)
    print("每个样本所属的簇：", model.labels_)

    # 可视化
    visualize(X, model.labels_)




if __name__ == '__main__':
    main()





















