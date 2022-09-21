# -*- encoding: utf-8 -*-
'''
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/20 21:13
@Version  :   1.0
@License  :   (C)Copyright 2022-
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score



def load_data():
    data = np.loadtxt(r"./data/cluster_data.csv", delimiter=',')
    plt.scatter(data[:, 0], data[:, 1], s=20)
    plt.show()

    return data



def generate_data():
    # 设置随机种子
    np.random.seed(0)

    # 随机生成服从指定期望和标准差的高斯分布的男性类别的身高数据
    mu_m = 1.71  # 期望
    sigma_m = 0.056  # 标准差
    num_m = 10000  # 生成的男性数据个数
    # 生成数据
    rand_data_m = np.random.normal(mu_m, sigma_m, num_m)
    # 生辰标签
    y_m = np.ones(num_m)

    # 随机生成服从指定期望和标准差的高斯分布的女性类别的身高数据
    mu_w = 1.58  # 期望
    sigma_w = 0.051  # 标准差
    num_w = 10000  # 生成的女性数据个数
    # 生成数据
    rand_data_w = np.random.normal(mu_w, sigma_w, num_w)
    # 生辰标签
    y_w = np.zeros(num_w)

    # 把男性数据和女性数据合在一起
    data = np.append(rand_data_m, rand_data_w)
    data = data.reshape((-1, 1))
    # 把男性标签和女性标签合在一起
    y = np.append(y_m, y_w)

    return data, y




def visualize(X, idx):
    # 定义类别颜色
    cm_dark = mpl.colors.ListedColormap(["r", "b", "c"])
    # 画样本点
    plt.scatter(X[:, 0], X[:, 1], c=idx, cmap=cm_dark, s=20)
    # 画质心
    plt.show()




def main():
    # # 生成数据
    # X, y = generate_data()
    #
    # # 训练高斯混合模型
    # g = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=1000)
    # g.fit(X)
    # print("类别概率：\n", g.weights_)
    # print("均值：\n", g.means_, '\n')
    # print("协方差矩阵：\n", g.covariances_)
    #
    # # 预测
    # y_hat = g.predict(X)
    # print("正确率=", accuracy_score(y, 1 - y_hat))


    # 加载数据
    X = load_data()

    # 训练高斯混合模型
    g = GaussianMixture(n_components=3, covariance_type='full', tol=1e-6, max_iter=1000)
    g.fit(X)
    print("类别概率：\n", g.weights_)
    print("均值：\n", g.means_, '\n')
    print("协方差矩阵：\n", g.covariances_)

    # 可视化
    y_hat = g.predict(X)
    visualize(X, y_hat)








if __name__ == '__main__':
    main()















