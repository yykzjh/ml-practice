# -*- encoding: utf-8 -*-
'''
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/20 19:19
@Version  :   1.0
@License  :   (C)Copyright 2022-
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.stats import multivariate_normal



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



# 高斯混合模型
def mixture_of_gaussian(X, k, num_iter=1000):
    m, d = X.shape
    # 初始化参数
    mu = np.random.uniform(0, 3, (k, d))  # k个均值向量(k, d)
    sigma = np.stack([np.identity(d)] * k, axis=0)  # k个协方差矩阵(k, d, d)
    pi = np.ones((k, )) / k  # k个混合系数(k, )

    # 迭代
    for epoch in range(num_iter):
        print(epoch)
        # 定义高斯密度函数
        norm = [multivariate_normal(mu[i], sigma[i]) for i in range(k)]  # (k, )
        # 计算高斯条件概率（似然）矩阵
        prob = np.stack([norm_func.pdf(X).reshape((-1, )) for norm_func in norm], axis=1)  # (m, k)
        # 全概率公式计算每个样本的概率
        px = np.sum(pi.reshape((1, -1)) * prob, axis=1).reshape((-1, 1))  # (m, 1)
        # 计算后验概率矩阵
        gamma = pi.reshape((1, -1)) * prob / px  # (m, k)

        # 计算Nk矩阵
        Nk = np.sum(gamma, axis=0)  # (k, )
        # 计算新的均值
        mu_new = np.stack([np.sum(gamma[:, i].reshape((-1, 1)) * X, axis=0) / Nk[i] for i in range(k)], axis=0)  # (k, d)
        # 计算新的协方差矩阵
        sigma_new = np.stack(  # (k, d, d)
            [np.sum(  # (d, d)
                gamma[:, i].reshape(-1, 1, 1) * np.stack(  # (m, d, d)
                    [(X[j] - mu[i]).reshape(-1, 1) @ (X[j] - mu[i]).reshape(1, -1) for j in range(m)],
                    axis=0),
                axis=0) / Nk[i] for i in range(k)],
            axis=0)
        # 计算新的混合系数
        pi_new = Nk / m
        # 更新
        mu = mu_new
        sigma = sigma_new
        pi = pi_new

    return mu, sigma, pi




def predict(X, mu, sigma, pi):
    k = mu.shape[0]
    # 定义高斯密度函数
    norm = [multivariate_normal(mu[i], sigma[i]) for i in range(k)]  # (k, )
    # 计算高斯条件概率（似然）矩阵
    prob = np.stack([norm_func.pdf(X).reshape((-1,)) for norm_func in norm], axis=1)  # (m, k)
    # 计算后验概率矩阵
    gamma = pi.reshape((1, -1)) * prob  # (m, k)
    # 获得每个样本后验概率最大的混合成分的索引
    y_hat = np.argmax(gamma, axis=1)

    return y_hat








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

    # # 训练高斯混合模型
    # mu, sigma, pi = mixture_of_gaussian(X, 2, num_iter=100)
    # print("类别概率：", pi)
    # print("均值：", mu)
    # print("协方差矩阵：", sigma)


    # 加载数据
    X = load_data()

    # 训练高斯混合模型
    mu, sigma, pi = mixture_of_gaussian(X, 3, num_iter=100)
    print("类别概率：", pi)
    print("均值：", mu)
    print("协方差矩阵：", sigma)

    # 可视化
    y_hat = predict(X, mu, sigma, pi)
    visualize(X, y_hat)





if __name__ == '__main__':
    main()








