# -*- encoding: utf-8 -*-
'''
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/22 0:04
@Version  :   1.0
@License  :   (C)Copyright 2022-
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA



def load_data():
    data = np.loadtxt(r"./data/pca_data.csv", delimiter=',')
    return data



# 画图
def plot_data(X_origin, X_rec):
    plt.scatter(X_origin[:, 0], X_origin[:, 1])
    plt.scatter(X_rec[:, 0], X_rec[:, 1], c="red")
    plt.show()



def main():
    # 加载数据
    X = load_data()

    # 定义PCA模型
    pca = PCA(n_components=1)
    # 生成降维后的数据
    Z = pca.fit_transform(X)

    # 显示一些信息
    print("主成分个数=", pca.n_components)
    print("贡献比=", pca.explained_variance_ratio_)
    print("特征的方差=", pca.explained_variance_)

    # 还原数据
    X_rec = pca.inverse_transform(Z)

    # 画图比较
    plot_data(X, X_rec)




if __name__ == '__main__':
    main()
















