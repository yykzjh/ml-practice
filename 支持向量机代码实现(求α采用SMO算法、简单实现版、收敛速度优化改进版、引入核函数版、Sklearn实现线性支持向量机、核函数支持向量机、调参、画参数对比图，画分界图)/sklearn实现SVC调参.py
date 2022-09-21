# -*- encoding: utf-8 -*-
'''
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/15 20:16
@Version  :   1.0
@License  :   (C)Copyright 2022-
'''
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors



# 读取数据
def load_dataset(file_path):
    # 初始化数据结构
    X = []
    y = []
    # 打开数据集文件
    f = open(file_path)
    # 遍历每行
    for line in f.readlines():
        # 每行字符串分割
        line_arr = line.strip().split('\t')
        # 前两列前加到X
        X.append([float(line_arr[0]), float(line_arr[1])])
        # 第三列添加到y
        y.append(float(line_arr[2]))

    return X, y




def main():
    # 加载数据
    X, y = load_dataset(r"./data/svm3.txt")

    # 转换数据格式
    X = pd.DataFrame(X)
    y = pd.Series(y)

    # 分类器
    clf_params = (('rbf', 1, 0.1), ('rbf', 1, 1), ('rbf', 1, 10), ('rbf', 1, 100),
                  ('rbf', 5, 0.1), ('rbf', 5, 1), ('rbf', 5, 10), ('rbf', 5, 100),
                  ('rbf', 1, 5), ('rbf', 50, 5), ('rbf', 100, 5), ('rbf', 1000, 5))
    # 获取数据集在两个维度上的边界值
    x1_min, x2_min = np.min(X, axis=0)
    x1_max, x2_max = np.max(X, axis=0)
    # 生成网格点
    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
    # 网格点的两个维度展开后对应拼接在一起
    grid_test = np.stack((x1.flat, x2.flat), axis=1)

    # 定义要用到的颜色列表
    cm_light = mpl.colors.ListedColormap(["#77E0A0", "#FFA0A0"])
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    # 设置字体和负号编码，解决中文乱码问题
    mpl.rcParams["font.sans-serif"] = [u"SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    # 设置画布大小和背景颜色
    plt.figure(figsize=(14, 10), facecolor='w')

    # 遍历要测试的参数
    for i, param in enumerate(clf_params):
        # 定义支持向量机模型
        clf = svm.SVC(C=param[1], kernel=param[0], gamma=param[2])
        # 训练
        clf.fit(X, y)
        # 获取训练集上的正确率
        y_hat = clf.predict(X)
        # 当前测试的子图标题
        title = "C = {0:.1f}, $\gamma$ = {1:.1f}, 准确率={2:.2f}".format(param[1], param[2], accuracy_score(y, y_hat))
        # 画子图
        plt.subplot(3, 4, i+1)

        # 预测网格点
        grid_hat = clf.predict(grid_test)
        # 变换网格点预测类别结果数组的形状
        grid_hat = grid_hat.reshape(x1.shape)
        # 画分界图
        plt.pcolormesh(x1, x2, grid_hat, shading="auto", cmap=cm_light, alpha=0.8)
        # 画样本点图
        plt.scatter(X[0], X[1], c=y, edgecolors='k', s=40, cmap=cm_dark)
        # 画支持向量
        plt.scatter(X.iloc[clf.support_, 0], X.iloc[clf.support_, 1], edgecolors='k', facecolors='none', s=100, marker='o')

        z = clf.decision_function(grid_test)
        z = z.reshape(x1.shape)
        plt.contour(x1, x2, z, colors=list("kbrbk"), linestyles=['--', '--', '-', '--', '--'],
                    linewidths=[1, 0.5, 1.5, 0.5, 1], levels=[-1, -0.5, 0, 0.5, 1])

        # 设置视野范围
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        # 设置子图标题
        plt.title(title, fontsize=14)

    # 设置主标题
    plt.suptitle("SVM不同参数的分类", fontsize=20)
    plt.show()






if __name__ == '__main__':
    main()



























