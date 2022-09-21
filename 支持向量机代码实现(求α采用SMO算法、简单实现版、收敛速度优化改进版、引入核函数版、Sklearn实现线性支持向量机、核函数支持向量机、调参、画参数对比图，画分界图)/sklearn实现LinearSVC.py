# -*- encoding: utf-8 -*-
'''
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/15 19:55
@Version  :   1.0
@License  :   (C)Copyright 2022-
'''
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle



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



def display(X, y, model):
    # 定义画布
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 画数据点
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    plt.scatter(np.array(X)[:, 0], np.array(X)[:, 1], c=np.array(y).squeeze(), cmap=cm_dark, s=30)

    # 画决策边界
    x1_min, x1_max = np.array(X)[:, 0].min() - .5, np.array(X)[:, 0].max() + .5
    x1 = np.arange(x1_min, x1_max, 0.1)
    w = model.coef_
    b = model.intercept_
    x2 = (-w[0][0] * x1 - b) / w[0][1]
    ax.plot(x1, x2)

    # 画间隔平面
    pos2 = (1 - w[0][0] * x1 - b) / w[0][1]
    ax.plot(x1, pos2, color="green")

    neg2 = (-1 - w[0][0] * x1 - b) / w[0][1]
    ax.plot(x1, neg2, color="green")

    # # 画支持向量
    # alphas_non_zeros_index = model.support_
    # for i in alphas_non_zeros_index:
    #     circle = Circle((X[i][0], X[i][1]), 0.05, facecolor='none', edgecolor=(0.7, 0.7, 0.7), linewidth=3, alpha=0.5)
    #     ax.add_patch(circle)

    ax.axis([-2, 12, -8, 6])
    plt.show()










def main():
    # 加载数据
    X, y = load_dataset(r"./data/svm1.txt")

    # 训练
    model = LinearSVC(C=1)
    model.fit(X, y)

    # 预测
    print(model.predict(np.array([[7.886242, 0.191813]])))

    # 获取常用的属性
    print("w = ", model.coef_)
    print("b = ", model.intercept_)
    # print("各类别各有多少个支持向量：", model.n_support_)
    # print("各类别的支持向量在训练样本中的索引：", model.support_)
    # print("各类别所有的支持向量：", model.support_vectors_)
    # print("支持向量的alpha值：", model.dual_coef_)
    # print(model.dual)


    display(X, y, model)






if __name__ == '__main__':
    main()



























