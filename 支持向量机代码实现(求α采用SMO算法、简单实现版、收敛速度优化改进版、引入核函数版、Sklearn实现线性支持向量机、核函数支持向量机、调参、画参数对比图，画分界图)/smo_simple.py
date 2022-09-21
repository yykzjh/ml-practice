# -*- encoding: utf-8 -*-
'''
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/14 11:05
@Version  :   1.0
@License  :   (C)Copyright 2022-
'''
import numpy as np
import random
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



# 画散点图
def display_scatter_diagram(X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    ax.scatter(np.array(X)[:, 0], np.array(X)[:, 1], c=np.array(y).squeeze(), cmap=cm_dark, s=30)
    plt.show()



# 传入alpha_i随机选择一个不等于i的alpha_j
def random_select_alpha_j(i ,m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))

    return j



# 裁剪更新的alpha的上下界
def clip_new_alpha(a, H, L):
    if a > H:
        a = H
    if a < L:
        a = L

    return a



# 计算g(xi)
def cal_gxi(xi, alphas, y_np, X_np, b):
    # 确保xi的维度为(n, 1)
    xi = xi.reshape(-1, 1)
    # 计算g(xi)
    gx = (alphas * y_np).T @ (X_np @ xi) + b

    return gx



# 计算Ei
def cal_Ei(gxi, yi):
    return gxi - yi



# 计算alpha_j更新的约束上下界
def cal_L_H(i, j, alphas, y_np, C):
    if y_np[i, 0] != y_np[j, 0]:
        L = max(0, alphas[j] - alphas[i])
        H = min(C, C + alphas[j] - alphas[i])
    else:
        L = max(0, alphas[j] + alphas[i] - C)
        H = min(C, alphas[j] + alphas[i])

    return L, H



# 计算eta
def cal_eta(i, j, X_np):
    # 转换格式
    xi = X_np[i, :].reshape((-1, 1))
    xj = X_np[j, :].reshape((-1, 1))
    # 计算
    return 2 * (xi.T @ xj) - xi.T @ xi - xj.T @ xj



# 更新alpha_j
def update_alpha_j(alpha_j_old, yj, Ei, Ej, eta):
    return alpha_j_old - (yj * (Ei - Ej)) / eta



# 更新alpha_i
def update_alpha_i(alpha_i_old, yi, yj, alpha_j_old, alpha_j_new):
    return alpha_i_old + yi * yj * (alpha_j_old - alpha_j_new)



# 计算b1, b2
def cal_b1_b2(i, j, Ei, Ej, X_np, y_np, alphas, alpha_i_old, alpha_j_old, b):
    b1 = -Ei \
         - y_np[i, 0] * (X_np[i, :] @ X_np[i, :]) * (alphas[i, 0] - alpha_i_old) \
         - y_np[j, 0] * (X_np[j, :] @ X_np[i, :]) * (alphas[j, 0] - alpha_j_old) \
         + b

    b2 = -Ej \
         - y_np[i, 0] * (X_np[i, :] @ X_np[j, :]) * (alphas[i, 0] - alpha_i_old) \
         - y_np[j, 0] * (X_np[j, :] @ X_np[j, :]) * (alphas[j, 0] - alpha_j_old) \
         + b

    return b1, b2




# smo算法更新迭代更新alpha
def smo_simple(X, y, C, toler, max_iter):
    """
    smo算法更新迭代更新alpha
    :param X: 训练数据
    :param y: 样本类别标签
    :param C: 惩罚因子python float()多少位
    :param toler: 误差值停止值
    :param max_iter: 迭代次数上限
    :return:
    """
    # 转换格式
    X_np = np.array(X, dtype=np.float64)
    y_np = np.array(y, dtype=np.float64).reshape((-1, 1))
    # 初始化b为0
    b = 0
    # 获取数据维度
    m, n = X_np.shape
    # 初始化所有alpha 为0
    alphas = np.zeros((m, 1), dtype=np.float64)

    # 迭代求解
    epoch = 0
    while epoch < max_iter:
        # 定义一个保存更新次数的变量
        alpha_pairs_changed = 0
        # 将每一个alpha依次作为alpha_i
        for i in range(m):
            # 计算g(xi)
            g_xi = cal_gxi(X_np[i, :], alphas, y_np, X_np, b)
            # 计算Ei
            Ei = cal_Ei(g_xi, y_np[i, 0])
            # 判断迭代是否提前停止
            # 若alpha_i < C表明该样本应该落在最大间隔边界及外部，但如果实际落在最大间隔边界内部深度大于toler，就需要继续优化
            # 若alpha_i > 0表明该样本应该落在最大间隔边界及内部，但如果实际落在最大间隔边界外部深度大于toler，也需要继续优化
            if ((y_np[i, 0] * Ei < -toler) and (alphas[i, 0] < C)) or ((y_np[i, 0] * Ei > toler) and (alphas[i, 0] > 0)):
                # 随机选择一个alpha_j
                j = random_select_alpha_j(i, m)
                # 计算g(xj)
                g_xj = cal_gxi(X_np[j, :], alphas, y_np, X_np, b)
                # 计算Ej
                Ej = cal_Ei(g_xj, y_np[j, 0])
                # 保存alpha_i和alpha_j的old值
                alpha_i_old = alphas[i, 0]
                alpha_j_old = alphas[j, 0]
                # 计算约束上下界
                L, H = cal_L_H(i, j, alphas, y_np, C)
                # 如果上下界相等则跳过更新参数
                if L == H:
                    print("L == H")
                    continue
                # 计算eta
                eta = cal_eta(i, j, X_np)
                # 如果eta>=0，跳过更新参数
                if eta >= 0:
                    print("eta >= 0")
                    continue
                # 更新alpha_j
                alphas[j, 0] = update_alpha_j(alphas[j, 0], y_np[j, 0], Ei, Ej, eta)
                # 裁剪alpha_j
                alphas[j, 0] = clip_new_alpha(alphas[j, 0], H, L)
                # 检查alpha_j是否有轻微改变
                if abs(alphas[j, 0] - alpha_j_old) < 0.00001:
                    print("j not moving enough")
                    continue
                # 更新alpha_i
                alphas[i, 0] = update_alpha_i(alphas[i, 0], y_np[i, 0], y_np[j, 0], alpha_j_old, alphas[j, 0])
                # 计算b1, b2
                b1, b2 = cal_b1_b2(i, j, Ei, Ej, X_np, y_np, alphas, alpha_i_old, alpha_j_old, b)
                # 根绝情况求解b
                if 0 < alphas[i, 0] < C:
                    b = b1
                elif 0 < alphas[j, 0] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                alpha_pairs_changed += 1
                print("epoch: {}, i: {}, j: {}, pairs changed: {}".format(epoch, i, j, alpha_pairs_changed))
        # 如果将所有alpha作为alpha_i都没有更新参数，则累计没有变化的迭代次数+1
        if alpha_pairs_changed == 0:
            epoch += 1
        else:  # 如果将所有alpha作为alpha_i有更新参数，则迭代次数重置为0
            epoch = 0
        print("iteration number: {}".format(epoch))

    return alphas, b



# 计算权重w
def cal_w(alphas, X, y):
    # 转换格式
    X_np = np.array(X, dtype=np.float64)
    y_np = np.array(y, dtype=np.float64).reshape((-1, 1))

    # 计算权重w
    w = np.sum(alphas * y_np * X_np, axis=0)

    return w


# 绘制支持向量
def plot_support(X, y, w, b, alphas):
    # 转换格式
    X_np = np.array(X, dtype=np.float64)
    y_np = np.array(y, dtype=np.float64).reshape((-1, 1))

    # 定义画布
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 定义平面的横坐标取值
    x = np.arange(-2.0, 12.0, 0.1)
    # 计算平面的纵坐标取值
    pred_y = ((-b - w[0] * x) / w[1]).reshape((-1, ))
    # 画出分割平面
    ax.plot(x, pred_y)

    # 画出散点图
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    ax.scatter(X_np[:, 0], X_np[:, 1], c=y_np.squeeze(), cmap=cm_dark, s=30)

    # 找到支持向量，并在图中用圈标出
    alphas = alphas.reshape((-1, ))
    for i, alpha in enumerate(alphas):
        if alpha > 0.0:
            circle = Circle((X_np[i, 0], X_np[i, 1]), 0.2, facecolor='none', edgecolor=(1, 0.84, 0),
                            linewidth=3, alpha=0.5)
            ax.add_patch(circle)

    # 限定X、Y轴的区间范围
    ax.axis([-2, 12, -8, 6])
    plt.show()






def main():
    # 加载数据
    X, y = load_dataset(r"./data/svm1.txt")
    # # 显示散点图
    # display_scatter_diagram(X, y)

    # 使用smo算法训练svm的参数
    alphas, b = smo_simple(X, y, 0.6, 0.001, 40)

    w = cal_w(alphas, X, y)

    plot_support(X, y, w, b, alphas)









if __name__ == '__main__':
    main()


































