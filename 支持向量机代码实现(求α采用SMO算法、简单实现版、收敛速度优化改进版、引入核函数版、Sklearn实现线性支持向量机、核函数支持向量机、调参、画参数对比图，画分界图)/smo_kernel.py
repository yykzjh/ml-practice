# -*- encoding: utf-8 -*-
'''
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/14 11:06
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
def random_select_alpha_j(i, m):
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



# 核函数
def kernel_trans(X, A, kTup):
    # 获取维度
    m, n = X.shape
    # 定义核矩阵
    K = np.zeros((m, m))
    if kTup[0] == "linear":  # 线性核
        K = X @ A.T
    elif kTup[0] == "rbf":  # 高斯核
        print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
        for j in range(m):
            deltaRow = X[j, :].reshape((1, -1)) - A
            K[j] = np.sum(np.power(deltaRow, 2), axis=1).reshape((m, ))
        K = np.exp(K / (-2 * kTup[1] ** 2))
    else:
        raise NameError("Houston We Have a Problem —— That Kernel is not recognized")
    print(K)
    return K




# 封装用到的参数
class OptimStruct:
    def __init__(self, X_np, y_np, C, toler, kTup):
        """
        封装用到的参数
        :param X_np: 训练样本数据
        :param y_np: 样本标签
        :param C: 惩罚因子
        :param toler: 误差值停止值
        """
        self.X = X_np
        self.y = y_np
        self.C = C
        self.tol = toler
        self.m = X_np.shape[0]
        self.alphas = np.zeros((self.m, 1))
        self.b = 0
        self.eCache = np.zeros((self.m, 2))  # 第一列是标志位，0无效，1有效；第二列是Ei
        self.K = kernel_trans(self.X, self.X, kTup)




# 计算Ek
def cal_Ek(oS, k):
    gxk = (oS.alphas * oS.y).T @ oS.K[:, k] + oS.b
    Ek = gxk - oS.y[k, 0]
    return Ek




# 选择第二个待优化的alpha_j，优化为选择一个误差Ej与误差Ei差距最大的j
def select_alpha_j(i, oS, Ei):
    # 初始化参数
    max_k = -1
    max_delta_E = 0
    Ej = 0
    # 将Ei设为有效
    oS.eCache[i][0] = 1
    oS.eCache[i][1] = Ei
    # 获得误差值Ek的有效位置有哪些
    valid_ecache_list = np.nonzero(oS.eCache[:, 0])[0]
    # 如果有除了i之外的有效位
    if len(valid_ecache_list) > 1:
        # 循环所有的有效缓存
        for k in valid_ecache_list:
            if k == i:  # 不能与i相等
                continue
            # 计算Ek
            Ek = cal_Ek(oS, k)
            # 计算Ek与Ei的差值的绝对值
            deltaE = abs(Ei - Ek)
            # 判断是否更新
            if deltaE > max_delta_E:
                max_k = k
                max_delta_E = deltaE
                Ej = Ek
        return max_k, Ej
    else:  # 第一次循环时是没有有效的缓存值的，所以随机选一个(仅会执行依次)
        j = random_select_alpha_j(i, oS.m)
        Ej = cal_Ek(oS, j)
        return j, Ej



# 更新Ek的缓存
def update_Ek(oS, k):
    Ek = cal_Ek(oS, k)
    oS.eCache[k][0] = 1
    oS.eCache[k][1] = Ek



# 内循环
def innerL(i, oS):
    # 计算Ei
    Ei = cal_Ek(oS, i)
    # 判断i是否满足违背KKT条件
    if ((oS.y[i, 0] * Ei < -oS.tol) and (oS.alphas[i, 0] < oS.C)) or ((oS.y[i, 0] * Ei > oS.tol) and (oS.alphas[i, 0] > 0))\
            or ((-oS.tol <= oS.y[i, 0] * Ei <= oS.tol) and (abs(oS.alphas[i, 0]) < oS.tol or abs(oS.alphas[i, 0]-oS.C) < oS.tol)):
        # 选择一个步长|Ei-Ej|最长的j
        j, Ej = select_alpha_j(i, oS, Ei)
        # 存储alpha_i和alpha_j的原始值
        alpha_i_old = oS.alphas[i, 0]
        alpha_j_old = oS.alphas[j, 0]
        # 计算上下边界
        L, H = cal_L_H(i, j, oS)
        # 如果上下界相等则跳过更新参数
        if L == H:
            print("L == H")
            return 0
        # 计算eta
        eta = cal_eta(i, j, oS)
        # 如果eta>=0，跳过更新参数
        if eta >= 0:
            print("eta >= 0")
            return 0
        # 更新alpha_j
        oS.alphas[j, 0] = update_alpha_j(oS.alphas[j, 0], oS.y[j, 0], Ei, Ej, eta)
        # 裁剪alpha_j
        oS.alphas[j, 0] = clip_new_alpha(oS.alphas[j, 0], H, L)
        # 更新Ej的缓存
        update_Ek(oS, j)
        # oS.alphas[j, 0] = alpha_j_old
        # 检查alpha_j是否有轻微改变
        if abs(oS.alphas[j, 0] - alpha_j_old) < 0.00001:
            print("j not moving enough")
            return 0
        # 更新alpha_i
        oS.alphas[i, 0] = update_alpha_i(oS.alphas[i, 0], oS.y[i, 0], oS.y[j, 0], alpha_j_old, oS.alphas[j, 0])
        # 更新Ei的缓存
        update_Ek(oS, i)
        # 计算b1, b2
        b1, b2 = cal_b1_b2(i, j, Ei, Ej, alpha_i_old, alpha_j_old, oS)
        # 根绝情况求解b
        if 0 < oS.alphas[i, 0] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[j, 0] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2
        return 1
    else:
        return 0




# 外循环
def smo(X, y, C, toler, max_iter, kTup):
    # 转换格式
    X_np = np.array(X, dtype=np.float64)
    y_np = np.array(y, dtype=np.float64).reshape((-1, 1))
    # 初始化内参缓存
    oS = OptimStruct(X_np, y_np, C, toler, kTup)
    # 定义迭代变量
    epoch = 0
    # 定义遍历模式标识符
    entire_set = True
    # 定义每次遍历修改一对alpha的累计计数
    alpha_pairs_changed = 0
    while (epoch < max_iter) and (alpha_pairs_changed > 0 or entire_set):
        # 计数归0
        alpha_pairs_changed = 0
        # 判断遍历的模式
        if entire_set:
            for i in range(oS.m):
                alpha_pairs_changed += innerL(i, oS)
                print("Full set, epoch: {}, i: {}, pairs changed: {}".format(epoch, i, alpha_pairs_changed))
        else:  # 遍历非边界数据集(0<alpha<C)，即在间隔边界上的数据
            # 获取所有非边界数据的索引
            non_bound_list = np.nonzero((0 < oS.alphas) * (oS.alphas < oS.C))[0]
            for i in non_bound_list:
                alpha_pairs_changed += innerL(i, oS)
                print("Non-bound set, epoch: {}, i: {}, pairs changed: {}".format(epoch, i, alpha_pairs_changed))
        # 迭代计数+1
        epoch += 1
        # 根据当前遍历的结果决定接下来的遍历方式
        if entire_set:  # 遍历完一次全集下一次就切换到非边界数据集
            entire_set = False
        elif alpha_pairs_changed == 0:  # 非边界数据集没有找到更新的alpha对，但全集中的alpha可能有变
            entire_set = True
        print("iteration number: {}".format(epoch))

    return oS.alphas, oS.b





# 计算alpha_j更新的约束上下界
def cal_L_H(i, j, oS):
    if oS.y[i, 0] != oS.y[j, 0]:
        L = max(0, oS.alphas[j] - oS.alphas[i])
        H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
    else:
        L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
        H = min(oS.C, oS.alphas[j] + oS.alphas[i])

    return L, H



# 计算eta
def cal_eta(i, j, oS):
    # 计算
    return 2 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]



# 更新alpha_j
def update_alpha_j(alpha_j_old, yj, Ei, Ej, eta):
    return alpha_j_old - (yj * (Ei - Ej)) / eta



# 更新alpha_i
def update_alpha_i(alpha_i_old, yi, yj, alpha_j_old, alpha_j_new):
    return alpha_i_old + yi * yj * (alpha_j_old - alpha_j_new)



# 计算b1, b2
def cal_b1_b2(i, j, Ei, Ej, alpha_i_old, alpha_j_old, oS):
    b1 = -Ei \
         - oS.y[i, 0] * oS.K[i, i] * (oS.alphas[i, 0] - alpha_i_old) \
         - oS.y[j, 0] * oS.K[j, i] * (oS.alphas[j, 0] - alpha_j_old) \
         + oS.b

    b2 = -Ej \
         - oS.y[i, 0] * oS.K[i, j] * (oS.alphas[i, 0] - alpha_i_old) \
         - oS.y[j, 0] * oS.K[j, j] * (oS.alphas[j, 0] - alpha_j_old) \
         + oS.b

    return b1, b2





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

    # # 定义平面的横坐标取值
    # x = np.arange(-2.0, 12.0, 0.1)
    # # 计算平面的纵坐标取值
    # pred_y = ((-b - w[0] * x) / w[1]).reshape((-1, ))
    # # 画出分割平面
    # ax.plot(x, pred_y)

    # 画出散点图
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    ax.scatter(X_np[:, 0], X_np[:, 1], c=y_np.squeeze(), cmap=cm_dark, s=30)

    # 找到支持向量，并在图中用圈标出
    alphas = alphas.reshape((-1, ))
    for i, alpha in enumerate(alphas):
        if 0.0 < alpha:
            circle = Circle((X_np[i, 0], X_np[i, 1]), 0.03, facecolor='none', edgecolor=(1, 0.84, 0),
                            linewidth=3, alpha=0.5)
            ax.add_patch(circle)
            # print(y_np[i] * (w@X_np[i] + b), y_np[i])
        # elif alpha == 0.0:
        #     circle = Circle((X_np[i, 0], X_np[i, 1]), 0.2, facecolor='none', edgecolor=(0, 0, 0.0),
        #                     linewidth=3, alpha=0.5)
        #     ax.add_patch(circle)
        # else:
        #     circle = Circle((X_np[i, 0], X_np[i, 1]), 0.2, facecolor='none', edgecolor=(0, 0.8, 0.8),
        #                     linewidth=3, alpha=0.5)
        #     ax.add_patch(circle)

    # 限定X、Y轴的区间范围
    # ax.axis([-2, 12, -8, 6])
    plt.show()



def display(svm):
    # 定义画布
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 画决策边界
    h = .01
    x1_min, x1_max = svm.X_train[:, 0].min() - .5, svm.X_train[:, 0].max() + .5
    x2_min, x2_max = svm.X_train[:, 1].min() - .5, svm.X_train[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))

    # 对所有网格点进行预测
    z = np.where(svm.predict(np.c_[xx.ravel(), yy.ravel()]) > 0, 1, -1)
    z = z.reshape(xx.shape)

    # 画分界图
    plt.pcolormesh(xx, yy, z, shading="auto", cmap=plt.cm.Paired)

    # 画数据点
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    plt.scatter(svm.X_train[:, 0], svm.X_train[:, 1], c=svm.y_train.squeeze(), cmap=cm_dark, s=30)

    # 画支持向量
    alphas_non_zeros_index = np.nonzero(svm.alphas > 0)[0]
    for i in alphas_non_zeros_index:
        circle = Circle((svm.X_train[i][0], svm.X_train[i][1]), 0.05, facecolor='none', edgecolor=(0.7, 0.7, 0.7), linewidth=3, alpha=0.5)
        ax.add_patch(circle)
    plt.show()





class SVM:
    def __init__(self, X, y, alphas, b, kTup=("rbf", 1.3)):
        super(SVM, self).__init__()
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y, dtype=np.float64).reshape((-1, 1))
        self.alphas = alphas
        self.b = b
        self.kTup = kTup


    def predict(self, X_test):
        m, n = self.X_train.shape
        k, _ = X_test.shape

        K = np.zeros((m, k))
        for i in range(m):
            for j in range(k):
                K[i][j] = np.sum(np.power(self.X_train[i, :] - X_test[j, :], 2))
        K = np.exp(K / (-2 * self.kTup[1] ** 2))

        return ((self.alphas * self.y_train).T @ K).squeeze() + self.b






def main():
    # 加载数据
    X, y = load_dataset(r"./data/svm2.txt")
    # # 显示散点图
    # display_scatter_diagram(X, y)

    # 使用smo算法训练svm的参数
    alphas, b = smo(X, y, 200, 0.0001, 10000, kTup=("rbf", 1.3))
    # print(alphas)

    svm = SVM(X, y, alphas, b, kTup=("rbf", 1.3))

    # plot_support(X, y, w, b, alphas)

    display(svm)






if __name__ == '__main__':
    main()







































