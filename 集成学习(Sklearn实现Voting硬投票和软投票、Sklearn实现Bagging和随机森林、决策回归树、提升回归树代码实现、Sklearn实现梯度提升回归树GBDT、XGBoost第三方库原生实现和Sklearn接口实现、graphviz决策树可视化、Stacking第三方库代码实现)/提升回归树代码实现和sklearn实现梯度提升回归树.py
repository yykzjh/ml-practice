# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/23 0:17
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor




def load_data():
    data = np.loadtxt(r"./data/data.txt", delimiter=",")
    X = data[:, [0]]
    y = data[:, -1]
    return X, y




def main():
    # 加载数据
    X, y = load_data()

    # 定义第一颗决策回归树
    tree_reg1 = DecisionTreeRegressor(max_depth=5)
    # 训练第一颗决策回归树
    tree_reg1.fit(X, y)
    # 计算第一颗决策回归树的残差
    y2 = y - tree_reg1.predict(X)

    # 定义第二颗决策回归树
    tree_reg2 = DecisionTreeRegressor(max_depth=5)
    # 训练第二颗决策回归树
    tree_reg2.fit(X, y2)
    # 计算第二颗决策回归树的残差
    y3 = y2 - tree_reg2.predict(X)

    # 定义第三颗决策回归树
    tree_reg3 = DecisionTreeRegressor(max_depth=5)
    # 训练第三颗决策回归树
    tree_reg3.fit(X, y3)

    # 结合测试
    y_hat = tree_reg1.predict(X) + tree_reg2.predict(X) + tree_reg3.predict(X)

    # 输出前5个预测结果
    print(y_hat[:5])

    # 按真实值和预测值画图
    plt.scatter(y, y_hat, label="test")
    plt.plot([y.min(), y.max()],
             [y.min(), y.max()],
             'r--',
             lw=3,
             label="predict")
    plt.show()





def sklearn_boosting_tree():
    # 加载数据
    X, y = load_data()

    # 定义梯度提升回归树
    gbrt = GradientBoostingRegressor(max_depth=5, n_estimators=3, learning_rate=1.0)
    # 训练梯度提升回归树
    gbrt.fit(X, y)

    # 结合测试
    y_hat = gbrt.predict(X)

    # 输出前5个预测结果
    print(y_hat[:5])

    # 按真实值和预测值画图
    plt.scatter(y, y_hat, label="test")
    plt.plot([y.min(), y.max()],
             [y.min(), y.max()],
             'r--',
             lw=3,
             label="predict")
    plt.show()





if __name__ == '__main__':
    main()

    sklearn_boosting_tree()



















