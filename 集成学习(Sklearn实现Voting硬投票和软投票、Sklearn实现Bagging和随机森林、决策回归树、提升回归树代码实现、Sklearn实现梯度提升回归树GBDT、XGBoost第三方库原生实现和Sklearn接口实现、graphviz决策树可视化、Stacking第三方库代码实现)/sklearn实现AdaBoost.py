# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/22 22:45
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier




def load_data():
    # 第一列是类别，其他属性值的含义描述在./data/wine_names中
    data = np.loadtxt(r"./data/wine.data", delimiter=',')
    X = data[:, 1:]
    y = data[:, 0]
    return X, y






def main():
    # 加载红酒数据
    X, y = load_data()
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # 定义弱分类器,数据做不做归一化对决策树影响不大
    dt = DecisionTreeClassifier()

    # 定义AdaBoost分类器
    adaboost = AdaBoostClassifier(
        base_estimator=dt,  # 指定弱分类器
        n_estimators=50,  # 指定弱分类器有多少个
        learning_rate=0.5,  # 减少每个弱分类器的贡献
    )

    # 训练
    adaboost.fit(X_train, y_train)

    # 测试在训练集上的正确率
    y_train_hat = adaboost.predict(X_train)
    print("train accuracy:", accuracy_score(y_train, y_train_hat))

    # 测试在测试集上的正确率
    y_test_hat = adaboost.predict(X_test)
    print("test accuracy:", accuracy_score(y_test, y_test_hat))





if __name__ == '__main__':
    main()






















