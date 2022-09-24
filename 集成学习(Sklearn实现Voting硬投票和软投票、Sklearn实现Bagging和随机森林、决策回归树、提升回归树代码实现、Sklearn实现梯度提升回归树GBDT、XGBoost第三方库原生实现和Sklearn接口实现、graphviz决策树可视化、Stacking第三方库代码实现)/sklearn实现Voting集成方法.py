# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/22 12:00
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier






def main():
    # 生成数据集
    X, y = make_moons(n_samples=7000, noise=0.1, random_state=123)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

    # 定义三个个体学习器
    lr = LogisticRegression()  # 逻辑回归
    dt = DecisionTreeClassifier()  # 决策树
    svm = SVC(probability=True)  # SVM，软投票分类器要加probability=True

    # # 定义投票分类器，硬投票分类器
    # voting = VotingClassifier(
    #     estimators=[("lr", lr), ("dt", dt), ("svm", svm)],
    #     voting="hard"
    # )

    # 定义投票分类器，软投票分类器
    voting = VotingClassifier(
        estimators=[("lr", lr), ("dt", dt), ("svm", svm)],
        voting="soft"
    )

    # 分别训练三个个体学习器和投票分类器
    for clf in (lr, dt, svm, voting):
        # 训练
        clf.fit(X_train, y_train)
        # 在测试集上预测
        y_hat = clf.predict(X_test)
        # 显示正确率
        print(clf.__class__.__name__, "=", accuracy_score(y_test, y_hat))





if __name__ == '__main__':
    main()




























