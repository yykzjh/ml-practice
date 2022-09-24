# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/22 21:41
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier



def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y




def main():
    # 读取鸢尾花数据集
    X, y = load_data()
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

    # # 定义Bagging分类器
    # bag_clf = BaggingClassifier(
    #     SVC(),  # 这里采用的是SVC作为基分类器
    #     n_estimators=500,  # 基分类器的个数
    #     bootstrap=True,  # True是自助法采样(bagging)，False为不放回采样(pasting)
    #     max_samples=0.8,  # 设置为整数表示的是采样的样本数，浮点数表示的是max_samples*X.shape[0]
    #     oob_score=True  # 用out-of-bag测试获得正确率分数
    # )

    # 定义Bagging分类器
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(),  # 这里采用的是决策树作为基分类器
        n_estimators=500,  # 基分类器的个数
        bootstrap=True,  # True是自助法采样(bagging)，False为不放回采样(pasting)
        max_samples=0.8,  # 设置为整数表示的是采样的样本数，浮点数表示的是max_samples*X.shape[0]
        oob_score=True  # 用out-of-bag测试获得正确率分数
    )

    # 训练Bagging分类器
    bag_clf.fit(X_train, y_train)
    # 显示out-of-bag测试获得的正确率分数
    print("out-of-bag score=", bag_clf.oob_score_)

    # 对测试集进行预测
    y_hat = bag_clf.predict(X_test)
    # 输出正确率
    print(bag_clf.__class__.__name__, "=", accuracy_score(y_test, y_hat))





def random_forest_main():
    # 读取鸢尾花数据集
    X, y = load_data()
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

    # 定义随机森林
    random_forest_clf = RandomForestClassifier(n_estimators=500, bootstrap=True, oob_score=True, max_samples=0.8)

    # 训练
    random_forest_clf.fit(X_train, y_train)
    # 显示out-of-bag测试获得的正确率分数
    print("out-of-bag score=", random_forest_clf.oob_score_)

    # 对测试集进行预测
    y_hat = random_forest_clf.predict(X_test)
    # 输出正确率
    print(random_forest_clf.__class__.__name__, "=", accuracy_score(y_test, y_hat))



if __name__ == '__main__':
    # main()

    random_forest_main()


