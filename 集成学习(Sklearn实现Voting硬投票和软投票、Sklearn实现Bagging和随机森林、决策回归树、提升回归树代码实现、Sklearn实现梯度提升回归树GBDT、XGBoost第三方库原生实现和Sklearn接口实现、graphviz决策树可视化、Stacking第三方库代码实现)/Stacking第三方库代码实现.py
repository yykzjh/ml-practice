# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/25 1:05
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from mlxtend.classifier import StackingClassifier





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
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

    # 定义三个组件分类器: KNN, 随机森林, 朴素贝叶斯
    clf1 = KNeighborsClassifier(n_neighbors=5)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()

    # 定义最后用的次级学习器(元学习器)：逻辑回归分类器
    lr = LogisticRegression(max_iter=5000)

    # 定义堆叠
    sclf = StackingClassifier(
        classifiers=[clf1, clf2, clf3],  # 传入组件分类器
        meta_classifier=lr,  # 传入元分类器
        use_probas=True  # 使用概率值
    )

    # 分别对组件分类器和Stacking分类器的性能进行评价
    for model in [clf1, clf2, clf3, lr, sclf]:
        # 训练
        model.fit(X_train, y_train)
        # 在测试集上预测
        y_test_hat = model.predict(X_test)
        # 显示测试正确率
        print(model.__class__.__name__, ", test accuracy:", accuracy_score(y_test, y_test_hat))






if __name__ == '__main__':
    main()




















