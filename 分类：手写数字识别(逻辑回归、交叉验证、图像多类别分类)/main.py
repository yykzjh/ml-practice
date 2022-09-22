# -*- encoding: utf-8 -*-
'''
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/11 21:25
@Version  :   1.0
@License  :   (C)Copyright 2022-
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from load_mnist import load_mnist




def show_images(x_train, y_train):
    x_train = x_train.reshape(x_train.shape[0], 28, 28)
    for i in range(5):
        print(y_train[i])
        plt.imshow(x_train[i], cmap="gray")
        plt.show()



def load_data():
    x_train, y_train, x_test, y_test = load_mnist(
        r"./data/train-images-idx3-ubyte.gz",
        r"./data/train-labels-idx1-ubyte.gz",
        r"./data/t10k-images-idx3-ubyte.gz",
        r"./data/t10k-labels-idx1-ubyte.gz",
        normalize=True,
        one_hot=False
    )
    # show_images(x_train, y_train)

    return x_train, y_train, x_test, y_test






def main():
    x_train, y_train, x_test, y_test = load_data()

    # 交叉验证训练
    # classifier = linear_model.LogisticRegression()
    # params_dict = {
    #     "C": list(np.linspace(1.0, 10000, 100)),
    #     "max_iter": [100, 1000, 5000, 10000]
    # }
    # gscv = GridSearchCV(estimator=classifier, param_grid=params_dict, cv=10, scoring="accuracy")
    # gscv.fit(x_train, y_train)
    # print(gscv.best_params_)
    # print(gscv.best_score_)

    # 普通训练
    classifier = linear_model.LogisticRegression(C=100, max_iter=1000)  # C越小，正则化越强，越不容易过拟合
    classifier.fit(x_train, y_train)

    # 训练集精度
    y_train_hat = classifier.predict(x_train)
    print("训练集精度=", accuracy_score(y_train, y_train_hat))
    # 测试集精度
    y_test_hat = classifier.predict(x_test)
    print("测试集精度=", accuracy_score(y_test, y_test_hat))







if __name__ == '__main__':
    main()







