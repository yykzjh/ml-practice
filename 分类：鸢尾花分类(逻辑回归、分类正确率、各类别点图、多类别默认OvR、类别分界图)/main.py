import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris



def load_data():
    # 加载鸢尾花数据集
    iris = load_iris()

    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")

    return X, y




def display_scatter(X, f1=2, f2=3):
    plt.scatter(X.iloc[0:50, f1], X.iloc[0:50, f2], color="red", marker='o', label="setosa")
    plt.scatter(X.iloc[50:100, f1], X.iloc[50:100, f2], color="blue", marker='x', label="versicolor")
    plt.scatter(X.iloc[100:, f1], X.iloc[100:, f2], color="green", marker='+', label="Virginica")
    plt.show()



def visualize(X, y, f1=2, f2=3):
    # 训练两个维度的模型
    model = linear_model.LogisticRegression(C=1000000.0)
    model.fit(X.iloc[:, [f1, f2]].values, y)


    # 设置网格步长
    stride = .02
    # 分别获取两个维度上的最小值和最大值
    x1_min, x1_max = X.iloc[:, f1].min() - .5, X.iloc[:, f1].max() + .5
    x2_min, x2_max = X.iloc[:, f2].min() - .5, X.iloc[:, f2].max() + .5
    # meshgrid函数生成两个网格矩阵
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, stride), np.arange(x2_min, x2_max, stride))
    # 两个维度展开后结合为网格点
    points = np.c_[xx.ravel(), yy.ravel()]

    # 网格点整体输入到网络中
    z = model.predict(points)
    # 还原维度
    z = z.reshape(xx.shape)

    # 输出两个维度的正确率
    y_hat = model.predict(X.iloc[:, [f1, f2]].values)
    print("两个维度的正确率为=", accuracy_score(y, y_hat))

    # 画图
    plt.pcolormesh(xx, yy, z, cmap=plt.cm.Paired)
    plt.scatter(X.iloc[0:50, f1], X.iloc[0:50, f2], color="red", marker='o', label="setosa")
    plt.scatter(X.iloc[50:100, f1], X.iloc[50:100, f2], color="blue", marker='x', label="versicolor")
    plt.scatter(X.iloc[100:, f1], X.iloc[100:, f2], color="green", marker='+', label="Virginica")
    plt.show()





def main():
    X, y = load_data()

    # display_scatter(X)

    model = linear_model.LogisticRegression(C=1000000.0)
    model.fit(X, y)
    print(model.coef_)
    print(model.intercept_)

    y_hat = model.predict(X)
    print("准确度=", accuracy_score(y, y_hat))

    visualize(X, y)









if __name__ == '__main__':
    main()


















