import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

from sklearn import linear_model
from sklearn.datasets import load_boston
from sklearn .preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def load_data():
    # 加载波士顿房价数据
    boston = load_boston()
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    y = pd.DataFrame({"MEDV": boston.target})

    # 查看数据字段
    print(X.columns, "\n")
    # 查看数据类型
    print(X.dtypes, "\n")
    # 查看数据集综合信息
    print(X.info(), "\n")
    # 查看数据集大小
    print(X.shape, "\n")
    # 查看数据集缺失值情况
    print(X.isnull().sum(), "\n")
    # 查看数据统计信息，中值、方差、最小值、四分之一分位数等
    print(X.describe(), "\n")

    # 相关性检验
    df = X.copy()
    df["MEDV"] = y
    corr = df.corr()
    # 绘制相关系数的热力分布图
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
    )
    plt.show()
    print(corr["MEDV"].abs().sort_values(), "\n")

    # 显示不同自变量之间、自变量和因变量之间的关系
    sns.pairplot(df[["LSTAT", "RM", "PTRATIO", "MEDV"]])  # 相关性绝对值靠前3的特征
    plt.show()

    return X, y


def split_data(X, y):
    # 实例化
    ss = StandardScaler()
    # 特征数据
    X = ss.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=114514)

    return X_train, X_test, y_train, y_test



def search_params(X_train, y_train, k=5):
    # 初始化模型对象
    ridge_model = linear_model.Ridge()
    # 定义需要搜索的参数区间
    params_dict = {
        "alpha": list(np.linspace(1, 10, 201))
    }
    # 定义交叉验证器
    gscv = GridSearchCV(estimator=ridge_model, param_grid=params_dict, cv=k, scoring="neg_mean_squared_error")
    # 做交叉验证训练
    gscv.fit(X_train, y_train)
    # 打印做好的参数和评价指标
    print(gscv.best_params_)
    print(gscv.best_score_)

    return gscv.best_estimator_, gscv.best_params_



def train(params, X_train, y_train):
    # 定义岭回归线性模型
    model = linear_model.Ridge(**params)
    # 训练
    model.fit(X_train, y_train)

    return model



def visualize(model, X_test, y_test):
    pass



def save_model(model, path=r"./models/ridge.m"):
    joblib.dump(model, path)




def main():
    # 加载数据
    X, y = load_data()

    # 划分数据集
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 使用交叉验证选择模型参数
    best_model, best_params = search_params(X_train, y_train)
    # print(best_model.coef_)
    # print(best_model.intercept_)

    # # 训练模型
    # final_model = train(best_params, X_train, y_train)
    # print(final_model.coef_)
    # print(final_model.intercept_)
    # print(np.allclose(best_model.coef_, final_model.coef_))

    # 在测试集上评价模型
    y_train_hat = best_model.predict(X_train)
    y_test_hat = best_model.predict(X_test)
    print("train_MSE=", mean_squared_error(y_train, y_train_hat))
    print("test_MSE=", mean_squared_error(y_test, y_test_hat))
    print("test_score=", best_model.score(X_test, y_test))

    # 保存模型
    save_model(best_model)




def inference():
    # 加载数据
    X, y = load_data()
    # 划分数据集
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 读取模型
    loaded_model = joblib.load(r"./models/ridge.m")
    # 在测试集上预测
    y_train_hat = loaded_model.predict(X_train)
    y_test_hat = loaded_model.predict(X_test)
    print("train_MSE=", mean_squared_error(y_train, y_train_hat))
    print("test_MSE=", mean_squared_error(y_test, y_test_hat))
    print("test_score=", loaded_model.score(X_test, y_test))

    # 保存结果
    res_df = pd.DataFrame({
        "test": y_test["MEDV"].tolist(),
        "predict": y_test_hat[:, 0]
    })
    print(res_df, "\n")

    # 显示结果与真实值的对比图
    res_df.plot(figsize=(18, 10))
    plt.show()
    print("真实值大于预测值的比例=", len(res_df.query("test > predict")) / len(res_df), "\n")

    # 按真实值和预测值画图
    plt.scatter(y_test, y_test_hat, label="test")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--',
             lw=3,
             label="predict")
    plt.show()






if __name__ == '__main__':
    # main()

    inference()






























