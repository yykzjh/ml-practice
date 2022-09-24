# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/24 15:21
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import matplotlib.pyplot as plt

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

import xgboost as xgb




def load_data():
    """
    读取数据(把libsvm格式读取成以前我们常用的二维数组形式)
    """
    X_train, y_train = load_svmlight_file(r"./data/agaricus.txt.train")
    X_test, y_test = load_svmlight_file(r"./data/agaricus.txt.test")

    # 将样本数据转换成二维数组格式
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    return X_train, y_train, X_test, y_test



# 5折交叉验证搜索
def grid_search(X_train, y_train):
    # 定义模型
    model = xgb.XGBClassifier(learning_rate=0.1, objective="binary:logistic")
    # 定义需要搜索的参数
    param_dict = {
        "n_estimators": range(1, 51, 1),
        "max_depth": range(1, 10, 1)
    }
    # 定义5折交叉验证搜索器
    gscv = GridSearchCV(estimator=model, param_grid=param_dict, scoring="accuracy", cv=5)
    # 训练
    gscv.fit(X_train, y_train)
    # 输出最好的参数
    print(gscv.best_params_)
    print(gscv.best_score_)

    return gscv.best_estimator_




# 提前停止迭代
def early_stop(X_train, y_train):
    # 划分验证集
    X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

    # 定义最大迭代计算次数
    num_round = 100
    # 定义XGBoost分类器
    xgb_clf = xgb.XGBClassifier(max_depth=2, learning_rate=0.1, n_estimators=num_round,
                                early_stopping_rounds=10, eval_metric=["error", "logloss"],
                                objective="binary:logistic")
    # 定义验证数据集集
    val_dataset = [(X_train_part, y_train_part), (X_val, y_val)]
    # 提前停止训练, 连续10次迭代没有降低错误率，则停止;verbose输出训练的迭代信息
    xgb_clf.fit(X_train_part, y_train_part, eval_set=val_dataset, verbose=False)

    return xgb_clf



# 画出迭代训练过程中的学习曲线
def display_val_error(xgb_clf):
    # 获得验证集错误率
    results = xgb_clf.evals_result()

    # 获得总的迭代次数
    epochs = len(results["validation_0"]["error"])
    # 获得x轴的标识
    x_axis = range(0, epochs)


    # 指定当前子图
    plt.subplot(121)
    # 画log loss图
    plt.plot(x_axis, results["validation_0"]["logloss"], label="Train")
    plt.plot(x_axis, results["validation_1"]["logloss"], label="Val")
    # 显示图例
    plt.legend()
    # y轴标注
    plt.ylabel("Log Loss")
    # x轴标注
    plt.xlabel("Iteration")
    # 标题
    plt.title("XGBoost Early Stop Log Loss")

    # 指定当前子图
    plt.subplot(122)
    # 画error图
    plt.plot(x_axis, results["validation_0"]["error"], label="Train")
    plt.plot(x_axis, results["validation_1"]["error"], label="Val")
    # 显示图例
    plt.legend()
    # y轴标注
    plt.ylabel("Error")
    # x轴标注
    plt.xlabel("Iteration")
    # 标题
    plt.title("XGBoost Early Stop Error")
    plt.show()






def main():
    # 读取数据
    X_train, y_train, X_test, y_test = load_data()


    # 定义XGBoost分类器模型并设置参数
    """
    silent=0                       设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
    nthread=4                      cpu 线程数 默认最大
    learning_rate= 0.3             如同学习率
    min_child_weight=1             这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
                                   假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
                                   这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    max_depth=6                    构建树的深度，越大越容易过拟合
    gamma=0                        树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
    subsample=1                    随机采样训练样本 训练实例的子采样比
    max_delta_step=0               最大增量步长，我们允许每个树的权重估计。
    colsample_bytree=1             生成树时进行的列采样 
    reg_lambda=1                   控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    reg_alpha=0                    L1 正则项参数
    scale_pos_weight=1             如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
    objective= 'multi:softmax'     多分类的问题 指定学习任务和相应的学习目标
        - reg:linear：线性回归
        - reg:logistic：逻辑回归
        - binary:logistic 二分类的逻辑回归，返回预测的概率
        - binary:logitraw：二分类逻辑回归，输出是逻辑为0/1的前一步的分数
        - multi:softmax：用于Xgboost 做多分类问题，需要设置num_class（分类的个数）
        - multi:softprob：和softmax一样，但是返回的是每个数据属于各个类别的概率。
        - rank:pairwise：让Xgboost 做排名任务，通过最小化(Learn to rank的一种方法)
    num_class=10                   类别数，多分类与 multisoftmax 并用
    n_estimators=100               树的个数
    seed=1000                      随机种子
    eval_metric= 'auc'
    """
    # xgb_clf = xgb.XGBClassifier(max_depth=2, learning_rate=1, n_estimators=6,
    #                            objective="binary:logistic")
    # # 训练
    # xgb_clf.fit(X_train, y_train)


    # # 交叉验证选择最优的模型
    # xgb_clf = grid_search(X_train, y_train)


    # 根据验证集提前停止迭代，迭代就是训练一个个决策树
    xgb_clf = early_stop(X_train, y_train)
    # 输出验证集在每次迭代时的结果
    # print(xgb_clf.evals_result())
    # 画出迭代训练过程中的学习曲线
    display_val_error(xgb_clf)


    # 对训练集进行预测
    train_prob = xgb_clf.predict(X_train)
    # 将训练集上的预测概率转化为类别标签
    train_pred = [round(prob) for prob in train_prob]
    # 计算训练集上准确率
    print("Train Accuracy=", accuracy_score(y_train, train_pred))

    # 对测试集进行预测
    test_prob = xgb_clf.predict(X_test)
    # 将测试集上的预测概率转化为类别标签
    test_pred = [round(prob) for prob in test_prob]
    # 计算测试集上准确率
    print("Test Accuracy=", accuracy_score(y_test, test_pred))










if __name__ == '__main__':
    main()






