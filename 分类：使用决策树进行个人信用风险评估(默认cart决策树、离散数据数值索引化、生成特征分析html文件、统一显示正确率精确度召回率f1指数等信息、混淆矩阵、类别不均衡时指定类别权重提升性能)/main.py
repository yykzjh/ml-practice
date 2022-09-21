# -*- encoding: utf-8 -*-
'''
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/12 18:22
@Version  :   1.0
@License  :   (C)Copyright 2022-
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandas_profiling as pp

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    classification_report, confusion_matrix




def numericalize(df):
    """
    将数据集中字符串类型的数据数值化
    :param df: 数据集
    :return:
    """
    String2Index = {}
    Index2String = {}
    for col in df.columns:
        if df[col].dtype in ['float64', 'int', 'int64']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype("int64")
        else:
            # 全部字符串化
            df[col] = df[col].apply(str)

            # 编码化
            LbE = LabelEncoder()
            LbE.fit(df[col])
            df[col] = LbE.transform(df[col].map(lambda x: x.strip())).astype("int64")

            # 导出映射表：开发逻辑中需要
            string_to_index = {encode: index for index, encode in enumerate(LbE.classes_)}
            index_to_string = {index: encode for index, encode in enumerate(LbE.classes_)}
            String2Index[col] = string_to_index
            Index2String[col] = index_to_string

    return String2Index, Index2String




def load_data():
    # 加载数据
    credit = pd.read_excel(r"./data/german_credit.xlsx")
    # print(credit.shape)
    # print(credit.columns)
    # print(credit.head(3))
    # print(credit["Status of existing checking account"].value_counts())
    # print(credit["Credit amount"].value_counts())

    # 分析数据类型
    credit.info()

    return credit





def main():
    credit = load_data()

    # 离散数据数值索引化
    String2Index, Index2String = numericalize(credit)
    print(String2Index)
    print(Index2String)
    print(credit.head())
    credit.info()

    # # 生成分析页面
    # profile = pp.ProfileReport(credit, title="german credit report")
    # profile.to_file(output_file=r"./german_credit_report.html")

    # 划分训练集和测试集
    X = credit.iloc[:, :-1]
    y = credit["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    # 比较一下训练集和测试集中类别是否接近
    print(y_train.value_counts() / len(y_train))
    print(y_test.value_counts() / len(y_test))

    # 训练
    classifier = tree.DecisionTreeClassifier(min_samples_leaf=6, random_state=1)
    classifier.fit(X_train, y_train)

    # 测试
    y_test_hat = classifier.predict(X_test)
    # print("正确率=", accuracy_score(y_test, y_test_hat))
    # print("精确度=", precision_score(y_test, y_test_hat))
    # print("召回率=", recall_score(y_test, y_test_hat))
    # print("f1指标=", f1_score(y_test, y_test_hat))
    print(classification_report(y_test, y_test_hat))
    # 显示混淆矩阵
    print(confusion_matrix(y_test, y_test_hat))

    # 性能提升
    class_weights = {1: 1, 2: 4}
    credit_model_cost = tree.DecisionTreeClassifier(max_depth=6, random_state=1, class_weight=class_weights)
    credit_model_cost.fit(X_train, y_train)
    y_test_hat_cost = credit_model_cost.predict(X_test)

    print(classification_report(y_test, y_test_hat_cost))
    # 显示混淆矩阵
    print(confusion_matrix(y_test, y_test_hat_cost))






if __name__ == '__main__':
    main()










