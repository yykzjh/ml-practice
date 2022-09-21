# -*- encoding: utf-8 -*-
'''
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/13 17:38
@Version  :   1.0
@License  :   (C)Copyright 2022-
'''
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import naive_bayes as nb
from sklearn.metrics import accuracy_score, classification_report

import scipy
from scipy import io


# 创建词汇表
def create_vocab_list(dataset):
    # 定义词汇表集合
    vocab_set = set([])
    # 遍历每一封邮件
    for document in dataset:
        # 取并集
        vocab_set = vocab_set | set(document)

    # 排序返回
    return sorted(list(vocab_set))



# 词集模型
def set_of_word_2_vec(vocabList, inputSet):
    # 初始化向量长度与词汇表长度一致
    returnVec = [0] * len(vocabList)
    # 遍历文件中的每一个词汇
    for word in inputSet:
        # 检测词汇是否在词汇表中
        assert word in vocabList, word + "not in vocabList"
        # 在词汇表中查找词汇出现的位置，将返回向量中的对应位置赋值为1
        returnVec[vocabList.index(word)] = 1

    return returnVec



# 词袋模型
def bag_of_word_2_vec(vocabList, inputSet):
    # 初始化向量长度与词汇表长度一致
    returnVec = [0] * len(vocabList)
    # 遍历文件中的每一个词汇
    for word in inputSet:
        # 检测词汇是否在词汇表中
        assert word in vocabList, word + "not in vocabList"
        # 在词汇表中查找词汇出现的位置，将返回向量中的对应位置出现的频数增加1
        returnVec[vocabList.index(word)] += 1

    return returnVec



# 邮件内容的预处理
def parse_text(bigString):
    # 以非字母、非数字、非汉字、非_的匹配内容作为分隔符
    list_of_tokens = re.split(r'\W+', bigString)
    # 过滤长度小于等于2的token，并且全都小写
    return [token.lower() for token in list_of_tokens if len(token) > 2]




# 加载数据
def load_data():
    # 初始化邮件内容列表
    docList = []
    # 初始化邮件类别列表
    classList = []

    # 定义邮件数量+1
    num = 26
    # 遍历读取所有邮件
    for i in range(1, num):
        # 读取垃圾邮件
        wordList = parse_text(open(r"./data/email/spam/{0}.txt".format(i)).read())
        docList.append(wordList)
        # 垃圾邮件的类别标签为1
        classList.append(1)

        # 读取垃圾邮件
        wordList = parse_text(open(r"./data/email/ham/{0}.txt".format(i)).read())
        docList.append(wordList)
        # 垃圾邮件的类别标签为0
        classList.append(0)

    # 获取词汇表
    vocabList = create_vocab_list(docList)

    # 初始化邮件内容向量
    X = []
    # 遍历计算每封邮件的词向量
    for docIndex in range(len(docList)):
        # 使用词袋模型
        X.append(bag_of_word_2_vec(vocabList, docList[docIndex]))
        # # 使用词集模型
        # X.append(set_of_word_2_vec(vocabList, docList[docIndex]))

    return X, classList, vocabList



def main():
    # 加载处理好的数据
    X, y, vocabList = load_data()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.20)

    # 定义模型
    naive_bayes_model = nb.MultinomialNB()
    # 训练
    naive_bayes_model.fit(X_train, y_train)

    # 预测训练集
    y_test_hat = naive_bayes_model.predict(X_test)
    # 打印测试结果
    print(classification_report(y_test, y_test_hat))




if __name__ == '__main__':
    main()




























