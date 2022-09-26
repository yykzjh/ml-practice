# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/26 21:03
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
from math import log
import numpy as np
import operator



def generate_data():
    # 定义数据集
    dataset = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    # 定义数据属性名称
    feature_names = ['age', 'job', 'house', 'credit']
    return dataset, feature_names



# 计算数据集的熵
def entropy(dataset):
    # 获取数据集维度信息
    m, n = len(dataset), len(dataset[0])
    # 定义不同标签值的计数字段
    label_counts = {}
    # 遍历所有样本的标签值
    for sample in dataset:
        # 获取标签值
        cur_label = sample[-1]
        if cur_label not in label_counts:
            label_counts[cur_label] = 0
        # 计数+1
        label_counts[cur_label] += 1
    # 初始化熵值
    e = 0.0
    # 遍历计数字典累加熵
    for k, v in label_counts.items():
        # 计算当前标签值的占比
        prob = v / m
        # 计算熵并累加
        e -= prob * log(prob, 2)
    return e



# 按照指定属性的指定值取出子数据集
def split_sub_dataset(dataset, axis, value):
    # 初始化子数据集
    sub_dataset = []
    # 遍历数据集
    for sample in dataset:
        # 判断当前样本的指定属性值是否满足条件
        if sample[axis] == value:
            # 前半部分
            reduced_sample = sample[:axis]
            # 拼接后半部分
            reduced_sample.extend(sample[axis+1:])
            # 将该拼接样本添加到子数据集
            sub_dataset.append(reduced_sample)
    return sub_dataset



# 选择使得数据增益最大的属性索引
def choose_best_feature(dataset):
    # 获取数据集维度信息
    m, n = len(dataset), len(dataset[0])
    # 计算整个数据集的熵
    base_entropy = entropy(dataset)
    # 初始化最大的信息增益和属性索引
    best_info_gain = 0.0
    best_feature = -1
    # 遍历所有的特征
    for i in range(n-1):
        # 获取当前数据集中当前特征列的所有值的集合
        feature_values = set([sample[i] for sample in dataset])
        # 定义条件熵
        condition_entropy = 0.0
        # 循环该特征列的每一个属性值
        for value in feature_values:
            # 按照该属性值划分出子数据集
            sub_dataset = split_sub_dataset(dataset, i, value)
            # 计算子数据集的占比
            prob = len(sub_dataset) / m
            # 累加条件熵
            condition_entropy += prob * entropy(sub_dataset)
        # 计算信息增益
        info_gain = base_entropy - condition_entropy
        # 判断当前信息增益是否是当前的最大信息增益
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    # 返回最优特征列索引
    return best_feature



# 根据样本的类别，投票选取最多的类别返回
def vote_label(label_list):
    # 定义计数字典
    label_cnt = {}
    # 循环遍历标签
    for _label in label_list:
        if _label not in label_cnt.keys():
            label_cnt[_label] = 0
        label_cnt[_label] += 1
    # 对计数字典从大到小怕排序
    sorted_label_cnt = sorted(label_cnt.items(), key=lambda x: x[1], reverse=True)
    return sorted_label_cnt[0][0]



# 递归地训练一颗ID3决策树
def training_ID3_decision_tree(dataset, feature_names, cur_depth, max_depth):
    # 获取数据集维度信息
    m, n = len(dataset), len(dataset[0])
    # 获取当前数据集的标签列
    label_list = [sample[-1] for sample in dataset]

    # 如果当前树的深度达到限定的最大深度，或者特征已经用完，则递归结束，将当前数据集中最多的标签返回
    if cur_depth >= max_depth or n <= 1:
        return vote_label(label_list)
    # 如果当前数据集都是同一个类别，则作为叶节点返回类别标签
    if label_list.count(label_list[0]) == m:
        return label_list[0]

    # 获得最优特征列索引
    best_feature_index = choose_best_feature(dataset)
    # 获得该最有特征列的名称
    best_feature_name = feature_names[best_feature_index]
    # 初始化当前结点
    cur_node = {best_feature_name: {}}
    # 获取最优特征列的所有值的集合
    feature_values = set([sample[best_feature_index] for sample in dataset])
    # 遍历所有属性值，根据每个属性值划分得到子数据集
    for value in feature_values:
        # 切分特征名称前半部分
        reduced_feature_names = feature_names[:best_feature_index]
        # 拼接后半部分
        reduced_feature_names.extend(feature_names[best_feature_index+1:])
        # 递归获得子树的根节点
        cur_node[best_feature_name][value] = training_ID3_decision_tree(
            split_sub_dataset(dataset, best_feature_index, value),
            reduced_feature_names,
            cur_depth+1,
            max_depth
        )
    return cur_node



# 使用训练好的决策树进行预测
def predict(node, feature_names, test_sample):
    # 获取当前节点用于划分的属性名称
    feature_name = list(node.keys())[0]
    # 获取以属性值划分的第二层字典
    second_dict = node[feature_name]
    # 获取当前特征名称对应的特征列索引
    feature_index = feature_names.index(feature_name)
    # 获取测试样本当前特征列的属性值
    value = test_sample[feature_index]
    # 根据属性值获得子树根结点
    child_node = second_dict[value]
    # 根据子树根节点是字典类型还是字符串类型判断是否到达叶节点
    if isinstance(child_node, dict):
        label = predict(child_node, feature_names, test_sample)
    else:
        label = child_node
    return label




def main():
    # 生成数据集
    dataset, feature_names = generate_data()

    # 训练一颗简单的ID3决策树
    ID3 = training_ID3_decision_tree(dataset, feature_names, 0, max_depth=1)

    # 用训练好的ID3决策树预测训练集
    y_hat = [predict(ID3, feature_names, sample) for sample in dataset]

    # 获取训练数据的标签
    y = [sample[-1] for sample in dataset]
    # 计算准确率
    print("Train Accuracy=", sum([y_hat[i] == y[i] for i in range(len(y))]) / len(y))





if __name__ == '__main__':
    main()



