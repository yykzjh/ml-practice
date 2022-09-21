# -*- encoding: utf-8 -*-
'''
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/21 1:55
@Version  :   1.0
@License  :   (C)Copyright 2022-
'''
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler




def load_data():
    # 读取数据
    df = pd.read_csv("./data/football_team_data.csv", index_col="国家")

    return df






def main():
    # 读取数据
    df = load_data()
    # 获取numpy格式的数据
    data = df.values
    # 标准化
    X = StandardScaler().fit_transform(data)

    # 定义k-means模型
    model = KMeans(n_clusters=3, max_iter=10)
    # 训练
    model.fit(X)

    # 获取聚类结果
    df["聚类结果"] = model.labels_
    print(df)






if __name__ == '__main__':
    main()




















