# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/22 0:18
@Version  :   1.0
@License  :   (C)Copyright 2022-
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler




def main():
    # 读取照片
    img = mpimg.imread(r"./data/jjq.jpg")
    # 保存原始图像维度
    src_shape = img.shape
    # 变换成二维
    X = img.reshape((img.shape[0], -1))
    # 数据标准化
    X = StandardScaler().fit_transform(X)

    # 定义PCA
    pca = PCA(n_components=50)
    # 降维
    Z = pca.fit_transform(X)
    # 输出贡献比
    print("贡献比=", np.sum(pca.explained_variance_ratio_))

    # 数据还原
    X_rec = pca.inverse_transform(Z)
    # 维度变换成原来的维度
    img_rec = X_rec.reshape(src_shape)
    # 显示还原后的图像
    plt.imshow(img_rec)
    plt.show()





if __name__ == '__main__':
    main()










