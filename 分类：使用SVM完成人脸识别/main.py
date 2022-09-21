# -*- encoding: utf-8 -*-
'''
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/15 21:36
@Version  :   1.0
@License  :   (C)Copyright 2022-
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # 默认风格

from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix





def load_face_data():
    faces = fetch_lfw_people(data_home=r"./data/lfw/scikit_learn_data/",
                             min_faces_per_person=60)
    print(faces.data.shape)
    print(faces.target_names)
    print(faces.images.shape)  # 原图大小为(250, 250),读取后自动剪切人脸为(62, 47)

    return faces




def show_some_faces(faces):
    fig, ax = plt.subplots(3, 5)
    # 调整子图之间的距离
    fig.subplots_adjust(left=0.0625, right=0.9, wspace=1)
    # 遍历各个子图
    for i, axi in enumerate(ax.flat):
        axi.imshow(faces.images[i], cmap='bone')
        axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])
    plt.show()



# 在SVC前面添加PCA主成分分析预处理，提取更有意义的特征，降维
def get_PCA_pipline_SVC_model():
    # 初始化PCA模块，降维到150维
    pca = PCA(n_components=150, whiten=True, random_state=42)
    # 定义SVC模型
    svc = SVC(kernel='rbf')
    # 将两个部分结合起来
    model = make_pipeline(pca, svc)

    return model



# 比较预测结果和真实结果
def compare_y(faces, X_test, y_hat, y_test):
    mpl.rcParams["font.sans-serif"] = [u"SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    # 定义子图
    fig, ax = plt.subplots(4, 6)
    # 选取测试集前几张图像
    for i, axi in enumerate(ax.flat):
        # 展示图象
        axi.imshow(X_test[i].reshape((62, 47)), cmap='bone')
        axi.set(xticks=[], yticks=[])
        # 根据预测正确与否，显示不同颜色的标注
        axi.set_ylabel(faces.target_names[y_hat[i]].split()[-1],
                       color="black" if y_hat[i] == y_test[i] else "red")
    fig.suptitle("预测错误的名字用红色标注", size=14)
    plt.show()



# 将混淆矩阵用热力图显示
def show_confusion_matrix_with_heatmap(faces, y_test, y_hat):
    # 计算混淆矩阵
    mat = confusion_matrix(y_test, y_hat)

    # 画出热力图
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=faces.target_names,
                yticklabels=faces.target_names)
    plt.xlabel("true label")
    plt.ylabel("predicted label")
    plt.show()




def main():
    # 加载人脸数据
    faces = load_face_data()
    # # 画一些人脸，看看需要处理的数据
    # show_some_faces(faces)

    # 创建模型
    model = get_PCA_pipline_SVC_model()

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, random_state=42)

    # 定义需要搜索的参数有哪些
    param_grid = {'svc__C': [1, 5, 10, 50], 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
    # 定义网格搜索交叉验证，默认5折交叉验证
    gscv = GridSearchCV(model, param_grid)

    # 训练
    gscv.fit(X_train, y_train)
    print(gscv.best_params_)

    # 获取最好的模型
    best_model = gscv.best_estimator_

    # 预测测试数据
    y_hat = best_model.predict(X_test)

    # 比较预测结果和真实结果
    compare_y(faces, X_test, y_hat, y_test)

    # 打印分类结果报告，列举每个标签的统计结果
    print(classification_report(y_test, y_hat, target_names=faces.target_names))

    # 画出混淆矩阵的热力图
    show_confusion_matrix_with_heatmap(faces, y_test, y_hat)







if __name__ == '__main__':
    main()



























