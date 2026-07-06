# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np


def plot_confusion_matrix(cm, classes, title='Confusion matrix2', cmap=plt.cm, fig_name='Confusion matrix.png'):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # cmap  'YlGnBu'   'viridis'  'Wistia' 'OrRd'
    cmap = plt.cm.get_cmap('YlGnBu')
    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    plt.colorbar()  # 放一条进度条
    # plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))  # 得到数组 [0, 1, 2, 3, 4, 5, 6]
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes,rotation=45)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):  # 会产生所有 (i, j) 这样的元组
        if cm[i, j] <= 0.00001:
            cm[i, j] = 0
            plt.text(j, i, '0'.format(cm[i, j]), horizontalalignment="center",fontsize=10,
                     color="white" if cm[i, j] > thresh else "black")  # 在坐标 (j, i) 位置写 '0'
        else:
            plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",fontsize=10,
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()  # 自动调整图像中 子图、坐标轴、标题、刻度标签、文字 的间距
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    # plt.savefig('confusion_matrix.png', dpi=300)  # 保存图片
    #
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')  # 自动调整图像边界，只保存图中实际有内容的部分，去掉多余空白
    print("混淆矩阵保存成功")
    plt.show()


def read_write_txt(file):
    pre_list = []
    true_list = []

    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 2:
                continue

            # 跳过表头 prediction true_label
            if parts[0] == 'prediction' and parts[1] == 'true_label':
                continue

            pre_list.append(int(parts[0]))
            true_list.append(int(parts[1]))

    return pre_list, true_list


# 标签，0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
if __name__ == '__main__':
    # read_txt_comp(true_labels)
    # txt_pred = read_write_txt(file = 'rafdb_pred.txt')
    # true_labels = read_write_txt(file = 'rafdb_truelabel.txt')
    txt_pred, true_labels = read_write_txt(file='../results/pre_labels.txt')

    conf_mat = confusion_matrix(
        y_true=true_labels,
        y_pred=txt_pred,
        labels=[0, 1, 2, 3]
    )

    plot_confusion_matrix(
        conf_mat,
        classes=['engraving', 'heat', 'ink', 'laser'],
        title='CNN-Mamba_confusion_matrix',
        fig_name='CNN-Mamba_Confusion_matrix.png'
    )

