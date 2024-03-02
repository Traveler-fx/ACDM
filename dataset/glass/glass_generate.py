#生成glass数据集
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def generate_glass_data(path):
    df=pd.read_csv(path, header=None)
    data = np.array(df)
    labels=data[:,10]
    labels=labels_processing(labels)
    data=data[:,1:10]
    # 创建MinMaxScaler对象
    scaler = MinMaxScaler()
    # 对数据进行归一化
    data = scaler.fit_transform(data)
    return data,labels

def labels_processing(raw_labels):
    new_lables=[]
    labels_dict={}
    count=0
    for i in raw_labels:
        if i not in labels_dict:
            labels_dict[i]=count
            count=count+1
    for j in raw_labels:
        new_lables.append(labels_dict[j])
    return new_lables


# #可视化聚类结果
# from itertools import cycle
# cycol = cycle('bgrcmk')
# def real_data_vis(data, labels):
#     #遍历labels中每个不同类别数字的位置
#     index_mapping={}
#     for i in range(len(labels)):
#         if labels[i] not in index_mapping.keys():
#             index_mapping[labels[i]]=[]
#             index_mapping[labels[i]].append(i)
#         else:
#             index_mapping[labels[i]].append(i)
#     #记录好下标之后，现在就差将数据点画出来的啦
#     ax = plt.subplot(1, 1, 1)  # 子图初始化
#     for cluster in index_mapping:
#         index=index_mapping[cluster]
#         ax.scatter(data[index,0],data[index,1],c=next(cycol))
#         for i in index:
#             ax.text(data[i,0],data[i,1],i,fontsize=7)
#     plt.savefig("F:\interactive_clustering_algorithms\交互式聚类-基于矩阵传播\experiement1/f_datasets\glass\glass_2d.png")
#     plt.close()

if __name__ == '__main__':
    data,labels=generate_glass_data(path="glass.data")
    print(np.shape(data))
    print(len(set(labels)))
    # real_data_vis(data, labels)