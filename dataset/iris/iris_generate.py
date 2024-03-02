#生成鸢尾花数据集
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def generate_iris_data(path):
    df=pd.read_csv(path, header=None)
    data = np.array(df)
    labels=data[:,4]
    labels=labels_processing(labels)
    data=data[:,:4]
    data=np.array(data)
    labels=np.array(labels)
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


#可视化聚类结果
from itertools import cycle
cycol = cycle('bgrcmk')
def real_data_vis(data, labels):
    #遍历labels中每个不同类别数字的位置
    index_mapping={}
    for i in range(len(labels)):
        if labels[i] not in index_mapping.keys():
            index_mapping[labels[i]]=[]
            index_mapping[labels[i]].append(i)
        else:
            index_mapping[labels[i]].append(i)
    #记录好下标之后，现在就差将数据点画出来的啦
    ax = plt.subplot(1, 1, 1)  # 子图初始化
    for cluster in index_mapping:
        index=index_mapping[cluster]
        ax.scatter(data[index,0],data[index,1],c=next(cycol))
    plt.show()
if __name__ == '__main__':
    data,labels=generate_iris_data(path="iris.csv")
    print(np.shape(data))
    print(len(set(labels)))