import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def generate_seeds_data(path):
    df=pd.read_csv(path, header=None)
    data = np.array(df)
    labels=data[:,7]
    labels=labels_processing(labels)
    data=data[:,0:7]
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


if __name__ == '__main__':
    data,labels=generate_seeds_data(path="seeds.data")
    print(np.shape(data))
    print(len(set(labels)))
