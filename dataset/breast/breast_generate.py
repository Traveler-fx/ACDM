import csv

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def generate_breast_data(path):
    df = pd.read_csv(path, header=None)
    df_labels=df.iloc[:,10]
    df_data=df.iloc[:,1:10]
    #将问号用np.nan代替
    df_data.replace("?", np.nan, inplace=True)
    missing_data = df_data.isnull()
    for column in missing_data.columns.values.tolist():
        avg_norm_loss = df_data[column].astype("float").mean(axis=0)
        df_data[column].replace(np.nan, avg_norm_loss, inplace=True)
    data = np.array(df_data)
    data=data.astype(float)
    labels=np.array(df_labels)
    labels=labels_processing(labels)
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

def csv_output(data,labels,path):
    # 指定要保存的CSV文件的路径
    csv_file = path
    # 将数据和标签合并为一个数组，方便写入CSV文件
    combined_data = np.column_stack((data, labels))
    # 使用csv.writer来写入CSV文件
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # 写入数据
        for row in combined_data:
            writer.writerow(row)

if __name__ == '__main__':
    data, labels = generate_breast_data(path="breast.data")
    csv_output(data, labels, path="breast.csv")
    print(np.shape(data))
    print(len(set(labels)))