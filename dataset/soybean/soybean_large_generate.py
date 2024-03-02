import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def generate_soybean_data(path):
    df=pd.read_csv(path, header=None)
    df_labels=df.iloc[:,0]
    df_data=df.iloc[:,1:]
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

if __name__ == '__main__':
    data, labels = generate_soybean_data(path="soybean.data")
    print(np.shape(data))
    print(len(set(labels)))