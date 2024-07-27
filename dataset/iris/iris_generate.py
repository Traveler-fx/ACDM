
import csv

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
    scaler = MinMaxScaler()
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

    csv_file = path
    combined_data = np.column_stack((data, labels))
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in combined_data:
            writer.writerow(row)


from itertools import cycle
cycol = cycle('bgrcmk')
def real_data_vis(data, labels):
    index_mapping={}
    for i in range(len(labels)):
        if labels[i] not in index_mapping.keys():
            index_mapping[labels[i]]=[]
            index_mapping[labels[i]].append(i)
        else:
            index_mapping[labels[i]].append(i)
    ax = plt.subplot(1, 1, 1)
    for cluster in index_mapping:
        index=index_mapping[cluster]
        ax.scatter(data[index,0],data[index,1],c=next(cycol))
    plt.show()
if __name__ == '__main__':
    data,labels=generate_iris_data(path="iris.data")