

import pandas as pd


from ACDM.a_DMCons import data_preprocess
from ACDM.e_neighborhood_learning import NNGAMD
from dataset.iris.iris_generate import generate_iris_data


def result_to_csv(ARI_record, title):
    record = []
    for i in range(len(ARI_record)):
        if i<len(ARI_record)-1:
            repeat=ARI_record[i+1][0]["interaction"]-ARI_record[i][0]["interaction"]
            for j in range(repeat):
                record.append(ARI_record[i][0]["ari"])
        else:
            record.append(ARI_record[i][0]["ari"])
    df = pd.DataFrame({
        'ARI': record,
    })
    df.to_csv("output/%s_result.csv" % title)


if __name__ == '__main__':
    k = 10
    r = 100
    l  = 100
    data, real_labels = data,labels=generate_iris_data(path="../dataset/iris/iris.data")
    data = data_preprocess(data)
    b_1 = len(set(real_labels))
    b_2 = int(len(data) * (1 / r))
    ARI_record=NNGAMD(data, real_labels,b_1,k,l,b_2)
    result_to_csv(ARI_record, title="iris")

