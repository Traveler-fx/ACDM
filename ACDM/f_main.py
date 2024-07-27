from ACDM.a_DMCons import data_preprocess
from ACDM.e_neighborhood_learning import NNGAMD
from dataset.iris.iris_generate import generate_iris_data

if __name__ == '__main__':
    datasets =["iris"]
    # parameters
    k = 10
    beta = 100
    omega = 100
    data, real_labels = generate_iris_data(path="../dataset/iris/iris.data")
    data = data_preprocess(data)
    ARI_record=NNGAMD(data, real_labels,k,omega,beta)
    print(ARI_record)















