import copy
import heapq

import networkx as nx
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

import math

from ACDM.a_DMCons import graph_initialization
from ACDM.b_ranking_allocation import order_allocation
from ACDM.c_neighborhood_initialization import connections_cal, interaction_process, \
    neighborhood_initialization
from ACDM.d_influence_model_propagation import influence_model_propagation


def k_nearest_neighbor_cal(data,k):

    neighbors = NearestNeighbors(n_neighbors=k).fit(data)
    k_nearest_neighbors = neighbors.kneighbors(data, return_distance=False)
    return k_nearest_neighbors


def uncertainty_oneNode(predict_labels, k_nearest_neighbor,k):
    dict={}
    for i in range(len(k_nearest_neighbor)):
        point=k_nearest_neighbor[i]
        if predict_labels[point] not in dict.keys():
            dict[predict_labels[point]]=[point]
        else:
            dict[predict_labels[point]].append(point)
    sum=0
    for m in dict.keys():
        proportion=len(dict[m])/k
        if proportion != 0:
            sum = sum + proportion * math.log2(proportion)
    sum = -sum
    if sum==-0.0:
        sum=0.0
    return sum

def uncertainty_cal(predict_labels,k_nearest_neighbors,candidates,k):
    uncertainty_dict=dict()
    for candidate in candidates:
        k_nearest_neighbor=k_nearest_neighbors[candidate]
        uncertainty=uncertainty_oneNode(predict_labels, k_nearest_neighbor,k)
        uncertainty_dict[candidate]=uncertainty
    return uncertainty_dict

def first_n_nodes_cal(my_dict,n):
    if n>len(my_dict):
        n=len(my_dict)
    sliced_list=[]

    heap = [(-value, key) for key, value in my_dict.items()]
    heapq.heapify(heap)
    count=0
    for _ in range(n):
        if heap:
            neg_value, key = heapq.heappop(heap)
            value = -neg_value
            if value==0.0:
                break
            sliced_list.append(key)
            del my_dict[key]
            count=count+1
        else:
            break
    remaining_count = n - count
    if remaining_count > 0:
        for i in range(remaining_count):
            first_key, first_value = next(iter(my_dict.items()))
            del my_dict[first_key]
            sliced_list.append(first_key)
            count = count + 1
    return sliced_list,my_dict



def neighborhood_learning(skeleton, data, predict_labels, neighborhood,neighborhood_r,neighborhood_r_behind, k_nearest_neighbors, count, order,real_labels, record, n,k,l):
    candidates = dict()
    for i in range(len(order)):
        candidates[order[i]] = 0
    flag = False
    iter=2
    while (True):
        candidates = uncertainty_cal(predict_labels, k_nearest_neighbors, candidates,k)
        sliced_list,candidates=first_n_nodes_cal(candidates, n)
        for node in sliced_list:
            connections = connections_cal(data, node, neighborhood_r)
            neighborhood,neighborhood_r,neighborhood_r_behind,count=interaction_process(connections, real_labels, neighborhood, count, neighborhood_r,neighborhood_r_behind, skeleton,l)
        if candidates == dict():
            flag = True
        predict_labels=influence_model_propagation(skeleton, neighborhood)
        ari = adjusted_rand_score(real_labels, predict_labels)
        record.append([{"iter": iter, "interaction": count, "ari": ari}])
        print("iteration: %d, queries: %d, ari: %s" % (iter, count, ari))
        iter=iter+1
        if flag == True:
            break
        if ari ==1:
            break
    return record



def clusters_to_predict_vec(clusters):
    tranversal_dict = {}
    predict_vec = []
    for i in range(len(clusters)):
        for j in clusters[i]:
            tranversal_dict[j] = i
    for i in range(len(tranversal_dict)):
        predict_vec.append(tranversal_dict[i])
    return predict_vec


def initialization_cut(skeleton,m,start_node):
    G=copy.deepcopy(skeleton)
    traversed_nodes = [start_node]
    candidate_edge = []
    for edge in G.edges(start_node, data=True):
        heapq.heappush(candidate_edge, (-edge[2]['weight'], edge[0], edge[1]))
    while candidate_edge:
        max_edge = heapq.heappop(candidate_edge)
        G.remove_edge(max_edge[1],max_edge[2])
        if len(traversed_nodes)==m:
            break
        weight, current_node, new_node = -max_edge[0], max_edge[1], max_edge[2]
        traversed_nodes.append(new_node)
        for edge in G.edges(new_node, data=True):
            if edge[1] != current_node:
                heapq.heappush(candidate_edge, (-edge[2]['weight'], edge[0], edge[1]))
    clusters = []
    S = [G.subgraph(c) for c in nx.connected_components(G)]
    for i in S:
        clusters.append(list(i.nodes))
    predict_labels = clusters_to_predict_vec(clusters)
    return predict_labels



def NNGAMD(data, real_labels,k,l,beta):
    m = int(len(data) * (1 / beta))
    n = int(len(data) * (1 / beta))
    k_nearest_neighbors = k_nearest_neighbor_cal(data, k)
    skeleton, representative = graph_initialization(data)
    record = [[{"iter": 0, "interaction": 0, "ari": 0}]]
    skeleton, order = order_allocation(skeleton, representative)
    neighborhood,neighborhood_r,neighborhood_r_behind,count,order=neighborhood_initialization(data, order, representative, real_labels, skeleton, m, l)
    predict_labels = influence_model_propagation(skeleton, neighborhood)
    record.append([{"iter": 1, "interaction": count, "ari": adjusted_rand_score(real_labels, predict_labels)}])
    record = neighborhood_learning(skeleton, data, predict_labels, neighborhood,neighborhood_r,neighborhood_r_behind, k_nearest_neighbors, count, order,
                                   real_labels, record, n,k,l)

    return record




