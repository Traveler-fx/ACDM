import numpy as np
from scipy.spatial.distance import euclidean


def connections_cal(data, node, neighborhood_r):
    connections = []
    for i in range(len(neighborhood_r)):
        distances = []
        for neighbor in neighborhood_r[i]:
            distances.append(euclidean(data[node], data[neighbor]))
        index = np.argmin(distances)
        connections.append([node, neighborhood_r[i][index], distances[index],i])
    connections = np.array(connections)
    sorted_indices = np.argsort(connections[:, 2])
    connections = connections[sorted_indices]
    return connections

def interaction_process(connections, real_labels, neighborhood, count, neighborhood_r,neighborhood_r_behind, skeleton,l):
    flag = False
    for i in range(len(connections)):
        node1 = int(connections[i][0])
        node2 = int(connections[i][1])
        neighborhood_index = int(connections[i][3])
        if real_labels[node1] == real_labels[node2]:
            count=count+1
            flag = True
            if len(neighborhood[neighborhood_index])<l:
                neighborhood[neighborhood_index].append(node1)
                neighborhood_r[neighborhood_index].append(node1)
                if skeleton.nodes[node1]['ranking']>skeleton.nodes[neighborhood_r_behind[neighborhood_index][0]]['ranking']:
                    neighborhood_r_behind[neighborhood_index]=[node1]
            if len(neighborhood[neighborhood_index])>=l:
                neighborhood[neighborhood_index].append(node1)

                if skeleton.nodes[node1]['ranking']<skeleton.nodes[neighborhood_r_behind[neighborhood_index][0]]['ranking']:
                    neighborhood_r[neighborhood_index].remove(neighborhood_r_behind[neighborhood_index][0])
                    neighborhood_r[neighborhood_index].append(node1)
                    a=[]
                    for j in neighborhood_r[neighborhood_index]:
                        a.append(skeleton.nodes[j]['ranking'])
                    c=neighborhood_r[neighborhood_index][np.argmax(a)]
                    neighborhood_r_behind[neighborhood_index]=[c]
            break
        if real_labels[node1] != real_labels[node2]:
            count = count + 1
    if flag == False:
        neighborhood.append([node1])
        neighborhood_r.append([node1])
        neighborhood_r_behind.append([node1])
    return neighborhood,neighborhood_r,neighborhood_r_behind,count


def neighborhood_initialization(data, decision_list, representative, real_labels, skeleton, m,l):
    count = 0
    neighborhood = []
    neighborhood.append([representative])
    neighborhood_r = []
    neighborhood_r.append([representative])
    neighborhood_r_behind=[]
    neighborhood_r_behind.append([representative])
    nodes = decision_list[1:m]
    decision_list.remove(representative)
    for node in nodes:
        decision_list.remove(node)
        connections = connections_cal(data, node, neighborhood_r)
        neighborhood,neighborhood_r,neighborhood_r_behind,count=interaction_process(connections, real_labels, neighborhood, count, neighborhood_r,neighborhood_r_behind, skeleton,l)
    return neighborhood,neighborhood_r,neighborhood_r_behind,count,decision_list



