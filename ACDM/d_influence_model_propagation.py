import networkx as nx
from sklearn.metrics import adjusted_rand_score

from ACDM.a_DMCons import data_preprocess, graph_initialization
from ACDM.b_ranking_allocation import order_allocation
from ACDM.c_neighborhood_initialization import neighborhood_initialization



def influence_model_propagation(skeleton,neighborhood):
    for i in range(len(skeleton.nodes)):
        skeleton.nodes[i]['state'] = 'inactive'
        skeleton.nodes[i]['label'] = 'unclear'
    nodes_in_neighborhood = []
    for i in range(len(neighborhood)):
        for node in neighborhood[i]:
            skeleton.nodes[node]['state'] = 'active'
            skeleton.nodes[node]['label'] = i
            nodes_in_neighborhood.append(node)
    for node in nodes_in_neighborhood:
        container = [node]
        label=skeleton.nodes[node]['label']
        while(True):
            neighbors = list(skeleton.neighbors(container[-1]))
            container.pop()
            for neighbor in neighbors:
                if (skeleton.nodes[neighbor]['ranking'] > skeleton.nodes[node]['ranking']) and (skeleton.nodes[neighbor]['state']=='inactive'):
                    container.append(neighbor)
                    skeleton.nodes[neighbor]['state'] = 'active'
                    skeleton.nodes[neighbor]['label'] = label
            if len(container)==0:
                break
    predicted_labels=[]
    for i in range(len(skeleton.nodes)):
        label=skeleton.nodes[i]['label']
        predicted_labels.append(label)
    return predicted_labels


