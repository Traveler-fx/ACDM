import heapq




def centrality_ranking(G, start_node):
    traversed_nodes = [start_node]
    candidate_edge = []
    for edge in G.edges(start_node, data=True):
        heapq.heappush(candidate_edge, (-edge[2]['weight'], edge[0], edge[1]))
    while candidate_edge:
        max_edge = heapq.heappop(candidate_edge)
        weight, current_node, new_node = -max_edge[0], max_edge[1], max_edge[2]
        traversed_nodes.append(new_node)
        for edge in G.edges(new_node, data=True):
            if edge[1] != current_node:
                heapq.heappush(candidate_edge, (-edge[2]['weight'], edge[0], edge[1]))
    return traversed_nodes



def order_allocation(skeleton, representative):
    decision_list = centrality_ranking(skeleton, start_node=representative)
    for i in range(len(decision_list)):
        skeleton.nodes[decision_list[i]]['ranking'] = i
    return skeleton,decision_list


