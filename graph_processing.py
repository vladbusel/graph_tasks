import csv
from heapq import heappush, heappop
from functools import reduce
from itertools import count
from lxml import etree as et
from math import sin, cos, sqrt, atan2, radians
import matplotlib.pyplot as plt
import networkx as nx
from random import random, sample
import numpy as np
import osmnx as ox
from scipy.cluster.hierarchy import dendrogram, linkage
import collections

#on file upload
def create_graph_from_osm(obj, osm_file): 
    f = open('houses.txt', 'r')
    houses_id = [house[:-1] for house in f.readlines()]
    f.close()
    obj_id = []
    if obj == 'medicine':
        f = open('med.txt', 'r')
        obj_id = [med[:-1] for med in f.readlines()]   
        f.close()
    elif obj == 'firewatch':
        f = open('firewatch.txt', 'r')
        obj_id = [firewatch[:-1] for firewatch in f.readlines()]   
        f.close()
    elif obj == 'market':
        f = open('market.txt', 'r')
        obj_id = [market[:-1] for market in f.readlines()]   
        f.close()
    else:
        print('wrong structure')

    G = nx.DiGraph()
    context = et.iterparse(osm_file, events=('end',), tag='node')
    nodes = {}
    for event, elem in context:
        #Поменял lon и lat местами для разворота
        nodes[elem.get('id')] = [float(elem.get('lon')),float(elem.get('lat'))]
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    del context
    G.add_nodes_from(nodes)
    
    def distance(node1,node2):
        R = 6373.0
        lat1 = radians(nodes[node1][1])#lat
        lon1 = radians(nodes[node1][0])#lon
        lat2 = radians(nodes[node2][1])
        lon2 = radians(nodes[node2][0])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c
        return distance

    def nearest_node_id(node_a_id, pass_list = []):    
        min_dist = float('inf')
        if pass_list.count(node_a_id) == 0:
            pass_list.append(node_a_id)
        for node_b_id in list(G.nodes()):
            d = distance(node_a_id,node_b_id)
            if d < min_dist and node_b_id not in pass_list:
                min_dist = d
                nni = node_b_id
        return nni
   
    context = et.iterparse(osm_file,events=('end',),tag='way')
    edges = []
    for event, elem in context:
        oneway = 0
        #Определим свойства
        for child in elem.iterchildren('tag'):
            key = child.get('k')      
            #Возможно, дорога односторонняя
            if key == 'oneway' and child.get('v') == 'yes':
                oneway = 1
        #Узнаем, какие вершины состоят в пути
        nodes_list = []
        for child in elem.iterchildren('nd'):
            _id = child.get('ref')
            nodes_list.append(_id)
        #Добавляем ребра с весами в список смежности
        if False:#oneway == 1:
            for i in range(len(nodes_list)-1):
                #Добавляем вершины
                from_node, to_node = nodes_list[i:i+2]
                d = distance(from_node, to_node)
                edges.append((from_node, to_node, {'weight': d}))
        else:
            for i in range(len(nodes_list)-1):
                from_node, to_node = nodes_list[i:i+2]
                d = distance(from_node, to_node)
                edges.append((from_node, to_node, {'weight': d}))
                edges.append((to_node, from_node, {'weight': d}))
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    del context
    G.add_edges_from(edges)    
    
    #Удалим лишнее
    for node in nodes:
        if (len(list(G.successors(node))) == 1 or len(list(G.predecessors(node)))) == 1 and G.degree(node) != 1:
            remove_list,add_list = [],[]
            for from_node in G.predecessors(node):
                for to_node in G.successors(node):
                    remove_list.append((from_node,node))
                    remove_list.append((node,to_node))
                    add_list.append((from_node, to_node, {'weight':G[from_node][node]['weight']+G[node][to_node]['weight']}))
            G.add_edges_from(add_list)
            G.remove_edges_from(remove_list)
            G.remove_node(node)
    
    remove_nodes = [node for node in G.nodes() if int(G.degree(node)) == 0]
    

    G.remove_nodes_from(remove_nodes)

    return G,houses_id,obj_id,nodes

#on params change
def set_weights(G,obj_id,obj_max_weight):
    for o_id in obj_id:
        w = 1 + random()*(obj_max_weight-1)
        for node in G.successors(o_id):
            G[o_id][node]['weight'] *= w
        for node in G.predecessors(o_id):
            G[node][o_id]['weight'] *= w    

def dijkstra(G,_from,to_list = 'empty', max_dist = None):
    if to_list == 'empty':
        to_list = list(G.nodes())
    
    push, pop = heappush, heappop  
    c,finished_to_nodes = count(),to_list.copy()
    finish = len(to_list)
    heap, checked = [], {}
    checked[_from] = {'weight':0,'way':[_from]}
    push(heap, (0, next(c), _from))
    
    while heap:
        (d, _, v) = pop(heap)
        if v in finished_to_nodes:
            finished_to_nodes.remove(v)
        if finished_to_nodes == []:
            break
        for u in G.successors(v):
            _weight = checked[v]['weight'] + G[v][u]['weight']
            if max_dist is not None:
                if vu_dist > max_dist:
                    continue
            if u not in checked:
                checked[u] = {'weight': _weight, 'way': checked[v]['way'] + [u]}
                push(heap, (_weight, next(c), u))
            elif _weight < checked[u]['weight']:
                checked[u]['weight'] = _weight
                push(heap, (_weight, next(c), u))
                checked[u]['way'] = checked[v]['way'] + [u]

    return checked

def create_adj_list(G,filename = 'adj_list2.csv'):
    with open(filename,'w',newline='') as csv_file:
        csv_file.write('from_id to_id,edge_weight\n')
        for _from in list(G.nodes):
            s = str(_from)
            for _to in G.successors(_from):
                s = s + ' ' + str(_to) + ',' + str(G[_from][_to]['weight'])
            csv_file.write(s + '\n')

def write_dijkstra_csv(G,filename,from_list,to_list):
    with open(filename,'w',newline='') as f:
        f.write('from,to,way_weight,way\n')
        for _from in from_list:
            a = dijkstra(G,_from,to_list)
            for _to in to_list:
                if _to in a.keys():
                    s = _from + ',' + _to + ',' + str(a[_to]['weight'])
                    way = ','.join(a[_to]['way'])
                    s = s + ',' + way
                    f.write(s + '\n')

def read_dijkstra_csv(filename):
    min_ways = {}         
    with open(filename,'r') as f:
        lines = f.readlines()
        lines = lines[1:] 
        for line in lines:
            split_line = line[:-1].split(',')
            _from = split_line[0]
            _to = split_line[1]
            weight = float(split_line[2])
            if weight != float('inf'):
                way = split_line[3:]
            else: 
                way = []             
            if min_ways.get(_from) == None:
                min_ways[_from] = {_to:{'weight':weight,'way':way}}
            else:
                min_ways[_from][_to] = {'weight':weight,'way':way}
                
    return min_ways

def write_ways(filename,min_way):
    with open(filename,'w',newline='') as f:  
        f.write('from_id,to_id,way_weight\n')
        s = ''
        for _from in min_way.keys():
            s = s + _from + ',' + min_way[_from]['to'] + ',' + str(min_way[_from]['weight'])+'\n'
        f.write(s)

def search_min_oneways(min_ways,from_list,to_list):
    min_way = {}
    for _from in from_list:
        for _to in to_list:
            if _from != _to:
                if _from in min_ways.keys() and _to in min_ways[_from].keys():
                    if min_way.get(_from) == None:
                        min_way[_from] = {'to':_to,'weight':min_ways[_from][_to]['weight'],'way':min_ways[_from][_to]['way']}
                    elif min_ways[_from][_to]['weight'] < min_way[_from]['weight']:
                        min_way[_from]['to'] = _to
                        min_way[_from]['weight'] = min_ways[_from][_to]['weight']                    
                        min_way[_from]['way'] =  min_ways[_from][_to]['way']
    return min_way

def search_min_ways_there_and_back(min_ways,from_list,to_list):
    min_way = {}
    for _from in from_list:
        for _to in to_list:
            if _from != _to:
                if _from in min_ways.keys() and _to in min_ways[_from].keys() and _to in min_ways.keys() and _from in min_ways[_to].keys():
                    if min_way.get(_from) == None:
                        min_way[_from] = {'to':_to,'weight':min_ways[_from][_to]['weight'] + min_ways[_to][_from]['weight'],'way':min_ways[_from][_to]['way'],'way_back':min_ways[_to][_from]['way']}
                    elif min_way[_from]['weight'] > min_ways[_from][_to]['weight'] + min_ways[_to][_from]['weight']:
                        min_way[_from]['to'] = _to
                        min_way[_from]['weight'] = min_ways[_from][_to]['weight'] + min_ways[_to][_from]['weight']
                        min_way[_from]['way'] =  min_ways[_from][_to]['way']
                        min_way[_from]['way_back'] = min_ways[_to][_from]['way']
    return min_way

def search_near_ways(min_ways,max_weight,from_list,to_list):
    near_ways = []
    for _from in from_list:
        for _to in to_list:
            if _from in min_ways.keys() and _to in min_ways[_from].keys():
                if min_ways[_from][_to]['weight'] <= max_weight and _from != _to:
                    near_ways.append([_from,_to,min_ways[_from][_to]['weight'],min_ways[_from][_to]['way']])
    return near_ways

def search_near_ways_there_and_back(min_ways,max_weight,from_list,to_list):
    near_ways = []
    for _from in from_list:
        for _to in to_list:
            if _from in min_ways.keys() and _to in min_ways[_from].keys() and _to in min_ways.keys() and _from in min_ways[_to].keys():
                weight = min_ways[_from][_to]['weight'] + min_ways[_to][_from]['weight']
                if weight <= max_weight and _from != _to:
                    near_ways.append([_from, _to, weight, min_ways[_from][_to]['way'], min_ways[_to][_from]['way']])
    return near_ways

def search_minmax_way(min_ways,from_list,to_list):
    obj = 'start'
    minmax_way = {obj:{'weight':float('inf')}}
    for _from in from_list:
        weight = 0
        for _to in to_list:
            if _from in min_ways.keys() and _to in min_ways[_from].keys():
                if min_ways[_from][_to]['weight'] > weight:
                    to = _to
                    weight = min_ways[_from][_to]['weight']
        if weight != 0 and weight < minmax_way[obj]['weight']:
            minmax_way = {_from:{'to':to,'weight':weight}}
            obj = _from
    if obj != 'start':
        minmax_way[obj]['way'] = min_ways[obj][minmax_way[obj]['to']]['way'].copy()
    
    return minmax_way

def search_minmax_way_there_and_back(min_ways,from_list,to_list):
    obj = 'start'
    minmax_way = []
    for _from in from_list:
        if _from in min_ways.keys():
            weight = 0
            for _to in to_list:
                if _to in min_ways[_from].keys() and _to in min_ways.keys() and _from in min_ways[_to].keys():
                    w0 = min_ways[_from][_to]['weight'] + min_ways[_to][_from]['weight']
                    if w0 > weight:
                        to = _to
                        weight = w0
                    if minmax_way == [] or weight < minmax_way[2]: 
                        obj = _from 
                        minmax_way = [obj,to,weight]
      
    if  obj != 'start':
        to = minmax_way[1]
        minmax_way.append(min_ways[obj][to]['way'])
        minmax_way.append(min_ways[to][obj]['way'])
        minmax_way = [minmax_way]
    
    return minmax_way

def search_min_distance_to_node(min_ways,from_list,to_list,mean=False, func = sum):
    #Для какого объекта инфраструктуры сумма кратчайших расстояний от него до всех домов минимальна
    min_sum = float('inf')
    ways = []
    obj = ''
    for _from in from_list:
        if _from in min_ways.keys():
            _sum_elem = []
            for _to in to_list:
                if _to in min_ways[_from].keys() and _from != _to:                  
                    _sum_elem.append(min_ways[_from][_to]['weight'])
            if _sum_elem != [] and func(_sum_elem) < min_sum:
                min_sum = func(_sum_elem)
                obj = _from
                
    for _to in to_list:
        if obj != _to and obj in min_ways.keys() and _to in min_ways[obj].keys():
            ways.append([obj,_to, min_ways[obj][_to]['weight'], min_ways[obj][_to]['way']])

    return obj,min_sum,ways

def search_min_weight_to_node(min_ways,from_list,to_list, func = sum):
    #Для какого объекта инфраструктуры построенное дерево кратчайших путей имеет минимальный вес.
    min_sum = float('inf')
    obj = ''
    for _from in from_list:
        if _from in min_ways.keys():
            _sum_elem = []
            edges_set = set()        
            for _to in to_list:
                if _to in min_ways[_from].keys() and _from != _to:
                    edges_set.add(_from + ' ' + _to)
            for elem in list(edges_set):
                i,j = elem.split(' ')
                _sum_elem.append(min_ways[i][j]['weight'])
            
            if _sum_elem != [] and func(_sum_elem) < min_sum:
                min_sum = func(_sum_elem)
                obj = _from
    ways = []
    for _to in to_list:
        if obj != _to and obj in min_ways.keys() and _to in min_ways[obj].keys():
            ways.append([obj,_to, min_ways[obj][_to]['weight'], min_ways[obj][_to]['way']])
    
    return obj,min_sum,ways


def draw_ways_on_graph(G,nodes,ways,filename='', draw_network=True, way_color='b', node_size=15):
    if type(ways) == dict:
        ways = [way['way'] for way in ways.values()]
    elif type(ways) == list:
        ways = [way[3] for way in ways]
    edgelist,nodedict,start,end = [],set(),[],[]
    for way in ways:
        start.append(way[0])
        end.append(way[-1])
        for edge in zip(way[:-1],way[1:]):
            edgelist.append(edge)
            nodedict.add(edge[1])
    nodedict = list(nodedict)
    
    fig = plt.gcf()
    fig.set_size_inches(16,22,forward = True)
    if(draw_network):
        nx.draw_networkx(G, pos = nodes, node_size = 0, width = 0.1, with_labels = False, arrows  = False)
  
    nx.draw_networkx_edges(G.subgraph(start+nodedict),pos = nodes,edgelist = edgelist,edge_color=way_color,width = 0.4,arrows  = False)
    nx.draw_networkx_nodes(G.subgraph(start),pos = nodes,nodelist = start,node_color='g',node_size=node_size)
    nx.draw_networkx_nodes(G.subgraph(end),pos = nodes,nodelist = end,node_color='b',node_size=node_size)
    
    if filename != '':
        plt.savefig(filename,dpi=1000)
        fig.clear()

def draw_graph(G,nodes,filename=''):
    fig = plt.gcf()
    fig.set_size_inches(16,22,forward = True)
    nx.draw_networkx(G, pos = nodes, node_size = 0, width = 0.1, with_labels = False, arrows  = False)
    if filename != '':
        plt.savefig(filename,dpi=1000)
        fig.clear()

def load_object(G,houses_id,obj_id,nodes,obj,osm_file='orig_graph.osm'):
    print('loading')
    G,houses_id,obj_id,nodes = create_graph_from_osm(obj,osm_file)
    draw_graph(G,nodes)

def run_dijkstra(G,from_list,to_list):
    print('start')
    write_dijkstra_csv(G,'dijkstra1',from_list,to_list)
    write_dijkstra_csv(G,'dijkstra2',to_list,from_list)
    print('calculated')

def run_culc(G,houses_id,obj_id,from_list,to_list,h_count,obj_count,obj_weight):
    set_weights(G,obj_id,obj_weight)
    from_list = sample(houses_id, h_count)
    to_list = sample(obj_id, obj_count)
    print('start')
    write_dijkstra_csv(G,'dijkstra1',from_list,to_list)
    write_dijkstra_csv(G,'dijkstra2',to_list,from_list)
    print('calculated')
    return from_list, to_list
    
#clusters
def get_tree_and_sum(G, _from, target_list):
    tree = dijkstra(G, _from, target_list)
    filtered_tree = {k:v for k,v in tree.items() if v["weight"] != float('inf') and k in target_list}
    tree_sum = sum(list(map(lambda x: x['weight'], list(filtered_tree.values()))))
    
    edge_set = list()
    for target in filtered_tree:
        node_list = filtered_tree[target]['way']
        for node_number in range(len(node_list)-1):           
            new_edge = [node_list[node_number], node_list[node_number + 1]]
            if new_edge not in edge_set:
                edge_set.append(new_edge)
    tree_small_sum = sum(list(map(lambda x: G[x[0]][x[1]]['weight'], edge_set)))
    return filtered_tree, tree_sum, tree_small_sum

def get_cluster_matrix(G,some_houses):
    matrix = []
    for house_id in some_houses:
        pre_vector = dijkstra(G, house_id, some_houses)
        filtered_vector = {k:v for k,v in pre_vector.items() if v["weight"] != float('inf') and k in some_houses}
        sorted_vector = collections.OrderedDict(sorted(filtered_vector.items()))
        vector = list(map(lambda x: x['weight'], list(sorted_vector.values())))
        matrix.append(vector)
    return matrix

def cluster_to_id_list(cluster_index, cluster_cash, list_id, count_of_values):
    cluster_indexes = [cluster_index]
    while len(list(filter(lambda x: x >= count_of_values, cluster_indexes))) > 0:
        cluster_to_open = list(filter(lambda x: x >= count_of_values, cluster_indexes))
        
        for cluster in cluster_to_open:
            cluster_indexes.remove(cluster)
            cluster_indexes.append(cluster_cash[int(cluster) - count_of_values, 0] )
            cluster_indexes.append(cluster_cash[int(cluster) - count_of_values, 1] )
            
    return list(map(lambda x: int(list_id[int(x)]), cluster_indexes))

def get_centroid_coords(cluster_nodes):
    x = sum(list(map(lambda x: x['x'], cluster_nodes))) / len(cluster_nodes)
    y = sum(list(map(lambda x: x['y'], cluster_nodes))) / len(cluster_nodes)
    return y, x

def get_cluster_cash_and_dendrogram(matrix):
    matrix_np = np.array(matrix)
    cluster_cash = linkage(matrix_np, 'complete')
    fig = plt.figure(figsize=(25, 20))
    dn = dendrogram(cluster_cash)
    return cluster_cash, dn

def get_cluster_info (G,cluster_index, cluster_cash, cluster_items_id, nodes_data,obj_id, houses_id):    
    cluster_nodes = cluster_to_id_list(cluster_index, cluster_cash, cluster_items_id, len(cluster_cash) + 1)
    str_cluster_nodes = list(map(lambda x: str(x),cluster_nodes))
    list_with_coords = list(map(lambda x: {'x': float(nodes_data[str(x)][1]), 'y': float(nodes_data[str(x)][0])} ,cluster_nodes))
    centroid = get_centroid_coords(list_with_coords)
    centroid_id = find_nearest_node(G,nodes_data,centroid,houses_id,obj_id)
    tree, tree_sum, tree_min_sum = get_tree_and_sum(G, str(centroid_id), str_cluster_nodes)
    return {'tree': tree, 'tree_sum':tree_sum, 'tree_min_sum':tree_min_sum, 'centroid_id':centroid_id, 'nodes': str_cluster_nodes}

def open_next_cluster(cluster_cash, clusters, count_of_values):
    cluster_to_open = max(clusters)
    clusters.remove(cluster_to_open)
    clusters.append(cluster_cash[int(cluster_to_open) - count_of_values, 0] )
    clusters.append(cluster_cash[int(cluster_to_open) - count_of_values, 1] )
    return clusters

def find_nearest_node(G,nodes,coords_node, houses_id, obj_id):
    buffer = []
    min_dist = float('inf')
    for node_id  in G.nodes():
        check_min = (nodes[node_id][0] - coords_node[0])**2 + (nodes[node_id][1] - coords_node[1])**2
        if check_min < min_dist and node_id not in houses_id and node_id not in obj_id:
            min_dist = check_min
            buffer.append(node_id)
            if len(buffer) > len(houses_id) + len(obj_id) + 5:
                buffer = buffer[2:]
    return(buffer[-1])     

def get_clusters_info_task(G,cluster_cash, some_houses, nodes, obj_id):
    clusters = [int(cluster_cash[-1,0]), int(cluster_cash[-1,1])]
    clusters_info_2 = list(map(lambda x: get_cluster_info(G,x, cluster_cash, some_houses, nodes, obj_id, some_houses), clusters))
    clusters = open_next_cluster(cluster_cash, clusters, len(some_houses))
    clusters_info_3 = list(map(lambda x: get_cluster_info(G,x, cluster_cash, some_houses, nodes, obj_id, some_houses), clusters))
    clusters = open_next_cluster(cluster_cash, clusters, len(some_houses))
    clusters = open_next_cluster(cluster_cash, clusters, len(some_houses))
    clusters_info_5 = list(map(lambda x: get_cluster_info(G,x, cluster_cash, some_houses, nodes, obj_id, some_houses), clusters))
    return clusters_info_2, clusters_info_3,clusters_info_5

def draw_clusters(G,nodes, clusters_info, filename=''):
    colors = ['b','g','r','#FF00FF','c']
    for cluster_number in range(len(clusters_info)):
        is_draw = cluster_number == 0
        save_filename = '' if cluster_number != len(clusters_info) - 1 else filename 
        draw_ways_on_graph(G,nodes, clusters_info[cluster_number]['tree'], filename=save_filename, draw_network=is_draw, way_color=colors[cluster_number],node_size=8)
        
def count_centroids_tree_from_cluster(G,clusters_info, obj_id):
    centroids = [str(cluster_info['centroid_id']) for cluster_info in clusters_info]
    return get_tree_and_sum(G, str(obj_id), centroids)

def clusters_info_to_routes(clusters_info):
    routes_clusters_str = reduce(lambda x,y: x + y, list(map(lambda x: [v['way'] for k,v in x['tree'].items()], clusters_info)))
    return list(map(lambda x: list(map(lambda y: int(y), x)), routes_clusters_str))

def cluster_info_to_csv(clusters_info):
    data = clusters_info_to_routes(clusters_info)
    for cluster in clusters_info:
        tree_data = [cluster['tree_sum'], cluster['tree_min_sum']]
        data.append(tree_data)
        data.append([cluster['centroid_id']])
        data.append(cluster["nodes"])
    return data

def dijkstra_to_routes(res):
    routes_int = [res[key]['way'] for key in res]
    return list(map(lambda x: [int(value) for value in x], routes_int))

def tree_info_to_csv(tree, tree_sum, tree_min_sum):
    data = dijkstra_to_routes(tree)
    data.append([tree_sum])
    data.append([tree_min_sum])
    return data

def cluster_cash_to_csv(cluster_cash, count_of_items):
    data = cluster_cash.copy()
    for i in range(len(cluster_cash)):
        if cluster_cash[i,0] < count_of_items:
            data[i,0] = some_houses[int(data[i,0])]
        if cluster_cash[i,1] < count_of_items:
            data[i,1] = some_houses[int(data[i,1])]
    return data