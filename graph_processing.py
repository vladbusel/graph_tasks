import csv
from heapq import heappush, heappop
from itertools import count
from lxml import etree as et
from math import sin, cos, sqrt, atan2, radians
import matplotlib.pyplot as plt
import networkx as nx
from random import random, sample
import numpy as np

#on file upload
def create_graph_from_osm(osm_file='orig_graph.osm'): 
    f = open('houses.txt', 'r')
    houses_id = [house[:-1] for house in f.readlines()]
    f.close()
    f = open('med.txt', 'r')
    med_id = [med[:-1] for med in f.readlines()]   
    f.close()

    # f = open('firewatch.txt', 'r')
    # firewatch_id = [firewatch[:-1] for firewatch in f.readlines()]   
    # f.close()
    # f = open('market.txt', 'r')
    # market_id = [market[:-1] for market in f.readlines()]   
    # f.close()

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

    return G,houses_id,med_id,nodes
    #return G,houses_id,med_id,firewatch_id,market_id,nodes

#on params change
#def set_weights(G,houses_id,med_id,firewatch_id,market_id, h_max_weight, med_max_weight, fw_max_weight, market_max_weight):
def set_weights(G,med_id,med_max_weight):
    for m_id in med_id:
        w = 1 + random()*(med_max_weight-1)
        for node in G.successors(m_id):
            G[m_id][node]['weight'] *= w
        for node in G.predecessors(m_id):
            G[node][m_id]['weight'] *= w    
    
    # for fw_id in firewatch_id:
    #     w = 1 + random()*(fw_max_weight-1)
    #     for node in G.successors(fw_id):
    #         G[fw_id][node]['weight'] *= w
    #     for node in G.predecessors(fw_id):
    #         G[node][fw_id]['weight'] *= w
        
    # for m_id in market_id:
    #     w = 1 + random()*(market_max_weight-1)
    #     for node in G.successors(m_id):
    #         G[m_id][node]['weight'] *= w
    #     for node in G.predecessors(m_id):
    #         G[node][m_id]['weight'] *= w  
    

def dijkstra(G,_from,to_list = 'empty', max_dist = None):
    if to_list == 'empty':
        to_list = list(G.nodes())

    c,target_count = count(),0
    finish = len(to_list)
    heap, checked = [], {}
    checked[_from] = {'weight':0,'way':[_from]}
    heappush(heap, (0, next(c), _from))
    
    while heap:
        (d, _, v) = heappop(heap)
        if v in to_list:
            target_count += 1
        if target_count == finish:
            break
        for u in G.successors(v):
            _weight = checked[v]['weight'] + G[v][u]['weight']
            if max_dist is not None:
                if vu_dist > max_dist:
                    continue
            if u not in checked:
                checked[u] = {'weight': _weight, 'way': checked[v]['way'] + [u]}
                heappush(heap, (_weight, next(c), u))
            elif _weight < checked[u]['weight']:
                checked[u]['weight'] = _weight
                heappush(heap, (_weight, next(c), u))
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


def draw_ways_on_graph(G,nodes,ways,filename=''):
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
    nx.draw_networkx(G, pos = nodes, node_size = 0, width = 0.1, with_labels = False, arrows  = False)
  
    nx.draw_networkx_edges(G.subgraph(start+nodedict),pos = nodes,edgelist = edgelist,edge_color='b',width = 0.4,arrows  = False)
    nx.draw_networkx_nodes(G.subgraph(start),pos = nodes,nodelist = start,node_color='g',node_size=15)
    nx.draw_networkx_nodes(G.subgraph(end),pos = nodes,nodelist = end,node_color='b',node_size=15)
    
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




#tasks
def draw_clear_graph(osm_file='orig_graph.osm'):
    G,houses_id,med_id,nodes = create_graph_from_osm(osm_file)
    draw_graph(G,nodes)

def draw_min_to_obj(h_count, obj_count, obj_weight, osm_file='orig_graph.osm'):
    G,houses_id,med_id,nodes = create_graph_from_osm(osm_file)
    set_weights(G,med_id,obj_weight)
    from_list = sample(houses_id, h_count)
    to_list = sample(med_id, obj_count)  
    write_dijkstra_csv(G,'dijkstra1',from_list,to_list)
    write_dijkstra_csv(G,'dijkstra2',to_list,from_list)
    min_ways = read_dijkstra_csv('dijkstra1')
    min_ways.update(read_dijkstra_csv('dijkstra2'))
    min_oneways1 = search_min_oneways(min_ways,from_list,to_list)
    draw_ways_on_graph(G,nodes,min_oneways1)

def draw_min_from_obj(h_count, obj_count, obj_weight, osm_file='orig_graph.osm'):
    G,houses_id,med_id,nodes = create_graph_from_osm(osm_file)
    set_weights(G,med_id,obj_weight)
    from_list = sample(houses_id, h_count)
    to_list = sample(med_id, obj_count)  
    write_dijkstra_csv(G,'dijkstra1',from_list,to_list)
    write_dijkstra_csv(G,'dijkstra2',to_list,from_list)
    min_ways = read_dijkstra_csv('dijkstra1')
    min_ways.update(read_dijkstra_csv('dijkstra2'))
    min_oneways2 = search_min_oneways(min_ways,to_list,from_list)
    draw_ways_on_graph(G,nodes,min_oneways2)

def draw_min_there_and_back(h_count, obj_count, obj_weight, osm_file='orig_graph.osm'):
    G,houses_id,med_id,nodes = create_graph_from_osm(osm_file)
    set_weights(G,med_id,obj_weight)
    from_list = sample(houses_id, h_count)
    to_list = sample(med_id, obj_count)  
    write_dijkstra_csv(G,'dijkstra1',from_list,to_list)
    write_dijkstra_csv(G,'dijkstra2',to_list,from_list)
    min_ways = read_dijkstra_csv('dijkstra1')
    min_ways.update(read_dijkstra_csv('dijkstra2'))
    min_ways_there_and_back = search_min_ways_there_and_back(min_ways,from_list,to_list)
    draw_ways_on_graph(G,nodes,min_ways_there_and_back)

def draw_close_to_obj(h_count, obj_count, obj_weight, distance, osm_file='orig_graph.osm'):
    G,houses_id,med_id,nodes = create_graph_from_osm(osm_file)
    set_weights(G,med_id,obj_weight)
    from_list = sample(houses_id, h_count)
    to_list = sample(med_id, obj_count)  
    write_dijkstra_csv(G,'dijkstra1',from_list,to_list)
    write_dijkstra_csv(G,'dijkstra2',to_list,from_list)
    min_ways = read_dijkstra_csv('dijkstra1')
    min_ways.update(read_dijkstra_csv('dijkstra2'))
    near_ways1 = search_near_ways(min_ways,distance,from_list,to_list)
    draw_ways_on_graph(G,nodes,near_ways1)
    
def draw_close_from_obj(h_count, obj_count, obj_weight, distance, osm_file='orig_graph.osm'):
    G,houses_id,med_id,nodes = create_graph_from_osm(osm_file)
    set_weights(G,med_id,obj_weight)
    from_list = sample(houses_id, h_count)
    to_list = sample(med_id, obj_count)  
    write_dijkstra_csv(G,'dijkstra1',from_list,to_list)
    write_dijkstra_csv(G,'dijkstra2',to_list,from_list)
    min_ways = read_dijkstra_csv('dijkstra1')
    min_ways.update(read_dijkstra_csv('dijkstra2'))
    near_ways2 = search_near_ways(min_ways,distance,from_list,to_list)
    draw_ways_on_graph(G,nodes,near_ways2)

def draw_close_there_and_back(h_count, obj_count, obj_weight, distance, osm_file='orig_graph.osm'):
    G,houses_id,med_id,nodes = create_graph_from_osm(osm_file)
    set_weights(G,med_id,obj_weight)
    from_list = sample(houses_id, h_count)
    to_list = sample(med_id, obj_count)  
    write_dijkstra_csv(G,'dijkstra1',from_list,to_list)
    write_dijkstra_csv(G,'dijkstra2',to_list,from_list)
    min_ways = read_dijkstra_csv('dijkstra1')
    min_ways.update(read_dijkstra_csv('dijkstra2'))
    near_ways_there_and_back = search_near_ways_there_and_back(min_ways,distance,from_list,to_list)
    draw_ways_on_graph(G,nodes,near_ways_there_and_back)

def draw_minmax_to_obj(h_count, obj_count, obj_weight, osm_file='orig_graph.osm'):
    G,houses_id,med_id,nodes = create_graph_from_osm(osm_file)
    set_weights(G,med_id,obj_weight)
    from_list = sample(houses_id, h_count)
    to_list = sample(med_id, obj_count)  
    write_dijkstra_csv(G,'dijkstra1',from_list,to_list)
    write_dijkstra_csv(G,'dijkstra2',to_list,from_list)
    min_ways = read_dijkstra_csv('dijkstra1')
    min_ways.update(read_dijkstra_csv('dijkstra2'))
    minmax_way1 = search_minmax_way(min_ways,from_list,to_list)
    draw_ways_on_graph(G,nodes,minmax_way1)

def draw_minmax_from_obj(h_count, obj_count, obj_weight, osm_file='orig_graph.osm'):
    G,houses_id,med_id,nodes = create_graph_from_osm(osm_file)
    set_weights(G,med_id,obj_weight)
    from_list = sample(houses_id, h_count)
    to_list = sample(med_id, obj_count)  
    write_dijkstra_csv(G,'dijkstra1',from_list,to_list)
    write_dijkstra_csv(G,'dijkstra2',to_list,from_list)
    min_ways = read_dijkstra_csv('dijkstra1')
    min_ways.update(read_dijkstra_csv('dijkstra2'))
    minmax_way2 = search_minmax_way(min_ways,from_list,to_list)
    draw_ways_on_graph(G,nodes,minmax_way2)

def draw_minmax_there_and_back(h_count, obj_count, obj_weight, osm_file='orig_graph.osm'):
    G,houses_id,med_id,nodes = create_graph_from_osm(osm_file)
    set_weights(G,med_id,obj_weight)
    from_list = sample(houses_id, h_count)
    to_list = sample(med_id, obj_count)  
    write_dijkstra_csv(G,'dijkstra1',from_list,to_list)
    write_dijkstra_csv(G,'dijkstra2',to_list,from_list)
    min_ways = read_dijkstra_csv('dijkstra1')
    min_ways.update(read_dijkstra_csv('dijkstra2'))
    minmax_way_there_and_back = search_minmax_way_there_and_back(min_ways,from_list,to_list)
    draw_ways_on_graph(G,nodes,minmax_way_there_and_back)

def draw_min_distance(h_count, obj_count, obj_weight, osm_file='orig_graph.osm'):
    G,houses_id,med_id,nodes = create_graph_from_osm(osm_file)
    set_weights(G,med_id,obj_weight)
    from_list = sample(houses_id, h_count)
    to_list = sample(med_id, obj_count)  
    write_dijkstra_csv(G,'dijkstra1',from_list,to_list)
    write_dijkstra_csv(G,'dijkstra2',to_list,from_list)
    min_ways = read_dijkstra_csv('dijkstra1')
    min_ways.update(read_dijkstra_csv('dijkstra2'))
    min_distance_to_node,min_distance,min_distance_way = search_min_distance_to_node(min_ways,from_list,to_list,sum)
    draw_ways_on_graph(G,nodes,min_distance_way)

def draw_min_weight(h_count, obj_count, obj_weight, osm_file='orig_graph.osm'):
    G,houses_id,med_id,nodes = create_graph_from_osm(osm_file)
    set_weights(G,med_id,obj_weight)
    from_list = sample(houses_id, h_count)
    to_list = sample(med_id, obj_count)  
    write_dijkstra_csv(G,'dijkstra1',from_list,to_list)
    write_dijkstra_csv(G,'dijkstra2',to_list,from_list)
    min_ways = read_dijkstra_csv('dijkstra1')
    min_ways.update(read_dijkstra_csv('dijkstra2'))
    min_weight_to_node,min_weight,min_weight_way = search_min_weight_to_node(min_ways,from_list,to_list,sum)
    draw_ways_on_graph(G,nodes,min_weight_way)