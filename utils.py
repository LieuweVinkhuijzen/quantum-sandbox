import random
import csv
import networkx as nx

def importGraph(filename):
    file = open(filename, 'r')
    reader = csv.reader(file, delimiter=',')
    data = []
    for line in reader:
        linelist = []
        for element in line:
            linelist.append(int(element))
        data.append(linelist)
    return data

def importGraphNx(filename) -> nx.graph.Graph:
    file = open(filename, 'r')
    reader = csv.reader(file, delimiter=',')
    graph : nx.graph.Graph = nx.graph.Graph()
    for line in reader:
        linelist = []
        for element in line:
            linelist.append(int(element))
        graph.add_edge(linelist[0], linelist[1])
    return graph


def getNumVertices(graph):
    n = 0
    for edge in graph:
        n = max(n, edge[0])
        n = max(n, edge[1])
    return n + 1

# Randomly divide the vertices into k partitions
def getRandomPartition(n, k):
    partition = []
    for j in range(0, k):
        partition.append([])
    for v in range(0, n):
        # Choose a partition
        partitionId = random.randint(0, k-1)
        # Add the vertex v to the chosen partition
        partition[partitionId].append(v)
    return partition

# 'vertices' is a list of integers in [0.. n -1]
def divideIntoRandomPartition(vertices, numBlocks):
    partition = []
    for j in range(0, numBlocks):
        partition.append([])
    for v in vertices:
        block = random.randint(0, numBlocks-1)
        # Add the vertex v to the chosen partition
        partition[block].append(v)
    return partition

# to do
def getRandomSubgraph(graph : nx.graph.Graph, probability) -> nx.graph.Graph:
    subgraph : nx.graph.Graph = nx.graph.Graph()
    # set the nodes
    for v in graph.nodes():
        subgraph.add_node(v)
    for e in graph.edges():
        if random.randint(0, 1000) >= int(1000.0 * probability):
            subgraph.add_edge(*e)
    return subgraph

