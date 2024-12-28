import random
import csv
import networkx as nx
import numpy as np

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

def getNumVerticesNx(graph : nx.graph.Graph):
    n = 0
    for edge in graph.edges():
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

# 'vertices' is a list of integers in [0 .. n-1]
# returns a list of lists of integers, i.e., a list of 'blocks'
# the blocks partition the list
# Each block has the same size, to the extent possible
def divideIntoRandomEqualPartition(vertices, numBlocks):
    partition = []
    for j in range(numBlocks):
        partition.append([])
    # Ensure that each block gets some
    unprocessedVerticesList = convertSetToListOfIntegers(vertices)
    random.shuffle(unprocessedVerticesList)
    for i in range(len(unprocessedVerticesList)):
        partition[i % numBlocks].append(unprocessedVerticesList[i])
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

def convertListToSet(a : list[int]) -> set[int]:
    b : set[int] = set()
    for i in range(len(a)):
        if a[i] == 1:
            b.add(i)
    return b

# Returns the characteristic vector of the set of numbers, i.e.,
# returns a list A of length n such that A[k] == 1 if k in a, and A[k] == 0 if k not in a
def convertSetToList(a : set[int], n : int) -> list[int]:
    b = n * [ 0 ]
    for i in range(n):
        if a.__contains__(i):
            b[i] = 1
    return b

def convertSetToListOfIntegers(a : set[int]) -> list[int]:
    b = []
    for k in a:
        b.append(k)
    return b

# sets each element of the list 'target' to a random 0/1 value
# modifies in-place to prevent dynamic memory allocation
def setListToRandomBooleans(target):
    for v in range(0, len(target)):
        target[v] = random.randint(0, 1)

def setArrayToRandomBooleans(target : np.ndarray):
    len = target.shape[0]
    for v in range(len):
        target[v] = random.randint(0, 1)

# Returns a list of length n, containing randomly chosen 0/1 values
def getListOfRandomBooleans(n : int):
    l = n * [ 0 ]
    setListToRandomBooleans(l)
    return l

# returns a numpy.array of random booleans
def getArrayOfRandomBooleans(n : int):
    a = np.ndarray([n], dtype=np.uint8)
    setArrayToRandomBooleans(a)
    return a

def writeSetToFile(f, s : set[int]):
    for e in s:
        f.write(f'{e},')

# n : largest index of any vertex in the graph
def chooseRandomVertex(graph : nx.graph.Graph, n : int):
    while True:
        v = random.randint(0, n-1)
        if graph.__contains__(v):
            return v

def flipVertex(solution, vertex : int):
    solution[vertex] = 1 - solution[vertex]