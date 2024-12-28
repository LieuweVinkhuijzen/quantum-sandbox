import csv
import networkx as nx

def getDataFromCsvFile(filename) -> list[tuple[int, int]]:
    file = open(filename, 'r')
    reader = csv.reader(file)
    data : list[tuple[int, int]] = []
    for line in reader:
        for i in range(0, len(line)):
            try:
                x = int(line[i])
                line[i] = x
            except:
                pass
        data.append(line)
    return data

def getColumn(mat, col):
    column = []
    for row in mat:
        column.append(row[col])
    return column

def getDifference(list1, list2):
    diff = []
    for j in range(min(len(list1), len(list2))):
        diff.append(list1[j] - list2[j])
    return diff

def getNXGraphFromCsvFile(filename) -> nx.graph.Graph:
    edgeSet = getDataFromCsvFile(filename)
    nxGraph : nx.graph.Graph = nx.graph.Graph()
    for edge in edgeSet:
        nxGraph.add_edge(edge[0], edge[1])
    return nxGraph