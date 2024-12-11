import csv
import networkx as nx

def convertSetOfArraysToArray(sets):
    data = []
    for e in sets:
        data.append(e)
    return data;

def convertEdgeViewToListOfLists(edges : nx.classes.reportviews.EdgeView) -> list[list[int]]:
    out : list[list[int]] = []
    for e in edges:
        out.append([ e[0], e[1] ])
    return out

def printGraphToCsv(graph, filename):
    data = convertSetOfArraysToArray(graph)
    csvfile = open(filename, 'w', newline='\n');
    writer = csv.writer(csvfile);
    writer.writerows(data);
    csvfile.close()

def printGraphToCsvNx(graph : nx.graph.Graph, filename):
    data = convertEdgeViewToListOfLists(graph.edges())
    csvfile = open(filename, 'w', newline='\n')
    writer = csv.writer(csvfile)
    writer.writerows(data)
    csvfile.close()

def getEdgeListFromEdgeView(edgeView) -> list[list[int]]:
    edgeList : list[list[int]] = []
    for edge in edgeView:
        edgeList.append(edge)
    return edgeList