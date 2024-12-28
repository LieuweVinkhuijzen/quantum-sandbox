import argparse
import string
import random
import csv
import sys
import write_graph_utils

def printUsageInstruction():
    # print('Usage: ' + sys.argv[0] + '  <number of vertices>  <number of edges>  <name of output csv file>');
    print('Usage: ' + sys.argv[0] + '  <number of vertices>  <number of edges>  <number of graphs>');

if (len(sys.argv) < 4):
    printUsageInstruction();
    exit();
numVertices = sys.argv[1];
try:
    numVertices = int(numVertices);
except:
    printUsageInstruction();
    print('Number of vertices is not an integer');
    exit()
numEdges = sys.argv[2];
try:
    numEdges = int(numEdges);
except:
    printUsageInstruction();
    print('Number of edges is not a number');
    exit()
numGraphs = sys.argv[3]
try:
    numGraphs = int(numGraphs)
except:
    printUsageInstruction()
    print('Number of graphs is not an integer')
    exit()
# outputFileName = sys.argv[3]

def getTwoDistinctRandomNumbers(low, high):
    a = random.randint(low, high)
    b = random.randint(low, high)
    while (a == b):
        b = random.randint(low, high)
    return (a, b);

def getRandomGraph(numVertices, numEdges):
    graph = set()
    if (numEdges > numVertices * (numVertices - 1) / 2):
        print('Impossible to have a graph with ' + str(numVertices) + ' vertices and ' + str(numEdges) + ' edges');
        return graph;
    while (len(graph) < numEdges):
        [v1, v2] = getTwoDistinctRandomNumbers(0, numVertices - 1);
        if (v2 < v1):
            temp = v1
            v1 = v2
            v2 = temp
        # print('Edge: (' + str(v1) + ', ' + str(v2) + ')');
        graph.add((v1, v2))
    return graph

for g in range(0, numGraphs):
    graph = getRandomGraph(numVertices, numEdges);
    write_graph_utils.printGraphToCsv(graph, 'max-cut-catalogue/graph-' + str(numVertices) + '-' + str(g) + '.csv')
