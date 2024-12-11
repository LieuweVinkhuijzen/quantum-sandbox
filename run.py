import solveMaxCut
import os.path
import utils

n = 200
j = 0
results = []
solvername = 'greedy_7_12'
numBlocks = 5
sizeFactor = 0.5
while (True):
    filename = 'max-cut-catalogue/graph-' + str(n)  + '-' + str(j) + '.csv'
    if (not os.path.isfile(filename)):
        break
    print('Solving instance ' + str(j))
    graph = utils.importGraph(filename)

    (s, ut) = solveMaxCut.solveMaxCutGreedy7(graph, numBlocks, 1, 1.2)
    results.append((filename, n, ut))

    j += 1

# write the results to a csv file
import csv
csvfile = open('results/' + solvername + '.csv', 'w')
writer = csv.writer(csvfile, lineterminator='\n')
writer.writerows(results)
