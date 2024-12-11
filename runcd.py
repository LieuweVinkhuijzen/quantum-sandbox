import communityDetection
import os.path
import utils

n = 200
numBlocks = 8
results = []
j = 0
while(True):
    filename = 'max-cut-catalogue/graph-' + str(n)  + '-' + str(j) + '.csv'
    if (not os.path.isfile(filename)):
        break
    graph = utils.importGraph(filename)

    (s, ut) = communityDetection.detectCommunities_v2(graph, numBlocks)
    results.append((filename, n, ut))

    j += 1
