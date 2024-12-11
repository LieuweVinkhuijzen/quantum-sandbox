import dwave_networkx as dnx
import utils
import write_graph_utils as write

num_graphs = 100
print(f'Generating {num_graphs} graphs')
chimera = dnx.chimera_graph(4,4,4)
for j in range(num_graphs):
    subgraph = utils.getRandomSubgraph(chimera, 0.5)
    write.printGraphToCsvNx(subgraph, f'max-cut-catalogue/chimera_subgraphs/chimera_subgraph-4-4-4-{j}.csv')
print('Graph geneartion complete.')