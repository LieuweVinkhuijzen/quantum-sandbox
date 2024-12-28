import dwave_networkx as dnx
import utils
import write_graph_utils as write

num_graphs = 10
print(f'Generating {num_graphs} graphs')
width : int = 2
height : int = 1
depth : int = 4
chimera = dnx.chimera_graph(width, height, depth)
for j in range(num_graphs):
    subgraph = utils.getRandomSubgraph(chimera, 0.5)
    write.printGraphToCsvNx(subgraph, f'max-cut-catalogue/chimera_subgraphs/chimera_subgraph-{width}-{height}-{depth}-{j}.csv')
print('Graph generation complete.')