from dwave.system import DWaveSampler, EmbeddingComposite
import networkx as nx
from dimod.reference.samplers import ExactSolver
import dwave_networkx as dnx
import utils
import solveMaxCut

# This example solves minimum-vertex-cover on a small star graph

# Define a star graph
s5 = nx.star_graph(4)

# Solve classically locally, on the cpu
classicalSampler = ExactSolver()
# DWave_Sampler = EmbeddingComposite(DWaveSampler())
# We choose the classical sampler
solver = classicalSampler

# solution = dnx.min_vertex_cover(s5, solver)
# print(solution)

#    Example from DWave documentation
# quantumSampler = EmbeddingComposite(DWaveSampler())
# # Now we query DWave. This is an API call over the internet, so it may take a while
# resultFromDwave = dnx.min_vertex_cover(s5, quantumSampler)
# print(resultFromDwave)

# sample a maximum cut from the Chimera graph
chimera = dnx.chimera_graph(2,1,4)
print('Chimera graph: ')
print(chimera)
chimera_subset = utils.getRandomSubgraph(chimera, 0.5)
print('Chimera subgraph: ')
print(chimera_subset)
result_from_dwave = dnx.maximum_cut(chimera_subset, solver)
print(result_from_dwave)
utility = solveMaxCut.evaluateSolutionNx(chimera_subset, result_from_dwave)
print(f'utility = {utility}')
