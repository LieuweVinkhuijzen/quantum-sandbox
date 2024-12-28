from dwave.system import DWaveSampler, EmbeddingComposite
import networkx as nx
from dimod.reference.samplers import ExactSolver
import dwave_networkx as dnx
import utils
import solveMaxCut
import csvutils
import time
import random

random.seed(2)

# This example solves minimum-vertex-cover on a small star graph

# Define a star graph
# s5 = nx.star_graph(4)

# Solve classically locally, on the cpu

# solution = dnx.min_vertex_cover(s5, solver)
# print(solution)

#    Example from DWave documentation
# quantumSampler = EmbeddingComposite(DWaveSampler())
# # Now we query DWave. This is an API call over the internet, so it may take a while
# resultFromDwave = dnx.min_vertex_cover(s5, quantumSampler)
# print(resultFromDwave)

# sample a maximum cut from the Chimera graph
# chimera = dnx.chimera_graph(2,1,4)
width = 16
height = 16
depth = 4
j = 0
# chimera_filename = f'./max-cut-catalogue/chimera_subgraphs/chimera_subgraph-{width}-{height}-{depth}-{j}.csv'
# chimera_filename = f'./max-cut-catalogue/random_graphs/graph-200-0.csv'
chimera_filename = f'./max-cut-catalogue/random_graphs/graph-2000-0.csv'
chimera_subgraph = utils.importGraphNx(chimera_filename)
n = utils.getNumVerticesNx(chimera_subgraph)
print('Chimera subgraph: ')
print(chimera_subgraph)

strategy   = 'greedy-12'
numBlocks  = 10
sizeFactor = 0.2
budget     = 30

if strategy == 'dwave':
    print('Setting up sampler...')
    solver = EmbeddingComposite(DWaveSampler())
    print('Calling D-Wave API...')
    result_from_dwave = dnx.maximum_cut(chimera_subgraph, solver)
    solution_set = result_from_dwave
    print('Result from dwave:\n')
    print(result_from_dwave)
if strategy == 'dwave-classical':
    print('We choose the classical, brute-force solver')
    # We choose the classical sampler
    solver = ExactSolver()
    print('Calling D-Wave API...')
    result_from_dwave = dnx.maximum_cut(chimera_subgraph, solver)
    solution_set = result_from_dwave
    print('Result from dwave:\n')
    print(result_from_dwave)
elif strategy == 'random':
    print(f'Num vertices: {n}')
    random_solution = utils.getListOfRandomBooleans(n)
    # print(f'Solution = {random_solution}')
    start_optimize_time = time.perf_counter()
    utility = solveMaxCut.evaluateSolutionNxList(chimera_subgraph, random_solution)
    (optimized_solution, optimized_utility) = solveMaxCut.optimizeAllEdgesGreedilyNx_v2(chimera_subgraph, random_solution, utility)
    end_time = time.perf_counter()
    optimize_time = (end_time - start_optimize_time) * 10**6
    print(f'Edge-optimization time: {optimize_time:.0f} microseconds, {optimize_time / 10**6:.2f} seconds ')
    solution_set = utils.convertListToSet(optimized_solution)
    # print(f'Solution set = {solution_set}')
    solution_list = utils.convertSetToList(solution_set, n)
    # print(f'Solution list = {solution_list}')
elif strategy == 'greedy-12':
    print('Solving graph using greedy-12 strategy')
    start_time = time.perf_counter()
    (greedy_solution, greedy_utility) = solveMaxCut.solve_max_cut_greedy_12(chimera_subgraph,numBlocks, budget, sizeFactor)
    greedy_utility_verified = solveMaxCut.evaluate_solution_array(chimera_subgraph, greedy_solution)
    start_optimize_time = time.perf_counter()
    (optimized_solution, optimized_utility) = solveMaxCut.optimizeAllEdgesGreedilyNx_v2(chimera_subgraph, greedy_solution, greedy_utility_verified)
    end_time = time.perf_counter()
    optimized_utility_verified = solveMaxCut.evaluate_solution_array(chimera_subgraph, optimized_solution)
    solution_set = utils.convertListToSet(optimized_solution)
    elapsed_time = (end_time - start_time)
    optimize_time = (end_time - start_optimize_time)
    greedy_time   = (start_optimize_time - start_time)

    print(f'Total optimize time:   {elapsed_time:.03f} seconds')
    print(f'Greedy optimize time:  {greedy_time:0.3f} seconds')
    print(f'Edge  Optimize time:   {optimize_time:0.3f} seconds ')
    print(f'greedy utility             = {greedy_utility}')
    print(f'greedy utility verified    = {greedy_utility_verified}')
    print(f'optimized utility          = {optimized_utility}')
    print(f'optimzied utility verified = {optimized_utility_verified}')
    if (greedy_utility != greedy_utility_verified):
        print(' <<  ERROR  >>  greedy utility was not computed correctly')
    if optimized_utility != optimized_utility_verified:
        print(' <<  ERROR  >>  optimized utiltiy was not computed correctly')
    

# Compute the utility of the solution found
utility = solveMaxCut.evaluateSolutionNxSet(chimera_subgraph, solution_set)
print(f'utility = {utility}')

# Write the utility to a file
resultsfile = open('outputs_dwave/outputs_dwave.csv', 'a')
resultsfile.write(f'{chimera_filename},{strategy},{utility}\n')
resultsfile.close()

# Write the solution to a file
solutionfile = open('outputs_dwave/solutions_dwave.csv', 'a')
solutionfile.write(f'{chimera_filename},{strategy},')
utils.writeSetToFile(solutionfile, solution_set)
solutionfile.write('\n')
solutionfile.close()