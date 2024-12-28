import time
import networkx as nx
import math
import random
import utils
from typing import Set, List
import numpy

class Profile:
    evaluate_solution_time = 0.0
    get_summary_graph_time = 0.0
    optimize_v4_time = 0.0
    optimize_v5_time = 0.0
    evaluate_after_flipping_partition_time = 0.0
    copying_time = 0.0
    optimize_edges_time = 0.0
    optimize_edges_v2_time = 0.0
    optimize_edges_v3_time = 0.0
    optimize_edges_v4_time = 0.0
    copy_to_candidate = 0.0
    get_neighbourhood_time = 0.0
    optimize_edges_v4_calls = 0
    def print():
        print(f'get_summary_graph ~ {Profile.get_summary_graph_time * 1000:.2f} ms  \
AFP ~ {       Profile.evaluate_after_flipping_partition_time * 1000:.0f} ms  \
eval ~ {      Profile.evaluate_solution_time * 1000:.0f} ms  \
getnbh ~ {      Profile.get_neighbourhood_time * 1000:.0f} ms  \
optimize4 ~ {  Profile.optimize_v4_time * 1000:.0f} ms  \
optimize5 ~ {  Profile.optimize_v5_time * 1000:.0f} ms  \
copy ~ {      Profile.copying_time * 1000:.2f} ms  \
copy ~ {      Profile.copy_to_candidate * 1000:.2f} ms  \
optEdges ~ {  Profile.optimize_edges_time * 1000:.2f} ms  \
optEdges2 ~ { Profile.optimize_edges_v2_time * 1000:.2f} ms  \
optEdges3 ~ { Profile.optimize_edges_v3_time * 1000:.2f} ms  \
optEdges4 ~ { Profile.optimize_edges_v4_time * 1000:.2f} ms  \
 ~ {          Profile.optimize_edges_v4_time * 1000 / (Profile.optimize_edges_v4_calls + 1):.2f} ms/call  \
({            Profile.optimize_edges_v4_calls} calls)')

def time_since(time_start : float) -> float:
    return time.perf_counter() - time_start

# graph is a list of lists
# solution is list of n Boolean values
# returns number of edges in the cut, i.e., between the two components
def evaluateSolution(graph, solution):
    utility = 0
    for edge in graph:
        if (solution[edge[0]] != solution[edge[1]]):
            utility += 1
    return utility;

def evaluateSolutionNxSet(graph : nx.graph.Graph, solution : Set[int]):
    utility : int = 0
    for edge in graph.edges:
        # Add a point of utility if the two vertices are in different sets
        c0 : bool = solution.__contains__(edge[0])
        c1 : bool = solution.__contains__(edge[1])
        if (c0 != c1):
            utility += 1
        # print(f'edge {edge[0]}-{edge[1]}; c0={c0} c1={c1};  utility := {utility}')
    return utility

# works for lists and numpy.ndarray
def evaluateSolutionNxList(graph : nx.graph.Graph, solution):
    time_start = time.perf_counter()
    utility = 0
    for edge in graph.edges:
        if solution[edge[0]] != solution[edge[1]]:
            utility += 1
    Profile.evaluate_solution_time += time.perf_counter() - time_start
    return utility

def evaluate_solution_array(graph : nx.graph.Graph, solution : numpy.ndarray) -> int:
    time_start = time.perf_counter()
    utility = 0
    for edge in graph.edges:
        if solution[edge[0]] != solution[edge[1]]:
            utility += 1
    Profile.evaluate_solution_time += time.perf_counter() - time_start
    return utility

def verify_utility(graph : nx.graph.Graph, solution, suspectedUtility : int):
    verifiedUtility = evaluateSolutionNxList(graph, solution)
    if verifiedUtility != suspectedUtility:
        print(f' <<  ERROR  >>  verified utility != expected utility:  {verifiedUtility} != {suspectedUtility}')
        return False
    return True

# copies data from one list to another list
def copy_data_inplace(list_to, list_from):
    if len(list_to) != len(list_from):
        print('  <<  ERROR  >>  lists have unequal sizes')
    for i in range(len(list_to)):
        list_to[i] = list_from[i]

# copy one 1-d numpy array to another, without allocating new memory
def copy_data_inplace_np(list_to : numpy.ndarray, list_from : numpy.ndarray):
    time_start = time.perf_counter()
    for i in range(list_to.shape[0]):
        list_to[i] = list_from[i]
    Profile.copying_time += time.perf_counter() - time_start

def get_inverse_partition(partition):
    inverse_partition = {}
    for i in range(len(partition)):
        for v in partition[i]:
            inverse_partition[v] = i
    return inverse_partition

# Naive solution to maxcut:
#   guess a random solution. No heuristics or clever tricks
#   the best of many attempts is returned
def solveMaxCutRandom(graph, attempts):
    n = utils.getNumVertices(graph)
    solution = n * [ 0 ]
    bestUtility = len(graph)
    bestSolution = solution
    for attempt in range(0, attempts):
        utils.setListToRandomBooleans(solution)
        utility = evaluateSolution(graph, solution)
        # print('Found random solution: ' + str(solution))
        # print('Solution penalty = ' + str(utility))
        if (utility > bestUtility):
            # print("That's better than before, so we adopt it")
            bestSolution = solution
            bestUtility = utility
    return (bestSolution, bestUtility)

def solveMaxCutGreedy1(graph):
    n = utils.getNumVertices(graph)
    m = len(graph)
    solution = n * [ 0 ]
    bestUtility = m # assume, worst case, that all edges are in violation
    utils.setListToRandomBooleans(solution)
    bestUtility = evaluateSolution(graph, solution)
    bestSolution = solution
    # for each vertex, see if changing it improves the solution
    # TODO this step can be made faster: we can quickly check whether v has more edges in "0" than in "1"
    for v in range(0, n):
        # Try moving vertex v to the other partition; keep the new solution if it's better
        solution[v] = 1 - solution[v]
        utility = evaluateSolution(graph, solution)
        if (utility > bestUtility):
            bestSolution = solution
            bestUtility = utility
        else:
            # undo the flip operation
            solution[v] = 1 - solution[v]
    # print('[Greedy] found solution ' + str(bestSolution))
    return (bestSolution, bestUtility)

def solveMaxCutGreedy2(graph):
    n = utils.getNumVertices(graph)
    m = len(graph)
    solution = n * [ 0 ]
    bestUtility = m # assume, worst case, that all edges are in violation
    utils.setListToRandomBooleans(solution)
    bestUtility = evaluateSolution(graph, solution)
    bestSolution = solution
    # for each vertex, see if changing it improves the solution
    # TODO this step can be made faster: we can quickly check whether v has more edges in "0" than in "1"
    solutionImprovedThisStep = True
    while (solutionImprovedThisStep):
        solutionImprovedThisStep = False
        for v in range(0, n):
            # Try moving vertex v to the other partition; keep the new solution if it's better
            solution[v] = 1 - solution[v]
            utility = evaluateSolution(graph, solution)
            if (utility > bestUtility):
                bestSolution = solution
                bestUtility = utility
                solutionImprovedThisStep = True
            else:
                # undo the flip operation
                solution[v] = 1 - solution[v]
    # print('[Greedy] found solution ' + str(bestSolution))
    return (bestSolution, bestUtility)

# Returns a list with m random elements, each element j has 0 <= j < n
def chooseRandomList(n, m):
    s = []
    for i in range(0, m):
        j = random.randint(0, n - 1)
        while (s.__contains__(j)):
            j = (j+1) % n
        s.append(j)
    return s

# In each round,
# Choose a random small set of vertices, and flip it
def solveMaxCutGreedy3(graph):
    n = utils.getNumVertices(graph)
    m = len(graph)
    maxSwapSize = int(math.log2(n))
    budget = 10*m # each edge can be changed 10 times (which is not a lot, imo)
    solution = n * [ 0 ]
    utils.setListToRandomBooleans(solution)
    bestUtility = evaluateSolution(graph, solution)
    for round in range(0, budget):
        # Guess a set of k vertices
        swapSetSize = random.randint(1, maxSwapSize)
        swapSet = chooseRandomList(n, swapSetSize)
        # Swap the verices in list
        for v in swapSet:
            solution[v] = 1 - solution[v]
        utility = evaluateSolution(graph, solution)
        if (utility > bestUtility):
            bestUtility = utility
        else:
            # undo the flip operation
            for v in swapSet:
                solution[v] = 1 - solution[v]
    # print('[Greedy] found solution ' + str(bestSolution))
    return (solution, bestUtility)

# Choose a random small set of vertices to flip
def greedy3Round(graph, startingSolution, n, maxSwapSize):
    # Guess a set of k vertices
    solution = startingSolution.copy()
    swapSetSize = random.randint(1, maxSwapSize)
    swapSet = chooseRandomList(n, swapSetSize)
    # Swap the verices in list
    for v in swapSet:
        solution[v] = 1 - solution[v]
    utility = evaluateSolution(graph, solution)
    return (solution, utility)

def optimizeVertex(graph, startingSolution, startingUtility, v):
    solution = startingSolution.copy()
    solution[v] = 1 - solution[v]
    utility = evaluateSolution(graph, solution)
    if (utility < startingUtility):
        return (solution, utility)
    else:
        return (startingSolution, startingUtility)

# optimize one random vertex by trying out putting it in both partitions
def optimizeRandomVertex(graph, startingSolution : int, startingUtility : int, n : int):
    v = random.randint(0, n - 1)
    return optimizeVertex(graph, startingSolution, startingUtility, v)

def optimizeAllVerticesGreedily(graph, startingSolution, n : int, startingUtility : int):
    foundImprovement = True
    while (foundImprovement):
        for v in range(0, n):
            foundImprovement = False
            (startingSolution, newUtility) = optimizeVertex(graph, startingSolution, startingUtility, v)
            if (newUtility > startingUtility):
                foundImprovement = True
    return (startingSolution, startingUtility)

# for each edge, try out all 4 ways of shuffling the two vertices of that edge
def optimizeAllEdgesGreedilyNx(graph : nx.graph.Graph, solution, startingUtility : int):
    bestUtility = startingUtility
    bestSolution = solution.copy()
    for edge in graph.edges:
        v0 = edge[0]
        v1 = edge[1]
        # TODO find a more efficient way to compute the utility function, without visiting ALL the edges
        # Try out flipping v0
        solution[v0] = 1 - solution[v0]
        utility = evaluateSolutionNxList(graph, solution)
        if (utility > startingUtility):
            bestUtility = utility
            bestSolution = solution.copy() # TO DO not very efficient, can be easily made better
        # Try out flipping v0 and v1
        solution[v1] = 1 - solution[v1]
        utility = evaluateSolutionNxList(graph, solution)
        if (utility > startingUtility):
            bestUtility = utility
            bestSolution = solution.copy() # TO DO not very efficient, can be easily made better
        # Try out flipping v1, i.e., flip v0 "back" to its original state
        solution[v0] = 1 - solution[v0]
        utility = evaluateSolutionNxList(graph, solution)
        if (utility > startingUtility):
            bestUtility = utility
            bestSolution = solution.copy() # TO DO not very efficient, can be easily made better
        # Lastly, flip v1 back to its original state, to guarantee no change happens to startingSolution
        # which was passed by reference
        solution[v1] = 1 - solution[v1]
    return (bestSolution, bestUtility)

# Same as above, but optimized to death
def optimizeAllEdgesGreedilyNx_v2(graph, solution, startingUtility : int):
    bestUtility = startingUtility
    print(f'[optimize edges]  suspected utility before optimzing: {startingUtility}')
    verify_utility(graph, solution, startingUtility)
    print(f'[optimize edges]  Verified the utility.')
    bestSolution = solution.copy()
    # TODO for now we just copy the new solution into the best solution, but we should optimize this, because that's not good
    for edge in graph.edges:
        v0 = edge[0]
        v1 = edge[1]
        # Try out flipping v0
        # So, I need to figure out how many edges would be affected, and in what way,
        # if v0 were flipped. 
        # So, I gather up all the neighbours of v0, and for each neighbour I ask: is it in 0 or in 1?
        v0p0 = 0
        v0p1 = 0
        v1p0 = 0
        v1p1 = 0
        # Let's count, for each vertex, how many neighbours are in various partitions
        for v0n in graph.adj[v0]:
            if v0n == v0:
                print(f'Node {v0} is its own neighbour???')
            if bestSolution[v0n]==bestSolution[v0]:
                v0p0 += 1
            else:
                v0p1 += 1
        for v1n in graph.adj[v1]:
            if bestSolution[v1n]==bestSolution[v1]:
                v1p0 += 1
            else:
                v1p1 += 1
        # print(f'Trying out edge ({v0},{v1}), with neighbour counts {v0p0}+{v0p1}  vs  {v1p0}+{v1p1}  and same-group ? = {bestSolution[v0] == bestSolution[v1]}')
        # We have 2+1  vs  2+3.
        # So flipping both

        # Let's make note of the fact that these counts INCLUDE the edge (v0, v1)
        # Then the difference in utility if we flipped v0, would be
        newUtility01 = bestUtility + (v0p0 - v0p1)
        newUtility10 = bestUtility + (v1p0 - v1p1)
        if bestSolution[v0] == bestSolution[v1]:
            newUtility11 = bestUtility + (v0p0 - v0p1) + (v1p0 - v1p1) - 2
        else:
            newUtility11 = bestUtility + (v0p0 - v0p1) + (v1p0 - v1p1) + 2
        # Now use the best solution
        # bestCandidateUtility = newUtility01
        # bestCandidateUtility = max(newUtility01, newUtility10)
        bestCandidateUtility = max(max(newUtility01, newUtility10), newUtility11)
        if bestCandidateUtility > bestUtility:
            bestUtility = bestCandidateUtility
            # Swap the proper vertices
            if newUtility01 == bestCandidateUtility:
                bestSolution[v0] = 1 - bestSolution[v0]
                # utils.flipVertex(bestSolution, v0)
            elif newUtility10 == bestCandidateUtility:
                bestSolution[v1] = 1 - bestSolution[v1]
                # utils.flipVertex(bestSolution, v1)
            else:
                bestSolution[v0] = 1 - bestSolution[v0]
                bestSolution[v1] = 1 - bestSolution[v1]
                # utils.flipVertex(bestSolution, v0)
                # utils.flipVertex(bestSolution, v1)
            # print(f'[optimize edges]  found good flip. Now utility suspected to be {bestUtility}')
            # if not verify_utility(graph, bestSolution, bestUtility):
            #     return
            # print('[optimize edges]  Verified utility is correctly computed.')
    return (bestSolution, bestUtility)

# Optimizes a single edge (u,v), i.e., tries out 
# modifies the solution in place
# TO DO not verified
def optimizeSingleEdge(graph, solution, solutionUtility : int, u : int, v : int):
    # Try out flipping v0
    # So, I need to figure out how many edges would be affected, and in what way,
    # if v0 were flipped. 
    # So, I gather up all the neighbours of v0, and for each neighbour I ask: is it in 0 or in 1?
    v0p0 = 0
    v0p1 = 0
    v1p0 = 0
    v1p1 = 0
    # Let's count, for each vertex, how many neighbours are in various partitions
    for v0n in graph.adj[u]:
        if v0n == u:
            print(f'Node {u} is its own neighbour???')
        if solution[v0n]==solution[u]:
            v0p0 += 1
        else:
            v0p1 += 1
    for v1n in graph.adj[v]:
        if solution[v1n]==solution[v]:
            v1p0 += 1
        else:
            v1p1 += 1
    # Let's make note of the fact that these counts INCLUDE the edge (v0, v1)
    # Then the difference in utility if we flipped v0, would be
    newUtility01 = solutionUtility + (v0p0 - v0p1)
    newUtility10 = solutionUtility + (v1p0 - v1p1)
    if solution[u] == solution[v]:
        newUtility11 = solutionUtility + (v0p0 - v0p1) + (v1p0 - v1p1) + 2
    else:
        newUtility11 = solutionUtility + (v0p0 - v0p1) + (v1p0 - v1p1) - 2
    # Now use the best solution
    bestCandidateUtility = max(max(newUtility01, newUtility10), newUtility11)
    if bestCandidateUtility > solutionUtility:
        bestUtility = bestCandidateUtility
        # Swap it
        if newUtility01 == bestCandidateUtility:
            utils.flipVertex(solution, u)
        elif newUtility10 == bestCandidateUtility:
            utils.flipVertex(solution, v)
        else:
            utils.flipVertex(solution, u)
            utils.flipVertex(solution, v)
    return bestUtility

# TODO modify the solution in place, instead of copying into a new solution
# TODO not verified
def optimizeAllEdgesInSubgraph(graph, solution, startingUtility : int, neighbourhood):
    best_utility = startingUtility
    time_start = time.perf_counter()
    best_solution = solution.copy()
    Profile.copying_time += time_since(time_start)
    # if not verify_utility(graph, best_solution, best_utility):
    #     print('[optimize edges]  failed sanity check before starting.')
    #     return
    # else:
    #     print('[optimize edges]  sanity check successful.')
    for vertex in neighbourhood:
        # optimize all edges of this vertex
        for neighbour in graph.adj[vertex]:
            v0 = vertex
            v1 = neighbour
            # Try out flipping v0
            # So, I need to figure out how many edges would be affected, and in what way,
            # if v0 were flipped. 
            # So, I gather up all the neighbours of v0, and for each neighbour I ask: is it in 0 or in 1?
            v0p0 = 0
            v0p1 = 0
            v1p0 = 0
            v1p1 = 0
            # Let's count, for each vertex, how many neighbours are in various partitions
            for v0n in graph.adj[v0]:
                if v0n == v0:
                    print(f'Node {v0} is its own neighbour???')
                if best_solution[v0n]==best_solution[v0]:
                    v0p0 += 1
                else:
                    v0p1 += 1
            for v1n in graph.adj[v1]:
                if best_solution[v1n]==best_solution[v1]:
                    v1p0 += 1
                else:
                    v1p1 += 1
            # Let's make note of the fact that these counts INCLUDE the edge (v0, v1)
            # Then the difference in utility if we flipped v0, would be
            new_utility_01 = best_utility + (v0p0 - v0p1)
            new_utility_10 = best_utility + (v1p0 - v1p1)
            if best_solution[v0] == best_solution[v1]:
                new_utility_11 = best_utility + (v0p0 - v0p1) + (v1p0 - v1p1) - 2
            else:
                new_utility_11 = best_utility + (v0p0 - v0p1) + (v1p0 - v1p1) + 2
            # Now use the best solution
            best_candidate_utility = max(max(new_utility_01, new_utility_10), new_utility_11)
            if best_candidate_utility > best_utility:
                best_utility = best_candidate_utility
                # Swap it
                if new_utility_01 == best_candidate_utility:
                    # print(f'[optimize edges]  swapping v0={v0}')
                    utils.flipVertex(best_solution, v0)
                elif new_utility_10 == best_candidate_utility:
                    # print(f'[optimize edges]  swapping v1={v1}')
                    utils.flipVertex(best_solution, v1)
                else:
                    # print(f'[optimize edges]  swapping v0={v0} and v1={v1}   with {v0p0}+{v0p1}  vs  {v1p0}+{v1p1}  and  solut[v0],solut[v1] = {best_solution[v0]} , {best_solution[v1]}')
                    utils.flipVertex(best_solution, v0)
                    utils.flipVertex(best_solution, v1)
                # if not verify_utility(graph, best_solution, best_utility):
                #     # print('[optimize edges]  I swapped some edges, but it didnt work out')
                #     return
    Profile.optimize_edges_time += time_since(time_start)
    return (best_solution, best_utility)

def optimize_all_edges_in_subgraph_v2(solution_write : list[int], graph, starting_solution : list[int], starting_utility : int, neighbourhood : numpy.array):
    best_utility = starting_utility
    time_start = time.perf_counter()
    copy_data_inplace(solution_write, starting_solution)
    Profile.copying_time += time_since(time_start)
    for vertex in neighbourhood:
        # optimize all edges of this vertex
        for neighbour in graph.adj[vertex]:
            v0 = vertex
            v1 = neighbour
            v0p0 = 0
            v0p1 = 0
            v1p0 = 0
            v1p1 = 0
            # Let's count, for each vertex, how many neighbours are in various partitions
            for v0n in graph.adj[v0]:
                if solution_write[v0n]==solution_write[v0]:
                    v0p0 += 1
                else:
                    v0p1 += 1
            for v1n in graph.adj[v1]:
                if solution_write[v1n]==solution_write[v1]:
                    v1p0 += 1
                else:
                    v1p1 += 1
            # Let's make note of the fact that these counts INCLUDE the edge (v0, v1)
            # Then the difference in utility if we flipped v0, would be
            new_utility_01 = best_utility + (v0p0 - v0p1)
            new_utility_10 = best_utility + (v1p0 - v1p1)
            if solution_write[v0] == solution_write[v1]:
                new_utility_11 = best_utility + (v0p0 - v0p1) + (v1p0 - v1p1) - 2
            else:
                new_utility_11 = best_utility + (v0p0 - v0p1) + (v1p0 - v1p1) + 2
            # Now use the best solution
            best_candidate_utility = max(max(new_utility_01, new_utility_10), new_utility_11)
            if best_candidate_utility > best_utility:
                best_utility = best_candidate_utility
                # Swap it
                if new_utility_01 == best_candidate_utility:
                    # print(f'[optimize edges]  swapping v0={v0}')
                    utils.flipVertex(solution_write, v0)
                elif new_utility_10 == best_candidate_utility:
                    # print(f'[optimize edges]  swapping v1={v1}')
                    utils.flipVertex(solution_write, v1)
                else:
                    # print(f'[optimize edges]  swapping v0={v0} and v1={v1}   with {v0p0}+{v0p1}  vs  {v1p0}+{v1p1}  and  solut[v0],solut[v1] = {best_solution[v0]} , {best_solution[v1]}')
                    utils.flipVertex(solution_write, v0)
                    utils.flipVertex(solution_write, v1)
                # if not verify_utility(graph, best_solution, best_utility):
                #     # print('[optimize edges]  I swapped some edges, but it didnt work out')
                #     return
    Profile.optimize_edges_v2_time += time_since(time_start)
    return best_utility

# improvement over version 2: use numpy.ndarray
#   also some miniscule performance optimizations
def optimize_all_edges_in_subgraph_v3(solution_write : numpy.ndarray, graph, starting_solution : numpy.ndarray, starting_utility : int, neighbourhood):
    best_utility = starting_utility
    time_start = time.perf_counter()
    copy_data_inplace_np(solution_write, starting_solution)
    Profile.copying_time += time_since(time_start)
    for v0 in neighbourhood:
        # optimize all edges of this vertex
        for v1 in graph.adj[v0]:
            v0p0 = 0
            v0p1 = 0
            v1p0 = 0
            v1p1 = 0
            # Let's count, for each vertex, how many neighbours are in various partitions
            for v0n in graph.adj[v0]:
                if solution_write[v0n]==solution_write[v0]:
                    v0p0 += 1
                else:
                    v0p1 += 1
            for v1n in graph.adj[v1]:
                if solution_write[v1n]==solution_write[v1]:
                    v1p0 += 1
                else:
                    v1p1 += 1
            # Let's make note of the fact that these counts INCLUDE the edge (v0, v1)
            # Then the difference in utility if we flipped v0, would be
            utility_diff_01 = (v0p0 - v0p1)
            utility_diff_10 = (v1p0 - v1p1)
            if solution_write[v0] == solution_write[v1]:
                utility_diff_11 = (v0p0 - v0p1) + (v1p0 - v1p1) - 2
            else:
                utility_diff_11 = (v0p0 - v0p1) + (v1p0 - v1p1) + 2
            # Now use the best solution
            best_utility_diff = max(max(utility_diff_01, utility_diff_10), utility_diff_11)
            if best_utility_diff > 0:
                best_utility = best_utility + best_utility_diff
                # Swap it
                if utility_diff_01 == best_utility_diff:
                    # print(f'[optimize edges]  swapping v0={v0}')
                    solution_write[v0] = 1 - solution_write[v0]
                    # utils.flipVertex(solution_write, v0)
                elif utility_diff_10 == best_utility_diff:
                    # print(f'[optimize edges]  swapping v1={v1}')
                    solution_write[v1] = 1 - solution_write[v1]
                    # utils.flipVertex(solution_write, v1)
                else:
                    # print(f'[optimize edges]  swapping v0={v0} and v1={v1}   with {v0p0}+{v0p1}  vs  {v1p0}+{v1p1}  and  solut[v0],solut[v1] = {best_solution[v0]} , {best_solution[v1]}')
                    solution_write[v0] = 1 - solution_write[v0]
                    # utils.flipVertex(solution_write, v0)
                    solution_write[v1] = 1 - solution_write[v1]
                    # utils.flipVertex(solution_write, v1)
                # if not verify_utility(graph, best_solution, best_utility):
                #     # print('[optimize edges]  I swapped some edges, but it didnt work out')
                #     return
    Profile.optimize_edges_v3_time += time_since(time_start)
    return best_utility

# improvement over version 3: neighbourhood is a numpy.ndarray
def optimize_all_edges_in_subgraph_v4(solution_write : numpy.ndarray, graph, starting_solution : numpy.ndarray, starting_utility : int, neighbourhood : numpy.ndarray):
    best_utility = starting_utility
    time_start = time.perf_counter()
    Profile.optimize_edges_v4_calls += 1
    copy_data_inplace_np(solution_write, starting_solution)
    Profile.copying_time += time_since(time_start)
    v0 : int = 0
    for v0 in neighbourhood:
    # for i in range(neighbourhood.shape[0]):
    #     v0 = neighbourhood[i]
        # optimize all edges of this vertex
        for v1 in graph.adj[v0]:
            v0p0 = 0
            v0p1 = 0
            v1p0 = 0
            v1p1 = 0
            # Let's count, for each vertex, how many neighbours are in various partitions
            for v0n in graph.adj[v0]:
                if solution_write[v0n]==solution_write[v0]:
                    v0p0 += 1
                else:
                    v0p1 += 1
            for v1n in graph.adj[v1]:
                if solution_write[v1n]==solution_write[v1]:
                    v1p0 += 1
                else:
                    v1p1 += 1
            # Let's make note of the fact that these counts INCLUDE the edge (v0, v1)
            # Then the difference in utility if we flipped v0, would be
            utility_diff_01 = (v0p0 - v0p1)
            utility_diff_10 = (v1p0 - v1p1)
            if solution_write[v0] == solution_write[v1]:
                utility_diff_11 = (v0p0 - v0p1) + (v1p0 - v1p1) - 2
            else:
                utility_diff_11 = (v0p0 - v0p1) + (v1p0 - v1p1) + 2
            # Now use the best solution
            best_utility_diff = max(max(utility_diff_01, utility_diff_10), utility_diff_11)
            if best_utility_diff > 0:
                best_utility = best_utility + best_utility_diff
                # Swap it
                if utility_diff_01 == best_utility_diff:
                    solution_write[v0] = 1 - solution_write[v0]
                elif utility_diff_10 == best_utility_diff:
                    solution_write[v1] = 1 - solution_write[v1]
                else:
                    solution_write[v0] = 1 - solution_write[v0]
                    solution_write[v1] = 1 - solution_write[v1]
    Profile.optimize_edges_v4_time += time_since(time_start)
    return best_utility


def solveMaxCutGreedy3Refactored(graph):
    n = utils.getNumVertices(graph)
    m = len(graph)
    maxSwapSize = int(math.log2(n))
    budget = 10*m # each edge can be changed 10 times (which is not a lot, imo)
    bestSolution = n * [ 0 ]
    utils.setListToRandomBooleans(bestSolution)
    bestUtility = evaluateSolution(graph, bestSolution)
    for round in range(0, budget):
        (solution, utility) = greedy3Round(graph, bestSolution, n, maxSwapSize)
        if (utility > bestUtility):
            bestSolution = solution
            bestUtility = utility
        else:
            if (solution == bestSolution):
                print("That's very coincidental!!!")
    # print('[Greedy] found solution ' + str(bestSolution))
    return (bestSolution, bestUtility)

def solveMaxCutGreedy4(graph):
    # Grow the swapset iteratively: keep adding vertices until you get a better solution
    n = utils.getNumVertices(graph)
    m = len(graph)
    budget = 10*m # each edge can be changed 10 times (which is not a lot, imo)
    solution = n * [ 0 ]
    utils.setListToRandomBooleans(solution)
    bestUtility = evaluateSolution(graph, solution)
    bestSolution = solution
    for round in range(0, budget):
        # Guess a set of k vertices
        solution = bestSolution
        improvementFound = False
        swapSet = []
        while (not improvementFound and len(swapSet) + 1 < n/2):
            # add a vertex to the swapset
            v = random.randint(0, n-1)
            while (swapSet.__contains__(v)):
                v = random.randint(0, n-1)
            swapSet.append(v)
            # Swap the verices in list
            solution[v] = 1 - solution[v]
            utility = evaluateSolution(graph, solution)
            if (utility > bestUtility):
                bestSolution = solution
                bestUtility = utility
                improvementFound = True
    return (bestSolution, bestUtility)

# is equivalent to 'optimizeRandomPartition'
# We pass n and m so we don't need to recompute, which saves time
def greedy5Round(graph, startingSolution, partitionSize, n, m):
    bestUtility = m
    bestSolution = startingSolution.copy()
    numCombinations = pow(2, partitionSize)
    partition = utils.getRandomPartition(n, partitionSize)
    for x in range(1, numCombinations-1):
        # try out the 'x' solution
        solution = bestSolution.copy()
        for p in range(0, partitionSize):
            if (x & (1 << p)):
                # flip the items in this partition
                for v in partition[p]:
                    solution[v] = 1 - solution[v]
        utility = evaluateSolution(graph, solution)
        if (utility > bestUtility):
            bestSolution = solution
            bestUtility = utility
    return (bestSolution, bestUtility)

def optimizePartition(graph, startingSolution, startingUtility : int, partition):
    bestSolution = startingSolution.copy()
    bestUtility = startingUtility
    numCombinations = pow(2, len(partition))
    for x in range(1, numCombinations - 1):
        solution = startingSolution.copy()
        for p in range(0, len(partition)):
            if (x & (1 << p)):
                for v in partition[p]:
                    solution[v] = 1 - solution[v]
        utility = evaluateSolution(graph, solution)
        if (utility > bestUtility):
            bestSolution = solution
            bestUtility = utility
    return (bestSolution, bestUtility)

# TODO use smart evaluation algorithm
def optimizePartitionNx(graph, startingSolution, startingUtility : int, partition):
    bestSolution = startingSolution.copy()
    bestUtility = startingUtility
    numCombinations = pow(2, len(partition))
    for x in range(1, numCombinations):
        solution = startingSolution.copy()
        for p in range(0, len(partition)):
            if (x & (1 << p)):
                for v in partition[p]:
                    solution[v] = 1 - solution[v] # TODO maybe use 1 & solution[v] ? is that noticeably faster...? probably not
        utility = evaluateSolutionNxList(graph, solution)
        if utility > bestUtility:
            bestSolution = solution
            bestUtility = utility
    return (bestSolution, bestUtility)

def applyPartitionFlip(solution, partition, x):
    flippedSolution = solution.copy()
    for p in range(len(partition)):
        if (x & (1 << p)):
            for v in partition[p]:
                flippedSolution[v] = 1 - flippedSolution[v]
    return flippedSolution

# mip = modify in place
def apply_partition_flip_mip(solution : numpy.ndarray, partition, x):
    for p in range(len(partition)):
        if x & (1 << p):
            for v in partition[p]:
                solution[v] = 1 - solution[v]

# Returns the index of the block that contains the target
def findItemInListOfLists(partition : list[list[int]], target : int) -> int:
    for j in range(len(partition)):
        if partition[j].__contains__(target):
            return j
    return -1

def evaluateSolutionAfterFlippingPartition(graph, solution, startingUtility, partition, x):
    # partition is a list of blocks, each block is a list of vertices
    # each p-th bit of x indicates whether the vertices in the block partition[p] are flipped
    # the partition is not necessarily "entire": there may be (usually are, in fact) vertices not in any block
    utilityDiff = 0
    for bit in range(len(partition)):
        if x & (1 << bit):
            block = partition[bit]
            for vertex in block:
                for neighbour in graph.adj[vertex]:
                    # utilityDiff goes up   if this neighbour is in a block that doesn't change and has the same  polarity
                    # utilityDiff goes down if this neighbour is in a block that doesn't change and has different polarity
                    # utilityDiff stays the same if this neighbour is in a block that is flipped
                    # find the neighbour
                    j = findItemInListOfLists(partition, neighbour)
                    if j == -1 or ((x & (1 << j)) == 0):
                        # neighbour is in a block that is not flipped
                        if solution[vertex] == solution[neighbour]:
                            utilityDiff += 1
                        else:
                            utilityDiff -= 1
    return startingUtility + utilityDiff

def evaluateSolutionAfterFlippingPartition_v2(graph, solution, startingUtility, partition, x, inverse_partition : dict[int, int]):
    # partition is a list of blocks, each block is a list of vertices
    # each p-th bit of x indicates whether the vertices in the block partition[p] are flipped
    # the partition is not necessarily "entire": there may be (usually are, in fact) vertices not in any block
    utilityDiff = 0
    for bit in range(len(partition)):
        if x & (1 << bit):
            block = partition[bit]
            for vertex in block:
                for neighbour in graph.adj[vertex]:
                    # utilityDiff goes up   if this neighbour is in a block that doesn't change and has the same  polarity
                    # utilityDiff goes down if this neighbour is in a block that doesn't change and has different polarity
                    # utilityDiff stays the same if this neighbour is in a block that is flipped
                    # find the neighbour
                    if inverse_partition.__contains__(neighbour):
                        j = inverse_partition[neighbour]
                    else:
                        j = -1
                    if j == -1 or ((x & (1 << j)) == 0):
                        # neighbour is in a block that is not flipped
                        if solution[vertex] == solution[neighbour]:
                            utilityDiff += 1
                        else:
                            utilityDiff -= 1
    return startingUtility + utilityDiff

# This method is specifically for evaluateSolutionAfterFlippingPartition
# It returns the 
def get_summary_graph(graph, solution, partition, inverse_partition):
    summary_graph = []
    for i in range(len(partition) + 1):
        summary_graph.append([])
        for j in range(len(partition) + 1):
            summary_graph[i].append( [[0,0],[0,0]] )
    for i in range(len(partition)):
        for vertex in partition[i]:
            for neighbour in graph[vertex]:
                if inverse_partition.__contains__(neighbour):
                    j = inverse_partition[neighbour]
                else:
                    j = len(partition)
                summary_graph[i][j][solution[vertex]][solution[neighbour]] += 1
    return summary_graph

# Improvement over v1: returns a numpy.ndarray instead of a list of lists of lists of lists
def get_summary_graph_v2(graph, solution, partition, inverse_partition : dict[int, int]) -> numpy.ndarray:
    time_start = time.perf_counter()
    num_blocks : int = len(partition)
    summary_graph = numpy.ndarray([num_blocks, num_blocks + 1, 2, 2], dtype = numpy.int16)
    j : int = 0
    for i in range(num_blocks):
        for j in range(num_blocks+1):
            summary_graph[i,j,0,0] = 0
            summary_graph[i,j,0,1] = 0
            summary_graph[i,j,1,0] = 0
            summary_graph[i,j,1,1] = 0
    for i in range(num_blocks):
        for vertex in partition[i]:
            for neighbour in graph[vertex]:
                if inverse_partition.__contains__(neighbour):
                    j = inverse_partition[neighbour]
                else:
                    j = num_blocks
                summary_graph[i, j, solution[vertex], solution[neighbour]] += 1
    Profile.get_summary_graph_time += time.perf_counter() - time_start
    return summary_graph

def set_summary_graph(graph, solution, partition : list[list[int]], inverse_partition : dict[int, int], summary_graph : numpy.ndarray):
    num_blocks : int = len(partition)
    # check that the summary_graph has the right dimension
    if summary_graph.shape != [num_blocks, num_blocks + 1, 2, 2]:
        print('  <<  ERROR  >>  the summary_graph : numpy.ndarray has the wrong shape')
    j : int = 0
    for i in range(num_blocks):
        for j in range(num_blocks+1):
            summary_graph[i,j,0,0] = 0
            summary_graph[i,j,0,1] = 0
            summary_graph[i,j,1,0] = 0
            summary_graph[i,j,1,1] = 0
    for i in range(num_blocks):
        for vertex in partition[i]:
            for neighbour in graph[vertex]:
                if inverse_partition.__contains__(neighbour):
                    j = inverse_partition[neighbour]
                else:
                    j = num_blocks
                summary_graph[i, j, solution[vertex], solution[neighbour]] += 1
    return summary_graph

def evaluateSolutionAfterFlippingPartition_v3(startingUtility, partition, x : int, summary_graph):
    utilityDiff : int = 0
    for block_a in range(len(partition)):
        if x & (1 << block_a):
            # compute the profit made by flipping the vertices in block_a
            for block_b in range(len(partition) + 1):
                if (x & (1 << block_b)) == 0:
                    diff = summary_graph[block_a][block_b][0][0] + summary_graph[block_a][block_b][1][1] - summary_graph[block_a][block_b][0][1] - summary_graph[block_a][block_b][1][0]
                    utilityDiff += diff
    return startingUtility + utilityDiff

# improvement over v3: use a numpy.ndarray instead of a list of lists
def evaluateSolutionAfterFlippingPartition_v4(startingUtility, partition, x : int, summary_graph : numpy.ndarray):
    time_start = time.perf_counter()
    utilityDiff : int = 0
    for block_a in range(len(partition)):
        if x & (1 << block_a):
            # compute the profit made by flipping the vertices in block_a
            for block_b in range(len(partition) + 1):
                if (x & (1 << block_b)) == 0:
                    diff = summary_graph[block_a, block_b, 0, 0] + summary_graph[block_a, block_b, 1, 1] \
                         - summary_graph[block_a, block_b, 0, 1] - summary_graph[block_a, block_b, 1, 0]
                    utilityDiff += diff
    Profile.evaluate_after_flipping_partition_time += time.perf_counter() - time_start
    return startingUtility + utilityDiff

def optimizePartitionNx_v2(graph, startingSolution, startingUtility, partition):
    bestUtility = startingUtility
    numCombinations = pow(2, len(partition))
    for x in range(1, numCombinations):
        # or better yet, don't apply a partition at all!
        utility = evaluateSolutionAfterFlippingPartition(graph, startingSolution, startingUtility, partition, x)
        # TODO verify that we're doing the right thing by keeping a bookkeeping solution
        # flippedSolution = startingSolution.copy()
        # flippedSolution = applyPartitionFlip(flippedSolution, partition, x)
        # verifiedUtility = evaluateSolutionNxList(graph, flippedSolution)
        # if (utility != verifiedUtility):
        #     print('[optimize partition]  ERROR verified utility != computed utility!!')
        if utility > bestUtility:
            bestUtility = utility
            bestX = x
    if bestUtility > startingUtility:
        solution = applyPartitionFlip(startingSolution, partition, bestX)
    else:
        solution = startingSolution.copy()
    return (solution, bestUtility)

# Verified correctness: done
def optimizePartitionAndEdges(graph : nx.graph.Graph, startingSolution, startingUtility : int, partition, neighbourhood):
    bestUtility = startingUtility
    bestX = 0
    numCombinations = pow(2, len(partition))
    for x in range(1, numCombinations):
        # Flip the partition relative to x
        # Then optimize all the individual edges in the total neighbourhood
        # Then compute the new utility.
        # This will present a very optimized candidate solution
        flippedSolution = startingSolution.copy()
        flippedSolution = applyPartitionFlip(flippedSolution, partition, x)
        # flippedUtility = evaluateSolutionNxList(graph, flippedSolution)
        flippedUtility = evaluateSolutionAfterFlippingPartition(graph, startingSolution, startingUtility, partition, x)
        (flippedSolution, flippedUtility) = optimizeAllEdgesInSubgraph(graph, flippedSolution, flippedUtility, neighbourhood)
        if flippedUtility > bestUtility:
            bestUtility = flippedUtility
            bestX = x
    if bestUtility > startingUtility:
        bestSolution = startingSolution.copy()
        bestSolution = applyPartitionFlip(bestSolution, partition, bestX)
        bestUtility  = evaluateSolutionAfterFlippingPartition(graph, startingSolution, startingUtility, partition, bestX)
        (bestSolution, bestUtility) = optimizeAllEdgesInSubgraph(graph, bestSolution, bestUtility, neighbourhood)
    else:
        bestSolution = startingSolution.copy()
    return (bestSolution, bestUtility)

def optimizePartitionAndEdges_v2(graph : nx.graph.Graph, starting_solution, starting_utility : int, partition, neighbourhood):
    best_utility = starting_utility
    bestX = 0
    num_combinations = pow(2, len(partition))
    flipped_solution = starting_solution.copy()
    best_solution = starting_solution.copy()
    for x in range(1, num_combinations):
        # Flip the partition relative to x
        # Then optimize all the individual edges in the total neighbourhood
        # Then compute the new utility.
        # This will present a very optimized candidate solution
        copy_data_inplace(flipped_solution, starting_solution)
        flipped_solution = applyPartitionFlip(flipped_solution, partition, x)
        # flippedUtility = evaluateSolutionNxList(graph, flippedSolution)
        flippedUtility = evaluateSolutionAfterFlippingPartition(graph, starting_solution, starting_utility, partition, x)
        (flipped_solution, flippedUtility) = optimizeAllEdgesInSubgraph(graph, flipped_solution, flippedUtility, neighbourhood)
        if flippedUtility > best_utility:
            best_utility = flippedUtility
            bestX = x
    if best_utility > starting_utility:
        copy_data_inplace(best_solution, starting_solution)
        best_solution = applyPartitionFlip(best_solution, partition, bestX)
        best_utility  = evaluateSolutionAfterFlippingPartition(graph, starting_solution, starting_utility, partition, bestX)
        (best_solution, best_utility) = optimizeAllEdgesInSubgraph(graph, best_solution, best_utility, neighbourhood)
    else:
        copy_data_inplace(best_solution, starting_solution)
    return (best_solution, best_utility)

# same result as v2, but faster
def optimizePartitionAndEdges_v3(graph : nx.graph.Graph, starting_solution, starting_utility : int, partition, neighbourhood):
    best_utility = starting_utility
    bestX = 0
    num_combinations = pow(2, len(partition))
    flipped_solution = starting_solution.copy()
    best_solution = starting_solution.copy()
    inverse_partition = get_inverse_partition(partition)
    summary_graph = get_summary_graph(graph, starting_solution, partition, inverse_partition)
    for x in range(1, num_combinations):
        copy_data_inplace(flipped_solution, starting_solution)
        flipped_solution = applyPartitionFlip(flipped_solution, partition, x)
        flipped_utility = evaluateSolutionAfterFlippingPartition_v3(starting_utility, partition, x, summary_graph)
        if not verify_utility(graph, flipped_solution, flipped_utility): return # TODO remove this once we're confident it gives the right answer
        (flipped_solution, flipped_utility) = optimizeAllEdgesInSubgraph(graph, flipped_solution, flipped_utility, neighbourhood)
        if flipped_utility > best_utility:
            best_utility = flipped_utility
            bestX = x
    if best_utility > starting_utility:
        copy_data_inplace(best_solution, starting_solution)
        best_solution = applyPartitionFlip(best_solution, partition, bestX)
        best_utility  = evaluateSolutionAfterFlippingPartition_v3(starting_utility, partition, bestX, summary_graph)
        (best_solution, best_utility) = optimizeAllEdgesInSubgraph(graph, best_solution, best_utility, neighbourhood)
    else:
        copy_data_inplace(best_solution, starting_solution)
    return (best_solution, best_utility)

# same as v3, but faster
def optimizePartitionAndEdges_v4(graph : nx.graph.Graph, starting_solution, starting_utility : int, partition, neighbourhood):
    time_start = time.perf_counter()
    best_utility = starting_utility
    bestX = 0
    num_combinations = pow(2, len(partition))
    copying_time_start = time.perf_counter()
    flipped_solution = starting_solution.copy()
    temp_solution = starting_solution.copy() # just a temp thing to avoid continuously allocating data
    best_solution = starting_solution.copy()
    Profile.copying_time += time_since(copying_time_start)
    inverse_partition = get_inverse_partition(partition)
    summary_graph = get_summary_graph_v2(graph, starting_solution, partition, inverse_partition)
    for x in range(1, num_combinations):
        copy_data_inplace(flipped_solution, starting_solution)
        flipped_solution = applyPartitionFlip(flipped_solution, partition, x)
        flipped_utility = evaluateSolutionAfterFlippingPartition_v4(starting_utility, partition, x, summary_graph)
        if not verify_utility(graph, flipped_solution, flipped_utility): print('[optimize v4]  ERROR utility is wrong'); exit(); # TODO remove this once we're confident it gives the right answer
        # (flipped_solution, flipped_utility) = optimizeAllEdgesInSubgraph(graph, flipped_solution, flipped_utility, neighbourhood)
        flipped_utility = optimize_all_edges_in_subgraph_v2(temp_solution, graph, flipped_solution, flipped_utility, neighbourhood)
        if flipped_utility > best_utility:
            best_utility = flipped_utility
            bestX = x
    if best_utility > starting_utility:
        copy_data_inplace(best_solution, starting_solution)
        best_solution = applyPartitionFlip(best_solution, partition, bestX)
        # (best_solution, best_utility) = optimizeAllEdgesInSubgraph(graph, best_solution, best_utility, neighbourhood)
        optimize_all_edges_in_subgraph_v2(best_solution, graph, best_solution, best_utility, neighbourhood)
    else:
        copy_data_inplace(best_solution, starting_solution)
    Profile.optimize_v4_time += time.perf_counter() - time_start
    return (best_solution, best_utility)

# TODO write this function
# improvement over version 4: use numpy arrays
#   and we have parameter 'solution_write' instead of allocating a new vector for output
def optimizePartitionAndEdges_v5(solution_write : numpy.ndarray, graph : nx.graph.Graph, starting_solution : numpy.ndarray, starting_utility : int, partition, neighbourhood):
    time_start = time.perf_counter()
    if not verify_utility(graph, starting_solution, starting_utility): print('[optimize v5] error starting utility is wrong.'); exit();
    best_utility = starting_utility
    bestX = 0
    num_combinations = pow(2, len(partition))
    copying_time_start = time.perf_counter()
    candidate_solution = starting_solution.copy()
    temp_solution = starting_solution.copy() # just a temp thing to avoid continuously allocating data
    Profile.copying_time += time_since(copying_time_start)
    inverse_partition = get_inverse_partition(partition)
    summary_graph = get_summary_graph_v2(graph, starting_solution, partition, inverse_partition)
    for x in range(1, num_combinations):
        # copy_to_candidate_start = time.perf_counter()
        copy_data_inplace_np(candidate_solution, starting_solution)
        # Profile.copy_to_candidate += time_since(copy_to_candidate_start)
        apply_partition_flip_mip(candidate_solution, partition, x)
        candidate_utility = evaluateSolutionAfterFlippingPartition_v4(starting_utility, partition, x, summary_graph)
        # if not verify_utility(graph, candidate_solution, candidate_utility): print('[optimize v5] ERROR'); exit(); # TODO remove this once we're confident it gives the right answer
        # else: print('[optimize v5]  success');
        candidate_utility = optimize_all_edges_in_subgraph_v3(temp_solution, graph, candidate_solution, candidate_utility, neighbourhood)
        if candidate_utility > best_utility:
            best_utility = candidate_utility
            bestX = x
        # undo the partition flip
        # apply_partition_flip_mip(candidate_solution, partition, x)
    if best_utility > starting_utility:
        copy_data_inplace_np(solution_write, starting_solution)
        apply_partition_flip_mip(solution_write, partition, bestX)
        optimize_all_edges_in_subgraph_v3(solution_write, graph, solution_write, best_utility, neighbourhood)
    else:
        copy_data_inplace_np(solution_write, starting_solution)
    Profile.optimize_v5_time += time.perf_counter() - time_start
    return best_utility

def convertSetToArray(S):
    a = numpy.ndarray([len(S)], dtype=numpy.int32)
    i = 0
    for e in S:
        a[i] = e
        i += 1
    return a

# improvement over v5: use a numpy.ndarray for neighbourhood
def optimizePartitionAndEdges_v6(solution_write : numpy.ndarray, graph : nx.graph.Graph, starting_solution : numpy.ndarray, starting_utility : int, partition, neighbourhood):
    neighbourhood_nda = convertSetToArray(neighbourhood)
    time_start = time.perf_counter()
    if not verify_utility(graph, starting_solution, starting_utility): print('[optimize v5] error starting utility is wrong.'); exit();
    best_utility = starting_utility
    bestX = 0
    num_combinations = pow(2, len(partition))
    copying_time_start = time.perf_counter()
    candidate_solution = starting_solution.copy()
    temp_solution = starting_solution.copy() # just a temp thing to avoid continuously allocating data
    Profile.copying_time += time_since(copying_time_start)
    inverse_partition = get_inverse_partition(partition)
    summary_graph = get_summary_graph_v2(graph, starting_solution, partition, inverse_partition)
    for x in range(1, num_combinations):
        # copy_to_candidate_start = time.perf_counter()
        copy_data_inplace_np(candidate_solution, starting_solution)
        # Profile.copy_to_candidate += time_since(copy_to_candidate_start)
        apply_partition_flip_mip(candidate_solution, partition, x)
        candidate_utility = evaluateSolutionAfterFlippingPartition_v4(starting_utility, partition, x, summary_graph)
        # if not verify_utility(graph, candidate_solution, candidate_utility): print('[optimize v5] ERROR'); exit(); # TODO remove this once we're confident it gives the right answer
        # else: print('[optimize v5]  success');
        candidate_utility = optimize_all_edges_in_subgraph_v4(temp_solution, graph, candidate_solution, candidate_utility, neighbourhood_nda)
        if candidate_utility > best_utility:
            best_utility = candidate_utility
            bestX = x
        # undo the partition flip
        # apply_partition_flip_mip(candidate_solution, partition, x)
    if best_utility > starting_utility:
        copy_data_inplace_np(solution_write, starting_solution)
        apply_partition_flip_mip(solution_write, partition, bestX)
        optimize_all_edges_in_subgraph_v4(solution_write, graph, solution_write, best_utility, neighbourhood_nda)
    else:
        copy_data_inplace_np(solution_write, starting_solution)
    Profile.optimize_v5_time += time.perf_counter() - time_start
    return best_utility


# 'numBlocks' is the number of blocks in the partition
def optimizeRandomPartition(graph, startingSolution : list, n : int, numBlocks : int):
    partition = utils.getRandomPartition(n, numBlocks)
    return optimizePartition(graph, startingSolution, partition)

def getNeighbourhood(graph, v : int, maxNeighbours : int):
    neigh = [v]
    for e in graph:
        if (e[0] == v):
            neigh.append(e[1])
        elif (e[1] == v):
            neigh.append(e[0])
        if (len(neigh) >= maxNeighbours):
            break
    return neigh;

# return a set S of vertices, each "close by" the vertex v
# in the sense that it holds that for each vertex u in S, let k := distance(u,v) = k, then all vertices at distance < k are also in S
def getExtendedNeighbourhood(graph, v : int, maxNeighbours : int):
    neigh = set()
    neigh.add(v)
    # get distance 1
    stopping = False
    changed = True
    while (not stopping and changed):
        # Get distance d
        boundary = set()
        changed = False
        for e in graph:
            if (neigh.__contains__(e[0]) and not boundary.__contains__(e[1]) and not neigh.__contains__(e[1])):
                boundary.add(e[1])
                changed = True
            elif (neigh.__contains__(e[1]) and not boundary.__contains__(e[0]) and not neigh.__contains__(e[0])):
                boundary.add(e[0])
                changed = True
            if (len(neigh) + len(boundary) >= maxNeighbours):
                stopping = True
                break
        for w in boundary:
            neigh.add(w)
    return neigh

# return a set S of vertices, each "close by" the vertex v
# in the sense that it holds that for each vertex u in S, let k := distance(u,v) = k, then all vertices at distance < k are also in S
# TODO this function can be optimized so that we loop over the vertices in the neighbourhood, instead of over all the edges in the graph.
def getExtendedNeighbourhoodNx(graph, v : int, maxNeighbours : int):
    time_start = time.perf_counter()
    neighbourhood = set()
    neighbourhood.add(v)
    stopping = False
    changed = True
    while changed and not stopping:
        boundary = set()
        changed = False
        for edge in graph.edges:
            # Add a point of utility if the two vertices are in different sets
            if (neighbourhood.__contains__(edge[0]) and not boundary.__contains__(edge[1]) and not neighbourhood.__contains__(edge[1])):
                boundary.add(edge[1])
                changed = True
            elif neighbourhood.__contains__(edge[1]) and not boundary.__contains__(edge[0]) and not neighbourhood.__contains__(edge[0]):
                boundary.add(edge[0])
                changed = True
            if len(neighbourhood) + len(boundary) >= maxNeighbours:
                stopping = True
                break
        for w in boundary:
            neighbourhood.add(w)
    Profile.get_neighbourhood_time += time_since(time_start)
    return neighbourhood

# Returns a numpy.ndarray containing the neighbourhood
# the array has exactly the right length, i.e., if the center vertex is in a connected component of less than max_neighbours vertices, then
# returns an array containing only its connected component
# TODO should we maybe refactor so that the set is maintained and the numpy array is just converted at the end...?
def getExtendedNeighbourhood_v2(graph : nx.graph.Graph, center : int, max_neighbours : int) -> numpy.ndarray:
    neighbourhood = numpy.ndarray([max_neighbours])
    neighbourhood[0] = center
    num_neighbours : int = 1
    stopping = False
    changed = True
    boundary = set()
    while changed and not stopping:
        boundary.clear()
        changed = False
        for v in neighbourhood:
            for neighbour in graph[v]:
                if not boundary.__contains__(neighbour):
                    boundary.add(neighbour)
                    if num_neighbours + len(boundary) >= max_neighbours:
                        stopping = True
                        break
            if stopping:
                break
        # Move the boundary to the neighbourhood
        for w in boundary:
            neighbourhood[num_neighbours] = w
            num_neighbours += 1
    if num_neighbours < max_neighbours:
        # Allocate a ndarray that's exactly large enough
        neighbourhood_temp = numpy.ndarray([num_neighbours])
        for i in range(num_neighbours):
            neighbourhood_temp[i] = neighbourhood[i]
        return neighbourhood_temp
    return neighbourhood

# collect the neighbourhood of a node; divide it into 'numBlocks' blocks;
#  optimize this partition
def optimizeNeighbourhood(graph, startingSolution, startingUtility : int, n : int, v : int, numBlocks : int):
    neigh = getNeighbourhood(graph, v, len(graph))
    numBlocks = min(numBlocks, n)
    partition = utils.divideIntoRandomPartition(neigh, numBlocks)
    return optimizePartition(graph, startingSolution, startingUtility, partition)

def optimizeExtendedNeighbourhood(graph, startingSolution, startingUtility : int, n : int, v: int, numBlocks : int, sizeFactor = 1.0):
    neighbourhoodSize = int(math.log2(n) * sizeFactor)
    neighbourhoodSize = min(neighbourhoodSize, n)
    numBlocks = min(numBlocks, n)
    neighbourhood = getExtendedNeighbourhood(graph, v, neighbourhoodSize)
    partition = utils.divideIntoRandomPartition(neighbourhood, numBlocks)
    return optimizePartition(graph, startingSolution, startingUtility, partition)

def optimizeExtendedNeighbourhoodNx(graph, startingSolution, startingUtility : int, n : int, v : int, numBlocks : int, neighbourhoodSize : int):
    neighbourhoodSize = min(neighbourhoodSize, n)
    neighbourhoodSize = max(neighbourhoodSize, numBlocks) # because you can't optimize fewer than the number of blocks
    numBlocks = min(numBlocks, n)
    neighbourhood = getExtendedNeighbourhoodNx(graph, v, neighbourhoodSize)
    partition = utils.divideIntoRandomEqualPartition(neighbourhood, numBlocks)
    return optimizePartitionNx_v2(graph, startingSolution, startingUtility, partition)

# More powerful than version 1, because we also optimize all the edges each time
def optimizeExtendedNeighbourhoodNx_v2(graph, startingSolution, startingUtility : int, n : int, center : int, numBlocks : int, neighbourhood_size):
    neighbourhood_size = min(neighbourhood_size, n)
    neighbourhood_size = max(neighbourhood_size, numBlocks) # because you can't optimize fewer than the number of blocks
    numBlocks = min(numBlocks, n)
    neighbourhood = getExtendedNeighbourhoodNx(graph, center, neighbourhood_size)
    partition = utils.divideIntoRandomEqualPartition(neighbourhood, numBlocks)
    return optimizePartitionAndEdges_v4(graph, startingSolution, startingUtility, partition, neighbourhood)

# improvement over version 2: use numpy.ndarray, and receive 'output' array as parameter
def optimizeExtendedNeighbourhoodNx_v3(solution_write : numpy.ndarray, graph, starting_solution : numpy.ndarray, starting_utility, n : int, center : int, num_blocks : int, neighbourhood_size : int):
    neighbourhood_size = min(neighbourhood_size, n)
    neighbourhood_size = max(neighbourhood_size, num_blocks) # because you can't optimize fewer than the number of blocks
    num_blocks = min(num_blocks, n)
    neighbourhood = getExtendedNeighbourhoodNx(graph, center, neighbourhood_size)
    partition = utils.divideIntoRandomEqualPartition(neighbourhood, num_blocks)
    # TODO the following two lines revert the code back to the 'old' way of doing things.
    #      uncomment the commented line, when the bug is fixed
    # (candidate_solution, candidate_utility) = optimizePartitionAndEdges_v4(graph, starting_solution, starting_utility, partition, neighbourhood)
    # copy_data_inplace_np(solution_write, candidate_solution)
    # TODO uncomment to revert back to v5
    return optimizePartitionAndEdges_v6(solution_write, graph, starting_solution, starting_utility, partition, neighbourhood)
    # return candidate_utility

def solveMaxCutGreedy5(graph, partitionSize = 4):
    if (partitionSize > 8):
        print("I really wouldn't recommend a swapsetsize this large")
        return ([], 0)
    n = utils.getNumVertices(graph)
    m = len(graph)
    budget = 10*m
    bestSolution = utils.getListOfRandomBooleans(n)
    bestUtility = evaluateSolution(graph, bestSolution)
    for round in range(0, budget):
        (solution, utility) = greedy5Round(graph, bestSolution, partitionSize, n, m)
        if (utility > bestUtility):
            bestSolution = solution
            bestUtility = utility
    return (bestSolution, bestUtility)

def solveMaxCutGreedy6(graph, numBlocks = 5, budgetFactor = 1):
    if (numBlocks > 8):
        print("I really wouldn't recommend a swapsetsize this large")
        return ([], 0)
    n = utils.getNumVertices(graph)
    m = len(graph)
    budget = budgetFactor*m
    bestSolution = utils.getListOfRandomBooleans(n)
    bestUtility = evaluateSolution(graph, bestSolution)
    for round in range(0, budget):
        (solution, utility) = optimizeNeighbourhood(graph, bestSolution, bestUtility, n, random.randint(0, n-1), numBlocks)
        if (utility > bestUtility):
            bestSolution = solution
            bestUtility = utility
    return (bestSolution, bestUtility)

def solveMaxCutGreedy7(graph, numBlocks = 5, budgetFactor = 1, sizeFactor = 1.0):
    n = utils.getNumVertices(graph)
    m = len(graph)
    budget = int(budgetFactor*m)
    bestSolution = utils.getListOfRandomBooleans(n)
    bestUtility = evaluateSolution(graph, bestSolution)
    for round in range(0, budget):
        # print('greedy_7 round = ' + str(round) + ' instancenum = ' + str(Instance.instancenum))
        v = utils.chooseRandomVertex(graph, n)
        (solution, utility) = optimizeExtendedNeighbourhood(graph, bestSolution, bestUtility, n, v, numBlocks, sizeFactor)
        if (utility > bestUtility):
            bestSolution = solution
            bestUtility = utility
    return (bestSolution, bestUtility)

def solveMaxCutGreedy7Nx(graph : nx.graph.Graph, numBlocks = 5, budgetFactor = 1.0, sizeFactor = 1.0):
    n = utils.getNumVerticesNx(graph)
    m = graph.number_of_edges()
    budget = int(budgetFactor * m)
    print(f'[solver 7]  Solving max cut with budget {budget}')
    bestSolution = utils.getListOfRandomBooleans(n)
    bestUtility = evaluateSolutionNxList(graph, bestSolution)
    for round in range(budget):
        v = utils.chooseRandomVertex(graph, n)
        (solution, utility) = optimizeExtendedNeighbourhoodNx(graph, bestSolution, bestUtility, n, v, numBlocks, sizeFactor)
        # verifiedUtility = evaluateSolutionNxList(graph, solution)
        # if (utility != verifiedUtility):
        #     print('[solver]  ERROR utility != verifiedUtility')
        if (utility > bestUtility):
            bestSolution = solution
            bestUtility = utility
    return (bestSolution, bestUtility)

def shakeUpNeighbourhood(graph : nx.graph.Graph, solution, center : int, neighbourhoodSize):
    n = utils.getNumVerticesNx(graph)
    neighbourhood = getExtendedNeighbourhoodNx(graph, center, neighbourhoodSize)
    # Flip each edge with 50% probability
    for vertex in neighbourhood:
        if random.randint(0, 1) == 1:
            utils.flipVertex(solution, vertex)

def shake_up_neighbourhood_v2(graph : nx.graph.Graph, solution : numpy.ndarray, neighbourhood : numpy.ndarray, probability : float):
    threshold = int(10000.0 * probability)
    for vertex in neighbourhood:
        if random.randint(0, 10000) <= threshold:
            solution[vertex] = 1 & solution[vertex]

# make the neighbourhood size adaptive:
# if it succeeds, make it 10% bigger; otherwise, 10% smaller
def solve_max_cut_greedy_8_nx(graph : nx.graph.Graph, numBlocks = 6, budgetFactor = 1.0, sizeFactor = 2.0):
    n = utils.getNumVerticesNx(graph)
    m = graph.number_of_edges()
    intermediate_result_file = open('results/intermediate_greedy_8.csv', 'w')
    neighbourhood_size : int = int( math.log2(n) * sizeFactor)
    neighbourhood_grow_factor = 1.1
    budget = int(budgetFactor * m)
    print(f'[solver 8]  Start. Solving max cut with budget {budget}, |neighbourhood| = {neighbourhood_size}  num blocks = {numBlocks}')
    current_solution = utils.getListOfRandomBooleans(n)
    current_utility = evaluateSolutionNxList(graph, current_solution)
    best_solution = current_solution.copy()
    best_utility = current_utility
    shaken_solution = current_solution.copy()
    intermediate_utilities = []
    numEvaluations = 0
    while numEvaluations < int(budget):
        center = utils.chooseRandomVertex(graph, n)
        utility_before_NO = current_utility
        (current_solution, current_utility) = optimizeExtendedNeighbourhoodNx(graph, current_solution, current_utility, n, center, numBlocks, neighbourhood_size)
        numEvaluations += 1
        intermediate_utilities.append(current_utility)
        if (current_utility > best_utility):
            print(f'[round {numEvaluations}] nbh_optimization improved by  {current_utility - utility_before_NO}\tto {current_utility} using |nbh|={neighbourhood_size}')
            copy_data_inplace(best_solution, current_solution)
            best_utility = current_utility
            # increase the size of the neighbourhood
            # make sure the neighbourhood_size increases by at least 1
            neighbourhood_size = max(int(neighbourhood_size * neighbourhood_grow_factor), neighbourhood_size + 2)
        else:
            # try shaking up the neighbourhood, then try again
            copy_data_inplace(shaken_solution, current_solution)
            shakeUpNeighbourhood(graph, shaken_solution, center, neighbourhood_size)
            shaken_utility = evaluateSolutionNxList(graph, shaken_solution)
            (shaken_solution, shaken_utility) = optimizeExtendedNeighbourhoodNx(graph, shaken_solution, shaken_utility, n, center, numBlocks, neighbourhood_size)
            numEvaluations += 1
            if (shaken_utility > best_utility):
                print(f'[round {numEvaluations}] nbh_shakeup      improved by  {shaken_utility - utility_before_NO}\tto {shaken_utility} using |nbh|={neighbourhood_size}')
                copy_data_inplace(best_solution, shaken_solution)
                copy_data_inplace(current_solution, shaken_solution)
                best_utility = shaken_utility
                current_utility = shaken_utility
                # increase the size of the neighbourhood
                # make sure the neighbourhood_size increases by at least 2
                neighbourhood_size = max(int(neighbourhood_size * neighbourhood_grow_factor), neighbourhood_size + 2)
            else:
                # reduce the size of the neighbourhood
                # make sure neighbourhood_size decreases by at least 1
                neighbourhood_size = min(int(neighbourhood_size / neighbourhood_grow_factor), neighbourhood_size - 1)
                # make sure the neighbourhood is not too small
                neighbourhood_size = max(neighbourhood_size, numBlocks)
            intermediate_utilities.append(current_utility)
    for ut in intermediate_utilities:
        intermediate_result_file.write(f'{ut}\n')
    intermediate_result_file.close()
    return (best_solution, best_utility)

# Progress since version 8: use optimizeExtendedNeighbourhood_v2
def solve_max_cut_greedy_9_nx(graph : nx.graph.Graph, numBlocks = 6, budgetFactor = 1.0, sizeFactor = 2.0):
    n = utils.getNumVerticesNx(graph)
    m = graph.number_of_edges()
    intermediate_result_file = open('results/intermediate_greedy_8.csv', 'w')
    neighbourhood_size : int = int( math.log2(n) * sizeFactor)
    neighbourhood_grow_factor = 1.1
    budget = int(budgetFactor * m)
    print(f'[solver 8]  Start. Solving max cut with budget {budget}, |neighbourhood| = {neighbourhood_size}  num blocks = {numBlocks}')
    current_solution = utils.getListOfRandomBooleans(n)
    current_utility = evaluateSolutionNxList(graph, current_solution)
    best_solution = current_solution.copy()
    best_utility = current_utility
    shaken_solution = current_solution.copy()
    intermediate_utilities = []
    numEvaluations = 0
    while numEvaluations < int(budget):
        center = utils.chooseRandomVertex(graph, n)
        utility_before_NO = current_utility
        (current_solution, current_utility) = optimizeExtendedNeighbourhoodNx_v2(graph, current_solution, current_utility, n, center, numBlocks, neighbourhood_size)
        numEvaluations += 1
        # verify_utility(graph, current_solution, current_utility)
        intermediate_utilities.append(current_utility)
        if (current_utility > best_utility):
            print(f'[round {numEvaluations}] nbh_optimization improved by  {current_utility - utility_before_NO}\tto {current_utility} using |nbh|={neighbourhood_size}')
            copy_data_inplace(best_solution, current_solution)
            best_utility = current_utility
            # increase the size of the neighbourhood
            # make sure the neighbourhood_size increases by at least 1
            neighbourhood_size = max(int(neighbourhood_size * neighbourhood_grow_factor), neighbourhood_size + 2)
        else:
            # try shaking up the neighbourhood, then try again
            copy_data_inplace(shaken_solution, current_solution)
            shakeUpNeighbourhood(graph, shaken_solution, center, neighbourhood_size)
            shaken_utility = evaluateSolutionNxList(graph, shaken_solution)
            (shaken_solution, shaken_utility) = optimizeExtendedNeighbourhoodNx_v2(graph, shaken_solution, shaken_utility, n, center, numBlocks, neighbourhood_size)
            # verify_utility(graph, shaken_solution, shaken_utility)
            numEvaluations += 1
            if (shaken_utility > best_utility):
                print(f'[round {numEvaluations}] nbh_shakeup      improved by  {shaken_utility - utility_before_NO}\tto {shaken_utility} using |nbh|={neighbourhood_size}')
                copy_data_inplace(best_solution, shaken_solution)
                copy_data_inplace(current_solution, shaken_solution)
                best_utility = shaken_utility
                current_utility = shaken_utility
                # increase the size of the neighbourhood
                # make sure the neighbourhood_size increases by at least 2
                neighbourhood_size = max(int(neighbourhood_size * neighbourhood_grow_factor), neighbourhood_size + 2)
            else:
                # reduce the size of the neighbourhood
                # make sure neighbourhood_size decreases by at least 1
                neighbourhood_size = min(int(neighbourhood_size / neighbourhood_grow_factor), neighbourhood_size - 1)
                # make sure the neighbourhood is not too small
                neighbourhood_size = max(neighbourhood_size, numBlocks)
            intermediate_utilities.append(current_utility)
    for ut in intermediate_utilities:
        intermediate_result_file.write(f'{ut}\n')
    intermediate_result_file.close()
    return (best_solution, best_utility)

# TODO copy data in place
# TODO compute utility faster than this
def shakeup_and_optimize_neighbourhood_v1(graph, numBlocks, n, neighbourhood_size, best_solution, shaken_solution, center):
    shaken_solution = best_solution.copy()
    shakeUpNeighbourhood(graph, shaken_solution, center, neighbourhood_size)
    shaken_utility = evaluateSolutionNxList(graph, shaken_solution)
    (candidate_solution, candidate_utility) = optimizeExtendedNeighbourhoodNx_v2(graph, shaken_solution, shaken_utility, n, center, numBlocks, neighbourhood_size)
    return (candidate_solution,candidate_utility)

def shakeup_and_optimize_neighbourhood_v2(solution_write : numpy.ndarray, graph : nx.graph.Graph, starting_solution : numpy.ndarray, num_blocks : int, n : int, neighbourhood_size : int, center : int):
    copy_data_inplace_np(solution_write, starting_solution)
    shakeUpNeighbourhood(graph, solution_write, center, neighbourhood_size)
    candidate_utility = evaluate_solution_array(graph, solution_write)
    # if not verify_utility(graph, solution_write, candidate_utility):
    #     print('[shakeup]  <<  ERROR  >>  utility is wrong'); exit();
    candidate_utility = optimizeExtendedNeighbourhoodNx_v3(solution_write, graph, solution_write, candidate_utility, n, center, num_blocks, neighbourhood_size)
    return candidate_utility

# choice between shaking up or not is made better: random
# refactored to be more readable
def solve_max_cut_greedy_10_nx(graph : nx.graph.Graph, numBlocks = 6, budget = 30, sizeFactor = 2.0):
    n = utils.getNumVerticesNx(graph)
    m = graph.number_of_edges()
    intermediate_result_file = open('results/intermediate_greedy_8.csv', 'w')
    neighbourhood_size : int = int( math.log2(n) * sizeFactor)
    neighbourhood_grow_factor = 1.1
    print(f'[solver 8]  Start. Solving max cut with budget {budget}, |neighbourhood| = {neighbourhood_size}  num blocks = {numBlocks}')
    best_solution = utils.getListOfRandomBooleans(n)
    best_utility = evaluateSolutionNxList(graph, best_solution)
    shaken_solution = best_solution.copy()
    intermediate_utilities = []
    numEvaluations = 0
    while numEvaluations < budget:
        numEvaluations += 1
        center = utils.chooseRandomVertex(graph, n)
        # randomly choose strategy to do with, or without shaking up
        strategy = random.randint(0,1)
        print(f'[round {numEvaluations}] using strategy {strategy}   utility = {best_utility}')
        Profile.print()
        if strategy == 0:
            # do not shake it up
            (candidate_solution, candidate_utility) = optimizeExtendedNeighbourhoodNx_v2(graph, best_solution, best_utility, n, center, numBlocks, neighbourhood_size)
        else:
            # shake it up
            (candidate_solution, candidate_utility) = shakeup_and_optimize_neighbourhood_v1(graph, numBlocks, n, neighbourhood_size, best_solution, best_utility, center)
            verify_utility(graph, candidate_solution, candidate_utility)
        if candidate_utility > best_utility:
            copy_data_inplace(best_solution, candidate_solution)
            utility_previous = best_utility
            best_utility = candidate_utility
            # increase the size of the neighbourhood
            # make sure the neighbourhood_size increases by at least 1
            neighbourhood_size = max(int(neighbourhood_size * neighbourhood_grow_factor), neighbourhood_size + 2)
            print(f'[round {numEvaluations}] optimization strat {strategy}      improved by  {best_utility - utility_previous}\tto {best_utility} using |nbh|={neighbourhood_size}')
        else:
            # reduce the size of the neighbourhood
            # make sure neighbourhood_size decreases by at least 1
            neighbourhood_size = min(int(neighbourhood_size / neighbourhood_grow_factor), neighbourhood_size - 1)
            # make sure the neighbourhood is not too small
            neighbourhood_size = max(neighbourhood_size, numBlocks)
    for ut in intermediate_utilities:
        intermediate_result_file.write(f'{ut}\n')
    intermediate_result_file.close()
    return (best_solution, best_utility)

# In this refactoring, we refactor everything to numpy arrays
# we also do not dynamically allocate any new data for use in intermediate processes
def solve_max_cut_greedy_11(graph : nx.graph.Graph, numBlocks = 6, budget = 30, sizeFactor = 2.0):
    n = utils.getNumVerticesNx(graph)
    m = graph.number_of_edges()
    intermediate_result_file = open('results/intermediate_greedy_8.csv', 'w')
    neighbourhood_size : int = int( math.log2(n) * sizeFactor)
    neighbourhood_grow_factor = 1.1
    print(f'[solver 8]  Start. Solving max cut with budget {budget}, |neighbourhood| = {neighbourhood_size}  num blocks = {numBlocks}')
    best_solution = utils.getArrayOfRandomBooleans(n)
    best_utility = evaluate_solution_array(graph, best_solution)
    if not verify_utility(graph, best_solution, best_utility): print('[solve v11] ERROR utility is wrong after first time')
    candidate_solution = best_solution.copy()
    shaken_solution = best_solution.copy()
    intermediate_utilities = []
    numEvaluations = 0
    while numEvaluations < budget:
        numEvaluations += 1
        center = utils.chooseRandomVertex(graph, n)
        # randomly choose strategy to do with, or without shaking up
        strategy = random.randint(0,1)
        print(f'[round {numEvaluations}] using strategy {strategy}   utility = {best_utility}')
        Profile.print()
        if strategy == 0:
            # do not shake it up
            candidate_utility = optimizeExtendedNeighbourhoodNx_v3(candidate_solution, graph, best_solution, best_utility, n, center, numBlocks, neighbourhood_size)
            if not verify_utility(graph, candidate_solution, candidate_utility): exit()
        else:
            # shake it up
            candidate_utility = shakeup_and_optimize_neighbourhood_v2(candidate_solution, graph, best_solution, numBlocks, n, neighbourhood_size, center)
            if not verify_utility(graph, candidate_solution, candidate_utility): exit()
        if candidate_utility > best_utility:
            copy_data_inplace(best_solution, candidate_solution)
            utility_previous = best_utility
            best_utility = candidate_utility
            # increase the size of the neighbourhood
            # make sure the neighbourhood_size increases by at least 1
            neighbourhood_size = max(int(neighbourhood_size * neighbourhood_grow_factor), neighbourhood_size + 2)
            print(f'[round {numEvaluations}] optimization strat {strategy}      improved by  {best_utility - utility_previous}\tto {best_utility} using |nbh|={neighbourhood_size}')
        else:
            # reduce the size of the neighbourhood
            # make sure neighbourhood_size decreases by at least 1
            neighbourhood_size = min(int(neighbourhood_size / neighbourhood_grow_factor), neighbourhood_size - 1)
            # make sure the neighbourhood is not too small
            neighbourhood_size = max(neighbourhood_size, numBlocks)
    for ut in intermediate_utilities:
        intermediate_result_file.write(f'{ut}\n')
    intermediate_result_file.close()
    return (best_solution, best_utility)

def optimizeExtendedNeighbourhoodAndEdges(solution_write : numpy.ndarray, graph : nx.graph.Graph, starting_solution : numpy.ndarray, starting_utility : int, budget : int, center : int, neighbourhood_size : int, p : float):
    neighbourhood = getExtendedNeighbourhoodNx(graph, center, neighbourhood_size)
    candidate_solution = starting_solution.copy()
    candidate_utility = starting_utility
    copy_data_inplace_np(solution_write, starting_solution)
    best_utility : int = starting_utility
    for round in range(budget):
        # shake up the neighbourhood, with probability p
        probability = random.randint(10,100) * p / 100.0
        shake_up_neighbourhood_v2(graph, candidate_solution, neighbourhood, probability)
        candidate_utility = evaluate_solution_array(graph, candidate_solution) # todo we can compute the new utility while we shake it up
        # optimize the edges
        # get the utility
        candidate_utility = optimize_all_edges_in_subgraph_v3(candidate_solution, graph, candidate_solution, candidate_utility, neighbourhood)
        # if not verify_utility(graph, candidate_solution, candidate_utility): print('[optimize nbh edges]  <<  ERROR  >> utility is wrong.'); exit(); # TODO remove when we are more confident
        # if the utility is better, then best solution is this
        if candidate_utility > best_utility:
            copy_data_inplace_np(solution_write, candidate_solution)
            best_utility = candidate_utility
    return best_utility

# We use a slightly different neighbourhood optimization technique:
#   we flip random vertices in a neighbourhood, then optimize each edge; repeat 1000 times for a single neighbourhood
# we scale down the size of the neighbourhood as the experiment goes on, by a predefined schedule
def solve_max_cut_greedy_12(graph : nx.graph.Graph, numBlocks = 6, budget = 30, sizeFactor = 2.0):
    n = utils.getNumVerticesNx(graph)
    m = graph.number_of_edges()
    budget_per_round : int = 50
    probability = 0.5
    neighbourhood_grow_factor = 1.1
    intermediate_result_file = open('results/intermediate_greedy_12.csv', 'w')
    neighbourhood_size : int = int(0.1 * n)
    # neighbourhood_size : int = int(math.log2(n) * sizeFactor)
    print(f'[solver 12]  Start. Solving max cut with budget {budget}, |neighbourhood| = {neighbourhood_size} budget={budget_per_round}')
    best_solution = utils.getArrayOfRandomBooleans(n)
    best_utility = evaluate_solution_array(graph, best_solution)
    candidate_solution = best_solution.copy()
    intermediate_utilities = []
    numEvaluations = 0
    while numEvaluations < budget:
        numEvaluations += 1
        Profile.print()
        center = utils.chooseRandomVertex(graph, n)
        # neighbourhood_size = int(n - numEvaluations * (n - 20) / float(budget))
        strategy = random.randint(0,2)
        print(f'[solve 12]  round {numEvaluations} | utility = {best_utility}  | |nbh| = {neighbourhood_size}  |  strategy = {strategy}')
        if strategy == 0:
            # shake it up
            candidate_utility = shakeup_and_optimize_neighbourhood_v2(candidate_solution, graph, best_solution, numBlocks, n, neighbourhood_size, center)
            # candidate_utility = optimizeExtendedNeighbourhoodAndEdges(candidate_solution, graph, best_solution, best_utility, budget_per_round, center, neighbourhood_size, probability)
        # elif strategy == 1:
        else:
            candidate_utility = optimizeExtendedNeighbourhoodNx_v3(candidate_solution, graph, best_solution, best_utility, n, center, numBlocks, neighbourhood_size)
            # if not verify_utility(graph, candidate_solution, candidate_utility): exit()
        # else:
        if candidate_utility > best_utility:
            copy_data_inplace_np(best_solution, candidate_solution)
            best_utility = candidate_utility
            neighbourhood_size = max(int(neighbourhood_size * neighbourhood_grow_factor), neighbourhood_size + 2)
            # nbh cannot grow to more than 10% of the vertices
            neighbourhood_size = min(neighbourhood_size, int(0.1 * n))
        else:
            # reduce the size of the neighbourhood
            # make sure neighbourhood_size decreases by at least 1
            neighbourhood_size = min(int(neighbourhood_size / neighbourhood_grow_factor), neighbourhood_size - 1)
            # make sure the neighbourhood is not too small
            neighbourhood_size = max(neighbourhood_size, numBlocks, 20)
        intermediate_utilities.append(best_utility)
    for ut in intermediate_utilities:
        intermediate_result_file.write(f'{ut}\n')
    intermediate_result_file.close()
    return (best_solution, best_utility)




# Pick a strategy at random
def solveMaxCutDiverse1(graph):
    n = utils.getNumVertices(graph)
    m = len(graph)
    budget = 10*m
    maxSwapSize = int(math.log2(n))
    bestSolution = n * [ 0 ]
    utils.setListToRandomBooleans(bestSolution)
    bestUtility = evaluateSolution(graph, bestSolution)
    for round in range(0, budget):
        # Either use 3, or 5
        strategy = random.randint(0, 1)
        if (strategy == 0):
            (solution, utility) = greedy3Round(graph, bestSolution, n, maxSwapSize)
        else:
            (solution, utility) = greedy5Round(graph, bestSolution, 4, n, m)
        if (utility > bestUtility):
            bestSolution = solution
            bestUtility = utility
    return (bestSolution, bestUtility)

# Start with one strategy. Then, if it succeeds in finding an improvement, try it again
# Otherwise, choose a random strategy
def solveMaxCutDiverse2(graph):
    n = utils.getNumVertices(graph)
    m = len(graph)
    budget = 10*m
    maxSwapSize = int(math.log2(n))
    numBlocks = 5
    bestSolution = n * [ 0 ]
    utils.setListToRandomBooleans(bestSolution)
    bestUtility = evaluateSolution(graph, bestSolution)
    forcedStrategy = -1 # -1 means the strategy should be chosen randomly
    numStrategies = 4
    improvementsCount = { 0: 0, 1: 0, 2: 0, 3:0}
    improvementsContrib = {0: 0, 1: 0, 2: 0, 3: 0}
    for round in range(0, budget):
        # Either use 3, or 5
        if (forcedStrategy == -1):
            strategy = random.randint(0, numStrategies - 1)
        else:
            strategy = forcedStrategy
        # Find a new solution
        if (strategy == 0):
            (solution, utility) = greedy3Round(graph, bestSolution, n, maxSwapSize)
        elif (strategy == 1):
            (solution, utility) = greedy5Round(graph, bestSolution, numBlocks, n, m)
        elif (strategy == 2):
            (solution, utility) = optimizeNeighbourhood(graph, bestSolution, bestUtility, n, random.randint(0, n-1), numBlocks)
        else:
            (solution, utility) = optimizeRandomVertex(graph, bestSolution, bestUtility, n)
        if (utility > bestUtility):
            # print('Strategy ' + str(strategy) + ' made it better; dif = -' + str(bestUtility - utility))
            improvementsCount[strategy] += 1
            improvementsContrib[strategy] += bestUtility - utility
            bestSolution = solution
            bestUtility = utility
            # forcedStrategy = strategy
        else:
            forcedStrategy = -1
    for s in range(0, numStrategies):
        print('Strategy ' + str(s) + ' improved ' + str(improvementsCount[s]) + ' times;\ttotal = ' + str(improvementsContrib[s]))
    return (bestSolution, bestUtility)

def solveMaxCutEvolutionary_1(graph):
    n = utils.getNumVertices(graph)
    m = len(graph)
    numBlocks = 4
    budget = m
    populationSize = 10
    population = populationSize * [ n * [ 0 ] ]
    utility = populationSize * [ 0 ]
    for i in range(len(population)):
        setListToRandomBooleans(population[i])
        utility[i] = evaluateSolution(graph, population[i])
    for round in range(budget):
        # mutate all the individuals
        for i in range(len(population)):
            (solution, utilityOptimized) = optimizeExtendedNeighbourhood(graph, population[i], utility[i], n, random.randint(0, n-1), numBlocks)
            if (utilityOptimized > utility[i]):
                population[i] = solution
                utility[i] = utilityOptimized

        # I want to end with basically just one individual
        # So I want to kill the weakest individual every so many rounds.
        # So I will kill the weakest individual with probability p each round;
        # and replace him with a copy of the winner
        # What should be p, given how many rounds I have?
        # after the round, the expected size of the population is
        # E[size] = p * (size - 1) + (1-p) * size = size - p
        # So the probability should be populationSize / budget

        # With probability populationSize / budget, kill the weakest one

