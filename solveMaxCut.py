import networkx as nx
import math
import random
import utils

# graph is a list of lists
# solution is list of n Boolean values
# returns number of edges in the cut, i.e., between the two components
def evaluateSolution(graph, solution):
    utility = 0
    for edge in graph:
        if (solution[edge[0]] != solution[edge[1]]):
            utility += 1
    return utility;

def evaluateSolutionNx(graph : nx.graph.Graph, solution : set[int]):
    utility = 0
    for edge in graph.edges():
        # Add a point of utility if the two vertices are in different sets
        c0 : bool = solution.__contains__(edge[0])
        c1 : bool = solution.__contains__(edge[1])
        if (c0 != c1):
            utility += 1
    return utility

# sets each element of the list 'target' to a random 0/1 value
# modifies in-place to prevent dynamic memory allocation
def setListToRandomBooleans(target):
    for v in range(0, len(target)):
        target[v] = random.randint(0, 1)
    return target

def getListOfRandomBooleans(n : int):
    l = n * [ 0 ]
    setListToRandomBooleans(l)
    return l

# Naive solution to maxcut:
#   guess a random solution. No heuristics or clever tricks
#   the best of many attempts is returned
def solveMaxCutRandom(graph, attempts):
    n = utils.getNumVertices(graph)
    solution = n * [ 0 ]
    bestUtility = len(graph)
    bestSolution = solution
    for attempt in range(0, attempts):
        setListToRandomBooleans(solution)
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
    setListToRandomBooleans(solution)
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
    setListToRandomBooleans(solution)
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
    setListToRandomBooleans(solution)
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

def optimizeAllVerticesGreedily(graph, startingSolution : int, n : int, startingUtility : int):
    foundImprovement = True
    while (foundImprovement):
        for v in range(0, n):
            foundImprovement = False
            (startingSolution, newUtility) = optimizeVertex(graph, startingSolution, startingUtility, v)
            if (newUtility > startingUtility):
                foundImprovement = True
    return (startingSolution, startingUtility)

def solveMaxCutGreedy3Refactored(graph):
    n = utils.getNumVertices(graph)
    m = len(graph)
    maxSwapSize = int(math.log2(n))
    budget = 10*m # each edge can be changed 10 times (which is not a lot, imo)
    bestSolution = n * [ 0 ]
    setListToRandomBooleans(bestSolution)
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
    setListToRandomBooleans(solution)
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

# Continue to get neighbours until maxNeighbours is hit
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

def solveMaxCutGreedy5(graph, partitionSize = 4):
    if (partitionSize > 8):
        print("I really wouldn't recommend a swapsetsize this large")
        return ([], 0)
    n = utils.getNumVertices(graph)
    m = len(graph)
    budget = 10*m
    bestSolution = n * [ 0 ]
    setListToRandomBooleans(bestSolution)
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
    bestSolution = getListOfRandomBooleans(n)
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
    bestSolution = getListOfRandomBooleans(n)
    bestUtility = evaluateSolution(graph, bestSolution)
    for round in range(0, budget):
        # print('greedy_7 round = ' + str(round) + ' instancenum = ' + str(Instance.instancenum))
        (solution, utility) = optimizeExtendedNeighbourhood(graph, bestSolution, bestUtility, n, random.randint(0, n-1), numBlocks, sizeFactor)
        if (utility > bestUtility):
            bestSolution = solution
            bestUtility = utility
    return (bestSolution, bestUtility)

# Pick a strategy at random
def solveMaxCutDiverse1(graph):
    n = utils.getNumVertices(graph)
    m = len(graph)
    budget = 10*m
    maxSwapSize = int(math.log2(n))
    bestSolution = n * [ 0 ]
    setListToRandomBooleans(bestSolution)
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
    setListToRandomBooleans(bestSolution)
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