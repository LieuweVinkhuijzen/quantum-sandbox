import random
import utils

def getEqualCommunities(n : int, numBlocks : int):
    c = []
    for v in range(n):
        c.append(v % numBlocks)
    return c

def getRandomEqualCommunities(n : int, numBlocks : int):
    c = getEqualCommunities(n, numBlocks)
    doRandomShuffle(c, numBlocks)
    return c

# put each vertex in a new random block
def doRandomShuffle(community, numBlocks : int):
    n = len(community)
    for v in range(len(community)):
        v2 = random.randint(0, n-1)
        swapVertices(community, v, v2)

def evaluateCommunities(graph, community):
    penalty = 0
    # Add a penalty for each inter-community bond
    for edge in graph:
        if community[edge[0]] != community[edge[1]]:
            penalty += 1
    # Add penalties if one community is too big... but scale that penalty in relation to the number of edges.
    # Otherwise, the trivial solution is 1 big community
    return penalty

def swapVertices(community, v1 : int, v2 : int):
    temp = community[v1]
    community[v1] = community[v2]
    community[v2] = temp

# Swap two random vertices
# repeat 'numMutations' times
# this maintains the relative sizes of the communities
def mutate_v1(graph, startingCommunity : list, numMutations : int, numBlocks : int):
    community = startingCommunity.copy()
    n = len(community)
    for j in range(numMutations):
        v1 = random.randint(0, n-1)
        v2 = random.randint(0, n-1)
        swapVertices(community, v1, v2)
    utility = evaluateCommunities(graph, community)
    return (community, utility)

def detectCommunities_v1(graph, numBlocks : int):
    n = utils.getNumVertices(graph)
    bestCommunity = getRandomEqualCommunities(n, numBlocks)
    bestUtility = evaluateCommunities(graph, bestCommunity)
    initialUtility = bestUtility
    budget = n
    for round in range(budget):
        (community, utility) = mutate_v1(graph, bestCommunity, int(0.1*n), numBlocks)
        if (utility < bestUtility):
            print(f'    improvement: {bestUtility} -> {utility}  = +{bestUtility - utility}')
            bestCommunity = community
            bestUtility = utility
    
    print(f'Solved instance. util: {initialUtility} -> {bestUtility}  = +{initialUtility - bestUtility}')
    return (bestCommunity, bestUtility)

def detectCommunities_v2(graph, numBlocks : int):
    n = utils.getNumVertices(graph)
    bestCommunity = getRandomEqualCommunities(n, numBlocks)
    bestUtility = evaluateCommunities(graph, bestCommunity)
    initialUtility = bestUtility
    budget = n
    for round in range(budget):
        swapSize = random.randint(1, int(0.1*n))
        (community, utility) = mutate_v1(graph, bestCommunity, swapSize, numBlocks)
        if (utility < bestUtility):
            bestCommunity = community
            bestUtility = utility
    
    print(f'Solved instance. util: {initialUtility} -> {bestUtility}  = +{initialUtility - bestUtility}')
    return (bestCommunity, bestUtility)
