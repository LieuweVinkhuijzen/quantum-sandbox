import sys
import math
import csvutils

# Compare:
#  - number of times A was better than B; B was better than A (in %)
#  - average improvement when A was better than B; vice versa
#  - standard deviations

class ResultComparison:
    def __init__(self):
        self.n1isbetter = 0
        self.n2isbetter = 0
        self.totalImprovement1beats2 = 0
        self.totalImprovement2beats1 = 0
        self.totalsquared1 = 0
        self.totalsquared2 = 0

    def getPercentage1beats2(self):
        return float(100 * self.n1isbetter) / (self.n1isbetter + self.n2isbetter)
    
    def getPercentage2beats1(self):
        return float(100 * self.n2isbetter) / (self.n1isbetter + self.n2isbetter)
    
    def averageImprovement1beats2(self):
        return float(self.totalImprovement1beats2) / self.n1isbetter
    
    def averageImprovement2beats1(self):
        return float(self.totalImprovement2beats1) / self.n2isbetter
    
    def getNumDatapoints(self):
        return self.n1isbetter + self.n2isbetter

    def getStdev(self, j : int):
        pass

def compareCounts(res1, res2):
    if len(res1) != len(res2):
        print('results do not have the same length!')
        return None
    comp = ResultComparison()
    for i in range(len(res1)):
        if res1[i] < res2[i]:
            comp.n1isbetter += 1
            comp.totalImprovement1beats2 += res2[i] - res1[i]
        elif res2[i] < res1[i]:
            comp.n2isbetter += 1
            comp.totalImprovement2beats1 += res1[i] - res2[i]
    return comp


file1 = sys.argv[1]
file2 = sys.argv[2]

# greedy2 = 'results/greedy_2.csv'
# greedy7 = 'results/greedy_7.csv'

results2All = csvutils.getDataFromCsvFile(file1)
results2utility = csvutils.getColumn(results2All, 2)
# results3All = getDataFromCsvFile('results/greedy_3.csv')
# results3utility = getColumn(results3All, 2)
# results6All = getDataFromCsvFile('results/greedy_6.csv')
# results6utility = getColumn(results6All, 2)
# results7All = getDataFromCsvFile('results/greedy_7.csv')
# results7utility = getColumn(results7All, 2)
results7_1All = csvutils.getDataFromCsvFile(file2)
results7_1utility = csvutils.getColumn(results7_1All, 2)

comp = compareCounts(results2utility, results7_1utility)
print(f'Comparison of A = {file1}  vs  B = {file2}')
print(f'A is better {comp.n1isbetter} times ({comp.getPercentage1beats2():.1f} %)')
print(f'B is better {comp.n2isbetter} times ({comp.getPercentage2beats1():.1f} %)')
print(f'A is better by {comp.averageImprovement1beats2():.1f} on average when better')
print(f'B is better by {comp.averageImprovement2beats1():.1f} on average when better')