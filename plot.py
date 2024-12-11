import matplotlib.pyplot as plot
import numpy as np
import csvutils

results2All = csvutils.getDataFromCsvFile('results/greedy_2.csv')
results2utility = csvutils.getColumn(results2All, 2)
results3All = csvutils.getDataFromCsvFile('results/greedy_3.csv')
results3utility = csvutils.getColumn(results3All, 2)
results6All = csvutils.getDataFromCsvFile('results/greedy_6.csv')
results6utility = csvutils.getColumn(results6All, 2)
results7All = csvutils.getDataFromCsvFile('results/greedy_7.csv')
results7utility = csvutils.getColumn(results7All, 2)
results7_1All = csvutils.getDataFromCsvFile('results/greedy_7_1.csv')
results7_1utility = csvutils.getColumn(results7_1All, 2)

plottype = 'diff'

if (plottype == 'cactus'):
    ####    CACTUS PLOT
    results3Sorted = results3utility.copy()
    results3Sorted.sort()
    results6Sorted = results6utility.copy()
    results6Sorted.sort()

    plot.plot(results3Sorted)
    plot.plot(results6Sorted)
    plot.title("Cactus plot")

####    DIFFERENCE PLOT
elif (plottype == 'diff'):
    diff_3_6 = csvutils.getDifference(results3utility, results7_1utility)
    diff_3_6.sort()
    plot.plot(diff_3_6, label="3 - 7_1")

    diff_2_6 = csvutils.getDifference(results2utility, results6utility)
    diff_2_6.sort()
    plot.plot(diff_2_6, label="2 - 7_1")

    diff_6_7 = csvutils.getDifference(results6utility, results7_1utility)
    diff_6_7.sort()
    plot.plot(diff_6_7, label="6 - 7_1")

    diff_7_71 = csvutils.getDifference(results7utility, results7_1utility)
    diff_7_71.sort()
    plot.plot(diff_7_71, label="7 - 7_1")

    plot.legend(loc="upper left")
    plot.title("Difference")

####    SCATTER PLOT
elif (plottype == 'scatter'):
    minresult = min(min(results3utility), min(results6utility))
    maxresult = max(max(results3utility), max(results6utility))

    print('results3 : ' + str(results3utility))

    plot.scatter(results3utility, results6utility)
    gca = plot.gca()
    gca.set_xlim(minresult - 5, maxresult + 5)
    gca.set_ylim(minresult - 5, maxresult + 5)
    plot.title("Scatter plot")
    # legend = plot.legend()
    # legend.set_title("legend title")

# TODO add LEGEND
plot.show()