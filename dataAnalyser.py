import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'nature'])
# plt.style.use(['science','grid'])

def calcMedMeanInter(data, metric,):
    mean = np.mean(data.loc[:, 'lengths'] * np.array(data.loc[:, metric])) / np.mean(data.loc[:, 'lengths'])
    median = np.median(data.loc[:, metric])
    inter25 = np.nanquantile(data.loc[:, metric], 0.25)
    inter75 = np.nanquantile(data.loc[:, metric], 0.75)
    return mean, median, inter25, inter75

def singleErrorBar(data, metric, x,color, marker):
    mean, median, inter25, inter75 = calcMedMeanInter(data, metric)
    plt.errorbar(x, median, yerr = np.array([[median-inter25], [inter75-median]]),color=color, capsize=3, fmt=marker)
#    plt.text(x + 0.2, median - 0.2, name, fontsize=8)

def medianTrend(plotData, metric, x, color):
    median = []
    inter25 = []
    inter75 = []
    for data in plotData:
        mean, medi, int25, int75 = calcMedMeanInter(data, metric)
        median.append(medi)
        inter25.append(int25)
        inter75.append(int75)
    #coef = np.polyfit(x,median,2)
    #poly1d_fn = np.poly1d(coef)
    plt.plot(x, median, color, linestyle='dashed' )
    # plt.fill_between(x,inter25, inter75,alpha=.1, color=color)





def main():
    clean = pd.read_csv('testResult/clean.csv', index_col=0)
    vocoded = pd.read_csv('testResult/vocoded.csv', index_col=0)
    encodec1_5 = pd.read_csv('testResult/encodec1_5.csv', index_col=0)
    encodec3 = pd.read_csv('testResult/encodec3.csv', index_col=0)
    encodec6 = pd.read_csv('testResult/encodec6.csv', index_col=0)
    encodec12 = pd.read_csv('testResult/encodec12.csv', index_col=0)
    lyra3_2 = pd.read_csv('testResult/lyra3_2.csv', index_col=0)
    lyra6 = pd.read_csv('testResult/lyra6.csv', index_col=0)
    lyra9_2 = pd.read_csv('testResult/lyra9_2.csv', index_col=0)
    opus6 = pd.read_csv('testResult/opus6.csv', index_col=0)
    opus10 = pd.read_csv('testResult/opus10.csv', index_col=0)
    opus14 = pd.read_csv('testResult/opus14.csv', index_col=0)
    variable8 = pd.read_csv('testResult/variable8.csv', index_col=0)
    variable12 = pd.read_csv('testResult/variable12.csv', index_col=0)
    variable16 = pd.read_csv('testResult/variable16.csv', index_col=0)
    variable24 = pd.read_csv('testResult/variable24.csv', index_col=0)
    variable32 = pd.read_csv('testResult/variable32.csv', index_col=0)
    variable64 = pd.read_csv('testResult/variable64.csv', index_col=0)

    data = [clean, vocoded, encodec1_5, encodec3, encodec6, encodec12, 
            lyra3_2, lyra6, lyra9_2, opus6, opus10, opus14, variable8,
            variable16, variable24, variable32, variable64]

    plot_array = [encodec1_5, encodec3, encodec6, lyra3_2, lyra6, lyra9_2, opus6, opus10, opus14, variable8, variable12, variable16, variable24, variable32, variable64 ]
    color_array = ['blue', 'blue', 'blue', 'orange', 'orange', 'orange' ,'green', 'green', 'green', 'red', 'red', 'red', 'red', 'red', 'red']
    # color_array = ['blue', 'blue', 'blue', 'orange', 'orange', 'orange' ,'green', 'green', 'red', 'red', 'red', 'red', 'red', 'red']
    names = ['Encodec', 'Encodec', 'Encodec', 'Lyra', 'Lyra', 'Lyra', 'Opus', 'Opus']
    kbps = [1.5, 3.0, 6, 3.2, 6, 9.2, 6, 10, 14, 0.69, 1.0, 1.37, 2.07, 2.76, 5.5]
    metric = 'visqol'

    plt.figure(dpi=300)
    for data, color, bit in zip(plot_array, color_array, kbps):
        singleErrorBar(data, metric, bit, color, 'o')
    medianTrend(plot_array[0:3], metric, kbps[0:3], 'blue')
    medianTrend(plot_array[3:6], metric, kbps[3:6], 'orange')
    medianTrend(plot_array[6:9], metric, kbps[6:9], 'green')
    medianTrend(plot_array[9:15], metric, kbps[9:15], 'red')
    plt.xlim([0 ,10.5])
    plt.ylim([0,5])
    plt.ylabel(metric)
    plt.xlabel('Bitrate in kbps')
    plt.grid()
    plt.savefig(metric + '.png')



if __name__ == "__main__":
    main()
