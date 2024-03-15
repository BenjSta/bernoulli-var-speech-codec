import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import scienceplots

plt.style.use('default')
# plt.style.use('science')
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
        _, medi, int25, int75 = calcMedMeanInter(data, metric)
        median.append(medi)
        inter25.append(int25)
        inter75.append(int75)
    # coef = np.polyfit(x,median,3)
    # poly1d_fn = np.poly1d(coef) 
    return plt.plot(x,median, color, linestyle='dashed' )
#    plt.fill_between(x,inter25, inter75,alpha=.1, color=color)





def main():
    clean = pd.read_csv('testResult/clean.csv', index_col=0)
    vocoded = pd.read_csv('testResult/vocoded.csv', index_col=0)

    var16 = pd.read_csv('testResult/variable16.csv', index_col=0)
    var24 = pd.read_csv('testResult/variable24.csv', index_col=0)
    var32 = pd.read_csv('testResult/variable32.csv', index_col=0)
    var64 = pd.read_csv('testResult/variable64.csv', index_col=0)

    var16_2 = pd.read_csv('testResult/variable16_2.csv', index_col=0)
    var24_2 = pd.read_csv('testResult/variable24_2.csv', index_col=0)
    var32_2 = pd.read_csv('testResult/variable32_2.csv', index_col=0)
    var64_2 = pd.read_csv('testResult/variable64_2.csv', index_col=0)

    encodec1_5 = pd.read_csv('testResult/encodec1_5.csv', index_col=0)
    encodec3 = pd.read_csv('testResult/encodec3.csv', index_col=0)
    encodec6 = pd.read_csv('testResult/encodec6.csv', index_col=0)
    encodec12 = pd.read_csv('testResult/encodec12.csv', index_col=0)
    lyra3_2 = pd.read_csv('testResult/lyra3_2.csv', index_col=0)
    lyra6 = pd.read_csv('testResult/lyra6.csv', index_col=0)
    lyra9_2 = pd.read_csv('testResult/lyra9_2.csv', index_col=0)
    opus6 = pd.read_csv('testResult/opus6.csv', index_col=0)
    opus10 = pd.read_csv('testResult/opus10.csv', index_col=0)
    opus14= pd.read_csv('testResult/opus14.csv', index_col=0)

    data = [clean, vocoded, encodec1_5, encodec3, encodec6, encodec12, 
            lyra3_2, lyra6, lyra9_2, var16, var24, var32, var64,  var16_2, var24_2, var32_2, var64_2]

    plot_array = [encodec1_5, encodec3, encodec6, encodec12, lyra3_2, lyra6, lyra9_2, var16, var24, var32, var64,  var16_2, var24_2, var32_2, var64_2]
    color_array = ['blue', 'blue', 'blue', 'blue', 'orange', 'orange', 'orange' ,'green', 'green', 'green', 'green', 'red', 'red', 'red', 'red']
    kbps = [1.5, 3.0, 6, 12, 3.2, 6, 9.2, 1.38, 2.07, 2.76, 5.51, 1.38, 2.07, 2.76, 5.51]
    metric = 'pesq'

    plt.figure(dpi=300)
    for data, color, bit in zip(plot_array, color_array, kbps):
        singleErrorBar(data, metric, bit, color, 'o')
    h1 = medianTrend(plot_array[0:4], metric, kbps[0:4], 'blue')
    h2 = medianTrend(plot_array[4:7], metric, kbps[4:7], 'orange')
    h3 = medianTrend(plot_array[7:11], metric, kbps[7:11], 'green')
    h4 = medianTrend(plot_array[11:15], metric, kbps[11:15], 'red')
    plt.xlim([1,12.5])
    #plt.ylim([2.5,4.3])
    plt.legend([h1[0], h2[0], h3[0], h4[0]], ['Encodec', 'Lyra', 'Proposed', 'ProposedBigVGAN'], loc='lower right')
    plt.ylabel('PESQ')
    plt.xlabel('Bitrate in kbps')
    plt.grid()
    plt.savefig('test.png')



if __name__ == "__main__":
    main()
