import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import scienceplots

plt.style.use(['science', 'nature'])
plt.rcParams.update({'figure.autolayout': True})
# plt.style.use(['science','grid'])

def calcMedMeanInter(data, metric,):
    mean = np.mean(data.loc[:, 'lengths'] * np.array(data.loc[:, metric])) / np.mean(data.loc[:, 'lengths'])
    median = np.median(data.loc[:, metric])
    inter25 = np.nanquantile(data.loc[:, metric], 0.25)
    inter75 = np.nanquantile(data.loc[:, metric], 0.75)
    return mean, median, inter25, inter75

def singleErrorBar(data, metric, x,color, marker):
    mean, median, inter25, inter75 = calcMedMeanInter(data, metric)
    return plt.errorbar(x, mean, yerr = np.array([[median-inter25], [inter75-median]]),color=color, capsize=3, fmt=marker)
#    plt.text(x + 0.2, median - 0.2, name, fontsize=8)

def medianTrend(plotData, metric, x, color, distance=False):
    median = []
    inter25 = []
    inter75 = []
    if not distance:
        for data in plotData:
            mean, medi, int25, int75 = calcMedMeanInter(data, metric)
            median.append(medi)
            inter25.append(int25)
            inter75.append(int75)
    else:
        for data in metric:
            mean, medi, int25, int75 = calcMedMeanInter(plotData, data)
            median.append(medi)
            inter25.append(int25)
            inter75.append(int75)     
    #coef = np.polyfit(x,median,2)
    #poly1d_fn = np.poly1d(coef)
    return plt.plot(x, median, color, linestyle='dashed' )
    # plt.fill_between(x,inter25, inter75,alpha=.1, color=color)

def export_legend(legend, filename, expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)



def main():   
    # vocoded = pd.read_csv('testResult/old/vocoded.csv', index_col=0)
    encodec1_5 = pd.read_csv('testResult/encodec1_5.csv', index_col=0)
    encodec3 = pd.read_csv('testResult/encodec3.csv', index_col=0)
    encodec6 = pd.read_csv('testResult/encodec6.csv', index_col=0)
    encodec12 = pd.read_csv('testResult/encodec12.csv', index_col=0)
    lyra3_2 = pd.read_csv('testResult/lyra3_2.csv', index_col=0)
    lyra6 = pd.read_csv('testResult/lyra6.csv', index_col=0)
    lyra9_2 = pd.read_csv('testResult/lyra9_2.csv', index_col=0)
    opus6 = pd.read_csv('testResult/opus6.csv', index_col=0)
    opus8 = pd.read_csv('testResult/opus8.csv', index_col=0)
    opus10 = pd.read_csv('testResult/opus10.csv', index_col=0)
    opus12 = pd.read_csv('testResult/opus12.csv', index_col=0)
    # variable8 = pd.read_csv('testResult/variable8.csv', index_col=0)
    # variable12 = pd.read_csv('testResult/variable12.csv', index_col=0)
    # variable16 = pd.read_csv('testResult/variable16.csv', index_col=0)
    # variable24 = pd.read_csv('testResult/variable24.csv', index_col=0)
    # variable32 = pd.read_csv('testResult/variable32.csv', index_col=0)
    # variable64 = pd.read_csv('testResult/variable64.csv', index_col=0)
    distances = pd.read_csv('testResult/Distances_all.csv', index_col=0)
    quant_16_nft = pd.read_csv('testResult/quantizer16.csv', index_col=0)
    quant_24_nft = pd.read_csv('testResult/quantizer24.csv', index_col=0)
    quant_32_nft = pd.read_csv('testResult/quantizer32.csv', index_col=0)
    quant_64_nft = pd.read_csv('testResult/quantizer64.csv', index_col=0)
    quant_16_ft = pd.read_csv('testResult/quantizer16_ft.csv', index_col=0)
    quant_24_ft = pd.read_csv('testResult/quantizer24_ft.csv', index_col=0)
    quant_32_ft = pd.read_csv('testResult/quantizer32_ft.csv', index_col=0)
    quant_64_ft = pd.read_csv('testResult/quantizer64_ft.csv', index_col=0)

    quant_16_nft_val = pd.read_csv('testResult/quantizer16_val.csv', index_col=0)
    quant_24_nft_val = pd.read_csv('testResult/quantizer24_val.csv', index_col=0)
    quant_32_nft_val = pd.read_csv('testResult/quantizer32_val.csv', index_col=0)
    quant_64_nft_val = pd.read_csv('testResult/quantizer64_val.csv', index_col=0)
    quant_16_ft_val = pd.read_csv('testResult/quantizer16_ft_val.csv', index_col=0)
    quant_24_ft_val = pd.read_csv('testResult/quantizer24_ft_val.csv', index_col=0)
    quant_32_ft_val = pd.read_csv('testResult/quantizer32_ft_val.csv', index_col=0)
    quant_64_ft_val = pd.read_csv('testResult/quantizer64_ft_val.csv', index_col=0)

    # data = [clean, vocoded, encodec1_5, encodec3, encodec6, encodec12,
    #        lyra3_2, lyra6, lyra9_2, opus6, opus10, opus14, variable8,
    #        variable16, variable24, variable32, variable64]
    
    # data = [clean, vocoded, encodec1_5, encodec3, encodec6, encodec12,
    #        lyra3_2, lyra6, lyra9_2, opus6, opus10, opus14, variable8,
    #        variable16, variable24, variable32, variable64

    ## Metriken
    # plot_array = [encodec1_5, encodec3, encodec6, encodec12, lyra3_2, lyra6, lyra9_2, opus6, opus8, opus10, opus12]
    # color_array = ['blue', 'blue', 'blue', 'blue', 'orange', 'orange' ,'orange', 'red', 'red', 'red', 'red']
    # names = ['Encodec', 'Encodec', 'Encodec', 'Lyra', 'Lyra', 'Lyra', 'Opus', 'Opus']
    # kbps = [1.5, 3, 5.9, 11.9, 3.2, 6.1, 9.2, 6, 8, 10, 12.1]
    # metric = 'pesq'

    # plt.rcParams.update({'figure.autolayout': True})

    # plt.figure(dpi=300)
    # for data, color, bit in zip(plot_array, color_array, kbps):
    #     singleErrorBar(data, metric, bit, color, 'o')
    # h1 = medianTrend(plot_array[0:4], metric, kbps[0:4], 'blue')
    # h2 = medianTrend(plot_array[4:7], metric, kbps[4:7], 'orange')
    # h3 = medianTrend(plot_array[7:11], metric, kbps[7:11], 'red')
    # # h4 = medianTrend(plot_array[11:15], metric, kbps[11:15], 'red')
    # plt.xlim([1,12.5])
    # #plt.ylim([2.5,4.3])
    # # legend = plt.legend([h1[0], h2[0], h3[0]], ['Encodec', 'Lyra', 'Opus'], loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, ncol=1)
    # # export_legend(legend, 'metricsLegend.png')
    # plt.ylabel('PESQ')
    # plt.xlabel('bitrate in kbps')
    # plt.xticks([1.5, 3, 3.2, 6, 12, 9.2, 8, 10],['1.5', '3','\n 3.2', '6', '12','9.2','8','10'])
    # plt.grid()
    # plt.savefig(metric + '.png')

    ## vergleich normal zu fine tuned

    # plot_array = [quant_16_nft, quant_16_ft, quant_24_nft, quant_24_ft, quant_32_nft, quant_32_ft, quant_64_nft, quant_64_ft]
    plot_array = [quant_16_nft_val, quant_16_ft_val, quant_24_nft_val, quant_24_ft_val, quant_32_nft_val, quant_32_ft_val, quant_64_nft_val, quant_64_ft_val]
    color_array = ['blue', 'blue', 'blue', 'blue', 'red', 'red', 'red', 'red']
    kbps = [1.38, 2.07, 2.76, 5.51, 1.38, 2.07, 2.76, 5.51]
    metric = 'nisqa_mos48'

    plt.rcParams.update({'figure.autolayout': True})

    plt.figure(dpi=300)
    for data, color, bit in zip(plot_array, color_array, kbps):
        singleErrorBar(data, metric, bit, color, 'o')
    h1 = medianTrend(plot_array[0:4], metric, kbps[0:4], 'blue')
    h2 = medianTrend(plot_array[4:8], metric, kbps[4:8], 'red')
    plt.xlim([1,6.5])
    #plt.ylim([2.5,4.3])
    # legend = plt.legend([h1[0], h2[0], h3[0]], ['Encodec', 'Lyra', 'Opus'], loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, ncol=1)
    # export_legend(legend, 'metricsLegend.png')
    plt.ylabel('NISQA')
    plt.xlabel('bitrate in kbps')
    plt.xticks([1.38, 2.07, 2.76, 5.51])
    plt.grid()
    plt.savefig(metric +'_val_vq_comp' '.png')
    

    # # distance plot
    # plot_array_L1 = ['L1_fix_16','L1_fix_24','L1_fix_32','L1_fix_64','L1_var_16','L1_var_24','L1_var_32','L1_var_64', 'L1_quant_16', 'L1_quant_24', 'L1_quant_32', 'L1_quant_64']
    # plot_array_L2 = ['L2_fix_16','L2_fix_24','L2_fix_32','L2_fix_64','L2_var_16','L2_var_24','L2_var_32','L2_var_64', 'L2_quant_16', 'L2_quant_24', 'L2_quant_32', 'L2_quant_64']
    # kbps = [1.38, 2.07, 2.76, 5.51]

    # plt.figure(figsize=(3.5,2.8),dpi=300)
    # for name, bit in zip(plot_array_L1[0:4], kbps):
    #     h1 = singleErrorBar(distances, name, bit, 'red',   'o')
    # medianTrend(distances, plot_array_L1[0:4], kbps, 'red', True)
    # for name, bit in zip(plot_array_L1[4:8], kbps):
    #     h2 = singleErrorBar(distances, name, bit, 'green',   'o')
    # medianTrend(distances, plot_array_L1[4:8], kbps, 'green', True)
    # for name, bit in zip(plot_array_L1[8:12], kbps):
    #     h3 = singleErrorBar(distances, name, bit, 'orange' , 'o')
    # medianTrend(distances, plot_array_L1[8:12], kbps, 'orange', True)
    # plt.xlim([0 ,10.5])
    # plt.ylabel('mean absolute error in dB', fontsize=9)
    # # plt.ylabel('root mean square error in dB', fontsize=9)
    # plt.xlabel('bitrate in kps', fontsize=9)
    # plt.xlim([0,6])
    # locs, labels = plt.xticks() 
    # plt.xticks(kbps)
    # plt.legend()
    # plt.grid()
    # # plt.cla()
    # # plt.gca().set_axis_off()
    # # legend = plt.legend([h1[0], h2[0], h3[0]], ['proposed -- fixed bitrate', 'proposed -- variable bitrate', 'baseline -- vector quantization'],
    # #             loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)
    # # # plt.savefig('Legend.png', bbox_inches='tight')
    # # export_legend(legend)
    # # plt.show()
    # plt.savefig('L1.png')






if __name__ == "__main__":
    main()
