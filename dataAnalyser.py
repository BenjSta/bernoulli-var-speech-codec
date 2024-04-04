import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

def singleErrorBar(data, metric, x,color, marker, fcolor = None, markers = 3.0):
    if fcolor == None:
        fcolor = color
    mean, median, inter25, inter75 = calcMedMeanInter(data, metric)
    return plt.errorbar(x, median, yerr = np.array([[median-inter25], [inter75-median]]),color=color, capsize=3, marker=marker, markerfacecolor=fcolor, ms = markers )
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
    variable8 = pd.read_csv('testResult/old/variable8.csv', index_col=0)
    variable12 = pd.read_csv('testResult/old/variable12.csv', index_col=0)
    variable16 = pd.read_csv('testResult/old/variable16.csv', index_col=0)
    variable24 = pd.read_csv('testResult/old/variable24.csv', index_col=0)
    variable32 = pd.read_csv('testResult/old/variable32.csv', index_col=0)
    variable64 = pd.read_csv('testResult/old/variable64.csv', index_col=0)
    distances = pd.read_csv('testResult/Distances_all.csv', index_col=0)
    quant_16_nft = pd.read_csv('testResult/quantizer16.csv', index_col=0)
    quant_24_nft = pd.read_csv('testResult/quantizer24.csv', index_col=0)
    quant_32_nft = pd.read_csv('testResult/quantizer32.csv', index_col=0)
    quant_64_nft = pd.read_csv('testResult/quantizer64.csv', index_col=0)
    quant_16_ft = pd.read_csv('testResult/quantizer16_ft.csv', index_col=0)
    quant_24_ft = pd.read_csv('testResult/quantizer24_ft.csv', index_col=0)
    quant_32_ft = pd.read_csv('testResult/quantizer32_ft.csv', index_col=0)
    quant_64_ft = pd.read_csv('testResult/quantizer64_ft.csv', index_col=0)


    variable_16_small = pd.read_csv('testResult/variable16_small.csv', index_col=0)
    variable_24_small = pd.read_csv('testResult/variable24_small.csv', index_col=0)
    variable_32_small = pd.read_csv('testResult/variable32_small.csv', index_col=0)
    variable_64_small = pd.read_csv('testResult/variable64_small.csv', index_col=0)
    variable_16_big = pd.read_csv('testResult/variable16_big.csv', index_col=0)
    variable_24_big = pd.read_csv('testResult/variable24_big.csv', index_col=0)
    variable_32_big = pd.read_csv('testResult/variable32_big.csv', index_col=0)
    variable_64_big = pd.read_csv('testResult/variable64_big.csv', index_col=0)
    causal_small_vocoder = pd.read_csv('testResult/clean.csv', index_col=0)
    causal_big_vocoder = pd.read_csv('testResult/old/vocoded.csv', index_col=0)
    non_causal_vocoder = pd.read_csv('testResult/old/variable12.csv', index_col=0)

    savefigure = False
    savelegend = True


    # data = [clean, vocoded, encodec1_5, encodec3, encodec6, encodec12,
    #        lyra3_2, lyra6, lyra9_2, opus6, opus10, opus14, variable8,
    #        variable16, variable24, variable32, variable64]
    
    # data = [clean, vocoded, encodec1_5, encodec3, encodec6, encodec12,
    #        lyra3_2, lyra6, lyra9_2, opus6, opus10, opus14, variable8,
    #        variable16, variable24, variable32, variable64


    ###############################################################################################
    ## B - comparison of fixed BR, variable BR both small vocoder and VQ baseline small + small FT

    plot_array = [encodec1_5, encodec3, encodec6, encodec12,
                   variable16, variable24, variable32, variable64,
                   quant_16_nft, quant_24_nft, quant_32_nft, quant_64_nft,
                   quant_16_ft, quant_24_ft, quant_32_ft, quant_64_ft]
    color_array = ['darkgreen', 'darkgreen', 'darkgreen', 'darkgreen', 'darkviolet', 'darkviolet', 'darkviolet', 'darkviolet', 
                    'black', 'black', 'black', 'black', 'grey', 'grey', 'grey', 'grey']
    kbps = [1.48, 2.17, 2.86, 5.61, 1.43, 2.12, 2.81, 5.56, 1.23, 1.92, 2.61, 5.36, 1.33, 2.02, 2.71, 5.46]
    marker = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
    metric = 'nisqa_mos48'

    plt.rcParams.update({'figure.autolayout': True})
    plt.figure(dpi=300)
    m = []
    for data, color, bit in zip(plot_array, color_array, kbps):
        m.append(singleErrorBar(data, metric, bit, color, 'o'))
    h1 = medianTrend(plot_array[0:4], metric, kbps[0:4], 'darkgreen')
    h2 = medianTrend(plot_array[4:8], metric, kbps[4:8], 'darkviolet')
    h3 = medianTrend(plot_array[8:12], metric, kbps[8:12], 'black')
    h4 = medianTrend(plot_array[12:16], metric, kbps[12:16], 'grey')
    plt.xlim([1,6.5])
    if savelegend:
        # legend = plt.legend([h1[0], h2[0], h3[0], h4[0]], ['fixed bitrate', 'variable bitrate', 'vector quantization', ' vector quantization fine-tuned'], 
        #                     loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, ncol=1)
        # legend = plt.legend([h1[0], h2[0], h3[0], h4[0]], ['fixed bitrate', 'variable bitrate', 'vector quantization', 'vector quantization fine-tuned'],
        #             loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4)
        legend = plt.legend([m[0][0], m[4][0], m[8][0], m[12][0]], ['fixed bitrate', 'variable bitrate', 'vector quantization', 'vector quantization fine-tuned'],
                    loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4)
        export_legend(legend, 'B_Legend.png')
    plt.ylabel('NISQA')
    plt.xlabel('bitrate in kbps')
    plt.xticks([1.38, 2.07, 2.76, 5.51])
    plt.minorticks_off()
    plt.grid()
    if savefigure:
        plt.savefig('B_' + metric + '.png')

    metric = 'pesq'
    plt.figure(dpi=300)
    for data, color, bit in zip(plot_array, color_array, kbps):
        singleErrorBar(data, metric, bit, color, 'o')
    medianTrend(plot_array[0:4], metric, kbps[0:4], 'darkgreen')
    medianTrend(plot_array[4:8], metric, kbps[4:8], 'darkviolet')
    medianTrend(plot_array[8:12], metric, kbps[8:12], 'black')
    medianTrend(plot_array[12:16], metric, kbps[12:16], 'grey')
    plt.xlim([1,6.5])
    plt.ylabel('PESQ')
    plt.xlabel('bitrate in kbps')
    plt.xticks([1.38, 2.07, 2.76, 5.51])
    plt.minorticks_off()
    plt.grid()
    if savefigure:
        plt.savefig('B_' + metric + '.png')

    metric = 'visqol'
    plt.figure(dpi=300)
    for data, color, bit in zip(plot_array, color_array, kbps):
        singleErrorBar(data, metric, bit, color, 'o')
    medianTrend(plot_array[0:4], metric, kbps[0:4], 'darkgreen')
    medianTrend(plot_array[4:8], metric, kbps[4:8], 'darkviolet')
    medianTrend(plot_array[8:12], metric, kbps[8:12], 'black')
    medianTrend(plot_array[12:16], metric, kbps[12:16], 'grey')
    plt.xlim([1,6.5])
    plt.ylabel('ViSQOL')
    plt.xlabel('bitrate in kbps')
    plt.xticks([1.38, 2.07, 2.76, 5.51])
    plt.minorticks_off()
    plt.grid()
    if savefigure:
        plt.savefig('B_' + metric + '.png')
    
    ## distance plot
    plot_array_L1 = ['L1_fix_16','L1_fix_24','L1_fix_32','L1_fix_64','L1_var_16','L1_var_24','L1_var_32','L1_var_64', 'L1_quant_16', 'L1_quant_24', 'L1_quant_32', 'L1_quant_64']
    plot_array_L2 = ['L2_fix_16','L2_fix_24','L2_fix_32','L2_fix_64','L2_var_16','L2_var_24','L2_var_32','L2_var_64', 'L2_quant_16', 'L2_quant_24', 'L2_quant_32', 'L2_quant_64']
    kbps = [1.38, 2.07, 2.76, 5.51]

    plt.figure(figsize=(3.5,2.8),dpi=300)
    for name, bit in zip(plot_array_L1[0:4], kbps):
        h1 = singleErrorBar(distances, name, bit, 'darkgreen',   'o')
    medianTrend(distances, plot_array_L1[0:4], kbps, 'darkgreen', True)
    for name, bit in zip(plot_array_L1[4:8], kbps):
        h2 = singleErrorBar(distances, name, bit, 'darkviolet',   'o')
    medianTrend(distances, plot_array_L1[4:8], kbps, 'darkviolet', True)
    for name, bit in zip(plot_array_L1[8:12], kbps):
        h3 = singleErrorBar(distances, name, bit, 'black' , 'o')
    medianTrend(distances, plot_array_L1[8:12], kbps, 'black', True)
    plt.xlim([0 ,10.5])
    plt.ylabel('mean absolute error in dB', fontsize=9)
    plt.xlabel('bitrate in kps', fontsize=9)
    plt.xlim([0,6])
    locs, labels = plt.xticks() 
    plt.xticks(kbps)
    plt.minorticks_off()
    plt.grid()
    if savelegend:
        legend = plt.legend([h1[0], h2[0], h3[0]], ['proposed -- fixed bitrate', 'proposed -- variable bitrate', 'baseline -- vector quantization'],
                    loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)
        export_legend(legend, 'B_Distance_Legend.png')
    if savefigure:
        plt.savefig('B_L1.png')

    plt.figure(figsize=(3.5,2.8),dpi=300)
    for name, bit in zip(plot_array_L2[0:4], kbps):
        h1 = singleErrorBar(distances, name, bit, 'darkgreen',   'o')
    medianTrend(distances, plot_array_L2[0:4], kbps, 'darkgreen', True)
    for name, bit in zip(plot_array_L2[4:8], kbps):
        h2 = singleErrorBar(distances, name, bit, 'darkviolet',   'o')
    medianTrend(distances, plot_array_L2[4:8], kbps, 'darkviolet', True)
    for name, bit in zip(plot_array_L2[8:12], kbps):
        h3 = singleErrorBar(distances, name, bit, 'black' , 'o')
    medianTrend(distances, plot_array_L2[8:12], kbps, 'black', True)
    plt.xlim([0 ,10.5])
    plt.ylabel('root mean square error in dB', fontsize=9)
    plt.xlabel('bitrate in kps', fontsize=9)
    plt.xlim([0,6])
    locs, labels = plt.xticks() 
    plt.xticks(kbps)
    plt.minorticks_off()
    plt.grid()
    if savefigure:
        plt.savefig('B_L2.png')
    ###############################################################################################
    ## C - Effect of Vocoder Complexity and Causality 

    plot_array = [ variable_16_small, variable_24_small, variable_32_small, variable_64_small,
                   variable_16_big, variable_24_big, variable_32_big, variable_64_big,
                   variable16, variable24, variable32, variable64,
                   causal_small_vocoder, causal_big_vocoder, non_causal_vocoder]
    color_array = [ 'darkviolet', 'darkviolet', 'darkviolet', 'darkviolet', 'violet', 'violet', 'violet', 'violet',
                    'indigo', 'indigo', 'indigo', 'indigo', 'darkviolet', 'violet', 'indigo']
    kbps = [1.28, 1.97, 2.66, 5.41, 1.38, 2.07, 2.76, 5.51, 1.48, 2.17, 2.86, 5.61, 6.0, 6.3, 6.6]
    marker = ['o', 'o', 'o', 'o', 's', 's', 's', 's', 'D', 'D', 'D', 'D', 'o', 's', 'D']
    fcolor_array = [ 'darkviolet', 'darkviolet', 'darkviolet', 'darkviolet', 'white', 'white', 'white', 'white',
                    'white', 'white', 'white', 'white', 'darkviolet', 'white', 'white'] 
    ms = [3.0, 3.0, 3.0, 3.0, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 3.0, 4.5, 4.5] 

    

    metric = 'nisqa_mos48'
    plt.rcParams.update({'figure.autolayout': True})
    plt.figure(dpi=300)
    m = []
    for data, color, fcolor,  bit, mark, markers in zip(plot_array, color_array, fcolor_array, kbps, marker, ms):
        m.append(singleErrorBar(data, metric, bit, color, mark, fcolor, markers ))
    h1 = medianTrend(plot_array[0:4], metric, kbps[0:4], 'darkviolet')
    h2 = medianTrend(plot_array[4:8], metric, kbps[4:8], 'violet')
    h3 = medianTrend(plot_array[8:12], metric, kbps[8:12], 'indigo')

    plt.xlim([1,6.7])
    if savelegend:
        # legend = plt.legend([h1[0], h2[0], h3[0], h4[0]], ['fixed bitrate', 'variable bitrate', 'vector quantization', ' vector quantization fine-tuned'], 
        #                     loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, ncol=1)
        # legend = plt.legend([h1[0], h2[0], h3[0], h4[0]], ['fixed bitrate', 'variable bitrate', 'vector quantization', 'vector quantization fine-tuned'],
        #             loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4)
        legend = plt.legend([m[0][0], m[4][0], m[8][0], m[12][0], m[13][0], m[14][0]], 
                            ['variable bitrate causal small fine-tuned', 'variable bitrate causal big', 'variable bitrate non causal' 'causal small vocoder', 'causal big vocoder', 'non causal vocoder'],
                              loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4)
        export_legend(legend, 'C_Legend.png')
    plt.ylabel('NISQA')
    plt.xlabel('bitrate in kbps')
    plt.xticks([1.38, 2.07, 2.76, 5.51, 6.3], ['1.38', '2.07', '2.76', '5.51', 'Vocoder \n only'])
    plt.minorticks_off()
    plt.gca().set_clip_on(False)
    plt.grid()
    if savefigure:
        plt.savefig('C_' + metric + '.png')
        
    metric = 'visqol'
    plt.rcParams.update({'figure.autolayout': True})
    plt.figure(dpi=300)
    m = []
    for data, color, fcolor,  bit, mark, markers in zip(plot_array, color_array, fcolor_array, kbps, marker, ms):
        m.append(singleErrorBar(data, metric, bit, color, mark, fcolor, markers ))
    h1 = medianTrend(plot_array[0:4], metric, kbps[0:4], 'darkviolet')
    h2 = medianTrend(plot_array[4:8], metric, kbps[4:8], 'violet')
    h3 = medianTrend(plot_array[8:12], metric, kbps[8:12], 'indigo')
    plt.xlim([1,6.7])
    plt.ylabel('ViSQOL')
    plt.xlabel('bitrate in kbps')
    plt.xticks([1.38, 2.07, 2.76, 5.51])
    plt.minorticks_off()
    plt.gca().set_clip_on(False)
    plt.grid()
    if savefigure:
        plt.savefig('C_' + metric + '.png')

    metric = 'pesq'
    plt.rcParams.update({'figure.autolayout': True})
    plt.figure(dpi=300)
    m = []
    for data, color, fcolor,  bit, mark, markers in zip(plot_array, color_array, fcolor_array, kbps, marker, ms):
        m.append(singleErrorBar(data, metric, bit, color, mark, fcolor, markers ))
    h1 = medianTrend(plot_array[0:4], metric, kbps[0:4], 'darkviolet')
    h2 = medianTrend(plot_array[4:8], metric, kbps[4:8], 'violet')
    h3 = medianTrend(plot_array[8:12], metric, kbps[8:12], 'indigo')
    plt.xlim([1,6.7])
    plt.ylabel('PESQ')
    plt.xlabel('bitrate in kbps')
    plt.xticks([1.38, 2.07, 2.76, 5.51])
    plt.minorticks_off()
    plt.gca().set_clip_on(False)
    plt.grid()
    if savefigure:
        plt.savefig('C_' + metric + '.png')

    
    ###############################################################################################
    ## D - comparison with SOTA Codecs

    plot_array = [encodec1_5, encodec3, encodec6, encodec12, 
                  lyra3_2, lyra6, lyra9_2, 
                  opus6, opus8, opus10, opus12, 
                  variable16, variable24, variable32, variable64,
                  quant_16_nft, quant_24_nft, quant_32_nft, quant_64_nft]
    color_array = ['blue', 'blue', 'blue', 'blue', 'darkorange', 'darkorange' ,'darkorange', 'orangered', 'orangered', 'orangered', 'orangered', 
                   'darkviolet', 'darkviolet', 'darkviolet', 'darkviolet', 'violet', 'violet', 'violet', 'violet']
    kbps = [1.5, 3, 5.9, 11.9, 3.2, 6.1, 9.2, 6, 8, 10, 12.1, 1.38, 2.07, 2.76, 5.51, 1.38, 2.07, 2.76, 5.51]
    metric = 'pesq'

    plt.rcParams.update({'figure.autolayout': True})

    metric = 'pesq'
    m = []
    plt.figure(dpi=300)
    for data, color, bit in zip(plot_array, color_array, kbps):
        m.append(singleErrorBar(data, metric, bit, color, 'o'))
    h1 = medianTrend(plot_array[0:4], metric, kbps[0:4], 'blue')
    h2 = medianTrend(plot_array[4:7], metric, kbps[4:7], 'darkorange')
    h3 = medianTrend(plot_array[7:11], metric, kbps[7:11], 'orangered')
    h4 = medianTrend(plot_array[11:15], metric, kbps[11:15], 'darkviolet')
    h5 = medianTrend(plot_array[15:19], metric, kbps[15:19], 'violet')
    plt.xlim([1,12.5])
    if savelegend:
        # legend = plt.legend([h1[0], h2[0], h3[0], h4[0], h5[0]], ['Encodec', 'Lyra', 'Opus', 'variable bitrate small fine-tuned', 'variable bitrate big' ],
        #                      loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, ncol=1)
        legend = plt.legend([m[0][0], m[4][0], m[7][0], m[11][0], m[15][0]], ['EnCodec', 'Lyra v2', 'Opus', 'variable bitrate small fine-tuned', 'variable bitrate big' ],
                             loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, ncol=5)
        export_legend(legend, 'D_Legend.png')
    plt.ylabel('PESQ')
    plt.xlabel('bitrate in kbps')
    plt.xticks([1.38, 1.5, 2.07, 2.76, 3, 3.2, 5.5, 6, 12, 9.2, 8, 10],['1.38', '\n1.5', '\n\n2.07', '2.76', '\n3','\n\n3.2', '5.51', '\n6', '12','9.2','8','10'])
    plt.minorticks_off()
    plt.grid()
    if savefigure:
        plt.savefig('D_' + metric + '.png')

    metric = 'visqol'
    m = []
    plt.figure(dpi=300)
    for data, color, bit in zip(plot_array, color_array, kbps):
        m.append(singleErrorBar(data, metric, bit, color, 'o'))
    h1 = medianTrend(plot_array[0:4], metric, kbps[0:4], 'blue')
    h2 = medianTrend(plot_array[4:7], metric, kbps[4:7], 'darkorange')
    h3 = medianTrend(plot_array[7:11], metric, kbps[7:11], 'orangered')
    h4 = medianTrend(plot_array[11:15], metric, kbps[11:15], 'darkviolet')
    h5 = medianTrend(plot_array[15:19], metric, kbps[15:19], 'violet')
    plt.xlim([1,12.5])
    plt.ylabel('ViSQOL')
    plt.xlabel('bitrate in kbps')
    plt.xticks([1.38, 1.5, 2.07, 2.76, 3, 3.2, 5.5, 6, 12, 9.2, 8, 10],['1.38', '\n1.5', '\n\n2.07', '2.76', '\n3','\n\n3.2', '5.51', '\n6', '12','9.2','8','10'])
    plt.minorticks_off()
    plt.grid()
    if savefigure:
        plt.savefig('D_' + metric + '.png')

    metric = 'nisqa_mos16'
    m = []
    plt.figure(dpi=300)
    for data, color, bit in zip(plot_array, color_array, kbps):
        m.append(singleErrorBar(data, metric, bit, color, 'o'))
    h1 = medianTrend(plot_array[0:4], metric, kbps[0:4], 'blue')
    h2 = medianTrend(plot_array[4:7], metric, kbps[4:7], 'darkorange')
    h3 = medianTrend(plot_array[7:11], metric, kbps[7:11], 'orangered')
    h4 = medianTrend(plot_array[11:15], metric, kbps[11:15], 'darkviolet')
    h5 = medianTrend(plot_array[15:19], metric, kbps[15:19], 'violet')
    plt.xlim([1,12.5])
    plt.ylabel('NISQ')
    plt.xlabel('bitrate in kbps')
    plt.xticks([1.38, 1.5, 2.07, 2.76, 3, 3.2, 5.5, 6, 12, 9.2, 8, 10],['1.38', '\n1.5', '\n\n2.07', '2.76', '\n3','\n\n3.2', '5.51', '\n6', '12','9.2','8','10'])
    plt.minorticks_off()
    plt.grid()
    if savefigure:
        plt.savefig('D_' + metric + '.png')

    metric = 'nisqa_mos48'
    m = []
    plt.figure(dpi=300)
    for data, color, bit in zip(plot_array, color_array, kbps):
        m.append(singleErrorBar(data, metric, bit, color, 'o'))
    h1 = medianTrend(plot_array[0:4], metric, kbps[0:4], 'blue')
    h2 = medianTrend(plot_array[4:7], metric, kbps[4:7], 'darkorange')
    h3 = medianTrend(plot_array[7:11], metric, kbps[7:11], 'orangered')
    h4 = medianTrend(plot_array[11:15], metric, kbps[11:15], 'darkviolet')
    h5 = medianTrend(plot_array[15:19], metric, kbps[15:19], 'violet')
    plt.xlim([1,12.5])
    plt.ylabel('NISQ')
    plt.xlabel('bitrate in kbps')
    plt.xticks([1.38, 1.5, 2.07, 2.76, 3, 3.2, 5.5, 6, 12, 9.2, 8, 10],['1.38', '\n1.5', '\n\n2.07', '2.76', '\n3','\n\n3.2', '5.51', '\n6', '12','9.2','8','10'])
    plt.minorticks_off()
    plt.grid()
    if savefigure:
        plt.savefig('D_' + metric + '.png')


if __name__ == "__main__":
    main()
