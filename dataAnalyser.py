import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
#import scienceplots

#plt.style.use(['science', 'nature'])
plt.rcParams.update({'figure.autolayout': True})
#plt.style.use(['science','grid'])

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
    return plt.plot(x, median, color, linestyle='dashed', alpha=0.3)
    # plt.fill_between(x,inter25, inter75,alpha=.1, color=color)

def export_legend(legend, filename, expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename + '.png', dpi="figure", bbox_inches=bbox)
    fig.savefig(filename + '.pdf', bbox_inches=bbox)



def main():   
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

    distances = pd.read_csv('testResult/Distances_64.csv', index_col=0)
    
    quant_16_nft = pd.read_csv('testResult/quantizer16.csv', index_col=0)
    quant_24_nft = pd.read_csv('testResult/quantizer24.csv', index_col=0)
    quant_32_nft = pd.read_csv('testResult/quantizer32.csv', index_col=0)
    quant_64_nft = pd.read_csv('testResult/quantizer64.csv', index_col=0)
    quant_16_ft = pd.read_csv('testResult/quantizer16_ft.csv', index_col=0)
    quant_24_ft = pd.read_csv('testResult/quantizer24_ft.csv', index_col=0)
    quant_32_ft = pd.read_csv('testResult/quantizer32_ft.csv', index_col=0)
    quant_64_ft = pd.read_csv('testResult/quantizer64_ft.csv', index_col=0)

    fixed_16 = pd.read_csv('testResult/fixed16.csv', index_col=0)
    fixed_24 = pd.read_csv('testResult/fixed24.csv', index_col=0)
    fixed_32 = pd.read_csv('testResult/fixed32.csv', index_col=0)
    fixed_64 = pd.read_csv('testResult/fixed64.csv', index_col=0)

    fixed_16_ft = pd.read_csv('testResult/fixed16_ft.csv', index_col=0)
    fixed_24_ft = pd.read_csv('testResult/fixed24_ft.csv', index_col=0)
    fixed_32_ft = pd.read_csv('testResult/fixed32_ft.csv', index_col=0)
    fixed_64_ft = pd.read_csv('testResult/fixed64_ft.csv', index_col=0)

    variable_16 = pd.read_csv('testResult/variable16.csv', index_col=0)
    variable_24 = pd.read_csv('testResult/variable24.csv', index_col=0)
    variable_32 = pd.read_csv('testResult/variable32.csv', index_col=0)
    variable_64 = pd.read_csv('testResult/variable64.csv', index_col=0)
    
    variable_16_ft = pd.read_csv('testResult/variable16_ft.csv', index_col=0)
    variable_24_ft = pd.read_csv('testResult/variable24_ft.csv', index_col=0)
    variable_32_ft = pd.read_csv('testResult/variable32_ft.csv', index_col=0)
    variable_64_ft = pd.read_csv('testResult/variable64_ft.csv', index_col=0)

    variable_16_big = pd.read_csv('testResult/variable16_big.csv', index_col=0)
    variable_24_big = pd.read_csv('testResult/variable24_big.csv', index_col=0)
    variable_32_big = pd.read_csv('testResult/variable32_big.csv', index_col=0)
    variable_64_big = pd.read_csv('testResult/variable64_big.csv', index_col=0)
    variable_16_bigsym = pd.read_csv('testResult/variable16_big_sym.csv', index_col=0)
    variable_24_bigsym = pd.read_csv('testResult/variable24_big_sym.csv', index_col=0)
    variable_32_bigsym = pd.read_csv('testResult/variable32_big_sym.csv', index_col=0)
    variable_64_bigsym = pd.read_csv('testResult/variable64_big_sym.csv', index_col=0)
    
    causal_vocoder = pd.read_csv('testResult/vocoder.csv', index_col=0)
    causal_big_vocoder = pd.read_csv('testResult/vocoder_big.csv', index_col=0)
    non_causal_vocoder = pd.read_csv('testResult/vocoder_big_sym.csv', index_col=0)

    savefigure = True

    ###############################################################################################
    ## B - comparison of fixed BR, variable BR both small vocoder and VQ baseline small + small FT
    ## distance plot
    plot_array_L1 = ['L1_fix_16','L1_fix_24','L1_fix_32','L1_fix_64','L1_var_16','L1_var_24','L1_var_32','L1_var_64', 'L1_quant_16', 'L1_quant_24', 'L1_quant_32', 'L1_quant_64']
    plot_array_L2 = ['L2_fix_16','L2_fix_24','L2_fix_32','L2_fix_64','L2_var_16','L2_var_24','L2_var_32','L2_var_64', 'L2_quant_16', 'L2_quant_24', 'L2_quant_32', 'L2_quant_64']
    kbps = [1.38, 2.07, 2.76, 4]

    FACTOR1 = 0.8
    FONTSIZE1 = 7.7
    plt.figure(figsize=(FACTOR1*3.5,FACTOR1*2.8),dpi=300)
    for name, bit in zip(plot_array_L1[0:4], kbps):
        h1 = singleErrorBar(distances, name, bit, 'darkgreen',   'o')
    medianTrend(distances, plot_array_L1[0:4], kbps, 'darkgreen', True)
    for name, bit in zip(plot_array_L1[4:8], [1.38, 2.07, 2.76, 4+0.05]):
        h2 = singleErrorBar(distances, name, bit, 'darkviolet',   'o')
    medianTrend(distances, plot_array_L1[4:8], [1.38, 2.07, 2.76, 4+0.05], 'darkviolet', True)
    for name, bit in zip(plot_array_L1[8:12], [1.38, 2.07, 2.76, 4-0.05]):
        h3 = singleErrorBar(distances, name, bit, 'black' , 'o')
    medianTrend(distances, plot_array_L1[8:12], [1.38, 2.07, 2.76, 4-0.05], 'black', True)
    plt.ylabel('MAE in dB', fontsize=FONTSIZE1)
    plt.xlabel('bitrate in kbps', fontsize=FONTSIZE1)
    plt.xlim([1.1, 4.28])
    locs, labels = plt.xticks() 
    plt.xticks(kbps, ['1.38', '2.07', '2.76', '5.51'], fontsize=FONTSIZE1)
    plt.yticks(fontsize=FONTSIZE1)
    plt.minorticks_off()
    plt.grid()
       
    if savefigure:
        plt.savefig('B_L1.png')
        plt.savefig('B_L1.pdf')
        plt.figure()
        legend = plt.legend([h1[0], h2[0], h3[0]], ['proposed - fixed bitrate',
                                                    'proposed - variable bitrate', 'VQ baseline'],
                    loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
        export_legend(legend, 'B_Distance_Legend')


    plt.figure(figsize=(FACTOR1*3.5,FACTOR1*2.8),dpi=300)
    for name, bit in zip(plot_array_L2[0:4], kbps):
        h1 = singleErrorBar(distances, name, bit, 'darkgreen',   'o')
    medianTrend(distances, plot_array_L2[0:4], kbps, 'darkgreen', True)
    for name, bit in zip(plot_array_L2[4:8], [1.38, 2.07, 2.76, 4+0.05]):
        h2 = singleErrorBar(distances, name, bit, 'darkviolet',   'o')
    medianTrend(distances, plot_array_L2[4:8], [1.38, 2.07, 2.76, 4+0.05], 'darkviolet', True)
    for name, bit in zip(plot_array_L2[8:12], [1.38, 2.07, 2.76, 4-0.05]):
        h3 = singleErrorBar(distances, name, bit, 'black' , 'o')
    medianTrend(distances, plot_array_L2[8:12], [1.38, 2.07, 2.76, 4-0.05], 'black', True)
    plt.ylabel('RMSE in dB', fontsize=FONTSIZE1)
    plt.xlabel('bitrate in kbps', fontsize=FONTSIZE1)
    plt.xlim([1.1, 4.28])
    locs, labels = plt.xticks() 
    plt.xticks(kbps, ['1.38', '2.07', '2.76', '5.51'], fontsize=FONTSIZE1)
    plt.yticks(fontsize=FONTSIZE1)
    plt.minorticks_off()
    plt.grid()
    if savefigure:
        plt.savefig('B_L2.png')
        plt.savefig('B_L2.pdf')
    


    FACTOR1 = 0.65
    FONTSIZE1 = 9
    plot_array = [fixed_16, fixed_24, fixed_32, fixed_64,
                   variable_16, variable_24, variable_32, variable_64,
                   variable_16_ft, variable_24_ft, variable_32_ft, variable_64_ft,
                   quant_16_nft, quant_24_nft, quant_32_nft, quant_64_nft,
                   quant_16_ft, quant_24_ft, quant_32_ft, quant_64_ft]
    color_array = ['darkgreen', 'darkgreen', 'darkgreen', 'darkgreen',
                   'violet', 'violet', 'violet', 'violet',
                    'darkviolet', 'darkviolet', 'darkviolet', 'darkviolet', 
                    'grey', 'grey', 'grey', 'grey',
                    'black', 'black', 'black', 'black']
    kbps = (np.array([[1.38, 2.07, 2.76, 4]]) + np.linspace(-0.15, 0.15, 5, endpoint=True)[:, None]).flatten()
    marker = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
    for metric, metricname in zip(['nisqa_mos48', 'visqol',  'pesq'], ['NISQA', 'ViSQOL', 'PESQ']):
        plt.rcParams.update({'figure.autolayout': True})
        plt.figure(dpi=300, figsize=(FACTOR1*6.4,FACTOR1*4.8))
        m = []
        for data, color, bit in zip(plot_array, color_array, kbps):
            m.append(singleErrorBar(data, metric, bit, color, 'o', markers=4.0))
        h1 = medianTrend(plot_array[0:4], metric, kbps[0:4], 'darkgreen')
        h2 = medianTrend(plot_array[4:8], metric, kbps[4:8], 'violet')
        h2 = medianTrend(plot_array[8:12], metric, kbps[8:12], 'darkviolet')
        h3 = medianTrend(plot_array[12:16], metric, kbps[12:16], 'grey')
        h4 = medianTrend(plot_array[16:20], metric, kbps[16:20], 'black')
        plt.xlim([1.1, 4.28])
        plt.yticks(fontsize=FONTSIZE1)
        plt.ylabel(metricname, fontsize=FONTSIZE1)
        plt.xlabel('bitrate in kbps', fontsize=FONTSIZE1)
        plt.xticks([1.38, 2.07, 2.76, 4.0], ['1.38', '2.07', '2.76', '5.51'], fontsize=FONTSIZE1)
        plt.minorticks_off()
        plt.grid()
        if savefigure:
            plt.savefig('B_' + metric + '.png')
            plt.savefig('B_' + metric + '.pdf')

    if savefigure:
        plt.figure()
        legend = plt.legend([m[0][0], m[4][0], m[8][0], m[12][0],  m[16][0]], ['prop. fixed bitrate BigVGAN-tiny-causal',
                                                                               'prop. variable bitrate BigVGAN-tiny-causal',
                                                                               'prop. variable bitrate BigVGAN-tiny-causal fine-tuned',
                                                                               'VQ baseline',
                                                                               'VQ baseline fine-tuned'],
                    loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2)
        export_legend(legend, 'B_Legend')

    
    
    
    
    ###############################################################################################
    ## C - Effect of Vocoder Complexity and Causality 
    plot_array = [ variable_16_ft, variable_24_ft, variable_32_ft, variable_64_ft,
                   variable_16_big, variable_24_big, variable_32_big, variable_64_big,
                   variable_16_bigsym, variable_24_bigsym, variable_32_bigsym, variable_64_bigsym,
                   causal_vocoder, causal_big_vocoder, non_causal_vocoder]
    color_array = [ 'darkviolet', 'darkviolet', 'darkviolet', 'darkviolet', 'violet', 'violet', 'violet', 'violet',
                    'indigo', 'indigo', 'indigo', 'indigo', 'darkviolet', 'violet', 'indigo']
    kbps = [1.28, 1.97, 2.66, 3.9, 
            1.38, 2.07, 2.76, 4, 
            1.48, 2.17, 2.86, 4.1, 
            4.9, 5.0, 5.1]
    marker = ['o', 'o', 'o', 'o', 's', 's', 's', 's', 'D', 'D', 'D', 'D', 'o', 's', 'D']
    fcolor_array = [ 'darkviolet', 'darkviolet', 'darkviolet', 'darkviolet', 'white', 'white', 'white', 'white',
                    'white', 'white', 'white', 'white', 'darkviolet', 'white', 'white'] 
    ms = [4.0, 4.0, 4.0, 4.0, 5, 5, 5, 5, 5, 5, 5, 5, 4.0, 5, 5] 

    
    for metric, metricname in zip(['nisqa_mos48', 'visqol',  'pesq'], ['NISQA', 'ViSQOL', 'PESQ']):
        plt.rcParams.update({'figure.autolayout': True})
        plt.figure(dpi=300, figsize=(FACTOR1*6.4,FACTOR1*4.8))
        m = []
        for data, color, fcolor,  bit, mark, markers in zip(plot_array, color_array, fcolor_array, kbps, marker, ms):
            m.append(singleErrorBar(data, metric, bit, color, mark, fcolor, markers ))
        h1 = medianTrend(plot_array[0:4], metric, kbps[0:4], 'darkviolet')
        h2 = medianTrend(plot_array[4:8], metric, kbps[4:8], 'violet')
        h3 = medianTrend(plot_array[8:12], metric, kbps[8:12], 'indigo')
        plt.yticks(fontsize=FONTSIZE1)
        plt.ylabel(metricname, fontsize=FONTSIZE1)
        plt.xlabel('bitrate in kbps', fontsize=FONTSIZE1)
        plt.xticks([1.38, 2.07, 2.76, 4, 5], ['1.38', '2.07', '2.76', '5.51', 'vocoder\nonly'], fontsize=FONTSIZE1)
        plt.minorticks_off()
        plt.gca().set_clip_on(False)
        plt.grid()
        if savefigure:
            plt.savefig('C_' + metric + '.png')
            plt.savefig('C_' + metric + '.pdf')
    if savefigure:
        plt.figure()
        legend = plt.legend([m[0][0], m[4][0], m[8][0]], 
                             ['BigVGAN-tiny-causal (fine-tuned)', 'BigVGAN-base-causal', 'BigVGAN-base'],
                               loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=3)
        export_legend(legend, 'C_Legend')

    
    ###############################################################################################
    ## D - comparison with SOTA Codecs
    FACTOR1 = 0.85
    FONTSIZE1 = 8
    
    plot_array = [encodec1_5, encodec3, encodec6, encodec12, 
                  lyra3_2, lyra6, lyra9_2, 
                  opus6, opus8, opus10, opus12, 
                  variable_16_ft, variable_24_ft, variable_32_ft, variable_64_ft,
                  variable_16_big, variable_24_big, variable_32_big, variable_64_big]
    color_array = ['blue', 'blue', 'blue', 'blue', 'darkorange', 'darkorange' ,'darkorange', 'orangered', 'orangered', 'orangered', 'orangered', 
                   'darkviolet', 'darkviolet', 'darkviolet', 'darkviolet', 'violet', 'violet', 'violet', 'violet']
    
    kbps = [1.5, 3, 5.95, 11.95, 3.2, 6.05, 9.2, 6, 8, 10, 12.05, 1.38-0.07, 2.07-0.07, 2.76-0.07, 5.51-0.07,
            1.38+0.07, 2.07+0.07, 2.76+0.07, 5.51+0.07]
    ms = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5, 5, 5, 5] 
    fcolor_array = ['blue', 'blue', 'blue', 'blue', 'darkorange', 'darkorange' ,'darkorange', 'orangered', 'orangered', 'orangered', 'orangered', 
                   'darkviolet', 'darkviolet', 'darkviolet', 'darkviolet', 'white', 'white', 'white', 'white']
    markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 's', 's', 's', 's']


    plt.rcParams.update({'figure.autolayout': True})

    for metric, metricname in zip(['pesq', 'visqol',  'nisqa_mos48', 'nisqa_mos16'], ['PESQ', 'ViSQOL', 'NISQA', 'NISQA 16kHz']):
        plt.figure(dpi=300, figsize=(FACTOR1*6.4,FACTOR1*4.3))
        m = []
        for data, color, bit, msize, marker, fc in zip(plot_array, color_array, kbps, ms, markers, fcolor_array):
            m.append(singleErrorBar(data, metric, bit, color, marker, fc, msize))
        h1 = medianTrend(plot_array[0:4], metric, kbps[0:4], 'blue')
        h2 = medianTrend(plot_array[4:7], metric, kbps[4:7], 'darkorange')
        h3 = medianTrend(plot_array[7:11], metric, kbps[7:11], 'orangered')
        h4 = medianTrend(plot_array[11:15], metric, kbps[11:15], 'darkviolet')
        h5 = medianTrend(plot_array[15:19], metric, kbps[15:19], 'violet')
        plt.xlim([1,12.5])
        plt.ylabel(metricname, fontsize=FONTSIZE1)
        plt.xlabel('bitrate in kbps', fontsize=FONTSIZE1)
        plt.xticks([1.38, 1.5, 2.07, 2.76, 3, 3.2, 5.5, 6, 12, 9.2, 8, 10],
                   ['1.38', '\n1.5', '\n\n2.07', '2.76', '\n3','\n\n3.2', '5.51', '\n6', '12','9.2','8','10'], fontsize=FONTSIZE1)
        plt.minorticks_off()
        plt.grid()
        if savefigure:
            plt.savefig('D_' + metric + '.png')
            plt.savefig('D_' + metric + '.pdf')
    if savefigure:
        plt.figure()
        legend = plt.legend([m[0][0], m[4][0], m[7][0], m[11][0], m[15][0]],
                            ['EnCodec', 'Lyra v2', 'Opus',
                             'prop. variable bitrate BigVGAN-tiny-causal fine-tuned',
                             'prop. variable bitrate BigVGAN-base-causal' ],
                             loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, ncol=1)
        export_legend(legend, 'D_Legend')

    # metric = 'pesq'
    # m = []
    # plt.figure(dpi=300)
    # for data, color, bit in zip(plot_array, color_array, kbps):
    #     m.append(singleErrorBar(data, metric, bit, color, 'o'))
    # h1 = medianTrend(plot_array[0:4], metric, kbps[0:4], 'blue')
    # h2 = medianTrend(plot_array[4:7], metric, kbps[4:7], 'darkorange')
    # h3 = medianTrend(plot_array[7:11], metric, kbps[7:11], 'orangered')
    # h4 = medianTrend(plot_array[11:15], metric, kbps[11:15], 'darkviolet')
    # h5 = medianTrend(plot_array[15:19], metric, kbps[15:19], 'violet')
    # plt.xlim([1,12.5])
        
    # plt.ylabel('PESQ')
    # plt.xlabel('bitrate in kbps')
    # plt.xticks([1.38, 1.5, 2.07, 2.76, 3, 3.2, 5.5, 6, 12, 9.2, 8, 10],['1.38', '\n1.5', '\n\n2.07', '2.76', '\n3','\n\n3.2', '5.51', '\n6', '12','9.2','8','10'])
    # plt.minorticks_off()
    # plt.grid()
    # if savefigure:
    #     plt.savefig('D_' + metric + '.png')
    #     plt.savefig('D_' + metric + '.pdf')
    #     plt.figure()
    #     # legend = plt.legend([h1[0], h2[0], h3[0], h4[0], h5[0]], ['Encodec', 'Lyra', 'Opus', 'variable bitrate small fine-tuned', 'variable bitrate big' ],
    #     #                      loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, ncol=1)
    #     legend = plt.legend([m[0][0], m[4][0], m[7][0], m[11][0], m[15][0]], ['EnCodec', 'Lyra v2', 'Opus', 'variable bitrate small fine-tuned', 'variable bitrate big' ],
    #                          loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, ncol=5)
    #     export_legend(legend, 'D_Legend')

    # metric = 'visqol'
    # m = []
    # plt.figure(dpi=300)
    # for data, color, bit in zip(plot_array, color_array, kbps):
    #     m.append(singleErrorBar(data, metric, bit, color, 'o'))
    # h1 = medianTrend(plot_array[0:4], metric, kbps[0:4], 'blue')
    # h2 = medianTrend(plot_array[4:7], metric, kbps[4:7], 'darkorange')
    # h3 = medianTrend(plot_array[7:11], metric, kbps[7:11], 'orangered')
    # h4 = medianTrend(plot_array[11:15], metric, kbps[11:15], 'darkviolet')
    # h5 = medianTrend(plot_array[15:19], metric, kbps[15:19], 'violet')
    # plt.xlim([1,12.5])
    # plt.ylabel('ViSQOL')
    # plt.xlabel('bitrate in kbps')
    # plt.xticks([1.38, 1.5, 2.07, 2.76, 3, 3.2, 5.5, 6, 12, 9.2, 8, 10],['1.38', '\n1.5', '\n\n2.07', '2.76', '\n3','\n\n3.2', '5.51', '\n6', '12','9.2','8','10'])
    # plt.minorticks_off()
    # plt.grid()
    # if savefigure:
    #     plt.savefig('D_' + metric + '.png')

    # metric = 'nisqa_mos16'
    # m = []
    # plt.figure(dpi=300)
    # for data, color, bit in zip(plot_array, color_array, kbps):
    #     m.append(singleErrorBar(data, metric, bit, color, 'o'))
    # h1 = medianTrend(plot_array[0:4], metric, kbps[0:4], 'blue')
    # h2 = medianTrend(plot_array[4:7], metric, kbps[4:7], 'darkorange')
    # h3 = medianTrend(plot_array[7:11], metric, kbps[7:11], 'orangered')
    # h4 = medianTrend(plot_array[11:15], metric, kbps[11:15], 'darkviolet')
    # h5 = medianTrend(plot_array[15:19], metric, kbps[15:19], 'violet')
    # plt.xlim([1,12.5])
    # plt.ylabel('NISQA')
    # plt.xlabel('bitrate in kbps')
    # plt.xticks([1.38, 1.5, 2.07, 2.76, 3, 3.2, 5.5, 6, 12, 9.2, 8, 10],['1.38', '\n1.5', '\n\n2.07', '2.76', '\n3','\n\n3.2', '5.51', '\n6', '12','9.2','8','10'])
    # plt.minorticks_off()
    # plt.grid()
    # if savefigure:
    #     plt.savefig('D_' + metric + '.png')

    # metric = 'nisqa_mos48'
    # m = []
    # plt.figure(dpi=300)
    # for data, color, bit in zip(plot_array, color_array, kbps):
    #     m.append(singleErrorBar(data, metric, bit, color, 'o'))
    # h1 = medianTrend(plot_array[0:4], metric, kbps[0:4], 'blue')
    # h2 = medianTrend(plot_array[4:7], metric, kbps[4:7], 'darkorange')
    # h3 = medianTrend(plot_array[7:11], metric, kbps[7:11], 'orangered')
    # h4 = medianTrend(plot_array[11:15], metric, kbps[11:15], 'darkviolet')
    # h5 = medianTrend(plot_array[15:19], metric, kbps[15:19], 'violet')
    # plt.xlim([1,12.5])
    # plt.ylabel('NISQA')
    # plt.xlabel('bitrate in kbps')
    # plt.xticks([1.38, 1.5, 2.07, 2.76, 3, 3.2, 5.5, 6, 12, 9.2, 8, 10],['1.38', '\n1.5', '\n\n2.07', '2.76', '\n3','\n\n3.2', '5.51', '\n6', '12','9.2','8','10'])
    # plt.minorticks_off()
    # plt.grid()
    # if savefigure:
    #     plt.savefig('D_' + metric + '.png')


if __name__ == "__main__":
    main()
