import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
from scipy.stats import binomtest
from scipy.stats import t as tdist


# import scienceplots

# plt.style.use(['science', 'nature'])
plt.rcParams.update({'figure.autolayout': True})
# plt.style.use(['science','grid'])


def calcMedMeanInter(data, metric,):
    # mean = np.mean(data.loc[:, 'lengths'] * np.array(data.loc[:, metric])) / np.mean(data.loc[:, 'lengths'])
    median = np.median(data.loc[:, metric])
    inter25 = np.nanquantile(data.loc[:, metric], 0.25)
    inter75 = np.nanquantile(data.loc[:, metric], 0.75)
    return median, inter25, inter75


def singleErrorBar(data, metric, x, color, marker, fcolor=None, markers=3.0):
    if fcolor == None:
        fcolor = color
    median, inter25, inter75 = calcMedMeanInter(data, metric)
    return plt.errorbar(x, median, yerr=np.array([[median-inter25], [inter75-median]]), color=color, capsize=3, marker=marker, markerfacecolor=fcolor, ms=markers)
#    plt.text(x + 0.2, median - 0.2, name, fontsize=8)


def medianTrend(plotData, metric, x, color, distance=False):
    median = []
    inter25 = []
    inter75 = []
    if not distance:
        for data in plotData:
            medi, int25, int75 = calcMedMeanInter(data, metric)
            median.append(medi)
            inter25.append(int25)
            inter75.append(int75)
    else:
        for data in metric:

            medi, int25, int75 = calcMedMeanInter(plotData, data)
            median.append(medi)
            inter25.append(int25)
            inter75.append(int75)
    # coef = np.polyfit(x,median,2)
    # poly1d_fn = np.poly1d(coef)
    return plt.plot(x, median, color, linestyle='dashed', alpha=0.3)
    # plt.fill_between(x,inter25, inter75,alpha=.1, color=color)


def export_legend(legend, filename, expand=[-5, -5, 5, 5]):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename + '.png', dpi="figure", bbox_inches=bbox)
    fig.savefig(filename + '.pdf', bbox_inches=bbox)


def main():
    encodec1_5 = pd.read_csv('results_tsp/encodec1_5.csv', index_col=0)
    encodec3 = pd.read_csv('results_tsp/encodec3.csv', index_col=0)
    encodec6 = pd.read_csv('results_tsp/encodec6.csv', index_col=0)
    encodec12 = pd.read_csv('results_tsp/encodec12.csv', index_col=0)

    audiodec8 = pd.read_csv('results_tsp/audiodec8.csv', index_col=0)

    lyra3_2 = pd.read_csv('results_tsp/lyra3_2.csv', index_col=0)
    lyra6 = pd.read_csv('results_tsp/lyra6.csv', index_col=0)
    lyra9_2 = pd.read_csv('results_tsp/lyra9_2.csv', index_col=0)

    opus6 = pd.read_csv('results_tsp/opus6.csv', index_col=0)
    opus8 = pd.read_csv('results_tsp/opus8.csv', index_col=0)
    opus10 = pd.read_csv('results_tsp/opus10.csv', index_col=0)
    opus12 = pd.read_csv('results_tsp/opus12.csv', index_col=0)

    opus6_16 = pd.read_csv('results_tsp/opus6_16.csv', index_col=0)
    opus8_16 = pd.read_csv('results_tsp/opus8_16.csv', index_col=0)
    opus10_16 = pd.read_csv('results_tsp/opus10_16.csv', index_col=0)
    opus12_16 = pd.read_csv('results_tsp/opus12_16.csv', index_col=0)

    distances = pd.read_csv('results_tsp/Distances_64.csv', index_col=0)

    quant_16_nft = pd.read_csv('results_tsp/quantizer16.csv', index_col=0)
    quant_24_nft = pd.read_csv('results_tsp/quantizer24.csv', index_col=0)
    quant_32_nft = pd.read_csv('results_tsp/quantizer32.csv', index_col=0)
    quant_64_nft = pd.read_csv('results_tsp/quantizer64.csv', index_col=0)
    quant_16_ft = pd.read_csv('results_tsp/quantizer16_ft.csv', index_col=0)
    quant_24_ft = pd.read_csv('results_tsp/quantizer24_ft.csv', index_col=0)
    quant_32_ft = pd.read_csv('results_tsp/quantizer32_ft.csv', index_col=0)
    quant_64_ft = pd.read_csv('results_tsp/quantizer64_ft.csv', index_col=0)

    fixed_16 = pd.read_csv('results_tsp/fixed16.csv', index_col=0)
    fixed_24 = pd.read_csv('results_tsp/fixed24.csv', index_col=0)
    fixed_32 = pd.read_csv('results_tsp/fixed32.csv', index_col=0)
    fixed_64 = pd.read_csv('results_tsp/fixed64.csv', index_col=0)

    fixed_16_ft = pd.read_csv('results_tsp/fixed16_ft.csv', index_col=0)
    fixed_24_ft = pd.read_csv('results_tsp/fixed24_ft.csv', index_col=0)
    fixed_32_ft = pd.read_csv('results_tsp/fixed32_ft.csv', index_col=0)
    fixed_64_ft = pd.read_csv('results_tsp/fixed64_ft.csv', index_col=0)

    variable_16 = pd.read_csv('results_tsp/variable16.csv', index_col=0)
    variable_24 = pd.read_csv('results_tsp/variable24.csv', index_col=0)
    variable_32 = pd.read_csv('results_tsp/variable32.csv', index_col=0)
    variable_64 = pd.read_csv('results_tsp/variable64.csv', index_col=0)

    variable_16_ft = pd.read_csv('results_tsp/variable16_ft.csv', index_col=0)
    variable_24_ft = pd.read_csv('results_tsp/variable24_ft.csv', index_col=0)
    variable_32_ft = pd.read_csv('results_tsp/variable32_ft.csv', index_col=0)
    variable_64_ft = pd.read_csv('results_tsp/variable64_ft.csv', index_col=0)

    variable_16_big = pd.read_csv(
        'results_tsp/variable16_big.csv', index_col=0)
    variable_24_big = pd.read_csv(
        'results_tsp/variable24_big.csv', index_col=0)
    variable_32_big = pd.read_csv(
        'results_tsp/variable32_big.csv', index_col=0)
    variable_64_big = pd.read_csv(
        'results_tsp/variable64_big.csv', index_col=0)
    variable_16_bigsym = pd.read_csv(
        'results_tsp/variable16_big_sym.csv', index_col=0)
    variable_24_bigsym = pd.read_csv(
        'results_tsp/variable24_big_sym.csv', index_col=0)
    variable_32_bigsym = pd.read_csv(
        'results_tsp/variable32_big_sym.csv', index_col=0)
    variable_64_bigsym = pd.read_csv(
        'results_tsp/variable64_big_sym.csv', index_col=0)

    causal_vocoder = pd.read_csv('results_tsp/vocoder.csv', index_col=0)
    causal_big_vocoder = pd.read_csv(
        'results_tsp/vocoder_big.csv', index_col=0)
    non_causal_vocoder = pd.read_csv(
        'results_tsp/vocoder_big_sym.csv', index_col=0)

    savefigure = True

    # selected comparisons
    pairings = ([variable_16_ft, encodec1_5],
                [variable_32_ft, lyra3_2],
                [variable_32_ft, encodec3],
                [variable_64_ft, lyra6],
                [variable_64_ft, encodec6],
                [variable_64_ft, opus6],
                [variable_64_ft, audiodec8],
                )
    for a, b in pairings:

        for metric in ['pesq', 'visqol', 'nisqa_mos48', 'nisqa_mos16', 'f0_rmse']:
            if metric == 'f0_rmse':
                k = np.sum(a.loc[:, metric] < b.loc[:, metric])
            else:
                k = np.sum(a.loc[:, metric] > b.loc[:, metric])
            ci = binomtest(k, a.shape[0], 0.5).proportion_ci(
                confidence_level=0.95, method='exact')

            l, p, h = [ci.low, k / a.shape[0], ci.high]
            if h < 0.5:
                sign = '\\textcolor{BrickRed}{\\textbf'
            elif l > 0.5:
                sign = '{\\textbf'
            else:
                sign = '{'

            print("\\multirow{1}*{%s{%.2f  (%.2f, %.2f)}}} &" %
                  (sign, p, l, h))
        print("\\\\ \\hdashline[1pt/1pt]")

    ###############################################################################################
    # B - comparison of fixed BR, variable BR both small vocoder and VQ baseline small + small FT
    # distance plot
    plot_array_L1 = ['L1_fix_16', 'L1_fix_24', 'L1_fix_32', 'L1_fix_64', 'L1_var_16', 'L1_var_24',
                     'L1_var_32', 'L1_var_64', 'L1_quant_16', 'L1_quant_24', 'L1_quant_32', 'L1_quant_64']
    plot_array_L2 = ['L2_fix_16', 'L2_fix_24', 'L2_fix_32', 'L2_fix_64', 'L2_var_16', 'L2_var_24',
                     'L2_var_32', 'L2_var_64', 'L2_quant_16', 'L2_quant_24', 'L2_quant_32', 'L2_quant_64']
    kbps = [1.38, 2.07, 2.76, 4]

    FACTOR1 = 0.8
    FONTSIZE1 = 7.7
    plt.figure(figsize=(FACTOR1*3.5, FACTOR1*2.8), dpi=300)
    for name, bit in zip(plot_array_L1[0:4], kbps):
        h1 = singleErrorBar(distances, name, bit, 'darkgreen',   'o')
    medianTrend(distances, plot_array_L1[0:4], kbps, 'darkgreen', True)
    for name, bit in zip(plot_array_L1[4:8], [1.38, 2.07, 2.76, 4+0.05]):
        h2 = singleErrorBar(distances, name, bit, 'darkviolet',   'o')
    medianTrend(distances, plot_array_L1[4:8], [
                1.38, 2.07, 2.76, 4+0.05], 'darkviolet', True)
    for name, bit in zip(plot_array_L1[8:12], [1.38, 2.07, 2.76, 4-0.05]):
        h3 = singleErrorBar(distances, name, bit, 'black', 'o')

    medianTrend(distances, plot_array_L1[8:12], [
                1.38, 2.07, 2.76, 4-0.05], 'black', True)
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
        legend = plt.legend([h1[0], h2[0], h3[0]], ['proposed - fixed-bitrate',
                                                    'proposed - bitrate-scalable', 'VQ baseline'],
                            loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
        export_legend(legend, 'B_Distance_Legend')

    plt.figure(figsize=(FACTOR1*3.5, FACTOR1*2.8), dpi=300)
    for name, bit in zip(plot_array_L2[0:4], kbps):
        h1 = singleErrorBar(distances, name, bit, 'darkgreen',   'o')
    medianTrend(distances, plot_array_L2[0:4], kbps, 'darkgreen', True)
    for name, bit in zip(plot_array_L2[4:8], [1.38, 2.07, 2.76, 4+0.05]):
        h2 = singleErrorBar(distances, name, bit, 'darkviolet',   'o')
    medianTrend(distances, plot_array_L2[4:8], [
                1.38, 2.07, 2.76, 4+0.05], 'darkviolet', True)
    for name, bit in zip(plot_array_L2[8:12], [1.38, 2.07, 2.76, 4-0.05]):
        h3 = singleErrorBar(distances, name, bit, 'black', 'o')
    medianTrend(distances, plot_array_L2[8:12], [
                1.38, 2.07, 2.76, 4-0.05], 'black', True)
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
                  quant_16_ft, quant_24_ft, quant_32_ft, quant_64_ft,
                  ]
    color_array = ['darkgreen', 'darkgreen', 'darkgreen', 'darkgreen',
                   'darkviolet', 'darkviolet', 'darkviolet', 'darkviolet',
                   'magenta', 'magenta', 'magenta', 'magenta',
                   'black', 'black', 'black', 'black',
                   'grey', 'grey', 'grey', 'grey',
                   ]
    kbps = (np.array([[1.38, 2.07, 2.76, 4]]) +
            np.linspace(-0.15, 0.15, 5, endpoint=True)[:, None]).flatten()
    marker = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o',
              'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
    for metric, metricname in zip(['nisqa_mos48', 'visqol',  'pesq', 'f0_rmse'], ['NISQA', 'ViSQOL', 'PESQ', '$F_0$ RMSE (semitones)']):
        plt.rcParams.update({'figure.autolayout': True})
        plt.figure(dpi=300, figsize=(FACTOR1*6.4, FACTOR1*4.8))
        m = []
        for data, color, bit in zip(plot_array, color_array, kbps):
            m.append(singleErrorBar(data, metric, bit, color, 'o', markers=4.0))
        h1 = medianTrend(plot_array[0:4], metric, kbps[0:4], 'darkgreen')
        h2 = medianTrend(plot_array[4:8], metric, kbps[4:8], 'darkviolet')
        h2 = medianTrend(plot_array[8:12], metric, kbps[8:12], 'magenta')
        h3 = medianTrend(plot_array[12:16], metric, kbps[12:16], 'black')
        h4 = medianTrend(plot_array[16:20], metric, kbps[16:20], 'grey')
        plt.xlim([1.1, 4.28])
        plt.yticks(fontsize=FONTSIZE1)
        plt.ylabel(metricname, fontsize=FONTSIZE1)

        plt.xlabel('bitrate in kbps', fontsize=FONTSIZE1)
        plt.xticks([1.38, 2.07, 2.76, 4.0], ['1.38', '2.07',
                   '2.76', '5.51'], fontsize=FONTSIZE1)
        plt.minorticks_off()
        plt.grid()
        if savefigure:
            plt.savefig('B_' + metric + '.png')
            plt.savefig('B_' + metric + '.pdf')

    if savefigure:
        plt.figure()
        legend = plt.legend([m[0][0], m[4][0], m[8][0], m[12][0], m[16][0]], ['prop. fixed-bitrate BigVGAN-tiny-causal non-fine-tuned',
                                                                              'prop. bitrate-scalable BigVGAN-tiny-causal non-fine-tuned',
                                                                              'prop. bitrate-scalable BigVGAN-tiny-causal fine-tuned',
                                                                              'VQ baseline non-fine-tuned',
                                                                              'VQ baseline fine-tuned'],
                            loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2)
        export_legend(legend, 'B_Legend')

    ###############################################################################################
    # C - Effect of Vocoder Complexity and Causality
    plot_array = [variable_16, variable_24, variable_32, variable_64,
                  variable_16_big, variable_24_big, variable_32_big, variable_64_big,
                  variable_16_bigsym, variable_24_bigsym, variable_32_bigsym, variable_64_bigsym,
                  causal_vocoder, causal_big_vocoder, non_causal_vocoder]
    color_array = ['darkviolet', 'darkviolet', 'darkviolet', 'darkviolet', 'darkviolet', 'darkviolet', 'darkviolet', 'darkviolet',
                   'darkviolet', 'darkviolet', 'darkviolet', 'darkviolet', 'darkviolet', 'darkviolet', 'darkviolet']
    kbps = [1.28, 1.97, 2.66, 3.9,
            1.38, 2.07, 2.76, 4,
            1.48, 2.17, 2.86, 4.1,
            4.9, 5.0, 5.1]
    marker = ['o', 'o', 'o', 'o', 's', 's', 's',
              's', 'D', 'D', 'D', 'D', 'o', 's', 'D']
    fcolor_array = ['darkviolet', 'darkviolet', 'darkviolet', 'darkviolet', 'white', 'white', 'white', 'white',
                    'white', 'white', 'white', 'white', 'darkviolet', 'white', 'white']
    ms = [4.0, 4.0, 4.0, 4.0, 5, 5, 5, 5, 5, 5, 5, 5, 4.0, 5, 5]

    for metric, metricname in zip(['nisqa_mos48', 'visqol',  'pesq', 'f0_rmse'], ['NISQA', 'ViSQOL', 'PESQ', 'F0 RMSE (semitones)']):
        plt.rcParams.update({'figure.autolayout': True})
        plt.figure(dpi=300, figsize=(FACTOR1*6.4, FACTOR1*4.8))
        m = []
        for data, color, fcolor,  bit, mark, markers in zip(plot_array, color_array, fcolor_array, kbps, marker, ms):
            m.append(singleErrorBar(data, metric, bit,
                     color, mark, fcolor, markers))
        h1 = medianTrend(plot_array[0:4], metric, kbps[0:4], 'darkviolet')
        h2 = medianTrend(plot_array[4:8], metric, kbps[4:8], 'darkviolet')
        h3 = medianTrend(plot_array[8:12], metric, kbps[8:12], 'darkviolet')
        plt.yticks(fontsize=FONTSIZE1)
        plt.ylabel(metricname, fontsize=FONTSIZE1)

        plt.xlabel('bitrate in kbps', fontsize=FONTSIZE1)
        plt.xticks([1.38, 2.07, 2.76, 4, 5], ['1.38', '2.07',
                   '2.76', '5.51', 'vocoder\nonly'], fontsize=FONTSIZE1)
        plt.minorticks_off()
        plt.gca().set_clip_on(False)
        plt.grid()
        if savefigure:
            plt.savefig('C_' + metric + '.png')
            plt.savefig('C_' + metric + '.pdf')
    if savefigure:
        plt.figure()
        legend = plt.legend([m[0][0], m[4][0], m[8][0]],
                            ['BigVGAN-tiny-causal',
                                'BigVGAN-base-causal', 'BigVGAN-base'],
                            loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=3)
        export_legend(legend, 'C_Legend')

    ###############################################################################################
    # D - comparison with SOTA Codecs
    FACTOR1 = 0.85
    FONTSIZE1 = 8

    plot_array = [encodec1_5, encodec3, encodec6, encodec12,
                  lyra3_2, lyra6, lyra9_2,
                  opus6, opus8, opus10, opus12,
                  #   opus6_16, opus8_16, opus10_16, opus12_16,
                  variable_16_ft, variable_24_ft, variable_32_ft, variable_64_ft,
                  #   variable_16_big, variable_24_big, variable_32_big, variable_64_big,
                  audiodec8]

    # plot_array2 = [encodec1_5_tsp, encodec3_tsp, encodec6_tsp, encodec12_tsp,
    #               lyra3_2_tsp, lyra6_tsp, lyra9_2_tsp,
    #               opus6_tsp, opus8_tsp, opus10_tsp, opus12_tsp,
    #               #   opus6_16, opus8_16, opus10_16, opus12_16,
    #               variable_16_ft_tsp, variable_24_ft_tsp, variable_32_ft_tsp, variable_64_ft_tsp,
    #               #   variable_16_big, variable_24_big, variable_32_big, variable_64_big,
    #               audiodec8_tsp]
    
    color_array = ['blue', 'blue', 'blue', 'blue',
                   'darkorange', 'darkorange', 'darkorange',
                   'orangered', 'orangered', 'orangered', 'orangered',
                   #    'red', 'red', 'red', 'red',
                   'magenta', 'magenta', 'magenta', 'magenta',
                   #  'violet', 'violet', 'violet', 'violet',
                   'green']

    kbps = [1.5, 3, 5.9, 11.95, 3.2, 6, 9.2, 6.1,
            8, 10, 12.05, 1.38, 2.07, 2.76, 5.51, 6.4]
    ms = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
          4.0, 4.0, 5, 5, 5, 5, 4.0, 4.0, 4.0, 4.0, 4.0]

    fcolor_array = ['blue', 'blue', 'blue', 'blue', 'darkorange', 'darkorange', 'darkorange', 'orangered', 'orangered', 'orangered', 'orangered',
                    # 'white', 'white', 'white', 'white',
                    'magenta', 'magenta', 'magenta', 'magenta', 'green']
    markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o',
               'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']

    plt.rcParams.update({'figure.autolayout': True})

    for metric, metricname in zip(['pesq', 'visqol',  'nisqa_mos48', 'nisqa_mos16', 'f0_rmse'], ['PESQ', 'ViSQOL', 'NISQA', 'NISQA 16kHz', '$F_0$ RMSE (semitones)']):
        plt.figure(dpi=300, figsize=(FACTOR1*6.4, FACTOR1*3.2))
        m = []
        for data, color, bit, msize, marker, fc in zip(plot_array, color_array, kbps, ms, markers, fcolor_array):
            m.append(singleErrorBar(data, metric,
                     bit, color, 'o', fc, msize))
        # for data, color, bit, msize, marker, fc in zip(plot_array2, color_array, kbps, ms, markers, fcolor_array):
        #     m.append(singleErrorBar(data, metric,
        #              bit + 0.05, color, '>', fc, msize))

        h1 = medianTrend(plot_array[0:4], metric, np.array(kbps[0:4]), 'blue')
        h2 = medianTrend(plot_array[4:7], metric, np.array(kbps[4:7]), 'darkorange')
        h2 = medianTrend(plot_array[7:11], metric, np.array(kbps[7:11]), 'orangered')
        h3 = medianTrend(plot_array[11:15], metric, np.array(kbps[11:15]), 'magenta')
        h4 = medianTrend(plot_array[15:16], metric, np.array(kbps[15:16]), 'green')

        # h1 = medianTrend(plot_array[0:4], metric, np.array(kbps[0:4])+ 0.05, 'blue')
        # h2 = medianTrend(plot_array[4:7], metric, np.array(kbps[4:7])+ 0.05, 'darkorange')
        # h2 = medianTrend(plot_array[7:11], metric, np.array(kbps[7:11])+ 0.05, 'orangered')
        # h3 = medianTrend(plot_array[11:15], metric, np.array(kbps[11:15])+ 0.05, 'magenta')
        # h4 = medianTrend(plot_array[15:16], metric, np.array(kbps[15:16])+ 0.05, 'green')


        plt.xlim([1, 12.5])
        plt.ylabel(metricname, fontsize=FONTSIZE1)

        # if metric == 'f0_rmse': # invert the y-axis
        #     plt.gca().invert_yaxis()

        plt.xlabel('bitrate in kbps', fontsize=FONTSIZE1)
        plt.xticks([1.38, 1.5, 2.07, 2.76, 3, 3.2, 5.5, 6, 6.4, 12, 9.2, 8, 10],
                   ['1.38', '\n1.5', '\n\n2.07', '2.76', '\n3', '\n\n3.2', '5.51', '\n6', '6.4', '12', '9.2', '8', '10'], fontsize=FONTSIZE1)
        plt.minorticks_off()
        plt.grid()
        if savefigure:
            plt.savefig('D_' + metric + '.png')
            plt.savefig('D_' + metric + '.pdf')
    if savefigure:
        plt.figure()
        legend = plt.legend([m[0][0], m[15][0], m[4][0], m[7][0], m[11][0]],
                            ['EnCodec', 'AudioDec', 'Lyra v2', 'Opus',
                             'prop. bitrate-scalable BigVGAN-tiny-causal fine-tuned'],
                            loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, ncol=3)
        export_legend(legend, 'D_Legend')

    ###############################################################################################
    # E - listening test results
    plt.figure(dpi=300, figsize=(FACTOR1*6.4, FACTOR1*4))

    mean_ratings = np.loadtxt('mean_ratings.csv')
    mean_ranks = np.loadtxt('mean_ranks.csv')

    for testset in ['VCTK testset', 'TSP testset']:
        if testset == 'VCTK testset':
            mr = mean_ratings[:8, :]
            mrk = mean_ranks[:8, :]
        else:
            mr = mean_ratings[8:, :]
            mrk = mean_ranks[8:, :]

        pairings = [[mr[:, 1], mr[:, 4]],
                    [mr[:, 3 ], mr[:, 5]],
                    [mr[:, 2], mr[:, 6]]]
        print('###########')
        for a, b in pairings:
            k = np.sum(a > b)
            ci = binomtest(k, a.shape[0], 0.5).proportion_ci(
                confidence_level=0.95, method='exact')

            l, p, h = [ci.low, k / a.shape[0], ci.high]
            if h < 0.5:
                sign = '\\textcolor{BrickRed}{\\textbf'
            elif l > 0.5:
                sign = '\\textcolor{ForestGreen}{\\textbf'
            else:
                sign = '{'

            print("\\multirow{2}*{%s{%.2f  (%.2f, %.2f)}}} &" %
                  (sign, p, l, h))

            diff = a - b
            tppf = tdist.ppf(0.975, diff.shape[0] - 1)

            dpm = tppf / np.sqrt(diff.shape[0])
            d = np.mean(diff) / np.std(diff, ddof=1)
            if d + dpm < 0:
                sign = '\\textcolor{BrickRed}{\\textbf'
            elif d - dpm > 0:
                sign = '\\textcolor{ForestGreen}{\\textbf'

            print("\\multirow{2}*{%s{%.2f  (%.2f, %.2f)}}} &" %
                  (sign, d, d - dpm, d + dpm))



        # pairings = [[mrk[:, 1], mrk[:, 4]],
        #             [mrk[:, 3 ], mrk[:, 5]],
        #             [mrk[:, 2], mrk[:, 6]]]
        # print('###########')
        # for a, b in pairings:
        #     k = np.sum(a < b)
        #     ci = binomtest(k, a.shape[0], 0.5).proportion_ci(
        #         confidence_level=0.95, method='exact')

        #     l, p, h = [ci.low, k / a.shape[0], ci.high]
        #     if h < 0.5:
        #         sign = '\\textcolor{BrickRed}{\\textbf'
        #     elif l > 0.5:
        #         sign = '{\\textbf'
        #     else:
        #         sign = '{'

        #     print("\\multirow{2}*{%s{%.2f  (%.2f, %.2f)}}} &" %
        #           (sign, p, l, h))
        FONTSIZE1 = 6
        plot_array = []
        for i in range(mean_ratings.shape[1]):
            plot_array.append(pd.DataFrame({'MMUSHRA': mr[:8, i]}))
            # plot_array.append(pd.DataFrame({'MMUSHRA': mean_ratings[8:, i]}))

        kbps = [1.38-0.05, 1.38+0.05,
                5.51-0.05, 5.51+0.05,
                3, 6, 6.4, 9.5, 8]

        color_array = ['magenta',  'violet',
                       'magenta',  'violet',
                       'darkorange', 'darkorange',
                       'green',  'black',
                       'black']
        ms = [4.0] * 9
        fcolor_array = ['magenta',  'violet',
                        'magenta',  'violet',
                        'darkorange', 'darkorange',
                        'green',  'black',
                        'black']
        markers = ['o'] * 9

        plt.figure(dpi=300, figsize=(FACTOR1*2.9, FACTOR1*2.5))
        m = []
        for data, color, bit, msize, marker, fc in zip(plot_array, color_array, kbps, ms, markers, fcolor_array):
            m.append(singleErrorBar(data, 'MMUSHRA',
                     bit, color, marker, fc, msize))
        h1 = medianTrend([plot_array[0], plot_array[2]],
                         'MMUSHRA', np.array(kbps)[[0, 2]], 'magenta')
        h2 = medianTrend([plot_array[1], plot_array[3]],
                         'MMUSHRA', np.array(kbps)[[1, 3]], 'violet')
        h3 = medianTrend([plot_array[4], plot_array[5]],
                         'MMUSHRA', np.array(kbps)[[4, 5]], 'darkorange')

        plt.xlim([1, 10])
        plt.ylim([11, 105])

        plt.yticks([20, 40, 60, 80, 100], fontsize=FONTSIZE1)
        plt.xticks([1.38, 3, 5.51, 6, 6.4, 8, 9.5], ['1.38', '3', '5.51',
                                                '\n6', '6.4', 'Anchor', 'Ref.'], fontsize=FONTSIZE1)
        plt.minorticks_off()
        plt.ylabel('Mean MUSHRA Score', fontsize=FONTSIZE1)
        plt.xlabel('bitrate in kbps', fontsize=FONTSIZE1)
        plt.grid()
        plt.title(testset, fontsize=FONTSIZE1)
        if savefigure:
            plt.savefig('E_MMUSHRA_%s.png' % testset)
            plt.savefig('E_MMUSHRA_%s.pdf' % testset)

        plot_array = []
        for i in range(mean_ratings.shape[1]):
            plot_array.append(pd.DataFrame({'MMUSHRA': mrk[:8, i]}))
            # plot_array.append(pd.DataFrame({'MMUSHRA': mean_ratings[8:, i]}))

        kbps = [1.38-0.05, 1.38+0.05,
                5.51-0.05, 5.51+0.05,
                3, 6, 6.4, 9.5, 8]

        color_array = ['magenta',  'violet',
                       'magenta',  'violet',
                       'darkorange', 'darkorange',
                       'green',  'black',
                       'black']
        ms = [4.0] * 9
        fcolor_array = ['magenta',  'violet',
                        'magenta',  'violet',
                        'darkorange', 'darkorange',
                        'green',  'black',
                        'black']
        markers = ['o'] * 9

        plt.figure(dpi=300, figsize=(FACTOR1*2.9, FACTOR1*2.5))
        m = []
        for data, color, bit, msize, marker, fc in zip(plot_array, color_array, kbps, ms, markers, fcolor_array):
            m.append(singleErrorBar(data, 'MMUSHRA',
                     bit, color, marker, fc, msize))
        h1 = medianTrend([plot_array[0], plot_array[2]],
                         'MMUSHRA', np.array(kbps)[[0, 2]], 'magenta')
        h2 = medianTrend([plot_array[1], plot_array[3]],
                         'MMUSHRA', np.array(kbps)[[1, 3]], 'violet')
        h3 = medianTrend([plot_array[4], plot_array[5]],
                         'MMUSHRA', np.array(kbps)[[4, 5]], 'darkorange')

        plt.xlim([1, 10])
        plt.ylim([0.5, 9.5])

        plt.yticks([1, 2, 3, 4, 5, 6, 7, 8, 9], fontsize=FONTSIZE1)

        plt.xticks([1.38, 3, 5.51, 6, 6.4, 8, 9.5], ['1.38', '3', '5.51',
                                                '\n6', '6.4', 'Anchor', 'Ref.'], fontsize=FONTSIZE1)
        plt.minorticks_off()
        plt.gca().invert_yaxis()
        plt.ylabel('Mean MUSHRA Rank', fontsize=FONTSIZE1)
        plt.xlabel('bitrate in kbps', fontsize=FONTSIZE1)
        plt.grid()
        plt.title(' ', fontsize=FONTSIZE1)
        if savefigure:
            plt.savefig('E_MRkMUSHRA_%s.png' % testset)
            plt.savefig('E_MRkMUSHRA_%s.pdf' % testset)

    if savefigure:
        legend = plt.legend([m[0][0], m[1][0], m[4][0], m[6][0]], ['prop. bitrate-scalable BigVGAN-tiny-causal fine-tuned',
                                                                   'prop. bitrate-scalable BigVGAN-tiny-causal fine-tuned 16kHz',
                                                                   'Lyra v2', 'AudioDec'],
                            loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, ncol=2)
        export_legend(legend, 'E_Legend')


    ###F - Opus 16 vs 48 kHz
    FACTOR1 = 0.85
    FONTSIZE1 = 8

    plot_array = [opus6, opus8, opus10, opus12,
                  opus6_16, opus8_16, opus10_16, opus12_16,
                  ]

    # plot_array2 = [encodec1_5_tsp, encodec3_tsp, encodec6_tsp, encodec12_tsp,
    #               lyra3_2_tsp, lyra6_tsp, lyra9_2_tsp,
    #               opus6_tsp, opus8_tsp, opus10_tsp, opus12_tsp,
    #               #   opus6_16, opus8_16, opus10_16, opus12_16,
    #               variable_16_ft_tsp, variable_24_ft_tsp, variable_32_ft_tsp, variable_64_ft_tsp,
    #               #   variable_16_big, variable_24_big, variable_32_big, variable_64_big,
    #               audiodec8_tsp]
    
    color_array = ['orangered', 'orangered', 'orangered', 'orangered',
                   'orangered', 'orangered', 'orangered', 'orangered']

    kbps = [6-0.05, 8-0.05, 10-0.05, 12-0.05, 
            6+0.05, 8+0.05, 10+0.05, 12+0.05]
    ms = [4.0, 4.0, 4.0, 4.0,5, 5, 5, 5]

    fcolor_array = ['orangered', 'orangered', 'orangered', 'orangered',
                    'white', 'white', 'white', 'white']
    markers = ['o', 'o', 'o', 'o', 's', 's', 's', 's',]

    plt.rcParams.update({'figure.autolayout': True})

    for metric, metricname in zip(['pesq'], ['PESQ']):
        plt.figure(dpi=300, figsize=(FACTOR1*6.4, FACTOR1*3.2))
        m = []
        for data, color, bit, msize, marker, fc in zip(plot_array, color_array, kbps, ms, markers, fcolor_array):
            m.append(singleErrorBar(data, metric,
                     bit, color, 'o', fc, msize))
        # for data, color, bit, msize, marker, fc in zip(plot_array2, color_array, kbps, ms, markers, fcolor_array):
        #     m.append(singleErrorBar(data, metric,
        #              bit + 0.05, color, '>', fc, msize))

        
        h1 = medianTrend(plot_array[:4], metric, np.array(kbps[:4]), 'orangered')
        h2 = medianTrend(plot_array[4:], metric, np.array(kbps[4:]), 'orangered')
       

        # h1 = medianTrend(plot_array[0:4], metric, np.array(kbps[0:4])+ 0.05, 'blue')
        # h2 = medianTrend(plot_array[4:7], metric, np.array(kbps[4:7])+ 0.05, 'darkorange')
        # h2 = medianTrend(plot_array[7:11], metric, np.array(kbps[7:11])+ 0.05, 'orangered')
        # h3 = medianTrend(plot_array[11:15], metric, np.array(kbps[11:15])+ 0.05, 'magenta')
        # h4 = medianTrend(plot_array[15:16], metric, np.array(kbps[15:16])+ 0.05, 'green')


        plt.xlim([5.5, 12.5])
        plt.ylabel(metricname, fontsize=FONTSIZE1)

        # if metric == 'f0_rmse': # invert the y-axis
        #     plt.gca().invert_yaxis()
        legend = plt.legend([m[0][0], m[5][0]],
                            ['Opus 48kHz', 'Opus 16kHz'],
                            loc='best')
        plt.xlabel('bitrate in kbps', fontsize=FONTSIZE1)
        plt.xticks([6, 8, 10, 12],
                   ['6', '8', '10', '12'], fontsize=FONTSIZE1)
        plt.minorticks_off()
        plt.grid()
        if savefigure:
            plt.savefig('F_' + metric + '.png')
            plt.savefig('F_' + metric + '.pdf')


    # h1 = medianTrend(plot_array[0:4], metric, kbps[0:4], 'blue')
    # h2 = medianTrend(plot_array[4:7], metric, kbps[4:7], 'darkorange')
    # h2 = medianTrend(plot_array[7:11], metric, kbps[7:11], 'orangered')
    # h3 = medianTrend(plot_array[11:15], metric, kbps[11:15], 'magenta')
    # h4 = medianTrend(plot_array[15:16], metric, kbps[15:16], 'green')
    # plt.xlim([1,12.5])
    # plt.ylabel(metricname, fontsize=FONTSIZE1)

    # # if metric == 'f0_rmse': # invert the y-axis
    # #     plt.gca().invert_yaxis()

    # plt.xlabel('bitrate in kbps', fontsize=FONTSIZE1)
    # plt.xticks([1.38, 1.5, 2.07, 2.76, 3, 3.2, 5.5, 6, 12, 9.2, 8, 10],
    #             ['1.38', '\n1.5', '\n\n2.07', '2.76', '\n3','\n\n3.2', '5.51', '\n6', '12','9.2','8','10'], fontsize=FONTSIZE1)
    # plt.minorticks_off()
    # plt.grid()
    # if savefigure:
    #     plt.savefig('D_' + metric + '.png')
    #     plt.savefig('D_' + metric + '.pdf')

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
    #     # legend = plt.legend([h1[0], h2[0], h3[0], h4[0], h5[0]], ['Encodec', 'Lyra', 'Opus', 'bitrate-scalable small fine-tuned', 'bitrate-scalable big' ],
    #     #                      loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, ncol=1)
    #     legend = plt.legend([m[0][0], m[4][0], m[7][0], m[11][0], m[15][0]], ['EnCodec', 'Lyra v2', 'Opus', 'bitrate-scalable small fine-tuned', 'bitrate-scalable big' ],
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
