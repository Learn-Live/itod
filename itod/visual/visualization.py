"""Plot data distribution

"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from itod.utils.tool import stat_data
import seaborn as sns
import pandas as pd


# rcParams.update({'figure.autolayout': True})


def plot_bar(data=[], datasets=['PC3(UNB)', 'MAWI', 'SFrig(private)'],
             repres=['IAT', 'IAT-FFT', 'SIZE'],
             colors=['tab:brown', 'tab:green', 'm', 'c', 'b', 'r'],
             output_file="F1_for_all.pdf", xlim=[-0.1, 1], show_flg=False):
    # create plot
    fig, ax = plt.subplots()
    bar_width = 0.13
    opacity = 1

    def autolabel(rects, aucs=['max', 'min'], pre_aucs=[0, 0]):

        for rect in rects:
            height = rect.get_height()
            if aucs[0] == pre_aucs[0]:
                offset = np.random.randint(low=1, high=4) / 100  # avoid overlap of two same aucs
                # print(rect, aucs, pre_aucs, offset)
            else:
                offset = 0
            # ax.text(rect.get_x() + rect.get_width()/2., 1.01*height, '%d' % int(height * 100), fontsize=5, ha='center', va='bottom')
            ax.text(rect.get_x() + rect.get_width() / 2.3, 1.005 * height + offset, '%.2f' % (height * 1.0),
                    # label offset
                    fontsize=4.5, fontweight='bold',
                    ha='center',
                    va='bottom')

            if aucs[1] != '':
                ax.text(rect.get_x() + rect.get_width() / 2.3, 0.8 * height + offset, '%.2f' % (aucs[1] * 1.0),
                        # label offset
                        fontsize=4.5, fontweight='bold', color='white',
                        ha='center',
                        va='bottom')

    s = min(len(repres), len(colors))
    for ind, data_name in enumerate(datasets):
        # print(ind, data_name, datasets)
        for i, reprs in enumerate(repres[:s]):
            try:
                if 'SAMP-' in reprs.upper():
                    max_auc, min_auc = data[ind][i].split(')')[0].split('(')[1].split('-')
                    aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                else:
                    aucs = [float(data[ind][i]), '']
            except Exception as e:
                print(f'ind: {ind}, data_name: {data_name}, i: {i}, reprs: {reprs}, e: {e}, out_file: {output_file}')
            rects = plt.bar((ind) + (i) * bar_width, height=aucs[0], width=bar_width, alpha=opacity,
                            color=colors[reprs],
                            label='Frank' + str(i))
            # g = sns.barplot(y="diff", x='detector', data=df, hue='representation', ax=ax,
            #                 palette=new_colors)  # palette=show_clo
            if i == 0:
                pre_aucs = [0, 0]
            autolabel(rects, aucs=aucs, pre_aucs=pre_aucs)
            pre_aucs = aucs
    # plt.xlabel('Catagory')
    plt.ylabel('AUC')
    plt.ylim(0, 1.07)
    # # plt.title('F1 Scores by category')
    n_groups = len(datasets)
    index = np.arange(n_groups)
    # print(index)
    # plt.xlim(xlim[0], n_groups)
    plt.xticks(index + len(repres) // 2 * bar_width, labels=[v for v in datasets])
    plt.tight_layout()

    plt.legend(repres, loc='lower right')
    # plt.savefig("DT_CNN_F1"+".jpg", dpi = 400)
    plt.savefig(output_file)  # should use before plt.show()
    if show_flg:
        plt.show()
        # use plt.clf() after every plt.show() to just clear the current figure instead of closing and reopening it,
        # keeping the window size and giving you a better performance and much better memory usage.
    plt.clf()  #
    # plt.cla()
    plt.close(fig)  # release memory


def plot_bar2(data='', repres=['IAT', 'IAT-FFT']):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import rcParams

    rcParams.update({'figure.autolayout': True})

    cnndata = [[66, 0, 0, 0, 1, 1, 0, 2],  # data for cnn
               [0, 65, 0, 1, 3, 0, 2, 0],
               [0, 0, 52, 0, 0, 0, 0, 7],
               [3, 0, 0, 64, 0, 0, 0, 0],
               [0, 0, 0, 0, 66, 0, 6, 1],
               [0, 0, 0, 0, 2, 74, 4, 0],
               [0, 0, 0, 0, 7, 1, 60, 0],
               [0, 0, 3, 1, 0, 1, 1, 66]]

    dtdata = [[66, 0, 0, 0, 2, 1, 0, 1],
              [1, 62, 0, 0, 4, 0, 4, 0],
              [0, 0, 72, 0, 0, 1, 0, 4],
              [1, 1, 0, 61, 0, 3, 0, 1],
              [2, 8, 0, 2, 50, 2, 7, 2],
              [0, 2, 0, 0, 1, 77, 0, 0],
              [0, 0, 0, 1, 3, 2, 61, 1],
              [0, 0, 3, 2, 1, 0, 0, 48]]

    lrdata = [[66, 0, 0, 0, 2, 1, 1, 0],
              [0, 60, 2, 1, 3, 1, 4, 0],
              [0, 0, 41, 1, 4, 0, 6, 9],
              [1, 1, 1, 58, 0, 0, 3, 3],
              [0, 2, 5, 2, 38, 0, 17, 9],
              [0, 2, 2, 1, 6, 63, 4, 2],
              [0, 6, 0, 3, 15, 2, 38, 4],
              [0, 1, 19, 0, 5, 1, 6, 38]]

    GaussianNBdata = [[52, 0, 12, 0, 1, 3, 0, 2],
                      [1, 32, 2, 1, 0, 4, 1, 30],
                      [0, 6, 11, 2, 0, 0, 1, 41],
                      [0, 1, 1, 41, 0, 17, 0, 7],
                      [2, 7, 0, 1, 0, 4, 3, 56],
                      [2, 5, 0, 0, 0, 40, 1, 32],
                      [7, 5, 2, 1, 2, 5, 2, 44],
                      [6, 5, 0, 0, 0, 1, 0, 58]]

    KNNdata = [[65, 0, 1, 1, 0, 1, 1, 1],
               [0, 60, 3, 1, 0, 1, 2, 4],
               [0, 0, 40, 2, 7, 0, 2, 10],
               [1, 1, 4, 57, 1, 1, 1, 1],
               [0, 2, 13, 0, 42, 0, 8, 8],
               [1, 5, 9, 0, 5, 52, 5, 3],
               [1, 3, 22, 4, 8, 0, 27, 3],
               [5, 1, 32, 0, 10, 1, 3, 18]]

    svmdata = [[65, 0, 0, 1, 2, 1, 0, 1],
               [0, 60, 3, 1, 2, 0, 5, 0],
               [0, 1, 43, 1, 3, 0, 2, 11],
               [2, 1, 6, 55, 0, 0, 1, 2],
               [0, 4, 4, 1, 45, 0, 12, 7],
               [0, 2, 1, 3, 7, 59, 6, 2],
               [1, 6, 4, 2, 17, 1, 35, 2],
               [2, 1, 18, 0, 6, 0, 7, 36]]

    test_results_dict = {'NB': GaussianNBdata, "SVM": svmdata, 'KNN': KNNdata, 'LR': lrdata, 'DT': dtdata,
                         'CNN': cnndata}
    app = ['Google', 'Twitter', 'Youtube', 'Outlook', 'Github', 'Facebook', 'Slack', 'Bing']

    f1_dict = {}
    for idx, (key, data) in enumerate(test_results_dict.items()):
        f1_lst = []
        for i in range(0, 8):
            tpfn = sum(data[i])
            tp = data[i][i]
            tpfp = 0
            for j in range(0, 8):
                tpfp += data[j][i]
            precision = tp / tpfp
            recall = tp / tpfn
            if precision + recall == 0:
                f1 = 0.0
                print(f'{key, i}')
            else:
                f1 = 2 * precision * recall / (precision + recall)
            f1_lst.append([precision, recall, f1])
        f1_dict[key] = f1_lst
    # print("{0}\t{1}\t{2}".format(precision,recall,f1))

    i = 0

    # data to plot
    n_groups = len(app)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    print(index)
    bar_width = 0.15
    opacity = 1

    # # test_results_dict={'NB':GaussianNBdata,"SVM":svmdata,'KNN':KNNdata, 'LR':lrdata,'DT':dtdata,'CNN':cnndata}
    key = 'NB'
    f1 = list(map(lambda x: x[-1], f1_dict[key]))
    rects1 = plt.bar(index, f1, bar_width, alpha=opacity, color='tab:brown', label='Frank')
    key = 'KNN'
    f1 = list(map(lambda x: x[-1], f1_dict[key]))
    rects2 = plt.bar(index + (2 - 1) * bar_width, f1, bar_width, alpha=opacity, color='tab:green', label='Frank2')
    key = 'SVM'
    f1 = list(map(lambda x: x[-1], f1_dict[key]))
    rects3 = plt.bar(index + (3 - 1) * bar_width, f1, bar_width, alpha=opacity, color='m', label='Frank3')
    key = 'LR'
    f1 = list(map(lambda x: x[-1], f1_dict[key]))
    rects4 = plt.bar(index + (4 - 1) * bar_width, f1, bar_width, alpha=opacity, color='c', label='Frank4')
    key = 'DT'
    f1 = list(map(lambda x: x[-1], f1_dict[key]))
    rects5 = plt.bar(index + (5 - 1) * bar_width, f1, bar_width, alpha=opacity, color='b', label='Frank5')
    key = 'CNN'
    f1 = list(map(lambda x: x[-1], f1_dict[key]))
    rects6 = plt.bar(index + (6 - 1) * bar_width, f1, bar_width, alpha=opacity, color='r', label='Frank6')

    #
    # # rects1 = plt.bar(index , f1, bar_width,alpha=opacity,color='b')
    # # rects2 = plt.bar(index + bar_width, dtf1, bar_width,alpha=opacity,color='r')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            # ax.text(rect.get_x() + rect.get_width()/2., 1.01*height, '%d' % int(height * 100), fontsize=5, ha='center', va='bottom')
            ax.text(rect.get_x() + rect.get_width() / 2.3, 1.025 * height, '%.2f' % (height * 1.0),  # label offset
                    fontsize=8, fontweight='bold',
                    ha='center',
                    va='bottom')

    # for i in range(6):
    # autolabel(rects+i)
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    autolabel(rects6)
    # # plt.xlabel('Catagory')
    plt.xlim(-0.2, n_groups)
    plt.ylabel('AUC')
    # plt.title('F1 Scores by category')
    plt.xticks(index + 2.5 * bar_width,
               (app[i], app[i + 1], app[i + 2], app[i + 3], app[i + 4], app[i + 5], app[i + 6], app[i + 7]))
    plt.tight_layout()

    # plt.legend([key for key in test_results_dict.keys() if key !='NB'], loc='lower right')
    plt.legend(['NB', 'KNN', 'SVM', 'LR', 'DT', '1D-CNN'], loc='lower right')
    # plt.savefig("DT_CNN_F1"+".jpg", dpi = 400)
    plt.savefig("F1_for_all.pdf")  # should use before plt.show()
    plt.show()


def plot_histgram(data, bins=100, title=''):
    # data = np.histogram(data, bins=bins)

    plt.hist(data, bins=bins)  # arguments are passed to np.histogram
    plt.title(f"{title}")
    plt.ylabel('Counts')
    plt.xlabel('Number of packets in each flow')
    # plt.xlim(0,15)
    plt.show()


def main():
    from itod.data.dataset import Dataset
    srcIP_lst = ['192.168.10.5', '192.168.10.8', '192.168.10.9', '192.168.10.14', '192.168.10.15']
    # sampling_lst = [4.5899, 6.7859, 7.2307, 5.7866, 3.5814]         #q=0.9
    # sampling_lst = [2.0662943522135417e-07, 1.6829546760110295e-07, 1.7881393432617188e-07,
    #                 1.430511474609375e-07, 1.6829546760110295e-07]  # q=0.1
    sampling_lst = [2.702077229817708e-07, 1.8232008990119484e-07,
                    1.9371509552001953e-07, 2.026557922363281e-07, 2.384185791015625e-07]  # q=0.3
    # sampling_lst = [0.0034731308619181315, 2.384185791015625e-07, 2.9355287551879883e-06,
    #                 0.0015740692615509033, 0.0018327656914206112]  # q=0.5
    # sampling_lst = [ 0.013366915384928386,  1.2888627893784468e-05,  4.0084123611450195e-06,
    #                   0.004426524639129641, 0.005506066715016085]     # q=0.6
    rescale_flg = False
    bins = 20
    data_aug = False
    for i, srcIP in enumerate(srcIP_lst):
        print(f'\ni: {i}, srcIP: {srcIP}')

        plt.subplot(2, 3, i + 1)  # plt.subplot(131)
        if data_aug:
            input_file = f'input_data/CICIDS2017/srcIP_{srcIP}/Friday-WorkingHours/srcIP_{srcIP}.pcap-IAT.dat_augmented.dat'
        else:
            input_file = f'input_data/CICIDS2017/srcIP_{srcIP}/Friday-WorkingHours/srcIP_{srcIP}.pcap-IAT.dat'
            # input_file = f'-input_data/DS10_UNB_IDS/srcIP_{srcIP}/Friday-WorkingHours/srcIP_{srcIP}.pcap-SampBaseline-number_10.dat'
            # input_file = f'-input_data/DS10_UNB_IDS/srcIP_{srcIP}/Friday-WorkingHours/srcIP_{srcIP}.pcap-SampBaseline-interval_0.1.dat'
            input_file = f'input_data/CICIDS2017/srcIP_{srcIP}/Friday-WorkingHours/srcIP_{srcIP}.pcap-SampBaseline-rate_{sampling_lst[i]}.dat'

        print(f'input_file: {input_file}')
        data_inst = Dataset(input_file=input_file)
        print(f'Counter(data_inst.labels): {Counter(data_inst.labels)}')

        individual_flg = False
        if individual_flg:
            anomaly_IAT_len_arr = []
            normal_IAT_len_arr = []
            other_IAT_len_arr = []
            for i, iat in enumerate(data_inst.features):
                if data_inst.labels[i] not in ['NORMAL', 'BENIGN', 0, None, 'None']:
                    anomaly_IAT_len_arr.append(len(iat))
                elif data_inst.labels[i] in ['NORMAL', 'BENIGN']:
                    normal_IAT_len_arr.append(len(iat))
                else:
                    other_IAT_len_arr.append(len(iat))

            anomaly_IATs_lens_flg = True  # only show attack IATs dimensions distribution
            norm_IATs_lens_flg = True
            if anomaly_IATs_lens_flg:
                IAT_len_arr = anomaly_IAT_len_arr
                title = f'srcIP:{srcIP},\nmin:{min(IAT_len_arr)}, max:{max(IAT_len_arr)},\ntotal anomaly: {len(IAT_len_arr)}'
            elif norm_IATs_lens_flg:
                IAT_len_arr = normal_IAT_len_arr
                title = f'srcIP:{srcIP},\nmin:{min(IAT_len_arr)}, max:{max(IAT_len_arr)},\ntotal normal: {len(IAT_len_arr)}'
            else:
                IAT_len_arr = other_IAT_len_arr
                title = f'srcIP:{srcIP},\nmin:{min(IAT_len_arr)}, max:{max(IAT_len_arr)},\ntotal other: {len(IAT_len_arr)}'
        else:
            IAT_len_arr = [len(iat) for i, iat in enumerate(data_inst.features) if
                           data_inst.labels[i] not in [None, 'None']]
            title = f'srcIP:{srcIP},\nmin:{min(IAT_len_arr)}, max:{max(IAT_len_arr)},\ntotal: {len(IAT_len_arr)}' + ',\nfft_bins: {fft_bins}, q: {quant}'
        print(f'{sorted(Counter(IAT_len_arr).items(), key=lambda item: item[0])}')
        quant = 0.9
        fft_bins = int(np.quantile(IAT_len_arr, q=quant))
        print(f'fft_bins: {fft_bins}, quantile = {quant}')
        if not individual_flg:
            title = title.format(fft_bins=fft_bins, quant=quant)
        stat_data(np.array(IAT_len_arr).reshape(-1, 1))
        if rescale_flg:
            IAT_len_arr = [value for value in IAT_len_arr if value < 25]
            bins = 30
        # plot_histgram(IAT_len_arr, title=f'srcIP:{srcIP}, quantile:{quant}, length_IAT:{thres}')
        plt.hist(IAT_len_arr, bins=bins)  # arguments are passed to np.histogram

        hist, bin_edges = np.histogram(IAT_len_arr, bins=bins)
        print(f'hist:{hist},\nbin_edges:{bin_edges}')
        max_idx = np.argmax(hist)
        if max_idx - 1 >= 0:
            max_range = f'[{int(bin_edges[max_idx - 1])}, {int(bin_edges[max_idx])}]'
        else:
            max_range = f'[0, {int(bin_edges[max_idx])}]'
        min_idx = np.argmin(hist)
        if min_idx - 1 >= 0:
            min_range = f'[{int(bin_edges[min_idx - 1])}, {int(bin_edges[min_idx])}]'
        else:
            min_range = f'[0, {int(bin_edges[min_idx])}]'

        # title = f'srcIP:{srcIP},\nmax:{max(hist)} in {max_range},' \
        #         f'\nmin:{min(hist)} in {min_range}'
        plt.title(f"{title}")
        plt.ylabel('Counts')
        # plt.xlabel('Number of packets in each flow')
        plt.xlabel('Dimension of new_features')
    plt.show()


if __name__ == '__main__':
    # main()
    plot_bar2(data='')
