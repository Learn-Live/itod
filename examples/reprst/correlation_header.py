"""
     for both MAWI, SFrig and UNB(PC1) ?
- Let Y in {0, 1} denote normal or novelty, and X1, .., X9 denote the packet header information.
- Compute the correlation(Xi, Y) for each of the Xi’s (do this on the test data).

"""
import os, sys

lib_path = os.path.abspath('../')
sys.path.append(lib_path)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

sns.set()

from itod_kjl.model.train_test import extract_data, preprocess_data
from itod_kjl.utils.utils import dump_data

# 0.4 get root logger
from itod_kjl import log

lg = log.get_logger(name=None, level='DEBUG', out_dir='./log/data_kjl', app_name='correlation')


def _get_each_correlation(x, y):
    rho = np.corrcoef(x, y)[0, 1]
    rho = 0 if np.isnan(rho) else rho
    return rho


def get_correlation(in_dir='data/data_reprst/csvs', out_dir='out/data_reprst', header=True):
    # datasets = ['DS10_UNB_IDS/DS11-srcIP_192.168.10.5', # UNB(PC1)
    #             'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50', # MAWI
    #             'DS60_UChi_IoT/DS63-srcIP_192.168.143.43'   # SFrig
    #             ]

    datasets = [  # 'DEMO_IDS/DS-srcIP_192.168.10.5',
        'DS10_UNB_IDS/DS11-srcIP_192.168.10.5',  # data_name is unique
        # 'DS10_UNB_IDS/DS12-srcIP_192.168.10.8',
        # 'DS10_UNB_IDS/DS13-srcIP_192.168.10.9',
        # 'DS10_UNB_IDS/DS14-srcIP_192.168.10.14',
        # 'DS10_UNB_IDS/DS15-srcIP_192.168.10.15',

        # 'DS20_PU_SMTV/DS21-srcIP_10.42.0.1',
        #
        # # 'DS30_OCS_IoT/DS31-srcIP_192.168.0.13',
        #
        'DS40_CTU_IoT/DS41-srcIP_10.0.2.15',
        #
        'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50',

        # 'DS60_UChi_IoT/DS61-srcIP_192.168.143.20',
        # 'DS60_UChi_IoT/DS62-srcIP_192.168.143.42',
        'DS60_UChi_IoT/DS63-srcIP_192.168.143.43',
        # 'DS60_UChi_IoT/DS64-srcIP_192.168.143.48'
    ]

    feat_sets = ['iat_size']  # all the features has the same header

    corr_results = {}
    for i, dataset in enumerate(datasets):
        for feat_set in feat_sets:
            key_pth = os.path.join(in_dir, dataset, feat_set, f"header:{header}")
            lg.info(f'i: {i}, key_path: {key_pth}')
            # 1. get data
            normal_file = os.path.join(key_pth, 'normal.csv')
            abnormal_file = os.path.join(key_pth, 'abnormal.csv')
            lg.info(f'normal_file: {normal_file}')
            lg.info(f'abnormal_file: {abnormal_file}')
            meta_data = {'idxs_feat': [0, -1], 'train_size': -1, 'test_size': -1}
            normal_data, abnormal_data = extract_data(normal_file, abnormal_file, meta_data)
            X_train, X_test, y_train, y_test, kjl_train_set_time, kjl_test_set_time = preprocess_data(normal_data,
                                                                                                      abnormal_data,
                                                                                                      kjl=False, d=0,
                                                                                                      n=0, quant=0,
                                                                                                      model_name=None,
                                                                                                      random_state=42)

            # 2 get correlation
            corrs = []
            for j in range(9):  # the first 9 columns: 8 tcp flags + 1 TTL
                _corr = _get_each_correlation(X_test[:, j], y_test)
                corrs.append(_corr)
            corr_results[(key_pth, dataset, feat_set, X_test.shape)] = corrs

        out_file = os.path.join(out_dir, dataset, feat_set, f"header:{header}", 'correlation.dat')
        lg.info(f'i: {i}, {dataset}, out_file: {out_file}')
        if not os.path.exists(os.path.dirname(out_file)): os.makedirs(os.path.dirname(out_file))
        dump_data((key_pth, corrs), out_file)

    # save all results
    out_file = os.path.splitext(out_file)[0] + '_all.dat'
    lg.info(f'out_file: {out_file}')
    dump_data(corr_results, out_file)

    return corr_results


def plot_correlation_multi(corr_results, out_dir, title=None, show=True):
    # only show the top 4 figures
    datasets = [
        ('DS10_UNB_IDS/DS11-srcIP_192.168.10.5', 'UNB(PC1)'),  # data_name is unique
        # ('DS10_UNB_IDS/DS14-srcIP_192.168.10.14', 'UNB(PC4)'),
        ('DS40_CTU_IoT/DS41-srcIP_10.0.2.15', 'CTU'),
        ('DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50', 'MAWI'),
        ('DS60_UChi_IoT/DS63-srcIP_192.168.143.43', 'SFrig'),
    ]
    new_corr_results = {}
    for i, (dataset, name) in enumerate(datasets):
        for j, (key, corrs) in enumerate(corr_results.items()):
            _key_path, _dataset, _feat_set, X_test_shape = key
            if dataset in key:
                new_corr_results[(_key_path, _dataset, name, _feat_set, X_test_shape)] = corrs
    t = 0
    cols = 2
    fontsize = 20
    ## http://jose-coto.com/styling-with-seaborn
    # colors = ["m", "#4374B3"]
    # palette = sns.color_palette('RdPu', 1)  # a list
    palette = [sns.color_palette('YlOrRd', 7)[4]]  # YlOrRd
    fig, axes = plt.subplots(2, cols, figsize=(18, 8))  # (width, height)
    lg.debug(new_corr_results)
    for i, (key, corrs) in enumerate(new_corr_results.items()):
        lg.info(f"i: {i}, {key}, corrs: {corrs}")  # hue = feat_set
        key_path, dataset, short_name, feat_set, X_test_shape = key
        # data = [[f'X{_i+1}_y', feat_set, corrs[_i]] if corrs[_i]!=np.nan else 0 for _i in range(9)]
        # data = [[f'X{_i + 1}_y', feat_set, corrs[_i]] for _i in range(9)]
        HEADER = ['FIN', 'SYN', 'RST', 'PSH', 'ACK', 'URG', 'ECE', 'CWR', '1st-TTL']

        data = sorted(range(len(corrs)), key=lambda i: abs(corrs[i]), reverse=True)[:6]  # top 6 values
        # data = [[f'({HEADER[_i]}, y)', feat_set, corrs[_i]] for _i in sorted(data, reverse=False)]
        data = [[f'({HEADER[_i]}, y)', feat_set, corrs[_i]] for _i in data]
        lg.info(f"i: {i}, {key}, corrs: {data}")

        new_yerrs = [1 / (np.sqrt(X_test_shape[0]))] * 6  # for err_bar
        lg.debug(f'i: {i}, {new_yerrs}')
        df = pd.DataFrame(data, columns=[f'Xi_y', 'feat_set', 'corr_rho'])
        if i % cols == 0 and i > 0:
            t += 1
        g = sns.barplot(x=f"Xi_y", y="corr_rho", ax=axes[t, i % cols], hue='feat_set', data=df,
                        palette=palette)  # palette=palette,
        g.set(xlabel=None)
        g.set(ylim=(-1, 1))
        if i % cols == 0:
            # g.set_ylabel(r'$\rho$', fontsize=fontsize + 4)
            g.set_ylabel(r'Correlation', fontsize=fontsize + 4)
            print(g.get_yticks())
            g.set_yticks([-1, -0.5, 0, 0.5, 1])
            g.set_yticklabels(g.get_yticks(), fontsize=fontsize + 6)  # set the number of each value in y axis
            print(g.get_yticks())
        else:
            g.set(ylabel=None)
            g.set_yticklabels(['' for v_tmp in g.get_yticks()])
            g.set_ylabel('')

        # g.set_title(dataset_name)
        g.get_legend().set_visible(False)
        g.set_xticklabels(g.get_xticklabels(), fontsize=fontsize + 4, rotation=30, ha="center")

        ys = []
        xs = []
        width = 0
        for i_p, p in enumerate(g.patches):
            height = p.get_height()
            width = p.get_width()
            ys.append(height)
            xs.append(p.get_x())
            if i_p == 0:
                pre = p.get_x() + p.get_width()
            if i_p > 0:
                cur = p.get_x()
                g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, ymin=0, ymax=1, alpha=0.3)
                pre = cur + p.get_width()
            ## https://stackoverflow.com/questions/34888058/changing-width-of-bars-in-bar-chart-created-using-seaborn-factorplot
            p.set_width(width / 3)  # set the bar width
            # we recenter the bar
            p.set_x(p.get_x() + width / 3)
        g.set_title(short_name, fontsize=fontsize + 8)

        # add error bars
        g.errorbar(x=xs + width / 2, y=ys,
                   yerr=new_yerrs, fmt='none', c='b', capsize=3)

    # # get the legend and modify it
    # handles, labels = g.get_legend_handles_labels()
    # fig.legend(handles, ['IAT+SIZE'], title=None, loc='lower center', ncol=1,
    #  prop={'size': fontsize-2})  # loc='lower right',  loc = (0.74, 0.13)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    # title = dataset
    # # fig.suptitle("\n".join(["a big long suptitle that runs into the title"]*2), y=0.98)
    # fig.suptitle(title)
    out_file = os.path.join(out_dir, feat_set, "header:True", 'correlation-bar.pdf')
    if not os.path.exists(os.path.dirname(out_file)): os.makedirs(os.path.dirname(out_file))
    lg.debug(out_file)
    plt.savefig(out_file)  # should use before plt.show()
    if show: plt.show()
    plt.close(fig)
    plt.close("all")
    # sns.reset_orig()
    # sns.reset_defaults()


def plot_correlation(corr_results, out_dir, title=None, show=True):
    for i, (key, corrs) in enumerate(corr_results.items()):
        lg.info(f"i: {i}, {key}")  # hue = feat_set
        key_path, dataset, feat_set, X_test_shape = key
        # data = [[f'X{_i+1}_y', feat_set, corrs[_i]] if corrs[_i]!=np.nan else 0 for _i in range(9)]
        # data = [[f'X{_i + 1}_y', feat_set, corrs[_i]] for _i in range(9)]
        HEADER = ['FIN', 'SYN', 'RST', 'PSH', 'ACK', 'URG', 'ECE', 'CWR', '1st-TTL']
        data = [[f'({HEADER[_i]}, y)', feat_set, corrs[_i]] for _i in range(9)]

        fig = plt.figure(figsize=(10, 5))  # (width, height)

        df = pd.DataFrame(data, columns=[f'Xi_y', 'feat_set', 'corr_rho'])
        g = sns.barplot(x=f"Xi_y", y="corr_rho", hue='feat_set', data=df)  # palette=palette,
        g.set(xlabel=None)
        g.set(ylabel='Rho')
        g.set_ylim(-1, 1)
        # g.set_title(dataset_name)
        g.get_legend().set_visible(True)  # False
        g.set_xticklabels(g.get_xticklabels(), fontsize=12, rotation=30, ha="center")

        ys = []
        xs = []
        width = 0
        for i_p, p in enumerate(g.patches):
            height = p.get_height()
            ys.append(height)
            xs.append(p.get_x())
            if i_p == 0:
                pre = p.get_x() + p.get_width()
            if i_p > 0:
                cur = p.get_x()
                g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, ymin=0, ymax=1, alpha=0.3)
                pre = cur + p.get_width()
        plt.legend(loc='upper right')
        # # get the legend and modify it
        # handles, labels = g.get_legend_handles_labels()
        # fig.legend(handles, labels, title=None, loc='upper right', ncol=1,
        #  prop={'size': 8})  # loc='lower right',  loc = (0.74, 0.13)

        plt.tight_layout()

        title = dataset
        g.set_title(title + ' (header:True)', fontsize=13)
        out_file = os.path.join(out_dir, dataset, feat_set, "header:True", '-corr-bar.png')
        lg.debug(out_file)
        plt.savefig(out_file)  # should use before plt.show()
        if show: plt.show()
        plt.close(fig)
        plt.close("all")


def main():
    corr_results = get_correlation(in_dir='data/data_reprst/csvs', out_dir='out/data_reprst', header=True)

    plot_correlation(corr_results, out_dir='out/data_reprst', show=True)
    plot_correlation_multi(corr_results, out_dir='out/data_reprst', show=True)


if __name__ == '__main__':
    main()
