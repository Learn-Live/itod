""" Display

"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE
# 1. system and built-in libraries
import os
import traceback
from collections import Counter
from copy import deepcopy
import os.path as pth

# 2. thrid-part libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score

# 3. local libraries
from itod.utils.tool import transform_params_to_str, pprint, func_notation


class DisplayFactory:

    def __init__(self, dataset_inst='', params={}):
        self.dataset_inst = dataset_inst
        self.params = params
        self.params['dataset_inst'] = dataset_inst  # save all results:  # {sub_name: SingleDataset, ... }

    @func_notation
    def get_out_dir(self):

        key = 'samp_num_dict'
        self.samp_rate = self.dataset_dict[key]['samp_rate']
        self.q_samp_rate = self.dataset_dict[key]['q_samp_rate']
        self.subflow = self.params['subflow']
        if self.subflow:
            subflow_interval = self.dataset_inst.subflow_interval  # different subdataset has different interval
            self.subflow_comb = f'{self.subflow}_{subflow_interval:.4f}'
        else:
            self.subflow_comb = f'{self.subflow}'  # subflow

        self.out_dir = os.path.join(self.params['opt_dir'],
                                    os.path.dirname(self.dataset_dict['iat_dict']['feat_file']),
                                    'sf-' + self.subflow_comb,
                                    self.params['detector_name'] + '/'
                                    )

        return self.out_dir

    @func_notation
    def get_title_name(self):
        self.detector_name = self.params['detector_name']
        self.header = self.params['header']
        # X_train, y_train, X_test, y_test = self.dataset_dict['iat_dict']['data']
        res = self.dataset_dict['iat_dict']['result']
        y_train = res['y_train']
        y_val = res['y_val']
        try:
            y_test = res['y_test']
        except:
            y_test = res['y_true']
        self.dataset_name = self.params['dataset_name']  # 'DS10_UNB_IDS', 'DS20_PU_SMTV'
        # srcIP = self.params['srcIP']  # self.dataset_inst.srcIP
        self.data_cat = self.params['data_cat']  # 'INDV', 'AGMT'
        self.gs = self.params['gs']

        title_name = f'{self.detector_name}, gs:{self.gs}, {self.data_cat}, sf:{self.subflow_comb}, hd:{self.header}, ' \
                     f'dataset:{self.dataset_name}, train:({dict(Counter(y_train))}), val:({dict(Counter(y_val))}), test:({dict(Counter(y_test))})'

        return title_name

    @func_notation
    def run(self):
        self.dataset_dict = self.dataset_inst.dataset_dict
        pprint(self.dataset_dict, name=f'DisplayFactory.dataset_inst.dataset_dict')
        self.out_dir = self.get_out_dir()
        self.title_name = self.get_title_name()
        self.output_fig = self.out_dir + f'{self.detector_name}-gs_{self.gs}-hd_{self.header}-sf_{self.subflow_comb}' \
                                         f'-sampling-q_{self.q_samp_rate}-rate_{self.samp_rate}.pdf'

        # save results in the following order
        order_key = ['iat_dict', 'size_dict', 'iat_size_dict',
                     'fft_iat_dict', 'fft_size_dict', 'fft_iat_size_dict',
                     'stat_dict',
                     'samp_num_dict', 'samp_size_dict', 'samp_num_size_dict',
                     'fft_samp_num_dict', 'fft_samp_size_dict', 'fft_samp_num_size_dict']
        try:
            self._plot_roc_auc(self.dataset_dict, output_file=self.output_fig, order_key=order_key,
                               title=self.title_name)
        except (KeyError, Exception) as e:
            print(f'Error: {e}')
        output_txt = self.out_dir + f'{self.detector_name}-gs_{self.gs}-hd_{self.header}-sf_{self.subflow_comb}' \
                                    f'-sampling-q_{self.q_samp_rate}-rate_{self.samp_rate}.txt'
        print(f'output_file: {output_txt}')
        self.save_results(self.dataset_dict, output_file=output_txt, order_key=order_key)

    @func_notation
    def save_results(self, dataset_dict, order_key=['iat_dict', 'size_dict', 'iat_size_dict'],
                     output_file='results.txt'):
        data_lst = []
        file_header = f'detector, gs, subflow(interval-q_flow_durations), header, data_cat, dataset_name,' \
                      f'[dataset_name|subflow(interval),iat_auc(q_iat|dim),size_auc(q_iat|dim),iat_size_auc(q_iat|dim), ' \
                      f'fft_iat_auc(q_iat|dim),fft_size_auc(q_iat|dim), fft_iat_size_auc(q_iat|dim)' \
                      f'stat_auc(q_iat=None|dim),' \
                      f'samp_num_auc(best_q_sampling_rate|dim|u|std|aucs|vote), ' \
                      f'samp_size_auc(best_q_sampling_rate|dim|u|std|aucs|vote),' \
                      f'samp_num_size_auc(best_q_sampling_rate|dim|u|std|aucs|vote),' \
                      f'fft_samp_num_auc(best_q_sampling_rate|dim|u|std|aucs|vote), ' \
                      f'fft_samp_size_auc(best_q_sampling_rate|dim|u|std|aucs|vote),' \
                      f'fft_samp_num_size_auc(best_q_sampling_rate|dim|u|std|aucs|vote)]'

        self.params['file_header'] = file_header
        tmp_lst = [f'{self.dataset_name}|{self.detector_name}|gs({self.gs})|header({self.header})|' \
                   f'subflow({self.subflow_comb})']  # train set and test set

        # print(f'dataset_dict: {dataset_dict}')
        # for i, (result_key, result_dict) in enumerate(dataset_dict.items()):
        data_flg = True
        for i, result_key in enumerate(order_key):  # save results in order
            # key = iat_dict, value = dict
            try:
                if result_key not in dataset_dict.keys():
                    tmp_lst.append('-')
                    continue
                if data_flg:
                    # X_train, y_train, X_test, y_test = dataset_dict[result_key]['data']
                    res = dataset_dict[result_key]['result']
                    y_train = res['y_train']
                    y_val = res['y_val']
                    try:
                        y_test = res['y_test']
                    except:
                        y_test = res['y_true']

                    t = Counter(y_test)
                    set_str = '0:' + str(len(y_train)) + ',0:' + str(t[0]) + ' 1:' + str(t[1])
                    tmp_lst.append(set_str)
                    data_flg = False

                result_dict = dataset_dict[result_key]
                auc = result_dict['result']['auc']
                q_iat = result_dict['q_iat']
                dim = result_dict['feat_dim']
                if result_key in ['iat_dict', 'size_dict', 'iat_size_dict',
                                  'fft_iat_dict', 'fft_size_dict', 'fft_iat_size_dict',
                                  'stat_dict',
                                  ]:
                    auc = f'{auc:.4f}(q={q_iat}|dim={dim})'
                elif result_key in ['samp_num_dict', 'samp_size_dict', 'samp_num_size_dict',
                                    'fft_samp_num_dict', 'fft_samp_size_dict', 'fft_samp_num_size_dict']:
                    q_samp_rate_j = result_dict['q_samp_rate']
                    aucs = np.asarray(result_dict['auc_lst'], dtype=float)
                    print(f'{result_key}: aucs: {aucs}')
                    aucs_str = '+'.join([f'{v:.4f}' for v in aucs])
                    vote_auc_str = result_dict['voted_auc']
                    auc = f'{auc:.4f}(q_samp={q_samp_rate_j}|dim={dim}|u={aucs.mean():.4f}|std={aucs.std():.4f}' \
                          f'|aucs_str={aucs_str}|voted_auc={vote_auc_str:.4f})'
                else:
                    print(f'result_key {result_key} is not correct.')
                tmp_lst.append(auc)
            except Exception as e:
                print(f'Error: {e}')
                tmp_lst.append('-')
                # traceback.print_tb(e.__traceback__)

        data_lst.append(','.join(tmp_lst))

        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        with open(output_file, 'w') as out_hdl:
            out_hdl.write(file_header + '\n')
            for i, line in enumerate(data_lst):
                out_hdl.write(line + '\n')

        self.data_lst = data_lst
        return data_lst

    @func_notation
    def _insert_newlines(self, data_str, step=40):
        """ format the title with fixed length

        :param data_str:
        :param step:
        :return:
        """
        return '\n'.join([data_str[i:i + step] for i in range(0, len(data_str), step)])

    @func_notation
    def _plot_roc_auc(self, dataset_dict, order_key=[],
                      output_file='output_data/figures/roc_of_different_algorithms.pdf',
                      title='ROC'):
        """ plot roc and auc

        :param result_dict_lst:
        :param out_file:
        :param title:
        :return:
        """
        detector_name = self.params['detector_name']
        # with plt.style.context(('ggplot')):
        # fig, ax = plt.subplots()
        ax = plt.subplot()

        colors_lst = ['r', 'm', 'b', 'g', 'y', 'c', '#0072BD', '#A2142F', '#EDB120', 'k', '#D95319', '#4DBEEE',
                      '#0072EE', '#00EEBD', '#EE72BD',
                      'C1']  # add more color values: https://www.mathworks.com/help/matlab/ref/plot.html

        # for idx, (key, result_dict) in enumerate(dataset_dict.items()):
        for idx, result_key in enumerate(order_key):
            # colors_dict[feature_set_key] = {'AE': 'r', 'DT': 'm', 'PCA': 'C1', 'IF': 'b', 'OCSVM': 'g'}
            result_dict = dataset_dict[result_key]
            feature_set_key = result_dict['feat_set']
            try:
                value_dict = result_dict['result']

                y_true = value_dict['y_true']
                y_scores = value_dict['y_scores']
                fpr, tpr, thres = roc_curve(y_true=y_true, y_score=y_scores)
                # IMPORTANT: first argument is true values, second argument is predicted probabilities (i.e., y_scores)
                auc = value_dict['best_score_']
                print(f'auc: {metrics.auc(fpr, tpr)} == {auc} ?')  # for verify
                auc = f'{auc:.4f}'
                print(f'result of {detector_name}: {feature_set_key}, auc={auc}, fpr={fpr}, tpr={tpr}')
                # print(best_dict[feature_set_key]['best_score_'] == auc)
                params_str = ''

                if feature_set_key in ['iat', 'size', 'iat_size',
                                       'fft_iat', 'fft_size', 'fft_iat_size']:
                    q_iat = result_dict['q_iat']
                    feat_dim = result_dict['feat_dim']
                    params_str = f'q_iat:{q_iat}, feat_dim:{feat_dim}, '
                elif feature_set_key in ['stat']:
                    feat_dim = result_dict['feat_dim']
                    params_str = f'feat_dim:{feat_dim}, '
                elif feature_set_key in ['samp_num', 'samp_size', 'samp_num_size',
                                         'fft_samp_num_dict', 'fft_samp_size_dict', 'fft_samp_num_size_dict']:
                    q_samp_rate = result_dict['q_samp_rate']
                    q_samp_rate = f'{q_samp_rate:.2f}'
                    samp_rate = result_dict['samp_rate']
                    samp_rate = f'{samp_rate:.4f}'
                    feat_dim = result_dict['feat_dim']
                    params_str = f'feat_dim:{feat_dim}, q_samp:{q_samp_rate}, samp_rate:{samp_rate}, '
                else:
                    print(f'feature_set_key {feature_set_key} is not correct.')

                params_str += transform_params_to_str(deepcopy(value_dict['best_params_']),
                                                      remove_lst=['means_init', 'verbose', 'random_state',
                                                                  None, 'auto', 'quantile'],
                                                      param_limit=100)
            except Exception as e:
                params_str = f'-, '
                auc = 0
                fpr = 0
                tpr = 0

            if detector_name == 'PCA':
                lw = 3
            else:
                lw = 2

            params_str = self._insert_newlines(str(params_str), step=40)
            feat_str = self._insert_newlines(f'{idx + 1}:{feature_set_key}, AUC:{auc}')
            # ax.plot(fpr, tpr, colors_lst.pop(0), label=f'{feat_str},\n {params_str[0:min(len(params_str), 10)]}', lw=lw,
            #         alpha=0.9,
            #         linestyle='-')
            ax.plot(fpr, tpr, colors_lst.pop(0), label=f'{feat_str}', lw=lw,
                    alpha=0.9,
                    linestyle='-')
            # ax.text(2, 7, 'this is\nyet another test',
            #         rotation=45,
            #         horizontalalignment='center',
            #         verticalalignment='top',
            #         multialignment='center')

        ax.plot([0, 1], [0, 1], 'k--', label='', alpha=0.9)
        plt.xlim([0.0, 1.0])
        plt.ylim([0., 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right', framealpha=0.1, fontsize=10)

        if len(title) > 300:
            title = title[:300]
        ax.set_title(self._insert_newlines(title, step=50))

        # plt.title(self._insert_newlines(title, step=50))
        # plt.subplots_adjust(bottom=0.25, top=0.75)
        plt.tight_layout()
        print(f'ROC:{output_file}')
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_file)  # should use before plt.show()
        plt.show()

        return output_file

    @func_notation
    def _plot_difference_seaborn(self, dataset_dict, order_key=[],
                                 output_file='output_data/figures/roc_of_different_algorithms.pdf',
                                 title='ROC'):
        """ plot roc and auc

        :param result_dict_lst:
        :param out_file:
        :param title:
        :return:
        """
        detector_name = self.params['detector_name']
        # with plt.style.context(('ggplot')):
        # fig, ax = plt.subplots()
        ax = plt.subplot(111)

        colors_lst = ['r', 'm', 'b', 'g', 'y', 'c', '#0072BD', '#A2142F', '#EDB120', 'k', '#D95319', '#4DBEEE',
                      '#0072EE', '#00EEBD', '#EE72BD',
                      'C1']  # add more color values: https://www.mathworks.com/help/matlab/ref/plot.html

        # for idx, (key, result_dict) in enumerate(dataset_dict.items()):
        for idx, result_key in enumerate(order_key):
            # colors_dict[feature_set_key] = {'AE': 'r', 'DT': 'm', 'PCA': 'C1', 'IF': 'b', 'OCSVM': 'g'}
            result_dict = dataset_dict[result_key]
            feature_set_key = result_dict['feat_set']
            try:
                value_dict = result_dict['result']

                y_true = value_dict['y_true']
                y_scores = value_dict['y_scores']
                fpr, tpr, thres = roc_curve(y_true=y_true, y_score=y_scores)
                # IMPORTANT: first argument is true values, second argument is predicted probabilities (i.e., y_scores)
                auc = value_dict['best_score_']
                print(f'auc: {metrics.auc(fpr, tpr)} == {auc} ?')  # for verify
                auc = f'{auc:.4f}'
                print(f'result of {detector_name}: {feature_set_key}, auc={auc}, fpr={fpr}, tpr={tpr}')
                # print(best_dict[feature_set_key]['best_score_'] == auc)
                params_str = ''

                if feature_set_key in ['iat', 'size', 'iat_size',
                                       'fft_iat', 'fft_size', 'fft_iat_size']:
                    q_iat = result_dict['q_iat']
                    feat_dim = result_dict['feat_dim']
                    params_str = f'q_iat:{q_iat}, feat_dim:{feat_dim}, '
                elif feature_set_key in ['stat']:
                    feat_dim = result_dict['feat_dim']
                    params_str = f'feat_dim:{feat_dim}, '
                elif feature_set_key in ['samp_num', 'samp_size', 'samp_num_size',
                                         'fft_samp_num_dict', 'fft_samp_size_dict', 'fft_samp_num_size_dict']:
                    q_samp_rate = result_dict['q_samp_rate']
                    q_samp_rate = f'{q_samp_rate:.2f}'
                    samp_rate = result_dict['samp_rate']
                    samp_rate = f'{samp_rate:.4f}'
                    feat_dim = result_dict['feat_dim']
                    params_str = f'feat_dim:{feat_dim}, q_samp:{q_samp_rate}, samp_rate:{samp_rate}, '
                else:
                    print(f'feature_set_key {feature_set_key} is not correct.')

                params_str += transform_params_to_str(deepcopy(value_dict['best_params_']),
                                                      remove_lst=['means_init', 'verbose', 'random_state',
                                                                  None, 'auto', 'quantile'],
                                                      param_limit=100)
            except Exception as e:
                params_str = f'-, '
                auc = 0
                fpr = 0
                tpr = 0

            if detector_name == 'PCA':
                lw = 3
            else:
                lw = 2

            params_str = self._insert_newlines(str(params_str), step=40)
            feat_str = self._insert_newlines(f'{idx + 1}:{feature_set_key}, AUC:{auc}')
            # ax.plot(fpr, tpr, colors_lst.pop(0), label=f'{feat_str},\n {params_str[0:min(len(params_str), 10)]}', lw=lw,
            #         alpha=0.9,
            #         linestyle='-')
            ax.plot(fpr, tpr, colors_lst.pop(0), label=f'{feat_str}', lw=lw,
                    alpha=0.9,
                    linestyle='-')
            # ax.text(2, 7, 'this is\nyet another test',
            #         rotation=45,
            #         horizontalalignment='center',
            #         verticalalignment='top',
            #         multialignment='center')

        ax.plot([0, 1], [0, 1], 'k--', label='', alpha=0.9)
        plt.xlim([0.0, 1.0])
        plt.ylim([0., 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right', framealpha=0.1)

        if len(title) > 300:
            title = title[:300]
        ax.set_title(self._insert_newlines(title, step=50))

        # plt.title(self._insert_newlines(title, step=50))
        # plt.subplots_adjust(bottom=0.25, top=0.75)
        print(f'ROC:{output_file}')
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_file)  # should use before plt.show()
        plt.show()

        return output_file

    @func_notation
    def _plot_data(self, x, y):
        # beta = 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.999:
        # beta = 0.1, num_clusters = 3962,
        # beta = 0.2, num_clusters = 3961,
        # beta = 0.4, num_clusters = 3952,
        # beta = 0.6, num_clusters = 3943,
        # beta = 0.8, num_clusters = 3939,
        # beta = 0.9, num_clusters = 3935,
        # beta = 0.999, num_clusters = 3911,

        # with plt.style.context(('ggplot')):
        fig, ax = plt.subplots()
        ax.plot(x, y, '*-',
                alpha=0.9)

        # ax.plot([0, 1], [0, 1], 'k--', label='', alpha=0.9)
        plt.xlim([0.0, 1.0])
        # plt.ylim([0., 1.05])
        plt.xlabel('Beta')
        plt.ylabel('Num_clusters')
        plt.legend(loc='lower right')
        plt.title('QuickShift_PP: k and beta')
        plt.show()
