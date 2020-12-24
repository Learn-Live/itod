"""Dataset process including flow2features, normalization, split_train_test

"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE

#############################################################################################
# 1. system and built-in libraries

# 2. thrid-part libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.utils import shuffle

# 3. local libraries
from itod.pparser.pcap import _flows_to_iats_sizes, _flows_to_stats, _flows_to_samps, _get_header_from_flows
from itod.utils.tool import *


class Dataset:
    """Flow process includes flow2features, normalization, split_train_test, and so on.

    """

    def __init__(self, srcIP='', single_data_name='', params={}, **kwargs):
        """

        Parameters
        ----------
        srcIP
        single_data_name
        params
        kwargs
        """
        self.kwargs = kwargs
        if len(kwargs) > 0:
            ## Implict set value to variable
            for i, (key, value) in enumerate(kwargs.items()):
                setattr(self, key, value)
        self.single_data_name = single_data_name
        self.srcIP = srcIP
        self.params = params
        self.num_pkt_thresh = 2
        self.params['num_pkt_thresh'] = self.num_pkt_thresh

    @func_notation
    def load_data(self):
        self.fids, self.features, self.labels = self.load_pickled_data(input_file=self.input_file)

    @func_notation
    def load_pickled_data(self, input_file):
        """ Especially for loading multi-objects stored in a file.

        Parameters
        ----------
        input_file

        Returns
        -------

        """

        fids = []
        features = []
        labels = []
        with open(input_file, 'rb') as in_hdl:
            while True:
                try:
                    fids_, features_, labels_ = pickle.load(in_hdl)
                    fids.extend(fids_)
                    features.extend(features_)
                    labels.extend(labels_)
                except EOFError:
                    break

        return fids, features, labels

    def generate_flows(self, flows, num=200, header=False):
        """

        Parameters
        ----------
        flows
        num
        header

        Returns
        -------

        """
        while len(flows) < num:
            new_flow = [0] * len(flows[0])
            k = 5
            for i in range(k):
                j = np.random.randint(0, len(flows))
                # new_flow += flows[j]
                new_flow = [v1 + v2 for v1, v2 in zip(new_flow, flows[j])]

            new_flow = [v / k for v in new_flow]

            for i, v in enumerate(new_flow):
                if header and i < 8:  # 8 TCP flags accumlated from packets of each flow
                    new_flow[i] = np.round(v)
                else:
                    break
            # flows.append([v/k for v in new_flow])   # average
            flows.append(new_flow)

        return flows

    @func_notation
    def split_train_test_flows(self, flows='', labels='', shuffle_flg=True, random_state=42):
        """ split train and test flows

        Parameters
        ----------
        flows
        labels
        shuffle_flg
        random_state

        Returns
        -------

        """
        X_normal = []
        y_normal = []
        X_anomaly = []
        y_anomaly = []
        none_num = 0

        for i, value in enumerate(labels):
            feat = list(flows[i])
            if value.upper() in ['NORMAL', 'BENIGN']:
                X_normal.append(feat)
                y_normal.append(0)
            elif value.upper() in ['BOT', 'ANOMALY', 'MALICIOUS']:
                X_anomaly.append(feat)
                y_anomaly.append(1)
            else:  # label in [None, 'None', ... ]
                print(f'i:{i}, {value}')
                none_num += 1
        print(f'{none_num} flows appear in pcap, however, don\'t appear in labels.csv')
        print(f'Normal(0):{len(y_normal)}, Anomaly(1):{len(y_anomaly)}')

        if shuffle_flg:  # random select a part of normal flows from normal flows
            c = list(zip(X_normal, y_normal))
            X_normal, y_normal = list(zip(*shuffle(c, random_state=random_state)))
            X_normal = list(X_normal)
            y_normal = list(y_normal)

        if len(y_anomaly) == 0:
            print(f'no anomaly data in this data_process')
            train_len = int(len(y_normal) // 2)  # use 100 normal flows as test set.
            X_train = X_normal[:train_len]
            y_train = y_normal[:train_len]
            X_test = X_normal[:train_len]
            y_test = y_normal[:train_len]
        else:
            if len(y_normal) <= len(y_anomaly):
                train_len = int(len(y_normal) // 2)
            else:
                train_len = len(y_normal) - len(y_anomaly)
            X_train = X_normal[:train_len]
            y_train = y_normal[:train_len]
            X_test = X_normal[train_len:] + X_anomaly
            y_test = y_normal[train_len:] + y_anomaly

        return X_train, y_train, X_test, y_test

    # @func_notation
    # def split_train_test_20190328(self, features='', labels='', shuffle_flg=True, random_state=42, header=False):
    #
    #     X_normal = []
    #     y_normal = []
    #     X_anomaly = []
    #     y_anomaly = []
    #     none_num = 0
    #
    #     for i, value in enumerate(labels):
    #         feat = list(features[i])
    #         if value.upper() in ['NORMAL', 'BENIGN']:
    #             X_normal.append(feat)
    #             y_normal.append(0)
    #         elif value.upper() in ['BOT', 'ANOMALY', 'MALICIOUS']:
    #             X_anomaly.append(feat)
    #             y_anomaly.append(1)
    #         else:  # label in [None, 'None', ... ]
    #             print(f'i:{i}, {value}')
    #             none_num += 1
    #     print(f'{none_num} flows appear in pcap, however, don\'t appear in labels.csv')
    #     print(f'Normal(0):{len(y_normal)}, Anomaly(1):{len(y_anomaly)}')
    #
    #     # num_anomaly = 200
    #     # if len(y_anomaly) < num_anomaly:
    #     #     X_anomaly= self.generate_flows(X_anomaly, num=num_anomaly, header=header)
    #     #     y_anomaly = [1]* num_anomaly
    #
    #     # num_normal = 10000 + num_anomaly
    #     # if len(y_normal) < (num_normal):
    #     #     X_normal = self.generate_flows(X_normal, num=num_normal, header=header)
    #     #     y_normal = [0]*num_normal
    #
    #     if shuffle_flg:  # random select a part of normal flows from normal flows
    #         c = list(zip(X_normal, y_normal))
    #         X_normal, y_normal = zip(*shuffle(c, random_state=random_state))
    #
    #     X_normal = np.array(X_normal, dtype=float)
    #     y_normal = np.array(y_normal, dtype=int)
    #     X_anomaly = np.array(X_anomaly, dtype=float)
    #     y_anomaly = np.array(y_anomaly, dtype=int)
    #
    #     if len(y_anomaly) == 0:
    #         print(f'no anomaly data in this data_process')
    #         train_len = int(len(y_normal) // 2)  # use 100 normal flows as test set.
    #         X_train = X_normal[:train_len, :]
    #         y_train = y_normal[:train_len]
    #         X_test = X_normal[:train_len, :]
    #         y_test = y_normal[:train_len]
    #     else:
    #         if len(y_normal) <= len(y_anomaly):
    #             train_len = int(len(y_normal) // 2)
    #         else:
    #             train_len = len(y_normal) - len(y_anomaly)
    #         X_train = X_normal[:train_len, :]
    #         y_train = y_normal[:train_len]
    #         X_test = np.concatenate([X_normal[train_len:, :], X_anomaly], axis=0)
    #         y_test = np.concatenate([y_normal[train_len:], y_anomaly], axis=0)
    #
    #     print(f'--- train set and test set --'
    #           f'\nX_train:{X_train.shape}, y_train:{Counter(y_train)}'
    #           f'\nX_test :{X_test.shape}, y_test :{Counter(y_test)}, in which, 0: normals and 1: anomalies.')
    #     # stat_data(X_train, name='X_train')
    #     # stat_data(X_test, name='X_test')
    #
    #     return X_train, y_train, X_test, y_test

    @func_notation
    def normalise_data(self, X_train, X_test, norm_method='std'):
        """ Normalise data or not

        Parameters
        ----------
        X_train
        X_test
        norm_method

        Returns
        -------

        """
        print(f'2) normalization ({norm_method})')
        if norm_method in ['min-max', 'std', 'robust']:
            if norm_method == 'min-max':  # (x-min)/(max-min)
                self.train_scaler = MinMaxScaler()
            elif norm_method == 'std':  # (x-u)/sigma
                self.train_scaler = StandardScaler()
            elif norm_method == 'robust':  # (x-Q2)/(Q3-Q1)
                self.train_scaler = RobustScaler()
            self.train_scaler.fit(X_train)
            X_train = self.train_scaler.transform(X_train)
            X_test = self.train_scaler.transform(X_test)

            print('after normalization: X_train distribution:')
            stat_data(X_train, name='X_train')
            print('after normalization:X_test distribution:')
            stat_data(X_test, name='X_test')

            return X_train, X_test

        elif norm_method == 'log':  # log scaler
            # X_train = np.log(X_train)         # np.log(v): v > 0
            min_trainvalues = np.min(X_train, axis=0)
            X_train = np.log(X_train - min_trainvalues + 1)  # X_train-in_values + 1: scale all values to [1, . ]
            # and np.log(values) will be [0, .]

            X_test = np.clip(X_test - min_trainvalues, a_min=0, a_max=10e+6)  # all values less than 0 are clipped to 0
            X_test = np.log(X_test + 1)

            print('after normalization: X_train distribution:')
            stat_data(X_train, name='X_train')
            print('after normalization:X_test distribution:')
            stat_data(X_test, name='X_test')

            return X_train, X_test

        elif norm_method == 'none' or norm_method == None:
            print('without normalization')
            return X_train, X_test

        else:
            print(f'norm_method: {norm_method} is not implemented yet, please check and retry')
            return -1

    @func_notation
    @execute_time
    def run(self):
        """

        Returns
        -------

        """
        # {"iat_dict":{}, "stat_dict":{}, ...}, and each item is "feat_set_dict:{}"
        self.dataset_dict = OrderedDict()

        # 1. pcap2flows: require normal and anomaly file, subflows
        self.get_flows_labels()

        # 2. flows2features.
        self.flows2features(flows=tuple(self.flows), labels=tuple(self.labels))  # update self.dataset_dict

        # print(self.dataset_dict.keys())
        # release memory
        if hasattr(self, 'flows'):
            # obj.attr_name exists.
            del self.flows
        if hasattr(self, 'labels'):
            del self.labels

        return self.dataset_dict

    @func_notation
    @execute_time
    def flows2features(self, flows, labels):
        """Flows to features

        Parameters
        ----------
        flows
        labels

        Returns
        -------

        """

        pcap_file = self.pcap_file  # dir_name for iat_file, stat_file, samp_file, ...
        q_iat = self.params['q_iat']  # use for fixing the feature dimension
        header = self.params['header']  # True or False

        # get subflow parameters to make up of output_data file
        sf = self.params['subflow']
        if sf:  # True
            sf_intv = self.params['subflow_interval']
            sf_dir = f'sf_{sf}_{sf_intv:.4f}'
        else:
            sf_dir = f'sf_{sf}'
        # split flows to train and test flows.
        x_train_flows, y_train, x_test_flows, y_test = self.split_train_test_flows(flows, labels=labels)
        if header:  # get header features
            train_header_fids_features = _get_header_from_flows(x_train_flows)
            train_header_fids = list(map(lambda x: x[0], train_header_fids_features))
            train_header_features = list(map(lambda x: x[1], train_header_fids_features))
            feat_dim_header = int(np.quantile([len(v) for v in train_header_features], q=q_iat))
            print(f'feat_dim_header: {feat_dim_header} only on train set (i.e., normal flows)')
            train_header_features = _fix_data(train_header_features, feat_dim=feat_dim_header)

            # test header
            test_header_fids_features = _get_header_from_flows(x_test_flows)
            test_header_fids = list(map(lambda x: x[0], test_header_fids_features))
            test_header_features = list(map(lambda x: x[1], test_header_fids_features))
            test_header_features = _fix_data(test_header_features, feat_dim=feat_dim_header)

        # get 'iat', 'size', 'iat_size' features and the corresponding fft features
        for i, feat_set in enumerate(['iat', 'size', 'iat_size']):
            # for i, feat_set in enumerate(['iat']):
            print(f'feat_set: {feat_set}')
            # 1) get raw features
            # assert (flows_cp == flows), f"flows_cp !=flows, {feat_set}"
            train_fids_features = _flows_to_iats_sizes(x_train_flows, feat_set=feat_set)  # header features
            train_fids = list(map(lambda x: x[0], train_fids_features))
            train_features = list(map(lambda x: x[1], train_fids_features))
            feat_dim = int(np.quantile([len(v) for v in train_features], q=q_iat))
            # feat_dim = max([len(v) for v in train_features])
            print(f'feat_dim: {feat_dim} only on train set (i.e., nomral flows)')
            print(f'num. of flows: {len(train_fids)}')
            # # store features to file, and the features have different dimensions
            feat_file = f'{pcap_file}-{sf_dir}-{feat_set}-{header}-q_iat={q_iat}.dat'

            test_fids_features = _flows_to_iats_sizes(x_test_flows, feat_set=feat_set)  # header features
            test_fids = list(map(lambda x: x[0], test_fids_features))
            test_features = list(map(lambda x: x[1], test_fids_features))
            print(f'num. of flows: {len(test_fids)}')

            self.dataset_dict[f'{feat_set}_dict'] = {'feat_set': feat_set, 'feat_file': feat_file,
                                                     'q_iat': q_iat, 'feat_dim': feat_dim,
                                                     # 'fixed_feat_file': fixed_feat_file,
                                                     'data': ''}

            if header:
                for i, feat in enumerate(train_features):
                    train_features[i] = list(train_header_features[i]) + list(feat)

                for i, feat in enumerate(test_features):
                    test_features[i] = list(test_header_features[i]) + list(feat)

                feat_dim = feat_dim_header + feat_dim
                self.dataset_dict[f'{feat_set}_dict']['feat_dim'] = feat_dim

            # # 2) fix features
            # fixed_feat_file = feat_file + f'-dim_{feat_dim}.dat'

            x_train = _fix_data(train_features, feat_dim)
            x_test = _fix_data(test_features, feat_dim)

            self.dataset_dict[f'{feat_set}_dict']['data'] = (x_train, y_train, x_test, y_test)
            file_name = 'header_' + str(self.params['header']) + '-gs_' + str(self.params['gs']) + f'-{feat_set}.dat'
            # dump_data((x_train, y_train, x_test, y_test), os.path.join(self.params['ipt_dir'], file_name))

            # 3) get fft features
            fft_feat_set = f"fft_{feat_set}"
            fft_part = 'real'
            fft_key = f"{fft_feat_set}_dict"
            self.dataset_dict[fft_key] = {'feat_set': fft_feat_set, 'fixed_feat_file': '', 'data': '',
                                          'fft_part': fft_part,
                                          'feat_dim': feat_dim, 'q_iat': q_iat}
            if fft_part == 'real+imaginary':
                fixed_feat_file = feat_file + f'-{fft_feat_set}-q_iat={q_iat}-dim_{feat_dim * 2}-fft_part_{fft_part}.dat'
            else:
                fixed_feat_file = feat_file + f'-{fft_feat_set}-q_iat={q_iat}-dim_{feat_dim}-fft_part_{fft_part}.dat'

            x_train = _get_fft_data(train_features, fft_bin=feat_dim, fft_part=fft_part, feat_set=fft_feat_set)
            x_test = _get_fft_data(test_features, fft_bin=feat_dim, fft_part=fft_part, feat_set=fft_feat_set)

            self.dataset_dict[fft_key]['data'] = (x_train, y_train, x_test, y_test)
            file_name = 'header_' + str(self.params['header']) + '-gs_' + str(
                self.params['gs']) + f'-{fft_feat_set}.dat'
            # dump_data((x_train, y_train, x_test, y_test), os.path.join(self.params['ipt_dir'], file_name))

        # get "stat" features
        for i, feat_set in enumerate(['stat']):
            print(f'feat_set: {feat_set}')
            # 1) get raw features
            fids_features = _flows_to_stats(x_train_flows)
            train_fids = list(map(lambda x: x[0], fids_features))
            train_features = list(map(lambda x: x[1], fids_features))
            # q_iat will affect the length of header, so if header=True, please consider about q_iat, too.
            # However, len(features[0]) is fixed after _flows_to_stats(flows, header=header, q_iat=q_iat)
            feat_dim = len(train_features[0])
            print(f'feat_set: {feat_set}, dimension: {feat_dim}')

            fids_features = _flows_to_stats(x_test_flows)
            test_fids = list(map(lambda x: x[0], fids_features))
            test_features = list(map(lambda x: x[1], fids_features))

            # feat_file = f'{pcap_file}-{sf_dir}-{feat_set}-{header}-q_iat={q_iat}-dim={feat_dim}.dat'
            # dump_data(data=(fids, features, labels), output_file=feat_file)
            self.dataset_dict[f'{feat_set}_dict'] = {'feat_set': feat_set, 'feat_file': feat_file,
                                                     'q_iat': q_iat, 'feat_dim': feat_dim}

            # 2) fix features
            # x_train, y_train, x_test, y_test = self.split_train_test(features=features, labels=labels,header=header)
            if header:
                for i, feat in enumerate(train_features):
                    train_features[i] = list(train_header_features[i]) + list(feat)

                for i, feat in enumerate(test_features):
                    test_features[i] = list(test_header_features[i]) + list(feat)

                feat_dim = feat_dim_header + feat_dim
                self.dataset_dict[f'{feat_set}_dict']['feat_dim'] = feat_dim

            x_train = np.asarray(train_features, dtype=float)
            x_test = np.asarray(test_features, dtype=float)

            self.dataset_dict[f'{feat_set}_dict']['data'] = (x_train, y_train, x_test, y_test)

            # 3) get fft features
            # pass

        # get 'samp_num', 'samp_size', 'samp_num_size' features and the corresponding fft features
        for i, feat_set in enumerate(['samp_num', 'samp_size', 'samp_num_size']):
            print(f'feat_set: {feat_set}')
            samp_method = self.params['sampling']  # self.sampling = 'rate', sampling method
            if feat_set == 'samp_num':
                # 'samp_size' and 'samp_num_size' has the same samp_rate as 'samp_num'
                feat_dim = self.dataset_dict['iat_dict']['feat_dim']
                # flow_duration/fft_bin
                flow_durs_arr = [(max(times) - min(times)) / feat_dim for fid, times, pkts in x_train_flows]
            elif feat_set == 'samp_size':
                # 'samp_size' and 'samp_num_size' has the same samp_rate as 'samp_num', i.e., the same flow_durs_arr
                feat_dim = self.dataset_dict['samp_num_dict']['feat_dim']
            elif feat_set == 'samp_num_size':
                # note: the dimension of samp_num_size doesn't equal that of IAT+SIZE
                # 'samp_size' and 'samp_num_size' has the same samp_rate as 'samp_num', i.e.,the same flow_durs_arr
                if header:
                    # features_header = []
                    # for (fid, times, pkts) in x_train_flows:
                    #     features_header.append((fid, get_header_features(pkts)))
                    # feat_dim_header = int(np.quantile([len(head) for (fid, head) in features_header], q=q_iat))
                    pass
                else:
                    feat_dim_header = 0
                feat_dim = self.dataset_dict['samp_num_dict']['feat_dim'] * 2 - (feat_dim_header)
            else:
                raise ValueError(f'{feat_set} is not correct.')

            samp_dict = OrderedDict()
            fft_samp_dict = OrderedDict()
            for j, q_samp_rate in enumerate(list(np.linspace(0.0, 1, 10, endpoint=False))[1:] + [0.95]):
                # get 10 q_samp_rates: list(np.linspace(0.0, 1, 10, endpoint=False))[1:] exclude 0

                # 1) get raw features
                q_samp_rate = float(f'{q_samp_rate:.2f}')
                samp_rate = np.quantile(flow_durs_arr, q=q_samp_rate)
                # The sampling rate is the 0.9 percentile of flow length/IAT size. Then fix all vectors to be that
                # length by padding and cutting.
                if samp_rate <= 0.0:
                    print(f'***samp_method: {samp_method}, q_samp_rate: {q_samp_rate}, samp_rate: {samp_rate}, '
                          f'weird, skipping this samp_rate')
                    continue
                print(f'samp_method: {samp_method}, q_samp_rate: {q_samp_rate}, samp_rate: {samp_rate}')
                fids_features = _flows_to_samps(x_train_flows, sampling_type=samp_method,
                                                sampling=samp_rate,
                                                sampling_feature=feat_set)
                train_fids = list(map(lambda x: x[0], fids_features))
                train_features = list(map(lambda x: x[1], fids_features))

                fids_features = _flows_to_samps(x_test_flows, sampling_type=samp_method,
                                                sampling=samp_rate,
                                                sampling_feature=feat_set)
                test_fids = list(map(lambda x: x[0], fids_features))
                test_features = list(map(lambda x: x[1], fids_features))

                print(f'feat_feat: {feat_set}, with different lengths. samp_method: {samp_method}, '
                      f'samp_rate: {samp_rate}')
                # # store sampled IATs which have different dimensions to file
                feat_file = f'{pcap_file}-{sf_dir}-{feat_set}-{header}-q_iat={q_iat}-{samp_method}-{samp_rate}-' \
                            f'q_samp_rate={q_samp_rate}.dat'
                # dump_data(data=(fids, features, labels), output_file=feat_file)

                samp_dict[q_samp_rate] = {'feat_set': feat_set, 'feat_file': feat_file,
                                          'samp_method': samp_method, 'samp_rate': samp_rate,
                                          'q_samp_rate': q_samp_rate,
                                          'q_iat': q_iat, 'feat_dim': feat_dim}

                # 2) fix features
                # fixed_feat_file = feat_file + f'-dim_{feat_dim}.dat'
                if header:
                    for i, feat in enumerate(train_features):
                        train_features[i] = list(train_header_features[i]) + list(feat)

                    for i, feat in enumerate(test_features):
                        test_features[i] = list(test_header_features[i]) + list(feat)

                    # feat_dim = self.dataset_dict['iat_dict']['feat_dim'] or ...

                x_train = _fix_data(train_features, feat_dim)
                x_test = _fix_data(test_features, feat_dim)

                samp_dict[q_samp_rate]['data'] = (x_train, y_train, x_test, y_test)

                # 3) get fft features
                fft_feat_set = f"fft_{feat_set}"
                fft_part = 'real'
                fft_samp_dict[q_samp_rate] = {'feat_set': fft_feat_set, 'feat_file': feat_file,
                                              'fixed_feat_file': '', 'data': '', 'fft_part': fft_part,
                                              'samp_rate': samp_rate, 'q_samp_rate': q_samp_rate,
                                              'feat_dim': feat_dim, 'q_iat': q_iat}

                if fft_part == 'real+imaginary':
                    fixed_feat_file = feat_file + f'-{fft_feat_set}-q_iat_{q_iat}-dim_{feat_dim * 2}-fft_part_{fft_part}.dat'
                else:
                    fixed_feat_file = feat_file + f'-{fft_feat_set}-q_iat_{q_iat}-dim_{feat_dim}-fft_part_{fft_part}.dat'

                x_train = _get_fft_data(train_features, fft_bin=feat_dim, fft_part=fft_part, feat_set=fft_feat_set)
                x_test = _get_fft_data(test_features, fft_bin=feat_dim, fft_part=fft_part, feat_set=fft_feat_set)
                fft_samp_dict[q_samp_rate]['data'] = (x_train, y_train, x_test, y_test)

            self.dataset_dict[f'{feat_set}_dict'] = {'feat_set': feat_set, f'{feat_set}_dict': samp_dict,
                                                     'q_iat': q_iat, 'feat_dim': feat_dim}
            self.dataset_dict[f'{fft_feat_set}_dict'] = {'feat_set': fft_feat_set,
                                                         f'{fft_feat_set}_dict': fft_samp_dict,
                                                         'q_iat': q_iat, 'feat_dim': feat_dim}
        return self.dataset_dict


@func_notation
def _fix_data(features, feat_dim):
    """ Fix data by appending '0' or cutting off

    Parameters
    ----------
    features

    feat_dim: int
        the fixed dimension of features

    Returns
    -------
    fixed_features:
        the fixed features
    """
    fixed_features = []
    for feat in features:
        feat = list(feat)
        if len(feat) > feat_dim:
            feat = feat[:feat_dim]
        else:
            feat += [0] * (feat_dim - len(feat))

        fixed_features.append(np.asarray(feat, dtype=float))

    return np.asarray(fixed_features, dtype=float)


@func_notation
def _get_fft_data(features, fft_bin='', fft_part='real', feat_set='fft_iat'):
    """ Do fft transform of features

    Parameters
    ----------
    features: features

    fft_bin: int
        the dimension of transformed features
    fft_part: str
        'real' or 'real+imaginary' transformation

    feat_set: str

    Returns
    -------
    fft_features:
        transformed fft features
    """
    if fft_part == 'real':  # default
        fft_features = [np.real(np.fft.fft(v, n=fft_bin)) for v in features]

    elif fft_part == 'real+imaginary':
        fft_features = []
        for i, v in enumerate(features):
            complex_v = np.fft.fft(v, fft_bin)
            if i == 0:
                print(f'dimension of the real part: {len(np.real(complex_v))}, '
                      f'dimension of the imaginary part: {len(np.imag(complex_v))}')
            v = np.concatenate([np.real(complex_v), np.imag(complex_v)], axis=np.newaxis)
            fft_features.append(v)

    else:
        print(f'fft_part: {fft_part} is not correct, please modify it and retry')
        return -1

    return np.asarray(fft_features, dtype=float)


@execute_time
def load_flows_label_pickle(output_flows_labels):
    with open(output_flows_labels, 'rb') as in_hdl:
        flows, labels, subflow_interval = pickle.load(in_hdl)

    return flows, labels, subflow_interval
