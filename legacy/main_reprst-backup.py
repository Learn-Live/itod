"""Analyze the effect of different representations to different models
    data:
    models: GMM, OCSVM, PCA, KDE, IF, and AE

Note:

    function name: shorter and clearer name preferred, and keep the detail in the docstring
    function: simple is better than complexity (and avoiding doing everything)
"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE

# 0. add work directory into sys.path
import os.path as pth
import sys

# add root_path into sys.path in order you can access all the folders
# avoid using os.getcwd() because it won't work after you change into different folders
# NB: avoid using relative path
root_path = pth.dirname(pth.dirname(pth.dirname(pth.abspath(__file__))))
print(root_path)
sys.path.insert(0, root_path)
# add root_path/examples into sys.path
sys.path.insert(1, f"{root_path}/examples")
# print(sys.path)   # for debug

# 1. put _config as the first line
try:
    # import reprst._config_kjl
    from reprst._config_reprst import *
except Exception as e:
    print(f'load default config')
    from itod._config import *

# 2. standard libraries
import argparse
import time
from datetime import datetime

# 3. third-party packages
from numpy import genfromtxt
from sklearn import metrics
from sklearn.metrics import pairwise_distances, average_precision_score, roc_curve

# 4. your own package
from itod.ndm.gmm_cls import GMMDetector
# from itod_reprst.ndm.kjl_cls import standardize, kernelJLInitialize, getGaussianGram, search_params_cv
from itod.utils.utils import dump_data, data_info

# all feat sets for each data
FEAT_SETS = ['iat', 'size', 'iat_size', 'fft_iat', 'fft_size', 'fft_iat_size',
             'stat', 'samp_num', 'samp_size', 'samp_num_size', 'fft_samp_num',
             'fft_samp_size', 'fft_samp_num_size']


def extract_data(normal_pth, abnormal_pth, meta_data={}):
    """Get normal and abnormal data from csv
    # NORMAL(inliers): 0, ABNORMAL(outliers): 1
    Returns
    -------
        normal_data
        abnormal_data

    """
    NORMAL = 0  # Use 0 to label normal data
    ABNORMAL = 1  # Use 1 to label abnormal data

    # Normal and abnormal are the same size in the test set
    if meta_data['train_size'] <= 0:
        n_normal = -1
        n_abnormal = -1
    else:
        n_abnormal = int(meta_data['test_size'] // 2)
        n_normal = meta_data['train_size'] + n_abnormal
    start = meta_data['idxs_feat'][0]
    end = meta_data['idxs_feat'][1]

    def _label_n_combine_data(X, size=-1, data_type='normal'):
        if size == -1:
            size = X.shape[0]
        idx = np.random.randint(0, high=X.shape[0], size=size)
        X = X[idx, :]
        if data_type.upper() == 'normal'.upper():
            y = np.ones((X.shape[0], 1)) * NORMAL
        elif data_type.upper() == 'abnormal'.upper():
            y = np.ones((X.shape[0], 1)) * ABNORMAL
        else:
            # todo
            print(f"KeyError: {data_type}")
            raise KeyError(f'{data_type}')
        _data = np.hstack((X, y))
        nans = np.isnan(_data).any(axis=1)  # remove NaNs
        _data = _data[~nans]
        return _data

    # Get normal data
    try:
        if end == -1:
            X_normal = genfromtxt(normal_pth, delimiter=',', skip_header=1)[:, start:]  # skip_header=1
            X_abnormal = genfromtxt(abnormal_pth, delimiter=',', skip_header=1)[:, start:]
        else:
            X_normal = genfromtxt(normal_pth, delimiter=',', skip_header=1)[:, start:end]  # skip_header=1
            X_abnormal = genfromtxt(abnormal_pth, delimiter=',', skip_header=1)[:, start:end]
    except FileNotFoundError as e:
        print(f'FileNotFoundError: {e}')
        raise FileNotFoundError(e)

    normal_data = _label_n_combine_data(X_normal, size=n_normal, data_type='normal')
    abnormal_data = _label_n_combine_data(X_abnormal, size=n_abnormal, data_type='abnormal')

    # data={'X_train':'', 'y_train':'', 'X_test':'', 'y_test':''}
    # data = {'normal_data': normal_data, 'abnormal_data': abnormal_data,
    #         'label': {'NORMAL': NORMAL, 'ABNORMAL': ABNORMAL}}

    return normal_data, abnormal_data


def _split_train_test(normal_data, abnormal_data, show=True):
    """Split train and test set

    Parameters
    ----------
    normal_data
    abnormal_data
    show

    Returns
    -------

    """
    # split train/test
    n_abnormal = abnormal_data.shape[0]
    normal_test_idx = np.random.choice(normal_data.shape[0], size=n_abnormal, replace=False)
    normal_test_idx = np.in1d(range(normal_data.shape[0]), normal_test_idx)  # return boolean idxes

    train_normal, test_normal = normal_data[~normal_test_idx], normal_data[normal_test_idx]
    test_abnormal = abnormal_data

    X_train_normal = train_normal[:, :-1]
    X_train = X_train_normal
    y_train = train_normal[:, -1]

    X_test_normal = test_normal[:, :-1]
    y_test_normal = test_normal[:, -1].reshape(-1, 1)
    X_test_abnormal = test_abnormal[:, :-1]
    y_test_abnormal = test_abnormal[:, -1].reshape(-1, 1)
    X_test = np.vstack((X_test_normal, X_test_abnormal))
    # normal and abnormal have the same size in the test set
    y_test = np.vstack((y_test_normal, y_test_abnormal)).flatten()

    print("train.shape: {}, test.shape: {}".format(X_train.shape, X_test.shape))
    if show:
        data_info(X_train, name='X_train')
        data_info(X_test, name='X_test')

    return X_train, X_test, y_train, y_test


def preprocess_data(normal_data, abnormal_data, kjl=True, n=100, d=10, quant=0.25, model_name='PCA', show=True):
    """Preprocessing data, such as split, kjl, and standardization

    Parameters
    ----------
    normal_data
    abnormal_data
    kjl
    n
    d
    quant
    model_name
    show

    Returns
    -------

    """
    # n: The number used to obtain Gaussian sketch
    # d: The projected dimension

    sigma = None
    # In every time, train set and test set will be different
    X_train, X_test, y_train, y_test = _split_train_test(normal_data, abnormal_data)

    if kjl:
        # project data
        # X_train_std, vec_std = standardize_features(X_train)
        X_train_std, scaler_kjl = standardize(X_train)
        if show: data_info(X_train_std, name='X_train_std')
        n = n or max([200, int(np.floor(X_train_std.shape[0] / 100))])  # n_v: rows; m_v: cols. 200, 100?
        m = n
        sigma = sigma or np.quantile(pairwise_distances(X_train_std), quant)
        if sigma == 0:
            sigma = 1e-7
        print("Sigma: {}".format(sigma))
        # project train data
        X_train, U, Xrow = kernelJLInitialize(X_train_std, sigma, d, m, n, centering=1,
                                              independent_row_col=0)
        if show: data_info(X_train, name='after KJL, X_train')

        print("Projecting test data")
        # X_test_std, _ = standardize_features(X_test, vec_std)
        X_test_std = scaler_kjl.transform(X_test)
        if show: data_info(X_test_std, name='X_test_std')
        K = getGaussianGram(X_test_std, Xrow, sigma)
        X_test = np.matmul(K, U)
        if show: data_info(X_test, name='after KJL, X_test')

    if model_name not in ['GMM', 'KDE']:
        # Aprintorithms, which need to do standardization before fitting data. However, if these aprintorithms
        # use 'rbf', then they don't need. Some aprintorithms (such as GMM and KDE) don't need.
        # aprints_need_std = ['OCSVM', 'PCA'] when the kernel is not 'rbf'.
        # In simplicity, we do standardization regardless kernel type, here.

        # if kjl ==False, we do standardization. Also, after KJL, X_train needs to std again
        X_train, scaler = standardize(X_train)
        if show: data_info(X_train, name=f'after KJL({kjl}) and std, X_train')
        # keep the process of standardization consistent with that of the train set.
        X_test = scaler.transform(X_test)
        if show: data_info(X_test, name=f'after KJL({kjl}) and std, X_test')

    return X_train, X_test, y_train, y_test


def _train(model, X_train, y_train=None):
    """Train ndm on the (X_train, y_train)

    Parameters
    ----------
    model
    X_train
    y_train

    Returns
    -------

    """
    start = datetime.now()
    try:
        model.fit(X_train)
    except Exception as e:
        msg = f'fit error: {e}'
        print(msg)
        raise ValueError(f'{msg}: {model.get_params()}')
    end = datetime.now()
    training_time = (end - start).total_seconds()
    print("Training took {} seconds".format(training_time))

    return model, training_time


def _test(model, X_test, y_test):
    """Evaulate the ndm on the X_test, y_test

    Parameters
    ----------
    model
    X_test
    y_test

    Returns
    -------
       y_score, testing_time, auc, apc
    """
    start = datetime.now()
    # For inlier, a small value is used; a larger value is for outlier (positive)
    if model.__class__.__name__.upper() in ['GMM', 'OCSVM']:
        # The original score from scikit-learn:
        # 1) For GMM, larger value is for an inlier
        # 2) For OCSVM,  signed distance is positive for an inlier and negative for an outlier
        y_score = model.score_samples(X_test)
        y_score = -1 * y_score
    else:
        y_score = model.score_samples(X_test)

    """
    if detector_name == "Gaussian" and n_components != 1:
        preds = ndm.predict_proba(X_test)
        pred = 1 - np.prod(1-preds, axis=1)
    else:
        pred = ndm.score_samples(X_test)
    """
    end = datetime.now()
    testing_time = (end - start).total_seconds()
    print("Testing took {} seconds".format(testing_time))

    apc = average_precision_score(y_test, y_score)
    # For binary  y_true, y_score is supposed to be the score of the class with greater label.
    # auc = roc_auc_score(y_test, y_score)  # NORMAL(inliers): 0, ABNORMAL(outliers: positive): 1
    fpr, tpr, _ = roc_curve(y_test, y_score,
                            pos_label=1)  # pos_label = 1, so y_score should be the corresponding score
    auc = metrics.auc(fpr, tpr)
    # f1, bestEp = selectThreshHold(test_y_i, pred)

    # if auc > max_auc:
    #     max_auc = auc
    #     best_pred = y_score

    print("APC: {}".format(apc))
    print("AUC: {}".format(auc))
    # print("F1: {}".format(f1))

    return y_score, testing_time, auc, apc


def train_test_intf(model, normal_data, abnormal_data, gs=False, kjl=False):
    """Main structure of calling different models
    
    Parameters
    ----------
    model
    normal_data
    abnormal_data
    gs
    kjl

    Returns
    -------
        result: dict
            save all the data
    """
    n_repeats = 5
    d = 10
    if model == 'GMM':
        if not gs:
            train_times = []
            test_times = []
            aucs = []
            apcs = []
            for i in range(n_repeats):
                model = GMMDetector()
                X_train, X_test, y_train, y_test = preprocess_data(normal_data, abnormal_data, kjl,
                                                                   model_name=model.__class__.__name__)
                model, train_time = _train(model, X_train)
                y_score, test_time, auc, apc = _test(model, X_test, y_test)

                train_times.append(train_time)
                test_times.append(test_time)
                aucs.append(auc)
                apcs.append(apc)

            result = {'train_times': train_times, 'test_times': test_times, 'aucs': aucs, 'apcs': apcs,
                      'ndm': model, 'params': model.get_params(),
                      'X_train_shape': X_train.shape, 'X_test_shape': X_test.shape}
        else:
            # Set the parameters by cross-validation
            # return int
            model = GMMDetector()
            X_train, X_test, y_train, y_test = preprocess_data(normal_data, abnormal_data, kjl,
                                                               model_name=model.__class__.__name__)
            _n_components = np.linspace(1, min(d, X_train.shape[1]), num=10, endpoint=True, dtype=int)
            n_components = []
            [n_components.append(v) for v in _n_components if v not in n_components]  # keept the order
            tuned_parameters = {'covariance_type': ['diag', 'spherical', 'tied'],
                                'n_components': n_components}
            model = search_params_cv(model, tuned_parameters=tuned_parameters, k=None,
                                     X_train=X_train, X_test=X_test,
                                     y_train=y_train, y_test=y_test, search_type='grid_search')
            model, train_time = _train(model, X_train)
            y_score, test_time, auc, apc = _test(model, X_test, y_test)

            result = {'train_times': [train_time], 'test_times': [test_time], 'aucs': [auc], 'apcs': [apc],
                      'ndm': model, 'params': model.get_params(),
                      'X_train_shape': X_train.shape, 'X_test_shape': X_test.shape
                      }

    elif model == 'OCSVM':
        if not gs:
            pass
        else:
            pass
        raise NotImplementedError
    else:
        result = ''

    return result


def _get_line(result_each, feat_set):
    """Get each feat_set result and format it

    Parameters
    ----------
    result_each: dict
         result_each=(_best, '')
    feat_set

    Returns
    -------

    """

    try:
        value = result_each[feat_set][0]
        X_train_shape = str(value['X_train_shape']).replace(', ', '-')
        X_test_shape = str(value['X_test_shape']).replace(', ', '-')

        mu_auc = np.mean(value['aucs'])
        std_auc = np.std(value['aucs'])

        mu_train_time = np.mean(value['train_times'])
        std_train_time = np.std(value['train_times'])

        mu_test_time = np.mean(value['test_times'])
        std_test_time = np.std(value['test_times'])

        line = f',{feat_set}(auc: ' + f'{mu_auc:0.5f}' + '+/-' + f'{std_auc:0.5f}' + \
               ',train_time: ' + f'{mu_train_time:0.5f}' + '+/-' + f'{std_train_time:0.5f}' + \
               ',test_time: ' + f'{mu_test_time:0.5f}' + '+/-' + f'{std_test_time:0.5f})'
    except (Exception, KeyError, ValueError) as e:
        X_train_shape = '(0-0)'
        X_test_shape = '(0-0)'
        line = f',{feat_set}(-)'
        msg = f'{_get_line.__name__}, error:{e}'
        print(msg)

    prefix = f'X_train_shape: {X_train_shape}, X_test_shape: {X_test_shape}'

    return prefix, line


def save_result(results, file_out='header-ndm-gs-kjl.csv'):
    """Save all data results in one file with the current parameters (header, ndm, gs, kjl)

    Parameters
    ----------
    results
    file_out

    Returns
    -------

    """
    with open(file_out, 'w') as f:
        for _, (key, result_each) in enumerate(results.items()):
            key_pth, data_name = key
            for i, feat_set in enumerate(FEAT_SETS):
                # for i, (feat_set, value) in enumerate(result_each.items()):
                _prefix, _line = _get_line(result_each, feat_set)
                if i == 0:
                    _prefix = f'{key_pth},{data_name},' + _prefix
                    line = _prefix + _line
                else:
                    line += f',{feat_set}(-)'

            f.write(line + '\n')


def save_result_each(result_each, file_out='header-ndm-gs-kjl.csv'):
    """Save each data result in each file with the current parameters (header, ndm, gs, kjl)

    Parameters
    ----------
    result_each
    file_out

    Returns
    -------

    """
    with open(file_out, 'w') as f:
        for i, feat_set in enumerate(FEAT_SETS):  # keep the result in order by feature set
            # for i, (feat_set, value) in enumerate(result_each.items()):
            _prefix, _line = _get_line(result_each, feat_set)
            line = _prefix + _line
            f.write(line + '\n')


def main(header, model, quant=0.9, kjl=''):
    """Get normal.csv and abnormal.csv from all data sources (pcap) with the current parameters(header, ndm, gs, kjl)

    Parameters
    ----------
    header
    quant=0.9: use the 0.9 quantile of flow durations to fix each flow, such as cut off or append 0

    Returns
    -------
        0: succeed
        otherwise: failed.
    """
    print(f'header-{header}-{model}-gs:{gs}-kjl:{kjl}')
    # dir_in = f'data_{ndm}'
    dir_in = f'../examples/data/reprst'
    dir_out = f'../examples/reprst/out/reprst'
    datasets = [
        # department/dataname_year/device

        'UNB/CICIDS_2017/pc_192.168.10.5',
        'UNB/CICIDS_2017/pc_192.168.10.8',
        'UNB/CICIDS_2017/pc_192.168.10.9',
        'UNB/CICIDS_2017/pc_192.168.10.14',
        'UNB/CICIDS_2017/pc_192.168.10.15',

        'CTU/IOT_2017/pc_10.0.2.15',

        'MAWI/WIDE_2019/pc_202.171.168.50',

        'UCHI/IOT_2019/smtv_10.42.0.1',

        'UCHI/IOT_2019/ghome_192.168.143.20',
        'UCHI/IOT_2019/scam_192.168.143.42',
        'UCHI/IOT_2019/sfrig_192.168.143.43',
        'UCHI/IOT_2019/bstch_192.168.143.488'

    ]
    # FEAT_SETS = ['iat']  # 'samp_num', for dubeg
    results = {}
    for data_name in datasets:
        result_each = {}  # store results obtained from the current dataset
        for feat_set in FEAT_SETS:
            key_pth = pth.join(dir_in, data_name, feat_set, f'header:{str(header)}')
            meta_data = {'idxs_feat': [0, -1], 'train_size': -1, 'test_size': -1}
            try:
                if 'samp' in feat_set:
                    _result = {}
                    _best = {'aucs': [0]}
                    for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                        _pth_key = pth.join(key_pth, str(q))
                        normal_file = pth.join(_pth_key, 'normal.csv')
                        abnormal_file = pth.join(_pth_key, 'abnormal.csv')
                        normal_data, abnormal_data = extract_data(normal_file, abnormal_file, meta_data)

                        _result = train_test_intf(model, normal_data, abnormal_data, gs, kjl)
                        result[q] = _result
                        if np.mean(_result['aucs']) > np.mean(_best['aucs']):  # find the best results
                            _best = _result

                    result_each[feat_set] = (_best, result)
                else:
                    # 1. get data
                    normal_file = pth.join(key_pth, 'normal.csv')
                    abnormal_file = pth.join(key_pth, 'abnormal.csv')
                    normal_data, abnormal_data = extract_data(normal_file, abnormal_file, meta_data)

                    # 2. get results
                    result = train_test_intf(model, normal_data, abnormal_data, gs, kjl)
                    result_each[feat_set] = (result, '')
            except Exception as e:
                msg = f'{main.__name__}, error: {e}'
                print(msg)
                result_each[feat_set] = ('', '')

        # 3. store results
        # results generated by current parameters (header, ndm, gs, kjl) on all datasets will be stored at
        # the same file
        file_out = pth.join(dir_out, data_name, f'header:{str(header)}',
                            f'{model}-gs:{gs}-kjl:{kjl}.csv')
        if not pth.exists(pth.dirname(file_out)): os.makedirs(pth.dirname(file_out))
        save_result_each(result_each, file_out)

        results[(key_pth, data_name)] = result_each  # results[data_name]={'feat_set':data}
    file_out = pth.join(dir_out, f'{model}-gs:{gs}-kjl:{kjl}.csv')
    save_result(results, file_out)
    dump_data(results, file_out + '.dat')
    print(f'out_file: {file_out}')

    return 0


def parse_cmd():
    """Parse commandline parameters

    Returns:
        args: parsed commandline parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--header", help="header", default=True, type=bool)
    parser.add_argument("-d", "--detector", help="outlier detection ndm", default="GMM", type=str)  # required=True
    parser.add_argument("-g", "--gs", help="tuning parameters", default=False, type=bool)
    parser.add_argument("-k", "--kjl", help="kjl", default=False, type=bool)
    parser.add_argument("-t", "--time", help="start time of the application",
                        default=time.strftime(TIME_FORMAT, time.localtime()))
    args = parser.parse_args()
    print(f"args: {args}")

    return args


if __name__ == '__main__':
    args = parse_cmd()
    header = args.header
    detector = args.detector.upper()
    gs = args.gs
    kjl = args.kjl
    main(header, detector, gs, kjl)
