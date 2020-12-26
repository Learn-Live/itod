"""Interfaces of all outlier detector algorithms

"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE

# from keras import backend as K
# import tensorflow as tf
import gc
# 1. system and built-in libraries
import os
import sys
from abc import abstractmethod
from collections import Counter

import numpy as np
import torch
# 3. local libraries
# from QuickshiftPP import QuickshiftPP   # it's the right way to use pyx file after building and installing
import sklearn
from QuickshiftPP import QuickshiftPP
from scipy.spatial import distance
from sklearn import metrics
from sklearn.model_selection import train_test_split
from datetime import datetime

from itod.pparser.dataset import Dataset
from itod.ndm.ae_cls import AEDetector
from itod.ndm.gmm_cls import GMMDetector
from itod.ndm.if_cls import IForestDetector
from itod.ndm.kde_cls import KDEDetector
from itod.ndm.model_selection.grid_search import GridSearch
from itod.ndm.ocsvm_cls import OCSVMDetector
from itod.ndm.pca_cls import PCADetector
from itod.utils.tool import stat_data, func_notation, execute_time


# 2. thrid-part libraries


class DetectorFactory:
    """Main entrance for outlier detection, i.e., interface

    """

    def __init__(self, detector_name='', dataset_inst='', params={}):
        self.detector_name = detector_name
        self.dataset_inst = dataset_inst
        self.params = params

        self.params['detector_name'] = detector_name
        self.params['dataset_inst'] = dataset_inst

    @func_notation
    def run(self):
        """Execute function

        Returns
        -------

        """
        if self.detector_name.upper() == 'GMM'.upper():
            self.detector = GMMFactory(dataset_inst=self.dataset_inst, params=self.params)
        elif self.detector_name.upper() == 'OCSVM'.upper():
            self.detector = OCSVMFactory(dataset_inst=self.dataset_inst, params=self.params)
        elif self.detector_name.upper() == 'KDE'.upper():
            self.detector = KDEFactory(dataset_inst=self.dataset_inst, params=self.params)
        elif self.detector_name.upper() == 'AE'.upper():
            self.detector = AEFactory(dataset_inst=self.dataset_inst, params=self.params)
        elif self.detector_name.upper() == 'IF'.upper():
            self.detector = IFFactory(dataset_inst=self.dataset_inst, params=self.params)
        elif self.detector_name.upper() == 'PCA'.upper():
            self.detector = PCAFactory(dataset_inst=self.dataset_inst, params=self.params)
        # elif self.detector_name.upper() == 'MO_GAAL'.upper():
        #     self.detector = MO_GAALFactory(dataset_inst=self.dataset_inst, params=self.params)
        else:
            msg = f"{self.detector_name} is not implemented."
            raise ValueError(msg)

        # update dataset_dict
        self.detector.run()

        return self.dataset_inst  # contains all results


@func_notation
@execute_time
# def obtain_means_init_quickshift_pp(X_train, k=''):
#     """Initialize GMM
#         1) Download quickshift++ from github
#         2) unzip and move the folder to your project
#         3) python3 setup.py build
#         4) python3 setup.py install
#         5) from QuickshiftPP import QuickshiftPP
#     :param X_train:
#     :param k:
#     :return:
#     """
#
#     print(f"k: {k}")
#     # Declare a Quickshift++ model with tuning hyperparameters.
#     if k == '':
#         k = int(np.sqrt(X_train.shape[0]))
#
#     # k: number of neighbors in k-NN
#     # beta: fluctuation parameter which ranges between 0 and 1.
#     beta = 0.9
#     model = QuickshiftPP(k=k, beta=beta)
#     try:
#         model.fit(X_train)
#     except:
#         print('quickshift failed.')
#     print('quickshift fit finished')
#     labels_ = model.memberships
#     cluster_centers_ = []
#     for i in range(np.max(labels_) + 1):
#         ind_i = np.where(labels_ == i)[0]  # get index of each cluster
#         cluster_i_center = np.mean(X_train[ind_i], axis=0)  # get center of each cluster
#         cluster_centers_.append(cluster_i_center)
#
#     means_init = np.asarray(cluster_centers_, dtype=float)
#     print(f'quickshift: number of clusters:{len(Counter(labels_))}, {Counter(labels_)}')
#
#     return means_init, len(set(labels_))

@func_notation
@execute_time
def obtain_means_init_quickshift_pp(X, k=None, beta=0.9, thres_n=20):
    """Initialize GMM
            1) Download quickshift++ from github
            2) unzip and move the folder to your project
            3) python3 setup.py build
            4) python3 setup.py install
            5) from QuickshiftPP import QuickshiftPP
        :param X_train:
        :param k:
            # k: number of neighbors in k-NN
            # beta: fluctuation parameter which ranges between 0 and 1.

        :return:
        """
    start = datetime.now()
    if k <= 0 or k > X.shape[0]:
        print(f'k {k} is not correct, so change it to X.shape[0]')
        k = X.shape[0]
    print(f"number of neighbors in k-NN: {k}")
    # Declare a Quickshift++ model with tuning hyperparameters.
    model = QuickshiftPP(k=k, beta=beta)

    # Note the try catch cannot capture the model.fit() error because it is cython. How to capture the exception?
    ret_code = 1
    try:
        ret_code = model.fit(X)
    except Exception as e:
        msg = f'quickshift++ fit error: {e}'
        raise ValueError(msg)

    if ret_code < 0:
        print(f'ret_code ({ret_code}) < 0, fit fail')
        raise ValueError('ret_code < 0, fit fail')

    end = datetime.now()
    quick_training_time = (end - start).total_seconds()
    # lg.info("quick_training_time took {} seconds".format(quick_training_time))

    start = datetime.now()
    # print('quickshift fit finished')
    all_labels_ = model.memberships
    all_n_clusters = len(set(all_labels_))
    cluster_centers = []
    for i in range(all_n_clusters):
        idxs = np.where(all_labels_ == i)[0]  # get index of each cluster. np.where return tuple
        if len(idxs) < thres_n:  # only keep the cluster which has more than "thres_n" datapoints
            continue
        center_cluster_i = np.mean(X[idxs], axis=0)  # here is X, not X_std.
        cluster_centers.append(center_cluster_i)

    means_init = np.asarray(cluster_centers, dtype=float)
    n_clusters = means_init.shape[0]
    end = datetime.now()
    ignore_clusters_time = (end - start).total_seconds()
    print(f'*** quick_training_time: {quick_training_time}, ignore_clusters_time: {ignore_clusters_time}')
    # print(f'--all clusters ({all_n_clusters}) when (k:({k}), beta:{beta}). However, only {n_clusters} '
    #       f'clusters have at least {thres_n} datapoints. Counter(labels_): {Counter(all_labels_)}, *** '
    #       f'len(Counter(labels_)): {all_n_clusters}')

    return means_init, n_clusters


def get_fpr_tpr(y_true, y_pred_labels):
    """Obtain FPR and TPR used for ROC and AUC
    only for binary labels

    Parameters
    ----------
    y_true
    y_pred_labels

    Returns
    -------
    fpr: list or arrary
    tpr: list or arrary
    """

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_labels).ravel()
    fpr = fp / (tn + fp)  # fp/Negatives
    tpr = tp / (tp + fn)  # recall  tp/Positives

    return fpr, tpr


def change_scores_to_quantiles_voted_auc(y_testing_scores_arrs, y_training_scores_arrs, y_true, num_quantile=20):
    """Obtain the voted AUC
    1. change scores to 0s or 1s labels
    2. get voted 0s and 1s labels
    3. get fpr and tpr from the labels for each q
    4. get the auc from all fprs and tprs

    Parameters
    ----------
    y_testing_scores_arrs
    y_training_scores_arrs
    y_true
    num_quantile

    Returns
    -------
    vote_auc: float
    """
    # mapping scores to 0-1 values
    # quantiles
    ps = list(np.linspace(0, 1, num=num_quantile, endpoint=True))  # proportion
    fpr_arr = []
    tpr_arr = []
    for i, p in enumerate(ps):
        majority_arrs = [[]] * y_testing_scores_arrs.shape[0]
        for j in range(y_testing_scores_arrs.shape[0]):
            # process the y_testing_scores generated by the i-th model
            arr = list(y_testing_scores_arrs[j, :])  # list can update its values, numpy array cannot do it.
            # the 0.01 quantile is the lowest score value ’s’ such that 99% (i.e. 1-0.01) of the training points
            # have score lower than s.
            # get the quantiles from trianing scores, not testing scores.
            quant = np.quantile(y_training_scores_arrs[j, :], q=1 - p)
            # mapping scores to 0-1
            majority_arrs[j] = [0] * len(arr)
            for t, v in enumerate(arr):
                if v >= quant:
                    majority_arrs[j][t] = 1
                else:
                    majority_arrs[j][t] = 0
        # get the majority of 0-1 predictions for one model for each sample
        majority_arrs = np.asarray(majority_arrs)
        final_labels = [0] * majority_arrs.shape[1]
        for t in range(majority_arrs.shape[1]):
            final_labels[t] = find_majority(majority_arrs[:, t])
        # print('y_true', list(y_true))
        # print('fina_', list(final_labels))
        fpr, tpr = get_fpr_tpr(list(y_true), list(final_labels))
        fpr_arr.append(fpr)
        tpr_arr.append(tpr)
    # print('fpr', len(fpr_arr), fpr_arr)
    # print('tpr', len(tpr_arr), tpr_arr)
    # sorted by the first list in ascending order
    fpr_arr, tpr_arr = zip(*sorted(zip(fpr_arr, tpr_arr), reverse=False))
    # please check the api to see the details: metrics.auc(fpr, tpr)
    auc = metrics.auc(fpr_arr, tpr_arr)

    return auc


def find_majority(arr):
    return Counter(arr).most_common(n=None)[0][0]


# def majority_quantiles(new_votes, inplace=False):
#     # avoid modify new_votes in the function
#     # new_votes = np.asarray(deepcopy(votes))
#     major_votes = []
#     for j in range(new_votes.shape[-1]):
#         major_votes.append(find_majority(new_votes[:, j]))
#
#     return major_votes


class BaseDetectorFactory():
    """Outlier detection interface

    """

    def __init__(self, dataset_inst='', params={}):
        self.dataset_inst = dataset_inst
        self.params = params
        self.params['dataset_inst'] = dataset_inst

    @func_notation
    def run(self):
        """Execute function
        """
        for i, (key, value_dict) in enumerate(self.dataset_inst.dataset_dict.items()):
            print(f'i:{i}, key:{key}, value_dict.keys():{value_dict.keys()}')
            if key in ['samp_num_dict', 'samp_size_dict', 'samp_num_size_dict',
                       'fft_samp_num_dict', 'fft_samp_size_dict', 'fft_samp_num_size_dict']:
                samp_best_auc = 0
                samp_set_dict = {}
                result_best_dict = {}
                auc_lst = []
                y_scores_lst = []  # testing scores
                y_training_scores_lst = []  # training scores
                for j, (q_samp_j, samp_dict_j) in enumerate(value_dict[key].items()):
                    try:
                        # value_dict={'samp_dict':{'feat_set':, 'data':,}}, 'fft_dict':{}, ...}
                        result_tmp = self.run_model(samp_dict_j)
                        y_scores_lst.append(result_tmp['y_scores'])
                        y_training_scores_lst.append(result_tmp['y_training_scores'])
                        # result_tmp={detector_name: {best_score:, best_params:,...}}
                    except (ValueError, Exception, KeyError, IndexError, AttributeError, MemoryError, NameError) as e:
                        print(f'{j}, q_samp_j: {q_samp_j}, {e}, fit failed.')
                        self.dataset_inst.dataset_dict[key]['result'] = {'auc': -1}
                        continue
                    except:
                        print(f'{j}, q_samp_j: {q_samp_j}, fit failed.')
                        self.dataset_inst.dataset_dict[key]['result'] = {'auc': -1}
                        continue
                    auc_j = result_tmp['auc']
                    auc_lst.append(auc_j)
                    print(f'q_samp_{j}:{q_samp_j}, auc_{j}: {auc_j}')
                    # feat_set = samp_dict_j['feat_set']
                    # only save the best auc on samp_set
                    if auc_j >= samp_best_auc:
                        samp_best_auc = auc_j
                        samp_set_dict = samp_dict_j
                        result_best_dict = result_tmp
                # print(f'key:{key}, auc_lst: {auc_lst}')
                self.dataset_inst.dataset_dict[key] = samp_set_dict
                self.dataset_inst.dataset_dict[key]['auc_lst'] = auc_lst
                self.dataset_inst.dataset_dict[key]['y_scores_lst'] = y_scores_lst
                self.dataset_inst.dataset_dict[key]['y_training_scores_lst'] = y_training_scores_lst
                self.dataset_inst.dataset_dict[key]['result'] = result_best_dict
                # print(key, 'y_scores_lst', self.dataset_inst.dataset_dict[key]['y_scores_lst'])

                # get vote_auc
                voted_auc = change_scores_to_quantiles_voted_auc(np.asarray(y_scores_lst, dtype=float),
                                                                 np.asarray(y_training_scores_lst, dtype=float),
                                                                 y_true=self.dataset_inst.dataset_dict[key]['result'][
                                                                     'y_true'],
                                                                 num_quantile=100)
                # print(f'voted_auc: {voted_auc}')
                self.dataset_inst.dataset_dict[key]['voted_auc'] = voted_auc

                print('size(self.dataset_inst):', sys.getsizeof(self.dataset_inst),
                      sys.getsizeof(self.dataset_inst.dataset_dict[key]))

            elif key in ['iat_dict', 'size_dict', 'iat_size_dict',
                         'fft_iat_dict', 'fft_size_dict', 'fft_iat_size_dict',
                         'stat_dict']:
                # value_dict={'iat_dict':{'feat_set':, 'data':,}}, 'fft_dict':{}, ...}
                try:
                    result_tmp = self.run_model(value_dict)
                    # result_tmp={detector_name: {best_score:, best_params:,...}}
                    self.dataset_inst.dataset_dict[key]['result'] = result_tmp
                except (ValueError, Exception, KeyError, IndexError, AttributeError, MemoryError, NameError) as e:
                    print(f'{key}, {e}.')
                    self.dataset_inst.dataset_dict[key]['result'] = {'auc': -1}
                    continue
                except:
                    print(f'{key}, fit failed.')
                    self.dataset_inst.dataset_dict[key]['result'] = {'auc': -1}
                    continue
                print('size(self.dataset_inst):', sys.getsizeof(self.dataset_inst),
                      sys.getsizeof(self.dataset_inst.dataset_dict[key]))
            else:
                msg = f'{key} is not correct, {os.path.relpath(self.run.__code__.co_filename)} ' \
                      f'at line {self.run.__code__.co_firstlineno}\''
                # raise ValueError(msg)
                print(msg)
                continue

    @abstractmethod
    def run_model(self, feat_dict={}):  # feat_dict={'feat_set':, 'feat_file':, 'data':(), ...}
        pass


class GMMFactory(BaseDetectorFactory):

    def __init__(self, dataset_inst='', params={}):
        super(GMMFactory, self).__init__()
        self.dataset_inst = dataset_inst
        self.params = params

        self.params['dataset_inst'] = dataset_inst

    @func_notation
    def run_model(self, feat_dict={}):  # feat_dict={'feat_set':, 'feat_file':, 'data':(), ...}
        gs = self.params['gs']
        X_train, y_train, X_test, y_test = feat_dict['data']

        size = 5000
        if len(y_train) > size:
            X_train, y_train = sklearn.utils.resample(X_train, y_train, n_samples=size, random_state=42, replace=False)

        detector_name = self.params['detector_name']
        assert self.params['detector_name'] == 'GMM'
        # result_dict = {detector_name: {}}
        sd_inst = Dataset()
        X_train, X_test = sd_inst.normalise_data(X_train, X_test,
                                                 norm_method=self.params['norm_method'])
        # select a part of val_data from test_set.
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=int(len(y_test) * 0.2),
                                                        stratify=y_test, random_state=42)
        if gs:
            print(f'with grid search: {detector_name.upper()}, gs: {gs}')
            # find the best parameters of the detector
            detector = GMMDetector(random_state=self.params['random_state'], verbose=self.params['verbose'])
            grid = GridSearch(estimator=detector, scoring='auc',
                              param_grid={
                                  'n_components': [1, 2, 5, 10, 15, 20, 25, 30, 35, 40],
                                  # [int(v) for v in list(np.linspace(2, 30, num=10, endpoint=True))],
                                  # [i + 1 for i in range(50)]
                                  'covariance_type': ['diag']})

            grid.fit(X_train, y_train, X_val, y_val)
            self.detector = grid.best_estimator_
            print(f'{detector_name}.params: {self.detector.get_params()}')

            self.detector.test(X_test, y_test)
            print(f'auc: {self.detector.auc}')

            self.y_training_scores = self.detector.gmm.predict_proba(X_train)[:, 1]
            result_dict = {'y_true': y_test, 'y_scores': self.detector.y_scores,
                           'y_pred': self.detector.y_pred,
                           'y_val': y_val, 'y_train': y_train,
                           'y_training_scores': self.y_training_scores,
                           'auc': self.detector.auc,
                           'best_score_': grid.best_score_, 'best_params_': grid.best_params_,
                           'best_estimator_': grid.best_estimator_
                           }

        else:  # gs: False
            # rule of thumb in practice
            print('without grid search')
            # distances = distance.pdist(norm_X_train, metric='euclidean')
            # stat_data(distances.reshape(-1, 1), name='distances')
            # q = 0.3   # cannot work for meanshift, so we use q=0.7
            # sigma = np.quantile(distances, q=q)
            # print(f'sigma: {sigma}, q (for setting bandwidth in MeanShift): {q}')
            #
            try:
                # # means_init, n_components = obtain_means_init(norm_X_train, bandwidth=sigma)
                means_init, n_components = obtain_means_init_quickshift_pp(X_train,
                                                                           k=int(np.sqrt(len(y_train))))
            except:
                raise ValueError('obtain_means_init_quickshift error.')
            if self.params['norm_method'] == 'std':
                means_init = means_init * sd_inst.train_scaler.scale_ + sd_inst.train_scaler.mean_
            self.detector = GMMDetector(n_components=n_components, means_init=means_init,
                                        covariance_type='diag',
                                        random_state=self.params['random_state'], verbose=self.params['verbose'])

            print(f'{detector_name}.params: {self.detector.get_params()}')
            try:
                self.detector.fit(X_train, y_train)
            except Exception as e:
                print(f'{e}')
                raise ValueError('gmm fit fails')
            self.detector.test(X_test, y_test)

            self.y_training_scores = self.detector.gmm.predict_proba(X_train)[:, 1]
            result_dict = {'y_true': y_test, 'y_scores': self.detector.y_scores,
                           'y_pred': self.detector.y_pred,
                           'y_val': y_val, 'y_train': y_train,
                           'y_training_scores': self.y_training_scores,
                           'auc': self.detector.auc,
                           'best_score_': self.detector.auc,
                           'best_params_': self.detector.get_params(),
                           'best_estimator_': self.detector
                           }

        return result_dict


class OCSVMFactory(BaseDetectorFactory):

    def __init__(self, kernel='rbf', dataset_inst='', params={}):
        super(OCSVMFactory, self).__init__()
        self.dataset_inst = dataset_inst
        self.params = params

        self.params['dataset_inst'] = dataset_inst

    @func_notation
    def run_model(self, feat_dict):
        gs = self.params['gs']
        X_train, y_train, X_test, y_test = feat_dict['data']

        size = 5000
        if len(y_train) > size:
            X_train, y_train = sklearn.utils.resample(X_train, y_train, n_samples=size, random_state=42, replace=False)

        detector_name = self.params['detector_name']
        # result_dict = {detector_name: {}}
        sd_inst = Dataset(params={})

        X_train, X_test = sd_inst.normalise_data(X_train, X_test,
                                                 norm_method=self.params['norm_method'])

        # select a part of val_data
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=int(len(y_test) * 0.2),
                                                        stratify=y_test,
                                                        random_state=42)
        if gs:
            print(f'with grid search: {detector_name.upper()}, gs: {gs}')
            # find the best parameters of the detector
            distances = distance.pdist(X_train, metric='euclidean')
            stat_data(distances.reshape(-1, 1), name='distances')
            # q_lst = list(np.linspace(0.0, 1, 20, endpoint=False))[1:]  # exclude 0
            q_lst = [float(f'{i * 0.1:.2f}') for i in range(1, 10, 1)] + [
                0.95]  # 10values: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
            sigma_lst = np.quantile(distances, q=q_lst)
            gamma_lst = list(1 / (2 * sigma_lst ** 2))  # sigma =np.quantile(distances, q=q_lst
            # print(f'gamma:{gamma}, q:{q}, sigma:{sigma}, Counter(distances):{sorted(Counter(distances).items(),
            # key=lambda item: item[0])}')
            print(f'gamma_lst:{gamma_lst},\nq_lst: {q_lst},\nsigma_lst: {sigma_lst}')
            detector = OCSVMDetector(random_state=self.params['random_state'], verbose=self.params['verbose'])
            grid = GridSearch(estimator=detector, scoring='auc',
                              param_grid={'gamma': gamma_lst, 'kernel': ['rbf']})
            # X_train = norm_X_train
            # X_test = norm_X_test

            grid.fit(X_train, y_train, X_val, y_val)
            self.detector = grid.best_estimator_
            print(f'{detector_name}.params: {self.detector.get_params()}')

            self.detector.test(X_test, y_test)
            print(f'auc: {self.detector.auc}')
            self.y_training_scores = self.detector.ocsvm.predict_proba(X_train)[:, 1]
            result_dict = {'y_true': y_test, 'y_scores': self.detector.y_scores,
                           'y_pred': self.detector.y_pred,
                           'y_val': y_val, 'y_train': y_train,
                           'y_training_scores': self.y_training_scores,
                           'auc': self.detector.auc,
                           'best_score_': grid.best_score_, 'best_params_': grid.best_params_,
                           'best_estimator_': grid.best_estimator_
                           }

        else:  # gs: False
            # rule of thumb in practice
            print('without grid search')
            distances = distance.pdist(X_train, metric='euclidean')
            stat_data(distances.reshape(-1, 1), name='distances')
            q = 0.3
            sigma = np.quantile(distances, q=q)
            if sigma == 0:  # find a new non-zero sigma
                print(f'sigma: {sigma}, q: {q}')
                q_lst = list(np.linspace(q + 0.01, 1, 10, endpoint=False))
                sigma_lst = np.quantile(distances, q=q_lst)
                sigma, q = [(s_v, q_v) for (s_v, q_v) in zip(sigma_lst, q_lst) if s_v > 0][0]
            print(f'sigma: {sigma}, q {q}')
            gamma = 1 / (2 * sigma ** 2)
            # print(f'gamma:{gamma}, q:{q}, sigma:{sigma}, Counter(distances):{sorted(Counter(distances).items(),
            # key=lambda item: item[0])}')
            print(f'gamma: {gamma}, q: {q}, sigma: {sigma}')
            self.detector = OCSVMDetector(gamma=gamma, random_state=self.params['random_state'],
                                          verbose=self.params['verbose'])

            # X_train = norm_X_train
            # X_test = norm_X_test

            print(f'{detector_name}.params: {self.detector.get_params()}')
            self.detector.fit(X_train, y_train)
            self.detector.test(X_test, y_test)
            self.y_training_scores = self.detector.ocsvm.predict_proba(X_train)[:, 1]
            result_dict = {'y_true': y_test, 'y_scores': self.detector.y_scores,
                           'y_pred': self.detector.y_pred,
                           'y_val': y_val, 'y_train': y_train,
                           'y_training_scores': self.y_training_scores,
                           'auc': self.detector.auc,
                           'best_score_': self.detector.auc,
                           'best_params_': self.detector.get_params(),
                           'best_estimator_': self.detector
                           }

        return result_dict


class KDEFactory(BaseDetectorFactory):

    def __init__(self, kernel='rbf', dataset_inst='', params={}):
        super(KDEFactory, self).__init__()
        self.dataset_inst = dataset_inst
        self.params = params

        self.params['dataset_inst'] = dataset_inst

    @func_notation
    def run_model(self, feat_dict):
        gs = self.params['gs']
        X_train, y_train, X_test, y_test = feat_dict['data']

        size = 5000
        if len(y_train) > size:
            X_train, y_train = sklearn.utils.resample(X_train, y_train, n_samples=size, random_state=42, replace=False)

        detector_name = self.params['detector_name']
        result_dict = {}
        sd_inst = Dataset(params={})

        X_train, X_test = sd_inst.normalise_data(X_train, X_test,
                                                 norm_method=self.params['norm_method'])
        # select a part of val_data
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=int(len(y_test) * 0.2),
                                                        stratify=y_test,
                                                        random_state=42)
        if gs:
            print(f'with grid search: {detector_name.upper()}, gs: {gs}')
            # find the best parameters of the detector
            distances = distance.pdist(X_train, metric='euclidean')
            stat_data(distances.reshape(-1, 1), name='distances')
            # q_lst = list(np.linspace(0.0, 1, 20, endpoint=False))[1:]  # exclude 0
            q_lst = [float(f'{i * 0.1:.2f}') for i in range(1, 10, 1)] + [0.95]  # 10 values
            sigma_lst = np.quantile(distances, q=q_lst)
            print(f'q_lst: {q_lst},\nsigma_lst:{list(sigma_lst)}')

            detector = KDEDetector(random_state=self.params['random_state'], verbose=self.params['verbose'])
            # grid = GridSearch(estimator=detector, scoring='auc',
            #                   param_grid={'bandwidth': list(np.linspace(0.0, 5, 501, endpoint=False))[1:]})

            grid = GridSearch(estimator=detector, scoring='auc',
                              param_grid={'bandwidth': sigma_lst})

            # X_train = norm_X_train
            # X_test = norm_X_test

            grid.fit(X_train, y_train, X_val, y_val)
            self.detector = grid.best_estimator_
            print(f'{detector_name}.params: {self.detector.get_params()}')

            self.detector.test(X_test, y_test)
            print(f'auc: {self.detector.auc}')
            self.y_training_scores = self.detector.kde.predict_proba(X_train)[:, 1]
            result_dict = {'y_true': y_test, 'y_scores': self.detector.y_scores,
                           'y_pred': self.detector.y_pred,
                           'y_val': y_val, 'y_train': y_train,
                           'y_training_scores': self.y_training_scores,
                           'auc': self.detector.auc,
                           'best_score_': grid.best_score_, 'best_params_': grid.best_params_,
                           'best_estimator_': grid.best_estimator_
                           }

        else:  # gs: False
            # rule of thumb in practice
            print('without grid search')
            distances = distance.pdist(X_train, metric='euclidean')
            stat_data(distances.reshape(-1, 1), name='distances')
            q = 0.3
            sigma = np.quantile(distances, q=q)  # distances >=0
            if sigma == 0:  # find a new non-zero sigma
                print(f'sigma: {sigma}, q: {q}')
                q_lst = list(np.linspace(q + 0.01, 1, 10, endpoint=False))
                sigma_lst = np.quantile(distances, q=q_lst)

                sigma, q = [(s_v, q_v) for (s_v, q_v) in zip(sigma_lst, q_lst) if s_v > 0][0]

            print(f'sigma: {sigma}, q {q}')

            self.detector = KDEDetector(bandwidth=sigma, random_state=self.params['random_state'],
                                        verbose=self.params['verbose'])

            # X_train = norm_X_train
            # X_test = norm_X_test

            print(f'{detector_name}.params: {self.detector.get_params()}')
            self.detector.fit(X_train, y_train)
            self.detector.test(X_test, y_test)
            self.y_training_scores = self.detector.kde.predict_proba(X_train)[:, 1]
            result_dict = {'y_true': y_test, 'y_scores': self.detector.y_scores,
                           'y_pred': self.detector.y_pred,
                           'y_val': y_val, 'y_train': y_train,
                           'y_training_scores': self.y_training_scores,
                           'auc': self.detector.auc,
                           'best_score_': self.detector.auc,
                           'best_params_': self.detector.get_params(),
                           'best_estimator_': self.detector
                           }

        return result_dict


class AEFactory(BaseDetectorFactory):

    def __init__(self, kernel='rbf', dataset_inst='', params={}):
        super(AEFactory, self).__init__()
        self.dataset_inst = dataset_inst
        self.params = params

        self.params['dataset_inst'] = dataset_inst

    @func_notation
    def run_model(self, feat_dict):
        gs = self.params['gs']
        X_train, y_train, X_test, y_test = feat_dict['data']

        size = 5000
        if len(y_train) > size:
            X_train, y_train = sklearn.utils.resample(X_train, y_train, n_samples=size, random_state=42, replace=False)

        detector_name = self.params['detector_name']
        result_dict = {}
        sd_inst = Dataset(params={})

        X_train, X_test = sd_inst.normalise_data(X_train, X_test,
                                                 norm_method=self.params['norm_method'])
        # select a part of val_data
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=int(len(y_test) * 0.2),
                                                        stratify=y_test,
                                                        random_state=42)

        if gs:
            # raise NotImplementedError('not implemented yet.')
            print(f'with grid search: {detector_name.upper()}, gs: {gs}')

            feat_dim = len(X_train[0])

            # adam_opt = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False, decay=0.1)
            # [64, 32, latent_num, 32, 64]
            # if feat_dim // 2 < 1:
            #     print(f'feat_dim: {feat_dim}')
            #     return -1

            def get_AE_parameters(d, num=10):
                latent_sizes = []
                for i in range(num):
                    v = np.ceil(1 + i * (d - 2) / 9).astype(int)
                    if v not in latent_sizes:
                        latent_sizes.append(v)

                hidden_sizes = [min((d - 1), np.ceil(2 * v).astype(int)) for v in latent_sizes]
                print(latent_sizes, len(latent_sizes), hidden_sizes)

                hidden_neurons = []
                for i, (hid, lat) in enumerate(zip(hidden_sizes, latent_sizes)):
                    v = [d, hid, lat, hid, d]
                    hidden_neurons.append(v)
                return hidden_neurons

            hidden_neurons = get_AE_parameters(feat_dim, num=10)
            print(f'For gridsearch, len:{len(hidden_neurons)}, hidden_neurons: {hidden_neurons}')
            detector = AEDetector(in_dim=feat_dim, params=self.params, random_state=self.params['random_state'],
                                  verbose=self.params['verbose'])
            # grid = GridSearch(estimator=detector, scoring='auc',
            #                   param_grid={'bandwidth': list(np.linspace(0.0, 5, 501, endpoint=False))[1:]})
            grid = GridSearch(estimator=detector, scoring='auc', params=self.params,
                              param_grid={'hidden_neurons': hidden_neurons})

            # X_train = norm_X_train
            # X_test = norm_X_test

            grid.fit(X_train, y_train, X_val, y_val)
            self.detector = grid.best_estimator_
            # dump_data((X_train, y_train, X_test, y_test), output_file=grid.best_model_file+'_Ghome.dat')
            print(f'--load the best model file obtain from grid search: {grid.best_model_file}')
            self.detector = torch.load(grid.best_model_file)
            self.detector.ae.model.eval()
            print(f'{detector_name}.params: {self.detector.get_params()}')

            # self.detector.ae.ndm.eval() should be added into ae.ndm.test()
            print('detector test:', self.detector.ae)
            self.detector.test(X_test, y_test)
            print(f'auc: {self.detector.auc}')
            self.y_training_scores = self.detector.ae.predict_proba(X_train)[:, 1]
            result_dict = {'y_true': y_test, 'y_scores': self.detector.y_scores,
                           'y_pred': self.detector.y_pred,
                           'y_val': y_val, 'y_train': y_train,
                           'y_training_scores': self.y_training_scores,
                           'auc': self.detector.auc,
                           'best_score_': grid.best_score_, 'best_params_': grid.best_params_,
                           'best_estimator_': grid.best_estimator_
                           }

            try:
                # avoid out of memory
                del self.detector.ae  # this is from global space - change this as you need
                del self.detector  # this is from global space - change this as you need
                os.remove(grid.best_model_file)
                print("gc.collect(): ", gc.collect())  # if it's done something you should see a number being outputted

            except Exception as e:
                print(f'Memory: {e}')

        else:
            # rule of thumb in practice
            print('without grid search')
            feat_dim = len(X_train[0])
            latent_dim = np.ceil(feat_dim / 2).astype(int)
            hid = min((feat_dim - 1), np.ceil(2 * latent_dim).astype(int))
            print(f'hidden_neurons: [{feat_dim}, {hid}, {latent_dim}, {hid}, {feat_dim}]')
            hidden_neurons = [feat_dim, hid, latent_dim, hid, feat_dim]

            self.detector = AEDetector(in_dim=feat_dim,
                                       hidden_neurons=hidden_neurons,
                                       random_state=self.params['random_state'],
                                       verbose=self.params['verbose'])

            # X_train = norm_X_train
            # X_test = norm_X_test

            print(f'{detector_name}.params: {self.detector.get_params()}')
            self.detector.fit(X_train, y_train)
            self.detector.test(X_test, y_test)
            self.y_training_scores = self.detector.ae.predict_proba(X_train)[:, 1]
            result_dict = {'y_true': y_test, 'y_scores': self.detector.y_scores,
                           'y_pred': self.detector.y_pred,
                           'y_val': y_val, 'y_train': y_train,
                           'y_training_scores': self.y_training_scores,
                           'auc': self.detector.auc,
                           'best_score_': self.detector.auc,
                           'best_params_': self.detector.get_params(),
                           'best_estimator_': self.detector
                           }

            # avoid out of memory
            del self.detector  # this is from global space - change this as you need
            print("gc.collect(): ", gc.collect())  # if it's done something you should see a number being outputted

        return result_dict


class IFFactory(BaseDetectorFactory):

    def __init__(self, dataset_inst='', params={}):
        super(IFFactory, self).__init__()
        self.dataset_inst = dataset_inst
        self.params = params

        self.params['dataset_inst'] = dataset_inst

    @func_notation
    def run_model(self, feat_dict={}):  # feat_dict={'feat_set':, 'feat_file':, 'data':(), ...}
        gs = self.params['gs']
        X_train, y_train, X_test, y_test = feat_dict['data']

        size = 5000
        if len(y_train) > size:
            X_train, y_train = sklearn.utils.resample(X_train, y_train, n_samples=size, random_state=42, replace=False)

        detector_name = self.params['detector_name']
        # assert self.params['detector_name'] == 'IF'
        # result_dict = {detector_name: {}}
        sd_inst = Dataset()
        X_train, X_test = sd_inst.normalise_data(X_train, X_test,
                                                 norm_method=self.params['norm_method'])
        # select a part of val_data
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=int(len(y_test) * 0.2),
                                                        stratify=y_test,
                                                        random_state=42)
        if gs:
            print(f'with grid search: {detector_name.upper()}, gs: {gs}')
            # find the best parameters of the detector
            detector = IForestDetector(random_state=self.params['random_state'], verbose=self.params['verbose'])
            grid = GridSearch(estimator=detector, scoring='auc',
                              param_grid={
                                  'n_estimators': [int(v) for v in list(np.linspace(30, 300, num=10, endpoint=True))]
                                  # 10 values: [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
                              })  # 'max_samples': [int, float, 'auto']

            # X_train = norm_X_train
            # X_test = norm_X_test

            grid.fit(X_train, y_train, X_val, y_val)
            self.detector = grid.best_estimator_
            print(f'{detector_name}.params: {self.detector.get_params()}')

            self.detector.test(X_test, y_test)
            print(f'auc: {self.detector.auc}')
            self.y_training_scores = self.detector.iforest.predict_proba(X_train)[:, 1]
            result_dict = {'y_true': y_test, 'y_scores': self.detector.y_scores,
                           'y_pred': self.detector.y_pred,
                           'y_val': y_val, 'y_train': y_train,
                           'y_training_scores': self.y_training_scores,
                           'auc': self.detector.auc,
                           'best_score_': grid.best_score_, 'best_params_': grid.best_params_,
                           'best_estimator_': grid.best_estimator_
                           }

        else:  # gs: False
            # rule of thumb in practice
            print('without grid search')
            # # distances = distance.pdist(norm_X_train, metric='euclidean')
            # # stat_data(distances.reshape(-1, 1), name='distances')
            # # q = 0.3   # cannot work for meanshift, so we use q=0.7
            # # sigma = np.quantile(distances, q=q)
            # # print(f'sigma: {sigma}, q (for setting bandwidth in MeanShift): {q}')
            # #
            # # # means_init, n_components = obtain_means_init(norm_X_train, bandwidth=sigma)
            # means_init, n_components = obtain_means_init_quickshift_pp(norm_X_train,
            #                                                            k=int(np.log(len(norm_X_train)) ** 4))
            #
            # if self.params['norm_method'] == 'std':
            #     means_init = means_init * sd_inst.train_scaler.scale_ + sd_inst.train_scaler.mean_

            self.detector = IForestDetector(
                random_state=self.params['random_state'], verbose=self.params['verbose'])

            print(f'{detector_name}.params: {self.detector.get_params()}')

            # X_train = norm_X_train
            # X_test = norm_X_test

            self.detector.fit(X_train, y_train)
            self.detector.test(X_test, y_test)
            self.y_training_scores = self.detector.iforest.predict_proba(X_train)[:, 1]
            result_dict = {'y_true': y_test, 'y_scores': self.detector.y_scores,
                           'y_pred': self.detector.y_pred,
                           'y_val': y_val, 'y_train': y_train,
                           'y_training_scores': self.y_training_scores,
                           'auc': self.detector.auc,
                           'best_score_': self.detector.auc,
                           'best_params_': self.detector.get_params(),
                           'best_estimator_': self.detector
                           }

        return result_dict


class PCAFactory(BaseDetectorFactory):

    def __init__(self, dataset_inst='', params={}):
        super(PCAFactory, self).__init__()
        self.dataset_inst = dataset_inst
        self.params = params

        self.params['dataset_inst'] = dataset_inst

    @func_notation
    def run_model(self, feat_dict={}):  # feat_dict={'feat_set':, 'feat_file':, 'data':(), ...}
        gs = self.params['gs']
        X_train, y_train, X_test, y_test = feat_dict['data']

        size = 5000
        if len(y_train) > size:
            X_train, y_train = sklearn.utils.resample(X_train, y_train, n_samples=size, random_state=42, replace=False)

        detector_name = self.params['detector_name']
        # assert self.params['detector_name'] == 'IF'
        # result_dict = {detector_name: {}}
        sd_inst = Dataset()
        X_train, X_test = sd_inst.normalise_data(X_train, X_test,
                                                 norm_method=self.params['norm_method'])

        # select a part of val_data
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=int(len(y_test) * 0.2),
                                                        stratify=y_test,
                                                        random_state=42)
        if gs:
            print(f'with grid search: {detector_name.upper()}, gs: {gs}')
            # find the best parameters of the detector
            detector = PCADetector(random_state=self.params['random_state'], verbose=self.params['verbose'])
            # n_components = np.log2(np.logspace(1, min(X_train.shape), num=5, endpoint=False, base=2))
            # n_components = sorted(set([int(v) for v in n_components if v >= 1.0]))

            # get n_components
            n_components = [int(v) for v in list(np.linspace(1, min(X_train.shape), num=10, endpoint=False))]
            print(f'n_components: {n_components}')

            # X_train = norm_X_train
            # X_test = norm_X_test

            grid = GridSearch(estimator=detector, scoring='auc',
                              param_grid={'n_components': n_components})  # 'max_samples': [int, float, 'auto']

            grid.fit(X_train, y_train, X_val, y_val)
            self.detector = grid.best_estimator_
            print(f'{detector_name}.params: {self.detector.get_params()}')

            self.detector.test(X_test, y_test)
            print(f'auc: {self.detector.auc}')
            self.y_training_scores = self.detector.pca.predict_proba(X_train)[:, 1]
            result_dict = {'y_true': y_test, 'y_scores': self.detector.y_scores,
                           'y_pred': self.detector.y_pred,
                           'y_val': y_val, 'y_train': y_train,
                           'y_training_scores': self.y_training_scores,
                           'auc': self.detector.auc,
                           'best_score_': grid.best_score_, 'best_params_': grid.best_params_,
                           'best_estimator_': grid.best_estimator_
                           }

        else:  # gs: False
            # rule of thumb in practice
            print('without grid search')

            self.detector = PCADetector(n_components='mle',
                                        random_state=self.params['random_state'], verbose=self.params['verbose'])
            # X_train = norm_X_train
            # X_test = norm_X_test

            self.detector.fit(X_train, y_train)
            self.detector.test(X_test, y_test)

            print(
                f'{detector_name}.params: {self.detector.get_params()},  n_components: {self.detector.pca.n_components}')
            self.y_training_scores = self.detector.pca.predict_proba(X_train)[:, 1]
            result_dict = {'y_true': y_test, 'y_scores': self.detector.y_scores,
                           'y_pred': self.detector.y_pred,
                           'y_val': y_val, 'y_train': y_train,
                           'y_training_scores': self.y_training_scores,
                           'auc': self.detector.auc,
                           'best_score_': self.detector.auc,
                           'best_params_': self.detector.get_params(),
                           'best_estimator_': self.detector
                           }

        return result_dict
