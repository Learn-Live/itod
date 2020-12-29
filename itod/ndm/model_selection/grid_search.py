""" grid search class
    'htop' check process status
"""
# Authors: kun.bj@outlook.com
#
# License: xxx
import os
import itertools
import pickle
from copy import deepcopy
# import tensorflow as tf
import sys
from sklearn import clone
import gc
# from keras import backend as K
from torch import save, load
import torch


# from pympler import asizeof


#
# def reset_seeds(reset_graph_with_backend=None):
#     if reset_graph_with_backend is not None:
#         K = reset_graph_with_backend
#         K.clear_session()
#         tf.compat.v1.reset_default_graph()
#         print("KERAS AND TENSORFLOW GRAPHS RESET")  # optional
#
#     # np.random.seed(1)
#     # random.seed(2)
#     tf.compat.v1.set_random_seed(3)
#     print("RANDOM SEEDS RESET")  # optional

class GridSearch:

    def __init__(self, estimator='', scoring='auc', params={},
                 param_grid={'n_components': [i + 1 for i in range(10)], 'covariance_type': ['full', 'diag']}):

        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.params = params

    def fit(self, X_train='', y_train='', X_test='', y_test=''):
        """ get the best parameters of the detector

        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :return:
        """

        keys, values = zip(*self.param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        print(f'len(combinations of parameters): {len(combinations)}')
        # print(combinations)
        self.best_score_ = -1
        self.best_params_ = {}
        self.best_estimator_ = None
        self.best_model_file = ''
        self.best_index = 0
        for idx, params in enumerate(combinations):
            print(f'grid search, idx: {idx + 1}, params: {params}')
            self.detector_idx = ''
            self.detector_idx = clone(
                self.estimator)  # constructs a new estimator with the same parameters, but not fit
            self.detector_idx.set_params(**params)  # set params
            print(f'idx: {idx + 1}, detector_params: {self.detector_idx.get_params()}')
            try:
                self.detector_idx.fit(X_train, y_train)
                print('test')
                self.detector_idx.test(X_test, y_test)
                # self.detector_idx.auc=0.1
            except Exception as e:
                print(f'{e}, skipping {params}')
                continue

            print(f'auc: {self.detector_idx.auc}')

            if self.scoring == 'auc':
                if self.detector_idx.auc > self.best_score_:
                    print(f'self.detector_idx.auc {self.detector_idx.auc} > self.best_score_ {self.best_score_}')
                    self.best_score_ = self.detector_idx.auc
                    self.best_params_ = params  # if key exists, update; otherwise, add new key

                    if self.detector_idx.name == 'AE':  # pleaes don't run multiple AE, because each of them will override the model file and causes the results unreliable.
                        model_dir = self.detector_idx.params['opt_dir']
                        header = self.params['header']
                        # dim = X_train.shape[1]
                        model_file = f'{model_dir}/idx_{idx}-grid_True-header_{header}-{self.detector_idx.name}_model.pth'  # .h5: save to HDF5 format; .pth: pytorch model
                        print(f'save AE model to disk: {model_file}')
                        # self.detector_idx.ae.model_.save(model_file)
                        if os.path.exists(model_file):
                            os.remove(model_file)
                        torch.save(self.detector_idx, model_file)  # overwrite

                        # print(f'deepcopy', asizeof.asizeof(deepcopy(self.detector_idx))//(1024**3), asizeof.asizeof(self.detector_idx))
                        # self.best_estimator_ = deepcopy(self.detector_idx)      # no matter deepcopy() or copy, both of them can cause memory leak for keras and pytorch
                        # self.best_estimator_ = clone(self.estimator)  # constructs a new estimator with the same parameters, but not fit
                        self.best_estimator_ = ''
                        # self.best_estimator_ .set_params(**params)  # set params
                        # print(f'deepcopy', asizeof.asizeof(self.best_estimator_) // (1024 ** 3),
                        #       asizeof.asizeof(self.best_estimator_.ae))

                        self.best_model_file = model_file
                        del self.detector_idx  # only remove the reference to the object, doesn't free memory
                        gc.collect()  # free memory immediately
                        print('---')
                    else:  # sklearn can use deepcopy()
                        self.best_estimator_ = deepcopy(self.detector_idx)  # it won't work to tensorflow and keras
                    self.best_index = idx
                    # print(f'best_auc: {self.best_estimator_.auc}, {self.best_score_}')
            else:
                print(f'scoring: {self.scoring} is not implemented yet, please check and retry')

            # del self.detector_idx

            print("gc.collect(): ", gc.collect())  # if it's done something you should see a number being outputted

        # # summarize the results of the grid search
        print(f'grid.best_score_: {self.best_score_}')
        print(f'grid.best_params_: {self.best_params_}')
        print(f'grid.best_estimator_: {self.best_estimator_}')
        print(f'grid.best_index: {self.best_index}')

        return self
