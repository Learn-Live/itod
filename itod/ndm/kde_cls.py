"""Kernel density estimation

"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE

import numpy as np
from pyod.models.base import BaseDetector
from pyod.utils import invert_order
from scipy.special import erf
from sklearn.compose._column_transformer import _check_X
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.neighbors import KernelDensity
from sklearn.neighbors._kde import VALID_KERNELS
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from itod.utils.tool import execute_time


class KernelDensityEstimator(KernelDensity, BaseDetector):

    def __init__(self, quantile=0.9, bandwidth=1.0, algorithm='auto',
                 kernel='gaussian', metric="euclidean", atol=0, rtol=0, contamination=0.1,
                 breadth_first=True, leaf_size=40, metric_params=None, random_state=42):
        self.quantile = quantile
        self.algorithm = algorithm
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.metric = metric
        self.atol = atol
        self.rtol = rtol
        self.breadth_first = breadth_first
        self.leaf_size = leaf_size
        self.metric_params = metric_params
        self.contamination = contamination
        self.random_state = random_state

        # run the choose algorithm code so that exceptions will happen here
        # we're using clone() in the GenerativeBayes classifier,
        # so we can't do this kind of logic in __init__
        self._choose_algorithm(self.algorithm, self.metric)

        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if kernel not in VALID_KERNELS:
            raise ValueError("invalid kernel: '{0}'".format(kernel))

        self._classes = 2  # default as binary classification

    def fit(self, X, y=None):

        X = _check_X(X)
        self.detector_ = KernelDensity(bandwidth=self.bandwidth,
                                       algorithm=self.algorithm,
                                       kernel=self.kernel,
                                       metric=self.metric,
                                       atol=self.atol,
                                       rtol=self.rtol,
                                       breadth_first=self.breadth_first,
                                       leaf_size=self.leaf_size,
                                       metric_params=self.metric_params)

        self.detector_.fit(X)

        self.decision_scores_ = invert_order(self.detector_.score_samples(X))  # invert_order: value * (-1)

        # print(f'quantile: {self.quantile}')
        # # stat_data(self.decision_scores_.reshape(-1, 1))
        # self.threshold_ = np.quantile(a=self.decision_scores_,
        #                               q=self.quantile)  # np.percentile will sort the input
        # print(f'threshold_: {self.threshold_}')
        #
        # self.labels_ = np.zeros(X.shape[0])
        # self.labels_[self.decision_scores_ > self.threshold_] = 1  # 0 is normal, 1 is abnormal

        self.threshold_ = np.percentile(self.decision_scores_,
                                        100 * (1 - self.contamination))
        self.labels_ = (self.decision_scores_ > self.threshold_).astype(
            'int').ravel()

        return self

    def decision_function(self, X):
        """Predict raw anomaly scores of X using the fitted detector.

        The anomaly score of an input sample is computed based on the fitted
        detector. For consistency, outliers are assigned with
        larger anomaly scores. so use invert_order

        After invert_order(): the higher score, the more probability of x that is predicted as abnormal

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        return invert_order(self.detector_.score_samples(X))

    def predict_proba(self, X, method='linear'):

        X = check_array(X)
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        train_scores = self.decision_scores_

        test_scores = self.decision_function(
            X)  # the probability density (not probability) of x is predicted as anomaly

        probs = np.zeros([X.shape[0], int(self._classes)])
        if method == 'linear':
            scaler = MinMaxScaler().fit(train_scores.reshape(-1, 1))
            probs[:, 1] = scaler.transform(
                test_scores.reshape(-1, 1)).ravel().clip(0, 1)
            probs[:, 0] = 1 - probs[:, 1]
            return probs
        elif method == 'unify':
            # turn output_data into probability
            pre_erf_score = (test_scores - self._mu) / (
                    self._sigma * np.sqrt(2))
            erf_score = erf(pre_erf_score)
            probs[:, 1] = erf_score.clip(0, 1).ravel()
            probs[:, 0] = 1 - probs[:, 1]
            return probs
        else:
            raise ValueError(method,
                             'is not a valid probability conversion method')

        return y_scores

    def predict(self, X='X_test', y=None):
        """Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted ndm. 0 stands for inliers and 1 for outliers.
        """
        # # y_pred = self._estimate_weighted_log_prob(X_test).argmax(axis=1)  # predict the x belongs to which component.
        # y_pred = np.zeros(X_test.shape[0])
        # y_pred[self.predict_proba(X_test) < self.detect_thres] = 1  # 0 is normal, 1 is abnormal

        X = check_array(X)
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])

        pred_score = self.decision_function(X)
        y_pred = np.zeros(X.shape[0])
        y_pred[pred_score > self.threshold_] = 1  # 0 is normal, 1 is abnormal

        return y_pred


class KDEDetector(KernelDensityEstimator):

    def __init__(self, quantile=0.9, bandwidth=1.0, algorithm='auto',
                 kernel='gaussian', metric="euclidean", atol=0, rtol=0,
                 breadth_first=True, leaf_size=40, metric_params=None, random_state=42, verbose=True):
        """

        :param quantile: to choose the threshold to get the predicted label
        :param bandwidth:
        :param algorithm:
        :param kernel:
        :param metric:
        :param atol:
        :param rtol:
        :param breadth_first:
        :param leaf_size:
        :param metric_params:
        :param random_state:
        :param verbose:
        """
        self.quantile = quantile
        self.algorithm = algorithm
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.metric = metric
        # self.atol = atol
        # self.rtol = rtol
        # self.breadth_first = breadth_first
        # self.leaf_size = leaf_size
        # self.metric_params = metric_params

        self.random_state = random_state
        self.verbose = verbose

        self.name = 'KDE'

    @execute_time
    def fit(self, X_train, y_train=None):
        # if len(X_train[0]) <= 1:
        #     raise ValueError(f'len(X_train[0]) <=1')
        #     return -1

        self.kde = KernelDensityEstimator(quantile=self.quantile, bandwidth=self.bandwidth, kernel=self.kernel,
                                          random_state=self.random_state, contamination=0.1)
        self.kde.fit(X_train)
        print(f'threshold_: {self.kde.threshold_}')
        return self

    @execute_time
    def test(self, X_test, y_test):
        """
            pyod.models.base.BaseDetector.labels_: The binary labels of the training data.
                                                0 stands for inliers and 1 for outliers/anomalies.
        :param X_test:
        :param y_test:
        :return:
        """

        # Predict if a particular sample is an outlier or not (0 is inlier and 1 for outlier).
        self.y_pred = self.kde.predict(X=X_test)
        self.cm = confusion_matrix(y_true=y_test, y_pred=self.y_pred)
        if self.verbose:
            print(f'kde.cm: \n=> predicted 0 and 1 (0 stands for normals and 1 for anomalies)\n{self.cm}')
        self.y_scores = self.kde.predict_proba(X_test)[:, 1]  # y_scores=positive_class, predict X as 1 (anomaly).
        # roc_curve(): When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1}, ``pos_label`` is set to 1,
        #  otherwise an error will be raised.
        self.auc = roc_auc_score(y_true=y_test, y_score=self.y_scores)
        print(f'auc: {self.auc}')
