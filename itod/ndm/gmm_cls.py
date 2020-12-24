"""GMM class for anomaly detection

    labels=['normal':0, 'anomaly':1]
    invert

"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE

import numpy as np
from pyod.models.base import BaseDetector
from pyod.utils import invert_order
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from sklearn.compose._column_transformer import _check_X
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from itod.utils.tool import execute_time


class GMMEstimator(BaseDetector, BaseEstimator):
    def __init__(self, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=42, warm_start=False,
                 verbose=0, verbose_interval=10, contamination=0.1, quantile=0.9):
        super(GMMEstimator, self).__init__()
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        # self.quantile = quantile  # for detecction threshold
        self.contamination = contamination

        self._classes = 2  # default as binary classification

    def fit(self, X, y=None):
        """Fit detector. y is optional for unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples,), optional (default=None)
            The ground truth of the input samples (labels).
        """
        # validate inputs X and y (optional)
        X = _check_X(X)
        self._set_n_classes(y)

        self.detector_ = GaussianMixture(n_components=self.n_components,
                                         covariance_type=self.covariance_type,
                                         tol=self.tol,
                                         reg_covar=self.reg_covar,
                                         max_iter=self.max_iter,
                                         n_init=self.n_init,
                                         init_params=self.init_params,
                                         weights_init=self.weights_init,
                                         means_init=self.means_init,
                                         precisions_init=self.precisions_init,
                                         random_state=self.random_state,
                                         warm_start=self.warm_start,
                                         verbose=self.verbose,
                                         verbose_interval=self.verbose_interval)
        self.detector_.fit(X=X, y=y)

        # log_prob_norm = logsumexp(self.detector_._estimate_weighted_log_prob(X), axis=1)
        # self.decision_scores_ = log_prob_norm  # log(sum(pi* normal()))

        # invert decision_scores_. Outliers comes with higher outlier scores
        self.decision_scores_ = invert_order(logsumexp(self.detector_._estimate_weighted_log_prob(X), axis=1))

        # print(f'quantile: {self.quantile}')
        # # stat_data(self.decision_scores_.reshape(-1, 1))
        # self.threshold_ = np.quantile(a=self.decision_scores_,
        #                               q=self.quantile)  # np.percentile will sort the input
        # print(f'threshold_: {self.threshold_}')
        # # idx = int(self.quantile * self.decision_scores_.shape[0])
        # # self.threshold_ = sorted(self.decision_scores_, reverse=False)[idx]
        # # # # self.gmm.detect_thres = get_thres(self.decision_scores_, percent=self.quantile)
        # # print(f'threshold_: {self.threshold_}')

        # # self.labels_ = (self.decision_scores_ < self.threshold_).astype('int').ravel()    # False=0, True = 1, so it should be '<'. normal=0, anomaly = 1
        # self.labels_ = np.zeros(X.shape[0])
        # self.labels_[self.decision_scores_ > self.threshold_] = 1  # 0 is normal, 1 is abnormal

        # self._process_decision_scores()   # get self.threshold_

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
        return invert_order(logsumexp(self.detector_._estimate_weighted_log_prob(X), axis=1))

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

    def predict_proba(self, X, method='linear'):
        """Predict the probability of a sample being outlier. Two approaches
        are possible:

        1. simply use Min-max conversion to linearly transform the outlier
           scores into the range of [0,1]. The ndm must be
           fitted first.
        2. use unifying scores, see :cite:`kriegel2011interpreting`.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        method : str, optional (default='linear')
            probability conversion method. It must be one of
            'linear' or 'unify'.

        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted ndm. Return the outlier probability, ranging
            in [0,1].
        """
        X = check_array(X)
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        train_scores = self.decision_scores_

        test_scores = self.decision_function(
            X)  # the probability density (not probability) of x is predicted as anomaly

        probs = np.zeros([X.shape[0], int(self._classes)])
        # probs[:,1] = test_scores      # for debug
        # probs[:,0] = -test_scores     # for debug
        # return probs
        if method == 'linear':
            scaler = MinMaxScaler().fit(train_scores.reshape(-1, 1))
            probs[:, 1] = scaler.transform(
                test_scores.reshape(-1, 1)).ravel().clip(0, 1)
            probs[:, 0] = 1 - probs[:, 1]
            return probs
        else:
            raise ValueError(method,
                             'is not a valid probability conversion method')


class GMMDetector(GMMEstimator):

    def __init__(self, n_components=1, covariance_type='full', quantile=0.9, means_init=None, random_state=42,
                 verbose=True):
        """

        :param n_components:  represents the number of gaussian distributions is used to form GMM.
                             In anomaly detection, we just use normal data to train our ndm ( no anomaly data is used),
                             Use pdf of GMM as the measure function to determine normal and anomaly.
        :param quantile: for choose the threshold : Use the q-th percent of pdf as detection threshold: quantile =[0,1], percentile = [0,100]
        """
        # print(f'n_components: {n_components}, covariance_type: {covariance_type}')
        super(GMMDetector, self).__init__()
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.quantile = quantile
        self.verbose = verbose
        self.means_init = means_init
        self.random_state = random_state
        self.verbose = verbose

        self.name = 'GMM'

    @execute_time
    def fit(self, X_train, y_train=None):
        """
            Using train set to find the parameters of GMM (such as mean, covariance and weight of each cluster)
        :param X_train:
        :return:
        """
        # if len(X_train[0]) <= 1:
        #     raise ValueError(f'len(X_train[0]) <=1')
        #     return -1
        self.gmm = GMMEstimator(n_components=self.n_components, covariance_type=self.covariance_type,
                                quantile=self.quantile,
                                means_init=self.means_init, contamination=0.1,
                                verbose=self.verbose, random_state=self.random_state)

        self.gmm.fit(X=X_train)
        print(f'threshold_: {self.gmm.threshold_}')
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
        self.y_pred = self.gmm.predict(X=X_test)
        self.cm = confusion_matrix(y_true=y_test, y_pred=self.y_pred)
        if self.verbose:
            print(f'gmm.cm: \n=> predicted 0 and 1 (0 stands for normals and 1 for anomalies)\n{self.cm}')
        self.y_scores = self.gmm.predict_proba(X_test)[:, 1]  # y_scores=positive_class, predict X as 1 (anomaly).
        # roc_curve(): When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1}, ``pos_label`` is set to 1,
        #  otherwise an error will be raised.
        self.auc = roc_auc_score(y_true=y_test, y_score=self.y_scores)
        print(f'auc: {self.auc}')
