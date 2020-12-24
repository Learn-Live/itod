"""PCA class

"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE

from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.metrics import confusion_matrix, roc_auc_score

from pyod.models.base import BaseDetector
from pyod.models.pca import PCA
from pyod.utils.utility import check_parameter, standardizer
from scipy.spatial import distance

from itod.utils.tool import execute_time


class MY_PCA(BaseDetector):
    """Principal component analysis (PCA) can be used in detecting outliers.
    PCA is a linear dimensionality reduction using Singular Value Decomposition
    of the data to project it to a lower dimensional space.

    In this procedure, covariance matrix of the data can be decomposed to
    orthogonal vectors, called eigenvectors, associated with eigenvalues. The
    eigenvectors with high eigenvalues capture most of the variance in the
    data.

    Therefore, a low dimensional hyperplane constructed by k eigenvectors can
    capture most of the variance in the data. However, outliers are different
    from normal data points, which is more obvious on the hyperplane
    constructed by the eigenvectors with small eigenvalues.

    Therefore, outlier scores can be obtained as the sum of the projected
    distance of a sample on d-k eigenvectors.
    See :cite:`shyu2003novel,aggarwal2015outlier` for details.

    Hard scores: in aggarwal2015outlier p78
    Score(X) = Sum of weighted euclidean distance between each sample to the
    hyperplane constructed by the k eigenvectors, i.e., using the remaining (d-k) eignvectors for scores.

    Parameters
    ----------
    n_components : int, float, None or string
        Number of components to keep.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

        if n_components == 'mle' and svd_solver == 'full', Minka\'s MLE is used
        to guess the dimension
        if ``0 < n_components < 1`` and svd_solver == 'full', select the number
        of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components
        n_components cannot be equal to n_features for svd_solver == 'arpack'.

    n_selected_components : int, optional (default=None)
        Number of selected principal components
        for calculating the outlier scores. It is not necessarily equal to
        the total number of the principal components. If not set, use
        all principal components.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    copy : bool (default True)
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    whiten : bool, optional (default False)
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    svd_solver : string {'auto', 'full', 'arpack', 'randomized'}
        auto :
            the solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < X.shape[1]
        randomized :
            run randomized SVD by the method of Halko et al.

    tol : float >= 0, optional (default .0)
        Tolerance for singular values computed by svd_solver == 'arpack'.

    iterated_power : int >= 0, or 'auto', (default 'auto')
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.

    random_state : int, RandomState instance or None, optional (default None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``svd_solver`` == 'arpack' or 'randomized'.

    weighted : bool, optional (default=True)
        If True, the eigenvalues are used in score computation.
        The eigenvectors with small eigenvalues comes with more importance
        in outlier score calculation.

    standardization : bool, optional (default=True)
        If True, perform standardization first to convert
        data to zero mean and unit variance.
        See http://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data. The components are sorted by
        ``explained_variance_``.

    explained_variance_ : array, shape (n_components,)
        The amount of variance explained by each of the selected components.

        Equal to n_components largest eigenvalues
        of the covariance matrix of X.

    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.

        If ``n_components`` is not set then all components are stored and the
        sum of explained variances is equal to 1.0.

    singular_values_ : array, shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.

    mean_ : array, shape (n_features,)
        Per-feature empirical mean, estimated from the training set.

        Equal to `X.mean(axis=0)`.

    n_components_ : int
        The estimated number of components. When n_components is set
        to 'mle' or a number between 0 and 1 (with svd_solver == 'full') this
        number is estimated from input data. Otherwise it equals the parameter
        n_components, or n_features if n_components is None.

    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA ndm
        from Tipping and Bishop 1999. See "Pattern Recognition and
        Machine Learning" by C. Bishop, 12.2.1 p. 574 or
        http://www.miketipping.com/papers/met-mppca.pdf. It is required to
        computed the estimated data covariance and score samples.

        Equal to the average of (min(n_features, n_samples) - n_components)
        smallest eigenvalues of the covariance matrix of X.

    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, n_components=None, n_selected_components=None,
                 contamination=0.1, copy=True, whiten=False, svd_solver='auto',
                 tol=0.0, iterated_power='auto', random_state=None,
                 weighted=True, standardization=True):

        super(MY_PCA, self).__init__(contamination=contamination)
        self.n_components = n_components
        self.n_selected_components = n_selected_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.weighted = weighted
        self.standardization = standardization
        self.score_name = "reconstructed"  # the way to obtain outlier scores

    # noinspection PyIncorrectDocstring
    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        # PCA is recommended to use on the standardized data (zero mean and unit variance).
        if self.standardization:
            X, self.scaler_ = standardizer(X, keep_scalar=True)

        self.detector_ = sklearn_PCA(n_components=self.n_components,
                                     copy=self.copy,
                                     whiten=self.whiten,
                                     svd_solver=self.svd_solver,
                                     tol=self.tol,
                                     iterated_power=self.iterated_power,
                                     random_state=self.random_state)

        # self.detector_.fit(X=X, y=y)
        U, S, V = self.detector_._fit(X)  # self.mean_ inside of it
        # copy the attributes from the sklearn PCA object
        # Get variance explained by singular values
        n_samples, n_features = X.shape
        explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        singular_values_ = S.copy()  # Store the singular values.
        self.components_ = V
        self.n_components = self.detector_.n_components_
        self.explained_variance_ = explained_variance_
        self.explained_variance_ratio_ = explained_variance_ratio_
        self.singular_values_ = singular_values_

        # validate the number of components to be used for outlier detection
        if self.n_selected_components is None:
            if self.score_name == 'reconstructed':
                self.n_selected_components_ = self.n_components  # top k
            else:
                self.n_selected_components_ = n_features - self.n_components  # d-k
        else:
            self.n_selected_components_ = self.n_selected_components
        # check_parameter(self.n_selected_components_, 1, self.n_components,
        #                 include_left=True, include_right=True,
        #                 param_name='n_selected_components_')
        print("----", self.n_selected_components_, self.n_components)
        # use eigenvalues as the weights of eigenvectors
        self.w_components_ = np.ones([len(self.explained_variance_ratio_), ])
        if self.weighted:
            self.w_components_ = self.explained_variance_ratio_

        if self.score_name == 'reconstructed':
            # outlier scores is the reconstruction error between X and reconstructed X' with top k eigenvectors.
            # a normal sample has a smaller reconstruction error than that of an anomaly.
            self.selected_components_ = self.components_[:self.n_selected_components_, :]
            self.selected_w_components_ = self.w_components_[: self.n_selected_components_]
            X_transformed = np.dot(X - self.detector_.mean_, np.transpose(self.selected_components_))
            # X=USV^T, Z = XV = USV^TV = US, reconst_X=Z*V^T
            reconst_X = np.dot(X_transformed, self.selected_components_) + self.detector_.mean_
            self.decision_scores_ = np.asarray([distance.euclidean(x, re_x) for (x, re_x) in zip(X, reconst_X)])
        elif self.score_name == 'hard':  # in aggarwal2015outlier p78
            # outlier scores is the sum of the weighted distances between each
            # sample to the (d-k) eigenvectors. The eigenvectors with smaller
            # eigenvalues have more influence
            # Not all eigenvectors are used, only n_selected_components_ or (d-k) smallest
            # are used since they better reflect the variance change
            # (d-k):d is n_features, k is n_components.
            self.selected_components_ = self.components_[-1 * self.n_selected_components_:, :]
            self.selected_w_components_ = self.w_components_[-1 * self.n_selected_components_:]
            # unit basis
            e_axis = np.transpose(self.selected_components_) / np.linalg.norm(np.transpose(self.selected_components_),
                                                                              axis=0).reshape(1, -1)
            self.decision_scores_ = np.sum(
                (np.dot(X - self.detector_.mean_, e_axis)) ** 2 / self.selected_w_components_,
                axis=1).ravel()
        else:  # self.score_name == 'soft': # in aggarwal2015outlier p78
            self.selected_components_ = self.components_
            self.selected_w_components_ = self.w_components_
            # unit basis
            e_axis = np.transpose(self.selected_components_) / np.linalg.norm(np.transpose(self.selected_components_),
                                                                              axis=0).reshape(1, -1)
            self.decision_scores_ = np.sum(
                (np.dot(X - self.detector_.mean_, e_axis)) ** 2 / self.selected_w_components_,
                axis=1).ravel()

        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['components_', 'w_components_'])

        X = check_array(X)
        if self.standardization:
            X = self.scaler_.transform(X)

        if self.score_name == 'reconstructed':
            print('self.detector_.mean_: ', self.detector_.mean_)
            # X=USV^T, Z = XV = USV^TV = US, reconst_X=Z*V^T
            reconst_X = np.dot(np.dot(X - self.detector_.mean_, np.transpose(self.selected_components_)),
                               self.selected_components_) + self.detector_.mean_
            return np.asarray([distance.euclidean(x, re_x) for (x, re_x) in zip(X, reconst_X)])
        elif self.score_name == 'hard':
            # unit basis
            e_axis = np.transpose(self.selected_components_) / np.linalg.norm(np.transpose(self.selected_components_),
                                                                              axis=0).reshape(1, -1)
            return np.sum(
                (np.dot(X - self.detector_.mean_, e_axis)) ** 2 / self.selected_w_components_, axis=1).ravel()
        else:  # self.score_name == 'soft':
            # unit basis
            e_axis = np.transpose(self.selected_components_) / np.linalg.norm(np.transpose(self.selected_components_),
                                                                              axis=0).reshape(1, -1)
            return np.sum(
                (np.dot(X - self.detector_.mean_, e_axis)) ** 2 / self.selected_w_components_, axis=1).ravel()


class PCADetector(MY_PCA):

    def __init__(self, n_components=None, random_state=42, verbose=True):
        """

        Parameters
        ----------
        n_components:  if n_components is not set all components are kept
        random_state
        verbose
        """
        self.n_components = n_components
        self.verbose = verbose
        self.random_state = random_state
        self.name = 'PCA'

    @execute_time
    def fit(self, X_train, y_train=None):
        # self.pca = MY_PCA(n_components=self.n_components,
        #                   random_state=self.random_state, n_selected_components=None,
        #                   contamination=0.1, copy=True, whiten=False, svd_solver='full',
        #                   tol=0.0, iterated_power='auto',
        #                   weighted=True, standardization=False)

        self.pca = PCA(n_components=self.n_components,
                       random_state=self.random_state, n_selected_components=None,
                       contamination=0.1, copy=True, whiten=False, svd_solver='full',
                       tol=0.0, iterated_power='auto',
                       weighted=True, standardization=False)

        self.pca.fit(X=X_train)
        self.n_components = self.pca.n_components

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
        self.y_pred = self.pca.predict(X=X_test)
        self.cm = confusion_matrix(y_true=y_test, y_pred=self.y_pred)
        print(f'pca.cm: \n=> predicted 0 and 1 (0 stands for normals and 1 for anomalies)\n{self.cm}')
        self.y_scores = self.pca.predict_proba(X_test)[:, 1]  # y_scores=positive_class, predict X as 1 (anomaly).
        # roc_curve(): When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1}, ``pos_label`` is set to 1,
        #  otherwise an error will be raised.
        self.auc = roc_auc_score(y_true=y_test, y_score=self.y_scores)
        print(f'auc: {self.auc}')
