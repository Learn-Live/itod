"""IForest class

"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE
from pyod.models.iforest import IForest
from sklearn.metrics import confusion_matrix, roc_auc_score

from itod.utils.tool import execute_time


class IForestDetector(IForest):

    def __init__(self, n_estimators=100, max_samples='auto', random_state=42, verbose=True):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.verbose = verbose
        self.random_state = random_state

        self.name = 'IF'

    @execute_time
    def fit(self, X_train, y_train=None):
        # if len(X_train[0]) <= 1:
        #     raise ValueError(f'len(X_train[0]) <=1')
        #     return -1

        """

        n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

        """
        self.iforest = IForest(n_estimators=self.n_estimators,
                               max_samples=self.max_samples,
                               contamination=0.1,
                               max_features=1.,
                               bootstrap=False,
                               n_jobs=-1,
                               behaviour='deprecated',  # no use any more in sklean 0.24.
                               random_state=self.random_state,
                               verbose=0)
        # self.detector = self.iforest
        self.iforest.fit(X=X_train)

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
        self.y_pred = self.iforest.predict(X=X_test)
        self.cm = confusion_matrix(y_true=y_test, y_pred=self.y_pred)
        print(f'iforest.cm: \n=> predicted 0 and 1 (0 stands for normals and 1 for anomalies)\n{self.cm}')
        self.y_scores = self.iforest.predict_proba(X_test)[:, 1]  # y_scores=positive_class, predict X as 1 (anomaly).
        # roc_curve(): When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1}, ``pos_label`` is set to 1,
        #  otherwise an error will be raised.
        self.auc = roc_auc_score(y_true=y_test, y_score=self.y_scores)
        print(f'auc: {self.auc}')
