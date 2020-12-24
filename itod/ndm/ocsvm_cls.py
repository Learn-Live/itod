"""One-Class SVM (ocsvm) class

"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE

from pyod.models.ocsvm import OCSVM
from sklearn.metrics import confusion_matrix, roc_auc_score

from itod.utils.tool import execute_time


class OCSVMDetector(OCSVM):

    def __init__(self, gamma='auto', kernel='rbf', verbose=True, random_state=42):
        self.gamma = gamma
        self.kernel = kernel
        self.verbose = verbose

        self.name = "OCSVM"
        self.random_state = random_state

    @execute_time
    def fit(self, X_train, y_train=None):
        # if len(X_train[0]) <= 1:
        #     raise ValueError(f'len(X_train[0]) <=1')
        #     return -1

        self.ocsvm = OCSVM(gamma=self.gamma, kernel=self.kernel, degree=3, coef0=0.0,
                           tol=1e-3, nu=0.5, shrinking=True, cache_size=200,
                           verbose=False, max_iter=-1, contamination=0.1)  # OCSVM does not have random_state parameter.
        self.ocsvm.fit(X=X_train)
        print(f'thres: {self.ocsvm.threshold_}')
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
        self.y_pred = self.ocsvm.predict(X=X_test)
        self.cm = confusion_matrix(y_true=y_test, y_pred=self.y_pred)
        print(f'ocsvm.cm: \n=> predicted 0 and 1 (0 stands for normals and 1 for anomalies)\n{self.cm}')
        self.y_scores = self.ocsvm.predict_proba(X_test)[:, 1]  # y_scores=positive_class, predict X as 1 (anomaly).
        # roc_curve(): When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1}, ``pos_label`` is set to 1,
        #  otherwise an error will be raised.
        self.auc = roc_auc_score(y_true=y_test, y_score=self.y_scores)
        print(f'auc: {self.auc}')
