"""MO_GAAL class

"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE
from pyod.models.mo_gaal import MO_GAAL
from sklearn.metrics import confusion_matrix, roc_auc_score


class MO_GAALDetector(MO_GAAL):

    def __init__(self, verbose=True, random_state=42):
        # super(MO_GAALDetector, self).__init__()
        self.verbose = verbose
        self.random_state = random_state
        self.name = 'MO_GAAL'

    def fit(self, X_train, y_train=None):
        # if len(X_train[0]) <= 1:
        #     raise ValueError(f'len(X_train[0]) <=1')
        #     return -1

        self.mo_gaal = MO_GAAL(k=10, stop_epochs=20, lr_d=0.01, lr_g=0.0001,
                               decay=1e-6, momentum=0.9, contamination=0.1)

        self.mo_gaal.fit(X=X_train)
        print(f'thres: {self.mo_gaal.threshold_}')

        return self

    def test(self, X_test, y_test):
        """
            pyod.models.base.BaseDetector.labels_: The binary labels of the training data.
                                                0 stands for inliers and 1 for outliers/anomalies.
        :param X_test:
        :param y_test:
        :return:
        """
        # Predict if a particular sample is an outlier or not (0 is inlier and 1 for outlier).
        self.y_pred = self.mo_gaal.predict(X=X_test)
        self.cm = confusion_matrix(y_true=y_test, y_pred=self.y_pred)
        print(f'mo_gaal.cm: \n=> predicted 0 and 1 (0 stands for normals and 1 for anomalies)\n{self.cm}')
        self.y_scores = self.mo_gaal.predict_proba(X_test)[:, 1]  # y_scores=positive_class, predict X as 1 (anomaly).
        # roc_curve(): When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1}, ``pos_label`` is set to 1,
        #  otherwise an error will be raised.
        self.auc = roc_auc_score(y_true=y_test, y_score=self.y_scores)
        print(f'auc: {self.auc}')
