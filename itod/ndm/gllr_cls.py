"""group-lasso for logistic regression class

"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE

from group_lasso import GroupLasso
from sklearn.metrics import confusion_matrix, roc_auc_score


class GLLRDetector(GroupLasso):

    def __init__(self, in_dim=10, hidden_neurons=None, verbose=True, div=4, random_state=42):

        self.in_dim = in_dim
        if hidden_neurons != None:
            self.hidden_neurons = hidden_neurons
        else:
            self.hidden_neurons = [in_dim, 16, 8, 16, in_dim]  # default value
        self.verbose = verbose
        self.random_state = random_state
        self.div = int(div)

        self.name = 'AE'

    def check_gpu(self):

        import tensorflow
        print(tensorflow.__version__)

        import keras
        print(keras.__version__)

        try:
            import tensorflow as tf
            print(tf.test.is_gpu_available())  # True/False

            # Or only check for gpu's with cuda support
            print(tf.test.is_gpu_available(cuda_only=True))

            # # confirm TensorFlow sees the GPU
            # from tensorflow.python.client import device_lib
            # assert 'GPU' in str(device_lib.list_local_devices())
            #
            # # confirm Keras sees the GPU (for TensorFlow 1.X + Keras)
            # from keras import backend
            # assert len(backend.tensorflow_backend._get_available_gpus()) > 0

            # # confirm PyTorch sees the GPU
            # from torch import cuda
            # assert cuda.is_available()
            # assert cuda.device_count() > 0
            # print(cuda.get_device_name(cuda.current_device()))
        except:
            pass

    def fit(self, X_train, y_train=None):
        self.check_gpu()
        # default: [64, 32, 32, 64], however, it require np.min(hidden_neurons) > the input size,
        # so it adds '8' into hidden_neurons.
        # latent_num = 8
        # if len(X_train[0]) < latent_num:
        #     latent_num = len(X_train[0])
        # self.ae = AutoEncoder(hidden_neurons=[64, 32, latent_num, 32, 64],
        #                       hidden_activation='relu', output_activation=None,
        #                       optimizer='adam',
        #                       epochs=100, batch_size=8, dropout_rate=0,
        #                       l2_regularizer=0.1, validation_size=0.1, postprocessing=False,     # we have done nomarlization before fit()
        #                       verbose=1, random_state=self.random_state, contamination=0.01)

        # feat_dim = len(X_train[0])
        # # adam_opt = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False, decay=0.1)
        # # [64, 32, latent_num, 32, 64]
        # # if feat_dim // 2 < 1:
        # #     print(f'feat_dim: {feat_dim}')
        # #     return -1
        #
        # if feat_dim // self.div < 1:
        #     hid = 1
        # else:
        #     hid = feat_dim // self.div
        # if feat_dim // (2*self.div) < 1:
        #     # hid // 4
        #     latent_num = 1
        # else:
        #     latent_num = feat_dim // (2*self.div)
        # print(f'feat_dim: {feat_dim}, hid: {hid}, latent_num: {latent_num}')
        # print(f'hidden_neurons: [{feat_dim}, {hid}, {latent_num}, {hid}, {feat_dim}]')

        self.ae = AutoEncoder(
            hidden_neurons=self.hidden_neurons,
            hidden_activation='relu', output_activation='sigmoid',
            optimizer='adam',
            epochs=10, batch_size=32, dropout_rate=0.2,
            l2_regularizer=0.1, validation_size=0.1, preprocessing=False,
            # we have done nomarlization before fit()
            verbose=1, random_state=self.random_state, contamination=0.1)

        print('--------AE Debug')
        # print(X_train.shape)      # not X_train.shape()
        print(self.ae)
        print(self.ae.get_params())
        self.ae.fit(X=X_train)
        print(f'thres: {self.ae.threshold_}')

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
        self.y_pred = self.ae.predict(X=X_test)
        self.cm = confusion_matrix(y_true=y_test, y_pred=self.y_pred)
        print(f'ae.cm: \n=> predicted 0 and 1 (0 stands for normals and 1 for anomalies)\n{self.cm}')
        self.y_scores = self.ae.predict_proba(X_test)[:, 1]  # y_scores=positive_class, predict X as 1 (anomaly).
        # roc_curve(): When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1}, ``pos_label`` is set to 1,
        #  otherwise an error will be raised.
        self.auc = roc_auc_score(y_true=y_test, y_score=self.y_scores)
        print(f'auc: {self.auc}')
