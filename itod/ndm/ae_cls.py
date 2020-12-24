"""Data process includes pcap2flows, flows2features

"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE

from sklearn.metrics import confusion_matrix, roc_auc_score
# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader

from itod.utils.tool import execute_time

"""Using Auto Encoder with Outlier Detection
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.regularizers import l2
# from keras.losses import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array

from pyod.utils.utility import check_parameter
from pyod.utils.stat_models import pairwise_distances_no_broadcast

from pyod.models.base import BaseDetector

import torch
from torch import nn


class autoencoder(nn.Module):

    def __init__(self, in_dim=0, hid_dim=0, lat_dim=0, p=0.2):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            # nn_pdf.ReLU(True),
            nn.LeakyReLU(True),
            # nn_pdf.Dropout(p=p),
            # nn_pdf.Linear(hid_dim, hid_dim),
            # # nn_pdf.ReLU(True),
            # nn_pdf.LeakyReLU(True),
            # nn_pdf.Linear(hid_dim, hid_dim),
            # # nn_pdf.ReLU(True),
            # nn_pdf.LeakyReLU(True),
            nn.Linear(hid_dim, lat_dim),
            # nn_pdf.ReLU(True),
            nn.LeakyReLU(True),
            # nn_pdf.Dropout(p=p),
        )
        self.decoder = nn.Sequential(
            nn.Linear(lat_dim, hid_dim),
            # nn_pdf.ReLU(True),
            nn.LeakyReLU(True),
            # nn_pdf.Dropout(p=p),
            # nn_pdf.Linear(hid_dim, hid_dim*2),
            # # nn_pdf.ReLU(True),
            # nn_pdf.LeakyReLU(True),
            # nn_pdf.Linear(hid_dim*2, hid_dim),
            # # nn_pdf.ReLU(True),
            # nn_pdf.LeakyReLU(True),
            nn.Linear(hid_dim, in_dim),
            nn.LeakyReLU(True),
            # nn_pdf.Tanh()
            # nn_pdf.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def predict(self, X=None):
        self.eval()  # Sets the module in evaluation mode.
        y = self.forward(torch.Tensor(X))

        return y.detach().numpy()


# noinspection PyUnresolvedReferences,PyPep8Naming,PyTypeChecker
class AutoEncoder(BaseDetector):

    def __init__(self, hidden_neurons=None,
                 hidden_activation='leakyrelu', output_activation='leakyrelu',
                 loss=None, optimizer='adam', lr=1e-3,
                 epochs=100, batch_size=32, dropout_rate=0.2,
                 l2_regularizer=0.1, validation_size=0.1, preprocessing=True,
                 verbose=1, random_state=None, contamination=0.1):
        super(AutoEncoder, self).__init__(contamination=contamination)
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.random_state = random_state
        self.lr = lr

        # # default values
        # if self.hidden_neurons is None:
        #     self.hidden_neurons = [64, 32, 32, 64]
        #
        # # Verify the network design is valid
        # if not self.hidden_neurons == self.hidden_neurons[::-1]:
        #     print(self.hidden_neurons)
        #     raise ValueError("Hidden units should be symmetric")

        self.hidden_neurons_ = self.hidden_neurons

        check_parameter(dropout_rate, 0, 1, param_name='dropout_rate',
                        include_left=True)

    def _build_model(self, X, y, hidden_neurons=''):

        self.model = autoencoder(in_dim=self.hidden_neurons[0], hid_dim=self.hidden_neurons[1],
                                 lat_dim=self.hidden_neurons[2], p=self.dropout_rate)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.l2_regularizer)  # weight_decay=1e-5

        # decay the learning rate
        decayRate = 0.99
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

        # # re-seed to make DataLoader() will have the same result.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # random.seed(42)
        # torch.manual_seed(42)
        # torch.cuda.manual_seed(42)
        # np.random.seed(42)
        val_size = int(self.validation_size * len(y))
        train_size = len(y) - val_size
        print(f'train_size: {train_size}, val_size: {val_size}')
        train_dataset, val_dataset = torch.utils.data.random_split(list(zip(X, y)), [train_size, val_size])

        # dataloader = DataLoader((X, y), batch_size=self.batch_size, shuffle=True)
        dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        # dataloader= val_dataloader
        for epoch in range(self.epochs):
            train_loss = 0
            for s, data in enumerate(dataloader):
                X_batch, y_batch = data
                # ===================forward=====================
                output = self.model(X_batch.float())
                loss = criterion(output, y_batch.float())
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.data
                # print(epoch, s, loss.data)
            # if epoch % 10 == 0:
            #     lr_scheduler.step()
            # ===================log========================
            with torch.no_grad():
                val_loss = 0
                for t, data in enumerate(val_dataloader):
                    X_batch, y_batch = data
                    output = self.model(X_batch.float())
                    loss = criterion(output, y_batch.float())
                    val_loss += loss.data
                print('epoch [{}/{}], loss:{:.4f}, eval: {:.4f}, lr: {}'
                      .format(epoch + 1, self.epochs, train_loss / (s + 1), val_loss / (t + 1),
                              lr_scheduler.get_last_lr()))

            # if epoch % 10 == 0:
            #     pic = to_img(output_data.cpu().data)
            #     save_image(pic, './mlp_img/image_{}.png'.format(epoch))

        # self.ndm.eval()  # Sets the module in evaluation mode.

        return self.model

    # noinspection PyUnresolvedReferences
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
        X = check_array(X)
        self._set_n_classes(y)

        # Verify and construct the hidden units
        self.n_samples_, self.n_features_ = X.shape[0], X.shape[1]

        # Standardize data for better performance
        if self.preprocessing:
            self.scaler_ = StandardScaler()
            X_norm = self.scaler_.fit_transform(X)
        else:
            X_norm = np.copy(X)

        # Shuffle the data for validation as Keras do not shuffling for
        # Validation Split
        np.random.shuffle(X_norm)

        # Validate and complete the number of hidden neurons
        if np.min(self.hidden_neurons) > self.n_features_:
            raise ValueError("The number of neurons should not exceed "
                             "the number of features")
        # self.hidden_neurons_.insert(0, self.n_features_)

        # Calculate the dimension of the encoding layer & compression rate
        self.encoding_dim_ = np.median(self.hidden_neurons)
        self.compression_rate_ = self.n_features_ // self.encoding_dim_

        # # Build AE ndm & fit with X
        self.model_ = self._build_model(X_norm, X_norm, hidden_neurons=self.hidden_neurons)
        # self.history_ = self.model_.fit(X_norm, X_norm,
        #                                 epochs=self.epochs,
        #                                 batch_size=self.batch_size,
        #                                 shuffle=True,
        #                                 validation_split=self.validation_size,
        #                                 verbose=self.verbose).history

        # Reverse the operation for consistency
        # self.hidden_neurons_.pop(0)
        # Predict on X itself and calculate the reconstruction error as
        # the outlier scores. Noted X_norm was shuffled has to recreate
        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        pred_scores = self.model_.predict(X_norm)
        self.decision_scores_ = pairwise_distances_no_broadcast(X_norm,
                                                                pred_scores)
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
        # check_is_fitted(self, ['model_', 'history_'])
        X = check_array(X)

        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        # Predict on X and return the reconstruction errors
        pred_scores = self.model_.predict(X_norm)
        return pairwise_distances_no_broadcast(X_norm, pred_scores)


class AEDetector(AutoEncoder):

    def __init__(self, in_dim=10, hidden_neurons=None, batch_size=32, lr=1e-3,
                 verbose=True, div=4, random_state=42, params=''):
        super(AEDetector, self).__init__()
        self.in_dim = in_dim
        if hidden_neurons != None:
            self.hidden_neurons = hidden_neurons
        else:
            self.hidden_neurons = [in_dim, 16, 8, 16, in_dim]  # default value
        self.verbose = verbose
        self.random_state = random_state
        self.div = int(div)
        self.params = params
        self.batch_size = batch_size
        self.lr = lr

        self.name = 'AE'

    def check_gpu(self):

        # import tensorflow
        # print(tensorflow.__version__)
        #
        # import keras
        # print(keras.__version__)
        #
        # try:
        #     import tensorflow as tf
        #     print(tf.test.is_gpu_available())  # True/False
        #
        #     # Or only check for gpu's with cuda support
        #     print(tf.test.is_gpu_available(cuda_only=True))
        #
        #     # # confirm TensorFlow sees the GPU
        #     # from tensorflow.python.client import device_lib
        #     # assert 'GPU' in str(device_lib.list_local_devices())
        #     #
        #     # # confirm Keras sees the GPU (for TensorFlow 1.X + Keras)
        #     # from keras import backend
        #     # assert len(backend.tensorflow_backend._get_available_gpus()) > 0
        #
        #     # # confirm PyTorch sees the GPU
        #     # from torch import cuda
        #     # assert cuda.is_available()
        #     # assert cuda.device_count() > 0
        #     # print(cuda.get_device_name(cuda.current_device()))
        # except:
        #     pass
        pass

    @execute_time
    def fit(self, X_train, y_train=None):
        # self.check_gpu()
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
            hidden_activation='leakyrelu', output_activation='leakyrelu',
            optimizer='adam', lr=self.lr,
            epochs=20, batch_size=self.batch_size, dropout_rate=0.2,
            l2_regularizer=1e-3, validation_size=0.01, preprocessing=False,
            # we have done nomarlization before fit()
            verbose=1, random_state=self.random_state, contamination=0.1)

        print('--------AE Debug')
        # print(X_train.shape)      # not X_train.shape()
        print(self.ae)
        print(self.ae.get_params())
        self.ae.fit(X=X_train)
        print(f'thres: {self.ae.threshold_}')

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
        self.y_pred = self.ae.predict(X=X_test)
        self.cm = confusion_matrix(y_true=y_test, y_pred=self.y_pred)
        print(f'ae.cm: \n=> predicted 0 and 1 (0 stands for normals and 1 for anomalies)\n{self.cm}')
        self.y_scores = self.ae.predict_proba(X_test)[:, 1]  # y_scores=positive_class, predict X as 1 (anomaly).
        # roc_curve(): When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1}, ``pos_label`` is set to 1,
        #  otherwise an error will be raised.
        self.auc = roc_auc_score(y_true=y_test, y_score=self.y_scores)
        print(f'auc: {self.auc}, even y_pred may different, however, auc could be the same value')
