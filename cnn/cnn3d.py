import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Model, Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import StratifiedShuffleSplit

from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils
from keras.regularizers import l2, activity_l2

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, merge
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_auc_score, roc_curve, auc


from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing
from skimage.transform import resize

import seaborn as sns
try:
    sns.set_style("ticks")
    # sns.set_style({"axes.axisbelow": False})
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
except:
    print("sns problem")

from ..utils.preprocess import *
from ..utils.io import *


class cnn3d:
    def __init__(self):
        self.__model__ = None

    def load_data(self, ids, base_dir = "./sample_images/"):
        self.base_dir = base_dir
        ys = get_labels(ids)
        xs = get_all_images(x=300, y=300, z=300, base_dir=base_dir, resample=True, all_scan_ids=ids)
        train_x, train_y, test_x, test_y = split_train_test(xs, ys)
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def get_model(self):
        if self.__model__ is None:
            raise ValueError('No model was trained/loaded...')
        return self.__model__

    def save_keras_model(self, filename, weight_file):
        """
        :param model:
        :param filename: e.g. 'model_sim5_moreData_architecture.json'
        :param weight_file: e.g. 'model_sim5_moreData_weights.h5'
        :return:
        """
        model = self.__model__
        json_string = model.to_json()
        with open(filename, 'w') as f:
            f.write(json_string)
        model.save_weights(weight_file)

    def load_keras_model(self, filename, weight_file):
        """
        :param filename: see above
        :param weight_file: see above
        :return: keras model obj
        """
        with open(filename, 'r') as f:
            self.__model__ = model_from_json(f.read())
        self.__model__.load_weights(weight_file)


    def load_weights(self, weight_file):
        self.__model__.load_weights(weight_file)


    def predict_stats(self, model, test_x, norm=1.):
        pred_ = model.predict_proba(test_x / norm)
        print("The mean predictions are", np.mean(pred_, axis=0))
        print("The std dev of the predictions are", np.std(pred_, axis=0))

    def init_model(self,
                   input_shape=(300, 300, 300), nb_classes=2, loss_func='binary_crossentropy',
                   lr=0.01, decay=1e-6, momentum=0.9, optimizer=None, **kwarg):
        print("Initializing a ConvNet model...")
        # __define_model__ is a template to be implemented in every model
        model = self.__define_model__(input_shape=input_shape, nb_classes=nb_classes, **kwarg)
        if optimizer is None:
            #optimizer = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
            #optimizer = RMSprop(lr=0.0001, rho=momentum, epsilon=decay)
            optimizer = RMSprop(lr=lr, rho=momentum, epsilon=decay)
        model.compile(loss=loss_func, optimizer=optimizer, metrics=["accuracy"])
        #model.fit_generator(construct_biased_dataset_generator(c_SamplesPerPatient, c_Bias, c_Patch[0], c_Images[0:2],
        #                                                       c_Segs[0:2]), samples_per_epoch=4000,
        #                    nb_epoch=c_Iterations[0], verbose=1, nb_worker=4)

        self.__model__ = model
        self.__nb_classes__ = nb_classes

    def __define_model__(self,
               input_shape=(300, 300, 300), nb_classes=2,
               n_filters=[32,32], filter_sizes=[6,6], filter_stride1=[1,1],
               border_modes=['valid','same'], pool_sizes=[2,2], filter_drops=[0.25, 0.25],
               dense_ns=[256,64], dense_drops=[0.5, 0.5], pool_method="max"):

        model = Sequential()
        model.add(Convolution3D(nb_filter=n_filters[0], len_conv_dim1=filter_sizes[0], len_conv_dim2=filter_sizes[0],
                                len_conv_dim3=filter_sizes[0], init='normal', W_regularizer=l2(0.4),
                                border_mode=border_modes[0], input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Dropout(filter_drops[0]))

        model.add(Flatten())
        model.add(Dense(dense_ns[0],init='normal'))
        model.add(Activation('relu'))
        model.add(Dropout(dense_drops[0]))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        return model

    def train(self, batch_size=128, early_stop=10, nb_epoch=5, norm_x=1.):
        model = self.__model__
        nb_classes = self.__nb_classes__
        train_x = self.train_x
        train_y = self.train_y
        test_x = self.test_x
        test_y = self.test_y
        if len(train_y.shape) == 1:
            # add a new axis to y
            train_y = np_utils.to_categorical(train_y, nb_classes)

        if len(test_y.shape) == 1:
            # add a new axis to y
            test_y = np_utils.to_categorical(test_y, nb_classes)

        if train_y.dtype != 'float64':
            train_y = train_y.astype('float64')
        if test_y.dtype != 'float64':
            test_y = test_y.astype('float64')

        # normalize x:
        if not isinstance(train_x, list):
            if train_x.dtype != 'float64':
                train_x = train_x.astype('float64')
            if test_x.dtype != 'float64':
                test_x = test_x.astype('float64')

            train_x = train_x / norm_x
            test_x = test_x / norm_x
        else:
            for i, tx in enumerate(train_x):
                if tx.dtype != 'float64':
                    train_x[i] = tx.astype('float64')
                train_x[i] = tx / norm_x
            for i, tx in enumerate(test_x):
                if tx.dtype != 'float64':
                    test_x[i] = tx.astype('float64')
                test_x[i] = tx / norm_x

        print("Training the ConvNet model...")

        checkpointer = ModelCheckpoint(filepath='check_points.hdf5', verbose=1, save_best_only=True)
        early_stop = EarlyStopping(patience=early_stop, verbose=1)
        m_history = model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                              validation_data=(test_x, test_y), callbacks=[checkpointer, early_stop])

        # m_history = model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True)
        # objective_score = model.evaluate(test_x, test_y, batch_size=batch_size)
        # print "The objective score on test data is", objective_score


        if nb_classes == 2:
            predict_train_y = model.predict_proba(train_x)[:, 1]
            predict_test_y = model.predict_proba(test_x)[:, 1]
            fpr, tpr, thresh = roc_curve(train_y[:, 1], predict_train_y)
            roc_auc = auc(fpr, tpr)
            fpr_test, tpr_test, thresh_test = roc_curve(test_y[:, 1], predict_test_y)
            roc_auc_test = auc(fpr_test, tpr_test)
            print('The training AUC score is {0}, and the test AUC score is: {1}'.format(roc_auc, roc_auc_test))
            print(
                'The mean prediction for training data is {0}, and for test data is: {1}'.format(
                    np.mean(predict_train_y),
                    np.mean(predict_test_y))
            )

        self.__m_history__ = m_history


def split_train_test(x, y, ratio=0.2, random_state=1234):
    sss = StratifiedShuffleSplit(y, test_size=ratio, random_state=random_state)
    for train_index, test_index in sss:
        break
    train_x, train_y = x[train_index], y[train_index]
    test_x, test_y = x[test_index], y[test_index]
    return train_x.astype('float32'), train_y.astype('float32'), test_x.astype('float32'), test_y.astype('float32')
