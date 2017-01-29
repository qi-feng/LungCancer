import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils

from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing

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

    def load_data(self, train_x, train_y, test_x, test_y):
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
                   input_shape=(4, 54, 54), nb_classes=2, loss_func='binary_crossentropy',
                   lr=0.01, decay=1e-6, momentum=0.9, optimizer=None, **kwarg):
        print("Initializing a ConvNet model...")
        # __define_model__ is a template to be implemented in every model
        model = self.__define_model__(input_shape=input_shape, nb_classes=nb_classes, **kwarg)
        if optimizer is None:
            optimizer = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
        model.compile(loss=loss_func, optimizer=optimizer, metrics=["accuracy"])

        self.__model__ = model
        self.__nb_classes__ = nb_classes

c_Filter = [32,64]
c_Kernel = [7,7]
c_Nodes = [128,256]
c_Patch = [7,13]
c_Bias = 0.5
c_SamplesPerPatient = 200
c_Iterations = [1000,200]

model_p1 = Sequential()
model_p1.add(Convolution3D(nb_filter=c_Filter[0], len_conv_dim1=c_Kernel[0], len_conv_dim2=c_Kernel[0], len_conv_dim3=c_Kernel[0], init='normal', W_regularizer=l2(0.4), border_mode='valid', input_shape=(1,c_Patch[0],c_Patch[0],c_Patch[0])))
model_p1.add(Activation('relu'))
model_p1.add(Dropout(0.5))

model_p1.add(Flatten())
model_p1.add(Dense(c_Nodes[0],init='normal'))
model_p1.add(Activation('relu'))
model_p1.add(Dropout(0.5))

model_p1.add(Dense(1))
model_p1.add(Activation('sigmoid'))

model_p1.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06))

model_p1.fit_generator(construct_biased_dataset_generator(c_SamplesPerPatient,c_Bias,c_Patch[0],c_Images[0:2],c_Segs[0:2]),samples_per_epoch=4000,nb_epoch=c_Iterations[0],verbose=1,nb_worker=4)

