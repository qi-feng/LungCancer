import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils

import cv2
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

from ..utils import preprocess



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


    # save model:
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

