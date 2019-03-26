# -*- coding: utf-8 -*-
# @Time    : 2018/7/19 16:15
# @Author  : Suzhenyu
# @File    : AlexNet.py
# @Email   : suzhenyu@qiyi.com
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dropout

def AlexNet(FLAGS):
    model = Sequential()
    input_shape = (FLAGS.normal_size,FLAGS.normal_size,FLAGS.channels)
    model.add(Convolution2D(96, (11, 11), strides=(4, 4), input_shape=input_shape, padding='valid', activation='relu',
                     kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Convolution2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Convolution2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Convolution2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Convolution2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(FLAGS.classes, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])

    return model
