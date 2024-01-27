"""
ResUNet architecture in Keras TensorFlow
"""

import os
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model


class ResUnet:
    def __init__(self, input_size=256):
        self.input_size = input_size

    def build_model(self):
        def stem_block(x, n_filter, strides):
            x_init = x

            # Conv 1
            x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv2D(n_filter, (3, 3), padding="same")(x)

            # Shortcut
            s = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
            s = tf.keras.layers.BatchNormalization()(s)

            # Add
            x = Add()([x, s])
            return x

        def conv_block(x, n_filter, strides):
            x_init = x

            # Conv 1
            x = tf.keras.layers.BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
            # Conv 2
            x = tf.keras.layers.BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv2D(n_filter, (3, 3), padding="same", strides=1)(x)

            # Shortcut
            s = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
            s = tf.keras.layers.BatchNormalization()(s)

            # Add
            x = Add()([x, s])
            return x

        def resnet_block(x, n_filter, strides, pool=True, stem=False):
            if stem:
                x1 = stem_block(x, n_filter, strides)
            else:
                x1 = conv_block(x, n_filter, strides)
            c = x1

            # Pooling
            if pool:
                x = MaxPooling2D((2, 2), (2, 2))(x1)
                return c, x
            else:
                return c

        n_filters = [32, 64, 128, 256]
        inputs = Input((self.input_size, self.input_size, 3))

        c0 = inputs
        c1 = resnet_block(c0, n_filters[0], strides=1, stem=True, pool=False)

        # Encoder
        c2 = resnet_block(c1, n_filters[1], strides=2, pool=False)
        c3 = resnet_block(c2, n_filters[2], strides=2, pool=False)
        # c4 = resnet_block(c3, n_filters[3], strides=2, pool=False)

        # Bridge
        b1 = resnet_block(c3, n_filters[3], strides=2, pool=False)

        # Decoder
        # d1 = Conv2DTranspose(n_filters[3], (3, 3), padding="same", strides=(2, 2))(b2)
        d1 = UpSampling2D((2, 2))(b1)
        d1 = Concatenate()([d1, c3])
        d1 = resnet_block(d1, n_filters[3], strides=1, pool=False)

        # d2 = Conv2DTranspose(n_filters[3], (3, 3), padding="same", strides=(2, 2))(d1)
        d2 = UpSampling2D((2, 2))(d1)
        d2 = Concatenate()([d2, c2])
        d2 = resnet_block(d2, n_filters[2], strides=1, pool=False)

        # d3 = Conv2DTranspose(n_filters[3], (3, 3), padding="same", strides=(2, 2))(d2)
        d3 = UpSampling2D((2, 2))(d2)
        d3 = Concatenate()([d3, c1])
        d3 = resnet_block(d3, n_filters[1], strides=1, pool=False)

        # output
        outputs = Conv2D(1, (1, 1), padding="same")(d3)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = Activation("sigmoid")(outputs)

        # Model
        model = Model(inputs, outputs)
        return model
