import os
import mne
import numpy as np
import scipy.io # To load .mat files
import matplotlib.pyplot as plt
import tensorflow as tf # I used Tensorflow-GPU 2.0.
import pandas as pd
import torch.nn as nn
import torch

class MSNN_rpts(tf.keras.Model):
    tf.keras.backend.set_floatx("float64")
    def __init__(self, num_channels):
        super(MSNN_rpts, self).__init__()
        self.C = num_channels
        self.fs = 16

        # Regularizer
        self.regularizer = tf.keras.regularizers.L1L2(l1=.001, l2=.01)

        # Activation functions
        self.activation = tf.keras.layers.LeakyReLU()
        self.softmax = tf.keras.layers.Softmax()
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        
        # Define convolutions
        conv = lambda D, kernel : tf.keras.layers.Conv2D(D, kernel, kernel_regularizer=self.regularizer)
        sepconv = lambda D, kernel : tf.keras.layers.SeparableConv2D(D, kernel, padding="same",
                                                                    depthwise_regularizer=self.regularizer,
                                                                    pointwise_regularizer=self.regularizer)
        
        # Spectral convoltuion
        self.conv0 = conv(4, (1, int(self.fs/2)))
        
        # Spatio-temporal convolution
        self.conv1t = sepconv(16, (1, 20))
        self.conv1s = conv(16, (self.C, 1))
        
        self.conv2t = sepconv(32, (1, 10))
        self.conv2s = conv(32, (self.C, 1))
        
        self.conv3t = sepconv(64, (1, 5))
        self.conv3s = conv(64, (self.C, 1))

        # Flatteninig
        self.flatten = tf.keras.layers.Flatten()

        # Dropout
        self.dropout = tf.keras.layers.Dropout(0.5)

        # Decision making
        self.dense = tf.keras.layers.Dense(2, activation="softmax", kernel_regularizer=self.regularizer)

    def call(self, x):
        # Extract spatio-spectral-temporal features
        x = self.activation(self.conv0(x)) # [batch, channel, 51, 4]
        
        x = self.activation(self.conv1t(x))
        f1 = self.activation(self.conv1s(x))
        
        x = self.activation(self.conv2t(x))
        f2 = self.activation(self.conv2s(x))

        x = self.activation(self.conv3t(x))
        input_feature = x # To estimate activation pattern
        f3 = self.activation(self.conv3s(x))
        
        feature = tf.concat((f1, f2, f3), -1) # Concat
        feature = tf.math.reduce_mean(feature, -2) # GAP
        feature = self.flatten(feature)
        feature = self.dropout(feature)
        prob = self.dense(feature)
        return tf.squeeze(prob), input_feature
        

import os
import mne
import numpy as np
import scipy.io # To load .mat files
import matplotlib.pyplot as plt
import tensorflow as tf # I used Tensorflow-GPU 2.0.
import pandas as pd
import torch.nn as nn
import torch

class MSNN(tf.keras.Model):
    tf.keras.backend.set_floatx("float64")
    def __init__(self, num_channels):
        super(MSNN, self).__init__()
        self.C = num_channels
        self.fs = 64

        # Regularizer
        self.regularizer = tf.keras.regularizers.L1L2(l1=.001, l2=.01)

        # Activation functions
        self.activation = tf.keras.layers.LeakyReLU()
        self.softmax = tf.keras.layers.Softmax()
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        
        # Define convolutions
        conv = lambda D, kernel : tf.keras.layers.Conv2D(D, kernel, kernel_regularizer=self.regularizer)
        sepconv = lambda D, kernel : tf.keras.layers.SeparableConv2D(D, kernel, padding="same",
                                                                    depthwise_regularizer=self.regularizer,
                                                                    pointwise_regularizer=self.regularizer)
        
        # Spectral convoltuion
        self.conv0 = conv(4, (1, int(self.fs/2)))
        
        # Spatio-temporal convolution
        self.conv1t = sepconv(16, (1, 20))
        self.conv1s = conv(16, (self.C, 1))
        
        self.conv2t = sepconv(32, (1, 10))
        self.conv2s = conv(32, (self.C, 1))
        
        self.conv3t = sepconv(64, (1, 5))
        self.conv3s = conv(64, (self.C, 1))

        # Flatteninig
        self.flatten = tf.keras.layers.Flatten()

        # Dropout
        self.dropout = tf.keras.layers.Dropout(0.5)

        # Decision making
        self.dense = tf.keras.layers.Dense(2, activation="softmax", kernel_regularizer=self.regularizer)

    def call(self, x):
        # Extract spatio-spectral-temporal features
        x = self.activation(self.conv0(x)) # [batch, channel, 51, 4]
        
        x = self.activation(self.conv1t(x))
        f1 = self.activation(self.conv1s(x))
        
        x = self.activation(self.conv2t(x))
        f2 = self.activation(self.conv2s(x))

        x = self.activation(self.conv3t(x))
        input_feature = x # To estimate activation pattern
        f3 = self.activation(self.conv3s(x))
        
        feature = tf.concat((f1, f2, f3), -1) # Concat
        feature = tf.math.reduce_mean(feature, -2) # GAP
        feature = self.flatten(feature)
        feature = self.dropout(feature)
        prob = self.dense(feature)
        return tf.squeeze(prob), input_feature
        