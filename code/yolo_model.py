import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten, LeakyReLU
import numpy as np


class YoloModel(tf.keras.Model):
    def __init__(self):
        super(YoloModel, self).__init__()

        # learning rate -> start at 10^-3, slowly raise to 10^-2
        #               -> raise to 10^-2 for 75 epochs
        #               -> lower to 10^-3 for 30 epochs
        #               -> lower to 10^-4 for 30 epochs

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.architecture = [
            # block 1
            Conv2D(64, 7, strides=(2, 2), padding='same'),
            LeakyReLU(alpha=0.1),
            MaxPool2D(pool_size=(2, 2), strides=2),
            # block 2
            Conv2D(192, 3, padding='same'),
            LeakyReLU(alpha=0.1),
            MaxPool2D(pool_size=(2, 2), strides=2),
            # block 3
            Conv2D(128, 1, padding='same'),
            LeakyReLU(alpha=0.1),
            Conv2D(256, 3, padding='same'),
            LeakyReLU(alpha=0.1),
            Conv2D(256, 1, padding='same'),
            LeakyReLU(alpha=0.1),
            Conv2D(512, 3, padding='same'),
            LeakyReLU(alpha=0.1),
            MaxPool2D(pool_size=(2, 2), strides=2),
            # block 4
            Conv2D(256, 1, padding='same'),
            LeakyReLU(alpha=0.1),
            Conv2D(512, 3, padding='same'),  # 1
            LeakyReLU(alpha=0.1),
            Conv2D(256, 1, padding='same'),
            LeakyReLU(alpha=0.1),
            Conv2D(512, 3, padding='same'),  # 2
            LeakyReLU(alpha=0.1),
            Conv2D(256, 1, padding='same'),
            LeakyReLU(alpha=0.1),
            Conv2D(512, 3, padding='same'),  # 3
            LeakyReLU(alpha=0.1),
            Conv2D(256, 1, padding='same'),
            LeakyReLU(alpha=0.1),
            Conv2D(512, 3, padding='same'),  # 4
            LeakyReLU(alpha=0.1),
            Conv2D(512, 1, padding='same'),
            LeakyReLU(alpha=0.1),
            Conv2D(1024, 3, padding='same'),
            LeakyReLU(alpha=0.1),
            MaxPool2D(pool_size=(2, 2), strides=2),
            # block 5
            Conv2D(512, 1, padding='same'),
            LeakyReLU(alpha=0.1),
            Conv2D(1024, 3, padding='same'),  # 1
            LeakyReLU(alpha=0.1),
            Conv2D(512, 1, padding='same'),
            LeakyReLU(alpha=0.1),
            Conv2D(1024, 3, padding='same'),  # 2
            LeakyReLU(alpha=0.1),
            Conv2D(1024, 3, padding='same'),
            LeakyReLU(alpha=0.1),
            Conv2D(1024, 3, strides=(2, 2), padding='same'),
            LeakyReLU(alpha=0.1),
            # block 6
            Conv2D(1024, 3, padding='same'),
            Conv2D(1024, 3, padding='same'),
            # block 7
            Flatten(),
            Dense(4096),
            Dropout(rate=0.5),
            Dense(1000, activation='softmax')  # must use softmax to use cross-entropy
        ]

    def call(self, x):
        for layer in self.architecture:
            x = layer(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
        return loss
