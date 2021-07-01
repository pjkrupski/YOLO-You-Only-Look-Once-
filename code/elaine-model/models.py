"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

import hyperparameters as hp


class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
       super(YourModel, self).__init__()

       # use the Adamax optimizer
       self.optimizer = tf.keras.optimizers.Adamax(
              learning_rate=0.001, beta_1=0.5, beta_2=0.9, epsilon=1e-07,
              name='Adamax')

       # TODO: Build your own convolutional neural network, using Dropout at
       #       least once. The input image will be passed through each Keras
       #       layer in self.architecture sequentially. Refer to the imports
       #       to see what Keras layers you can use to build your network.
       #       Feel free to import other layers, but the layers already
       #       imported are enough for this assignment.
       #
       #       Remember: Your network must have under 15 million parameters!
       #       You will see a model summary when you run the program that
       #       displays the total number of parameters of your network.
       #
       #       Remember: Because this is a 15-scene classification task,
       #       the output dimension of the network must be 15. That is,
       #       passing a tensor of shape [batch_size, img_size, img_size, 1]
       #       into the network will produce an output of shape
       #       [batch_size, 15].
       #
       #       Note: Keras layers such as Conv2D and Dense give you the
       #             option of defining an activation function for the layer.
       #             For example, if you wanted ReLU activation on a Conv2D
       #             layer, you'd simply pass the string 'relu' to the
       #             activation parameter when instantiating the layer.
       #             While the choice of what activation functions you use
       #             is up to you, the final layer must use the softmax
       #             activation function so that the output of your network
       #             is a probability distribution.
       #
       #       Note: Flatten is a very useful layer. You shouldn't have to
       #             explicitly reshape any tensors anywhere in your network.

       # first set of convolutions + max pooling 
       first_conv1 = tf.keras.layers.Conv2D(64, 5, activation='relu',)

       first_conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu')

       maxPool1 = tf.keras.layers.MaxPool2D( pool_size=(2, 2), strides=None)
       
       # suggested 0.0-0.1 then 0.1-0.25 orrr 0.2 orrr 0.5
       dropout1 = tf.keras.layers.Dropout(0.2, noise_shape=None, seed=None)

       # second set of convolutions + max pooling
       second_conv1 = tf.keras.layers.Conv2D(128, 3, activation='relu')

       second_conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu')

       maxPool2 = tf.keras.layers.MaxPool2D( pool_size=(2, 2), strides=None)

       dropout2 = tf.keras.layers.Dropout(0.2, noise_shape=None, seed=None)

       # third set of convolutions + max pooling
       third_conv1 = tf.keras.layers.Conv2D(256, 3, activation='relu')

       third_conv2 = tf.keras.layers.Conv2D(256, 3, activation='relu')

       maxPool3 = tf.keras.layers.MaxPool2D( pool_size=(2, 2), strides=None)

       dropout3 = tf.keras.layers.Dropout(0.2, noise_shape=None, seed=None)

       # fourth set of convolutions + max pooling
       fourth_conv1 = tf.keras.layers.Conv2D(512, 3, activation='relu')

       fourth_conv2 = tf.keras.layers.Conv2D(512, 3, activation='relu')

       maxPool4 = tf.keras.layers.MaxPool2D( pool_size=(2, 2), strides=None)

       dropout4 = tf.keras.layers.Dropout(0.2, noise_shape=None, seed=None)

       # fifth set of convolutions + max pooling
       fifth_conv1 = tf.keras.layers.Conv2D(512, 3, activation='relu')

       fifth_conv2 = tf.keras.layers.Conv2D(512, 3, activation='relu')

       maxPool5 = tf.keras.layers.MaxPool2D( pool_size=(2, 2), strides=None)

       dropout5 = tf.keras.layers.Dropout(0.2, noise_shape=None, seed=None)

       # reshape outputs for fully connected layers
       flatten = tf.keras.layers.Flatten()

       first_dense = tf.keras.layers.Dense(512, activation='relu')

       last_dense = tf.keras.layers.Dense(15, activation='softmax')
       
       self.architecture = [first_conv1, first_conv2, maxPool1, dropout1,
       second_conv1, second_conv2, maxPool2, dropout2, 
       third_conv1, third_conv2, maxPool3, dropout3,
       fourth_conv1, fourth_conv2, maxPool4,
       fifth_conv1, fifth_conv2, maxPool5,
       flatten, last_dense]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
       """ Loss function for the model. """

       # use tf.keras.losses.CategoricalCrossentropy as loss function for the network
       loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
       return loss

class VGGModel(tf.keras.Model):
    def __init__(self):
        super(VGGModel, self).__init__()

        # TODO: Select an optimizer for your network (see the documentation
        #       for tf.keras.optimizers)

        self.optimizer = tf.keras.optimizers.Adamax(
              learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
              name='Adamax')

        # Don't change the below:

        self.vgg16 = [
            # Block 1
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv1"),
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv2"),
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv1"),
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv2"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv1"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv2"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv3"),
            MaxPool2D(2, name="block3_pool"),
            # Block 4
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv3"),
            MaxPool2D(2, name="block4_pool"),
            # Block 5
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv3"),
            MaxPool2D(2, name="block5_pool")
        ]

        # TODO: Make all layers in self.vgg16 non-trainable. This will freeze the
        #       pretrained VGG16 weights into place so that only the classificaiton
        #       head is trained.

        for layer in self.vgg16:
               layer.trainable = False

        # TODO: Write a classification head for our 15-scene classification task.

        # reshape outputs for fully connected layers
        flatten = tf.keras.layers.Flatten()

        last_dense = tf.keras.layers.Dense(15, activation='softmax', use_bias=True,
        kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None)

        self.head = [flatten, last_dense]

        # Don't change the below:
        self.vgg16 = tf.keras.Sequential(self.vgg16, name="vgg_base")
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes the image through the network. """

        x = self.vgg16(x)
        x = self.head(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        # TODO: Select a loss function for your network (see the documentation
        #       for tf.keras.losses)

        # use tf.keras.losses.CategoricalCrossentropy as loss function for the network
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
        return loss
