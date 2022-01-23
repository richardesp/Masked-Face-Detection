"""
VGG16 Convolutional Network for masked face people classification

:author:
    Ricardo Espantaleón
"""

from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from utils import get_learning_rate
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten


class Model(BaseModel):
    def __init__(self, config):
        """
        Parametrized constructor for initialize config file variable

        :param config: Configuration file where read all models params required for the creation
        """
        super(Model, self).__init__(config)
        self.build_model(compilation=True)

    def build_model(self, compilation=False):
        """
        Implemented function of the abstract class base_model, to create the specific instance of the model

        This specific implementation is based in typical VGG16 architecture

        A convolutional block consist: in 2/3 convolutional layers and a max Pooling layer

        :param compilation: Boolean to compile the resulting model
        """
        self.model = Sequential()

        # Adding input_layers to te sequential model
        for input_layer in self.config.model.input_layers:
            self.model.add(tf.keras.Input(shape=tuple(map(int, input_layer.shape.split(', ')))))

        # Adding convolutional blocks to the sequential model
        for conv_block in self.config.model.conv_blocks:

            # Each block can contain between 2 and 3 layers of conv
            for num in range(conv_block.num_conv_layers):
                self.model.add(
                    Conv2D(filters=conv_block.filters,
                           kernel_size=tuple(map(int, conv_block.kernel_size.split(', '))),
                           padding=conv_block.padding,
                           activation=conv_block.activation))

            # Final maxPool layer in each convolution block
            self.model.add(MaxPool2D(pool_size=tuple(map(int, conv_block.pool_size.split(', '))),
                                     strides=tuple(map(int, conv_block.strides.split(', ')))))

        self.model.add(Flatten())

        # Adding final dense layers
        for dense_layer in self.config.model.dense_layers:
            self.model.add(Dense(units=dense_layer.units, activation=dense_layer.activation))

        # If user indicated that wants the model compiled
        if compilation:

            # If learning rate param was specified by the user
            if self.config.model.learning_rate is not None:

                # We need to create a specified model.optimizer for call the __init__ function
                if self.config.model.optimizer == "Adam":
                    optimizer = Adam(learning_rate=self.config.model.learning_rate)

                self.model.compile(optimizer=optimizer,
                                   loss=self.config.model.loss,
                                   metrics=['accuracy', get_learning_rate.get_lr_metric(optimizer)])

            # If learning rate wasn't specified, we can use default constructor
            else:

                # We need to create a specified model.optimizer for call the __init__ function
                if self.config.model.optimizer == "Adam":
                    optimizer = Adam()

                self.model.compile(optimizer=optimizer,
                                   loss=self.config.model.loss,
                                   metrics=['accuracy', get_learning_rate.get_lr_metric(optimizer)])

        # Printing final model summary
        print(self.model.summary())
