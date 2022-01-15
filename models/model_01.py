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
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten

class Model(BaseModel):
    def __init__(self, config):
        """
        Parametrized constructor for initialize config file variable

        :param config: Configuration file where read all models params required for the creation
        """
        super(Model, self).__init__(config)
        self.build_model()

    def build_model(self, compilation=False):
        """
        Implemented function of the abstract class base_model, to create the specific instance of the model

        :param compilation: Boolean to compile the resulting model
        """
        model = Sequential()
        model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())

        model.add(Dense(units=4096, activation="relu"))
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dense(units=2, activation="softmax"))

        self.model = model

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
