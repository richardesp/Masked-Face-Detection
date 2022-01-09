"""
VGG16 Convolutional Network for masked face people classification

:author:
    Ricardo Espantaleón
"""

from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from utils import get_learning_rate


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
        self.model = tf.keras.applications.VGG16(
            include_top=self.config.model.include_top,
            weights=self.config.model.weights,
            input_tensor=self.config.model.input_tensor,
            # Maximum size given my gpu
            input_shape=tuple(map(int, self.config.model.input_shape.split(', '))),
            pooling=self.config.model.pooling,
            classes=self.config.model.classes,
            classifier_activation=self.config.model.classifier_activation,
        )

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

                self.model.compile(optimizer=optimizer, loss=self.config.model.loss,
                                   metrics=['accuracy', get_learning_rate.get_lr_metric(optimizer)])
