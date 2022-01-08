"""
VGG16 Convolutional Network for masked face people classification

:author:
    Ricardo Espantaleón
"""

from base.base_model import BaseModel
import tensorflow as tf


class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.build_model()

    def build_model(self):
        model = tf.keras.applications.VGG16(
            include_top=self.config.model.include_top,
            weights=self.config.model.weights,
            input_tensor=self.config.model.input_tensor,
            # Maximum size given my gpu
            input_shape=tuple(map(int, self.config.model.input_shape.split(', '))),
            pooling=self.config.model.pooling,
            classes=self.config.model.classes,
            classifier_activation=self.config.model.classifier_activation,
        )

        return model
