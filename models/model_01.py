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
            input_tensor=self.config.input_tensor,
            input_shape=self.config.input_shape,
            pooling=self.config.pooling,
            classes=self.config.classes,
            classifier_activation=self.config.classifier.activation,
        )
