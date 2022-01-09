"""
Abstract class for train the model

:author: Ricardo Espantaleón Pérez
"""


class BaseTrain(object):
    def __init__(self, model, training_data, validation_data, config):
        """
        Parameterized constructor to load the configuration file of the instantiated model

        :param model: model to train
        :param training_data: train dataset for train the instantiated model
        :param validation_data: testing dataset for test the final trained model
        :param config: config file where to load all params for train the model
        """
        self.model = model
        self.training_data = training_data
        self.validation_data = validation_data
        self.config = config

    def train(self):
        """
        Abstract method for train a specified model
        """
        raise NotImplementedError

    def get_callbacks(self) -> list:
        return self.callbacks
