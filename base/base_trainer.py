"""
Abstract class for train the model

:author:
    Ricardo Espantaleón Pérez
"""


class BaseTrain(object):
    """
    Parameterized constructor to load the configuration file of the instantiated model
    :param model: model to train
    :param data_train: train dataset for train the instantiated model
    :param data_test: testing dataset for test the final trained model
    :param config: config file where to load all params for train the model
    """

    def __init__(self, model, data_train, data_test, config):
        self.model = model
        self.train_data = data_train
        self.test_data = data_test
        self.config = config

    def train(self):
        raise NotImplementedError
