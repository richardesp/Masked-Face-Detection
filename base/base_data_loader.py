"""
Abstract class for load a dataset

:author: Ricardo Espantaleón Pérez
"""


class BaseDataLoader:
    def __init__(self, config):
        """
        Parameterized constructor to load the configuration file of the instantiated model

        :param config: config file where to load all params for load a specified dataset
        """
        self.config = config

    def get_train_data(self):
        pass

    def get_test_data(self):
        pass
