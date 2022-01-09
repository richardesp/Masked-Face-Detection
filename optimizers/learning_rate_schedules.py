"""
Class for create differents learning rate schedules for training models
I've implemented the following options:

    1. -> constant
    2. -> time_based_decay


:author: Ricardo Espantaleón
"""


class LearningRateSchedules:
    def __init__(self, initial_learning_rate=0.01):
        self.initial_learning_rate = initial_learning_rate

    def get_lr_time_based_decay(self):
        decay = self.initial_learning_rate

        def lr_time_based_decay(epoch, lr):
            return lr * 1 / (1 + decay * epoch)

        return lr_time_based_decay
