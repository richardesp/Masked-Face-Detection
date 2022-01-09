"""
Utility for see learning rate in every epoch

:author: Ricardo Espantaleón
"""


def get_lr_metric(optimizer):
    """
    Function that given an optimizer, return the current learning rate

    :param optimizer: Optimizer from which it takes its learning rate
    :return: Current learning rate
    """

    def lr(y_true, y_pred):
        return optimizer.lr

    return lr
