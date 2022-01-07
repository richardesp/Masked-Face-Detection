"""
Abstract class for create the model

:author:
    Ricardo Espantaleón Pérez
"""


class BaseModel:
    """
    Parameterized constructor to load the configuration file of the instantiated model
    :param config: config file where to load all params for the specified model
    """
    def __init__(self, config):
        self.config = config
        self.model = None

    """
    Function that saves the instantiated model into the specified path, given in the
    config file
    
    :param checkpoint_path: Path where to save the specified checkpoint
    """
    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first!")

        print("Saving model...")
        self.model.save_weights(checkpoint_path)
        print("Model saved")

    """
    Function that load latest checkpoint from the experiment path defined in the config file
    
    :param checkpoint_path: Path where to save the specified checkpoint
    """
    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    def build_model(self):
        raise NotImplementedError
