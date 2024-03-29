"""
Abstract class for create the model

:author: Ricardo Espantaleón Pérez
"""


class BaseModel:
    def __init__(self, config):
        """
        Parameterized constructor to load the configuration file of the instantiated model

        :param config: config file where to load all params for the specified model
        """
        self.config = config
        self.model = None

    def save(self, checkpoint_path):
        """
        Function that saves the instantiated model into the specified path, given in the
        config file

        :param checkpoint_path: Path where to save the specified checkpoint
        """
        if self.model is None:
            raise Exception("You have to build the model first!")

        print("Saving model...")
        self.model.save(checkpoint_path)
        print("Model saved")

    def load(self, checkpoint_path):
        """
        Function that load the latest checkpoint from the experiment path defined in the config file

        :param checkpoint_path: Path where to save the specified checkpoint
        """
        if self.model is None:
            raise Exception("You have to build the model first")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    def build_model(self):
        """
        Abstract method for build a specified model
        """
        raise NotImplementedError

    def get_model(self):
        """
        Getter function for return the instanced model

        :return: The instanced model
        """
        if self.model is None:
            raise Exception("You have to build the model first!")

        return self.model
