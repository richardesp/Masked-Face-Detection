import unittest
from models import model_01
from utils import config
import tensorflow as tf
from utils import get_model_size


class MyTestCase(unittest.TestCase):
    @staticmethod
    def test_creation_model():
        config_file = config.process_config("../configs/maskedfacepeople_config_vgg16.json")
        class_model = model_01.Model(config_file)
        model = class_model.build_model()

        # Printing model summary for check if config file was loaded correctly
        model.summary()

        # For check if any GPU is available
        print(tf.config.list_physical_devices("GPU"))
        batch_size = 128
        print(f"{get_model_size.keras_model_memory_usage_in_bytes(model, batch_size) / 1000000000} GB")


if __name__ == '__main__':
    unittest.main()