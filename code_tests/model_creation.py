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
        vgg_model = class_model.build_model()

        # For check if any GPU is available
        print(tf.config.list_physical_devices("GPU"))
        # Batch_size = 10
        print(f"{get_model_size.keras_model_memory_usage_in_bytes(vgg_model, 10) / 1000000000} GB")

        vgg_model.summary()


if __name__ == '__main__':
    unittest.main()
