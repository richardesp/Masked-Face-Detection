import unittest
from trainers import vgg_trainer
from utils import config


class MyTestCase(unittest.TestCase):
    def test_load_callbacks(self):
        config_file = config.process_config("../configs/maskedfacepeople_config_vgg16.json")
        trainer = vgg_trainer.ModelTrainer(None, None, None, config=config_file)
        callbacks = trainer.get_callbacks()
        self.assertGreater(callbacks.__len__(), 0, "Callbacks list is empty")
        print(f"{callbacks.__len__()} callbacks were added to the trainer")


if __name__ == '__main__':
    unittest.main()
