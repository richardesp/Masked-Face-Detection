"""
Instantiated class for load the dataset for masked face people detection

:author: Ricardo Espantaleón
"""

from base.base_data_loader import BaseDataLoader
from keras.preprocessing.image import ImageDataGenerator


class DataLoader(BaseDataLoader):
    def __init__(self, config, data_dir):
        super(DataLoader, self).__init__(config)
        self.data_dir = data_dir
        self.training_data = None
        self.validation_data = None

        self.train_datagen = ImageDataGenerator(validation_split=self.config.data_loader.validation_split,
                                                # Splits the data into training (80%) and validation (20%)
                                                rescale=1. / 255,  # Normalizing all pixels values
                                                rotation_range=self.config.data_loader.rotation_range,
                                                # rotate the images
                                                width_shift_range=self.config.data_loader.width_shift_range,
                                                height_shift_range=self.config.data_loader.height_shift_range,
                                                zoom_range=self.config.data_loader.zoom_range,
                                                horizontal_flip=self.config.data_loader.horizontal_flip,
                                                # add new pixels when the image is rotated or shifted
                                                fill_mode=self.config.data_loader.fill_mode)

        self.load_training_data()
        self.load_validation_data()

    def load_training_data(self):
        self.training_data = self.train_datagen.flow_from_directory(
            self.data_dir,
            target_size=tuple(map(int, self.config.data_loader.target_size.split(', '))),
            batch_size=self.config.trainer.batch_size,
            color_mode=self.config.data_loader.color_mode,  # for coloured images
            class_mode=self.config.data_loader.class_mode,
            seed=self.config.data_loader.seed,  # to make the result reproducible
            subset='training')  # Specify this is training set

    def load_validation_data(self):
        self.validation_data = self.train_datagen.flow_from_directory(
            self.data_dir,
            target_size=tuple(map(int, self.config.data_loader.target_size.split(', '))),
            batch_size=self.config.trainer.batch_size,
            color_mode=self.config.data_loader.color_mode,  # for coloured images
            class_mode=self.config.data_loader.class_mode,
            subset='validation')

    def get_training_data(self):
        return self.training_data

    def get_validation_data(self):
        return self.validation_data
