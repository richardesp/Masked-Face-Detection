"""
This file is an extract from the notebook made in colab, for the testing phase prior to training the model

:author: Ricardo Espantaleón
"""
from models import model_01
from utils import config
from trainers import vgg_trainer
import tensorflow as tf
from utils import get_model_size
from keras.preprocessing.image import ImageDataGenerator
from data_loader import data_loader_01
from tensorflow.keras.optimizers import Adam

config_file = config.process_config("../configs/maskedfacepeople_config_vgg16.json")

# Allowing to get more GPU memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

class_model = model_01.Model(config_file)
class_model.build_model(compilation=True)
model = class_model.get_model()

directory = '/home/ricardo/Projects/maskedFaceDetection/dataset/mfd_dataset_reduced'

batch_size = 50

# Getting required model size
print(
    f"{get_model_size.keras_model_memory_usage_in_bytes(model, config_file.trainer.batch_size) / 1000000000} GB required")

class_names = ["with mask", "without mask"]

data_loader = data_loader_01.DataLoader(config_file, directory)

trainer = vgg_trainer.ModelTrainer(model, data_loader.get_training_data(), data_loader.get_validation_data(),
                                   config=config_file)
callbacks = trainer.get_callbacks()

# Model previously compiled
# model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Pillow is required for work with PIL.Images
# SciPy is required for image transformations
# history = model.fit(train_generator, epochs=5, batch_size=batch_size, validation_data=validation_generator,
#                    callbacks=callbacks)

trainer.train()
