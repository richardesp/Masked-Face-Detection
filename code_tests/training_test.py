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
from tensorflow.keras.optimizers import Adam

config_file = config.process_config("../configs/maskedfacepeople_config_vgg16.json")

# Allowing to get more GPU memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

class_model = model_01.Model(config_file)
class_model.build_model(compilation=True)
model = class_model.get_model()

# Printing model summary for check if config file was loaded correctly
model.summary()

directory = '/home/ricardo/Projects/maskedFaceDetection/dataset/dataset_1_0'

batch_size = 50

# Getting required model size
print(f"{get_model_size.keras_model_memory_usage_in_bytes(model, batch_size) / 1000000000} GB required")

train_datagen = ImageDataGenerator(validation_split=0.2,  # Splits the data into training (80%) and validation (20%)
                                   rescale=1. / 255,
                                   # Multiple the colors by a number between 0-1 to process data faster
                                   rotation_range=40,  # rotate the images
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')  # add new pixels when the image is rotated or shifted

train_generator = train_datagen.flow_from_directory(
    directory,
    target_size=(256, 256),
    batch_size=batch_size,
    color_mode="rgb",  # for coloured images
    class_mode='categorical',
    seed=2020,  # to make the result reproducible
    subset='training')  # Specify this is training set

validation_generator = train_datagen.flow_from_directory(
    directory,
    target_size=(256, 256),
    batch_size=batch_size,
    color_mode="rgb",  # for coloured images
    class_mode='categorical',
    subset='validation')  # Specify this is training set

class_names = ["with mask", "without mask"]

trainer = vgg_trainer.ModelTrainer(model, train_generator, validation_generator, config=config_file)
callbacks = trainer.get_callbacks()

# Model previously compiled
# model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Pillow is required for work with PIL.Images
# SciPy is required for image transformations
#history = model.fit(train_generator, epochs=5, batch_size=batch_size, validation_data=validation_generator,
#                    callbacks=callbacks)

trainer.train()
