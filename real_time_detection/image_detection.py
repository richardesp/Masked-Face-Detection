import numpy as np
import cv2
from keras.models import load_model
from utils import get_learning_rate
from tensorflow.keras.optimizers import Adam

optimizer = Adam(0.0001)
customMetric = get_learning_rate.get_lr_metric(optimizer)
model = load_model(
    '/home/ricardo/PycharmProjects/maskedFaceDetection/experiments/2022-01-27/m01-004-dl01-checkpoints/maskedfacepeople_exp_004-val_loss.h5',
    custom_objects={"lr": customMetric})

img_path = "/home/ricardo/Downloads/image_black.webp"
input_img = cv2.imread(img_path)
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
input_img = input_img.astype(np.float32)
input_img /= 255.

new_array = cv2.resize(input_img, (224, 224))

new_array = new_array.reshape(-1, 224, 224, 3)

prediction = model.predict(new_array)

class_names = ["With", "Without"]
print(prediction)
print(f"My model says: {class_names[prediction.argmax()]}")
