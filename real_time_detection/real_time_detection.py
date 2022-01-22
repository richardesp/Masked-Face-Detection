import warnings

import optimizers.learning_rate_schedules

warnings.filterwarnings('ignore')
import numpy as np
import cv2
from keras.models import load_model
from utils import get_learning_rate
from tensorflow.keras.optimizers import Adam

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
threshold = 0.90
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

best_model_path = '/home/ricardo/PycharmProjects/maskedFaceDetection/code_tests/experiments/2022-01-21/m01-vgg16-dl01-checkpoints/mfp22_time_based_decay_224-244-3-val_loss.h5'

optimizer = Adam(0.0001)
customMetric = get_learning_rate.get_lr_metric(optimizer)
model = load_model(
    best_model_path,
    custom_objects={"lr": customMetric})


def preprocessing(img_input):
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    img_input = img_input.astype(np.float32)
    img_input /= 255.
    return img_input


while True:
    sucess, imgOrignal = cap.read()
    faces = facedetect.detectMultiScale(imgOrignal, 1.3, 5)
    for x, y, w, h in faces:
        crop_img = imgOrignal[y:y + h, x:x + h]
        img = cv2.resize(crop_img, (128, 128))
        img = preprocessing(img)
        img = img.reshape(-1, 128, 128, 3)
        prediction = model.predict(img)
        print(prediction)
        class_names = ["With", "Without"]
        classIndex = prediction.argmax()
        print(classIndex)
        probabilityValue = np.amax(prediction)
        if probabilityValue > threshold:
            if classIndex == 0:
                cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)

                # Using 4 decimals for probability value
                cv2.putText(imgOrignal, str(class_names[classIndex]) + " " + str(round(probabilityValue, 4)),
                            (x, y - 10), font,
                            0.75, (255, 255, 255), 1,
                            cv2.LINE_AA)
            elif classIndex == 1:
                cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (50, 50, 255), 2)
                cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (50, 50, 255), -2)

                # Using 4 decimals for probability value
                cv2.putText(imgOrignal, str(class_names[classIndex]) + " " + str(round(probabilityValue, 4)),
                            (x, y - 10), font,
                            0.75,
                            (255, 255, 255), 1,
                            cv2.LINE_AA)

    cv2.imshow("Result", imgOrignal)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()