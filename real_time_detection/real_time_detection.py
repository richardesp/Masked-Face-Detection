import cv2
import mediapipe as mp
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from utils import get_learning_rate
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

class_names = ["Without", "With"]
model_path = '/home/ricardo/PycharmProjects/maskedFaceDetection/experiments/2022-01-27/m01-004-dl01-checkpoints/maskedfacepeople_exp_004-val_loss.h5'

optimizer = Adam()
customMetric = get_learning_rate.get_lr_metric(optimizer)
model = load_model(model_path, custom_objects={"lr": customMetric})
threshold = 0.85

# For webcam input:
cap = cv2.VideoCapture(0)
cap.set(3, 500)
cap.set(4, 500)
with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        height, width, _ = image.shape

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:

                x_min = int(detection.location_data.relative_bounding_box.xmin * width)
                y_min = int(detection.location_data.relative_bounding_box.ymin * height)
                w = int(detection.location_data.relative_bounding_box.width * width)
                h = int(detection.location_data.relative_bounding_box.height * height)
                crop_img = image[y_min:y_min + h, x_min:x_min + h]
                #cv2.rectangle(image, (x_min, y_min), (x_min + w, y_min + h), (0, 255, 0), 15)
                #cv2.imshow("Cara", crop_img)

                print(x_min, y_min, w, h)

                if crop_img.__len__() > 0:

                    print(crop_img)

                    crop_img = cv2.resize(crop_img, (224, 224))
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    crop_img = crop_img.astype(np.float32)
                    crop_img /= 255.

                    face_to_predict = crop_img.reshape(-1, 224, 224, 3)
                    prediction = model.predict(face_to_predict)
                    class_index = prediction.argmax()

                    confidence = np.amax(prediction)

                    if confidence > threshold:
                        if class_index == 0:
                            mp_drawing.draw_detection(image, detection,
                                                      mp_drawing.DrawingSpec(color=(255, 0, 255), circle_radius=2),
                                                      mp_drawing.DrawingSpec(color=(0, 255, 0)))

                        elif class_index == 1:
                            mp_drawing.draw_detection(image, detection,
                                                      mp_drawing.DrawingSpec(color=(255, 0, 255), circle_radius=2),
                                                      mp_drawing.DrawingSpec(color=(0, 0, 255)))

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
