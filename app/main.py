import cv2
import streamlit as st
import mediapipe as mp
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

class_names = ["Without", "With"]
model_path = '/home/ricardo/PycharmProjects/maskedFaceDetection/experiments/2022-01-27/m01-004-dl01-checkpoints/maskedfacepeople_exp_004-val_loss.h5'

optimizer = Adam()

# Compile=False, for preventing load any custom object,
# due the model it's only predicting, and it's not going to be trained
# = load_model(model_path, compile=False)
threshold = 0.85

model = load_model(model_path, compile=False)


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.threshold1 = 100
        self.threshold2 = 200

    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")

        with mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5) as face_detection:
            height, width, _ = frame.shape

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

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

                if crop_img is not None:
                    crop_img = cv2.resize(crop_img, (224, 224))
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    crop_img = crop_img.astype(np.float32)
                    crop_img /= 255.

                    face_to_predict = crop_img.reshape(-1, 224, 224, 3)
                    prediction = model.predict(face_to_predict)
                    class_index = prediction.argmax()

                    print(prediction)
                    print(class_index)

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

        return crop_img


ctx = webrtc_streamer(key="example", video_processor_factory=VideoTransformer)
