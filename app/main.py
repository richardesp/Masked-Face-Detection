import cv2
import streamlit as st
import mediapipe as mp
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import threading
from typing import Union
import av


def main():
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    class_names = ["Without", "With"]
    model_path = '/home/ricardo/PycharmProjects/maskedFaceDetection/experiments/2022-01-27/m01-004-dl01-checkpoints/maskedfacepeople_exp_004-val_loss.h5'

    optimizer = Adam()

    # Compile=False, for preventing load any custom object,
    # due the model it's only predicting, and it's not going to be trained
    threshold = 0.85

    model = load_model(model_path, compile=False)

    class VideoTransformer(VideoTransformerBase):
        frame_lock: threading.Lock  # `transform()` is running in another thread, then a lock object is used here for thread-safety.
        in_image: Union[np.ndarray, None]
        out_image: Union[np.ndarray, None]

        def __init__(self) -> None:
            self.frame_lock = threading.Lock()
            self.in_image = None
            self.out_image = None

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            in_image = frame.to_ndarray(format="bgr24")

            with self.frame_lock:
                self.in_image = in_image
                self.out_image = in_image

                with mp_face_detection.FaceDetection(
                        model_selection=0, min_detection_confidence=0.5) as face_detection:

                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    self.out_image.flags.writeable = False
                    self.out_image = cv2.cvtColor(self.in_image, cv2.COLOR_BGR2RGB)

                    results = face_detection.process(self.out_image)

                    # Draw the face detection annotations on the image.
                    self.out_image.flags.writeable = True
                    self.out_image = cv2.cvtColor(self.out_image, cv2.COLOR_RGB2BGR)

                    height, width, _ = self.out_image.shape

                    if results.detections:
                        for detection in results.detections:
                            x_min = int(detection.location_data.relative_bounding_box.xmin * width)
                            y_min = int(detection.location_data.relative_bounding_box.ymin * height)
                            w = int(detection.location_data.relative_bounding_box.width * width)
                            h = int(detection.location_data.relative_bounding_box.height * height)
                            crop_img = self.out_image[y_min:y_min + h, x_min:x_min + h]

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
                                    cv2.rectangle(self.out_image, (x_min, y_min), (x_min + w, y_min + h), (0, 255, 0),
                                                  2)
                                    cv2.rectangle(self.out_image, (x_min, y_min - 40), (x_min + w, y_min), (0, 255, 0),
                                                  -2)

                                    """
                                    # Using 4 decimals for probability value
                                    cv2.putText(self.out_image,
                                                str(class_names[class_index]) + " " + str(round(confidence, 4)),
                                                (x_min, y_mi - 10), font,
                                                0.75, (255, 255, 255), 1,
                                                cv2.LINE_AA)
                                    """

                                elif class_index == 1:
                                    cv2.rectangle(self.out_image, (x_min, y_min), (x_min + w, y_min + h), (50, 50, 255),
                                                  2)
                                    cv2.rectangle(self.out_image, (x_min, y_min - 40), (x_min + w, y_min),
                                                  (50, 50, 255), -2)

                                    """
                                    # Using 4 decimals for probability value
                                    cv2.putText(imgOrignal,
                                                str(class_names[classIndex]) + " " + str(round(probabilityValue, 4)),
                                                (x, y - 10), font,
                                                0.75,
                                                (255, 255, 255), 1,
                                                cv2.LINE_AA)
                                    """

            return self.out_image

    ctx = webrtc_streamer(key="start", video_processor_factory=VideoTransformer)


if __name__ == "__main__":
    main()
