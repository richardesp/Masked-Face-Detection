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
from streamlit_option_menu import option_menu
import youtube_dl


def main():
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    class_names = ["With", "Without"]
    model_path = '../experiments/2022-01-27/m01-004-dl01-checkpoints/maskedfacepeople_exp_004-val_loss.h5'

    optimizer = Adam()

    # Compile=False, for preventing load any custom object,
    # due the model it's only predicting, and it's not going to be trained
    threshold = 0.85

    model = load_model(model_path, compile=False)

    with st.sidebar:
        selected = option_menu("Input menu", ["Webcam", "Youtube video", "Image"],
                               icons=["webcam", "youtube", "images"], default_index=0)

    if selected == "Webcam":
        st.write("Webcam mode")

        class VideoTransformer(VideoTransformerBase):
            frame_lock: threading.Lock  # transform() is running in another thread, then a lock object is used here for thread-safety.
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

                                if crop_img.size > 0:

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
                                            cv2.rectangle(self.out_image, (x_min, y_min), (x_min + w, y_min + h),
                                                          (0, 255, 0),
                                                          2)
                                            cv2.rectangle(self.out_image, (x_min, y_min - 40), (x_min + w, y_min),
                                                          (0, 255, 0),
                                                          -2)

                                            # Using 4 decimals for probability value
                                            cv2.putText(self.out_image,
                                                        str(class_names[class_index]) + " " + str(round(confidence, 4)),
                                                        (x_min, y_min - 10), cv2.FONT_HERSHEY_DUPLEX,
                                                        0.75, (255, 255, 255), 1,
                                                        cv2.LINE_AA)

                                        elif class_index == 1:
                                            cv2.rectangle(self.out_image, (x_min, y_min), (x_min + w, y_min + h),
                                                          (50, 50, 255),
                                                          2)
                                            cv2.rectangle(self.out_image, (x_min, y_min - 40), (x_min + w, y_min),
                                                          (50, 50, 255), -2)

                                            # Using 4 decimals for probability value
                                            cv2.putText(self.out_image,
                                                        str(class_names[class_index]) + " " + str(round(confidence, 4)),
                                                        (x_min, y_min - 10), cv2.FONT_HERSHEY_DUPLEX,
                                                        0.75, (255, 255, 255), 1,
                                                        cv2.LINE_AA)

                return self.out_image

        ctx = webrtc_streamer(key="start", video_processor_factory=VideoTransformer)

    elif selected == "Youtube video":
        st.write("Youtube video mode")

        video_url = st.text_input("The URL link")

        if video_url != '':

            ydl_opts = {}

            frame_window = st.image([])

            # create youtube-dl object.
            ydl = youtube_dl.YoutubeDL(ydl_opts)

            # set video url, extract video information.
            info_dict = ydl.extract_info(video_url, download=False)

            # get video formats available.
            formats = info_dict.get('formats', None)

            for f in formats:

                # Here you can specify the resolution.
                if f.get('format_note', None) == '480p':

                    # get the video url
                    url = f.get('url', None)

                    # open url with opencv
                    cap = cv2.VideoCapture(url)

                    # check if url was opened
                    if not cap.isOpened():
                        st.write("The video could not be reproduced")
                        break

                    while True:
                        with mp_face_detection.FaceDetection(
                                model_selection=0, min_detection_confidence=0.6) as face_detection:
                            # read frame.
                            ret, image = cap.read()

                            # check if frame is empty.
                            if not ret:
                                break

                            height, width, _ = image.shape

                            # To improve performance, optionally mark the image as not writeable to
                            # pass by reference.
                            image.flags.writeable = False
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            results = face_detection.process(image)

                            # Draw the face detection annotations on the image.
                            image.flags.writeable = True
                            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                            if results.detections:
                                for detection in results.detections:

                                    x_min = int(detection.location_data.relative_bounding_box.xmin * width)
                                    y_min = int(detection.location_data.relative_bounding_box.ymin * height)
                                    w = int(detection.location_data.relative_bounding_box.width * width)
                                    h = int(detection.location_data.relative_bounding_box.height * height)
                                    crop_img = image[y_min:y_min + h, x_min:x_min + h]

                                    if crop_img.size > 0:

                                        crop_img = cv2.resize(crop_img, (224, 224))
                                        # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                                        crop_img = crop_img.astype(np.float32)
                                        crop_img /= 255.

                                        face_to_predict = crop_img.reshape(-1, 224, 224, 3)
                                        prediction = model.predict(face_to_predict)
                                        class_index = prediction.argmax()

                                        confidence = np.amax(prediction)

                                        if confidence > threshold:
                                            if class_index == 0:
                                                cv2.rectangle(image, (x_min, y_min), (x_min + w, y_min + h),
                                                              (0, 255, 0),
                                                              2)
                                                cv2.rectangle(image, (x_min, y_min - 40), (x_min + w, y_min),
                                                              (0, 255, 0),
                                                              -2)

                                                # Using 4 decimals for probability value
                                                cv2.putText(image,
                                                            str(class_names[class_index]) + " " + str(
                                                                round(confidence, 4)),
                                                            (x_min, y_min - 10), cv2.FONT_HERSHEY_DUPLEX,
                                                            0.75, (255, 255, 255), 1,
                                                            cv2.LINE_AA)

                                            elif class_index == 1:
                                                cv2.rectangle(image, (x_min, y_min), (x_min + w, y_min + h),
                                                              (255, 50, 50),
                                                              2)
                                                cv2.rectangle(image, (x_min, y_min - 40), (x_min + w, y_min),
                                                              (255, 50, 50), -2)

                                                # Using 4 decimals for probability value
                                                cv2.putText(image,
                                                            str(class_names[class_index]) + " " + str(
                                                                round(confidence, 4)),
                                                            (x_min, y_min - 10), cv2.FONT_HERSHEY_DUPLEX,
                                                            0.75, (255, 255, 255), 1,
                                                            cv2.LINE_AA)

                            frame_window.image(image)

                    # release VideoCapture.
                    cap.release()

                    # Deleting previously url entered.
                    video_url = ""

    else:
        st.write("Image mode")

        uploaded_file = st.file_uploader("Choose a image file", type="jpg")

        if uploaded_file is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            input_img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            input_img = input_img.astype(np.float32)
            input_img /= 255.

            input_array = cv2.resize(input_img, (224, 224))
            input_array = input_array.reshape(-1, 224, 224, 3)
            prediction = model.predict(input_array)

            confidence = np.amax(prediction)

            st.image(opencv_image, channels="BGR")
            st.write("The model says: {} (Confidence: {}%)".format(class_names[prediction.argmax()],
                                                                   round(confidence * 100, 3)))


if __name__ == "__main__":
    main()
