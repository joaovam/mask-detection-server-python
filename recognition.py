from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray
from tensorflow.python.keras.models import load_model, Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from face import Face

MODEL_PATH = "/home/leonardovallem/PycharmProjects/tensorflow-vgg16/trained_model"

model: Optional[Model] = None

classes = ["not wearing mask", "wearing mask"]


def update_model():
    global model

    if model is None:
        model = load_model(MODEL_PATH)


def detect(data):
    update_model()
    faces = detect_faces(data)
    for face in faces:
        #TODO
        #cortar a imagem inicial com base na posicao do rosto
        #adicionar booleano mostrando uso de mascara
        #adicionar credibilidade dessa predição
        #ambas as adições são ao objeto face

        prediction: NDArray = model.predict(data)

    return prediction.argmax(-1)[0]


def detect_faces(frame):
    print("[INFO] loading face detector model...")

    net = cv2.dnn.readNet("prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

    (h, w) = frame.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    faces = []

    # print(detections)
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = Face()
            face.initial_x = startX
            face.initial_y = startY
            face.final_x = endX
            face.final_y = endY
            faces.append(face)

    return faces
