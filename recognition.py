from typing import Optional

from numpy.typing import NDArray
from tensorflow.python.keras.models import load_model, Model

MODEL_PATH = "/home/leonardovallem/PycharmProjects/tensorflow-vgg16/trained_model"

model: Optional[Model] = None

classes = ["not wearing mask", "weaking mask"]


def update_model():
    global model

    if model is None:
        model = load_model(MODEL_PATH)


def detect(data):
    update_model()
    prediction: NDArray = model.predict(data)
    return prediction.argmax(-1)[0]
