import json
from typing import Optional
from tensorflow.python.keras.models import load_model, Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf

MODEL_PATH = "./trained_model.h5"

model: Optional[Model] = None

classes = ["not wearing mask", "wearing mask"]


def update_model():
    global model

    if model is None:
        model = load_model(MODEL_PATH)


def detect(data):
    update_model()

    img_array = tf.keras.utils.img_to_array(data)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    img_array = preprocess_input(img_array)
    mask, withoutMask = model.predict(img_array)[0]

    # determine the class label and color we will use to draw the bounding box and text
    label, score = ('Mask', mask) if mask > withoutMask else ('No Mask', withoutMask)
    print((mask, withoutMask))
    print(
        "This image most likely belongs to {} with a {} percent confidence."
        .format(label, score))

    return json.dumps(dict(label=label, score=score))

