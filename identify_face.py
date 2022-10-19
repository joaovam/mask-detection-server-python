import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# define a video capture object
vid = cv2.VideoCapture(0)

while True:
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 176)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 144)
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    global RGB_img
    # load our serialized face detector model from disk

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

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (176, 144))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            color = (0, 255, 0)
            print(f"{startX}.. {startY}, {endX} ..{endY}")
            cv2.putText(frame, "label", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            RGB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Display the resulting frame
    cv2.imshow('frame', RGB_img)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
