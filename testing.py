""" analyze video with trained model and output in real time """
from keras.models import load_model
from collections import deque
import numpy as np
import pickle
import cv2

model = load_model("./model/VideoClassificationModel2.keras")
ib = pickle.loads(open("./model/VideoClassificationBinarizer2.pickle", "rb").read())

output_path = "./model/demo_output.mp4"

mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")

Queue = deque(maxlen=128)

capture_video = cv2.VideoCapture("./model/tyson.mp4")

writer = None

(Width, Height) = (None, None)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter(output_path, fourcc, 30, (Width, Height), True)

while True:
    (taken, frame) = capture_video.read()
    if not taken:
        break
    if Width is None or Height is None:
        (Width, Height) = frame.shape[:2]
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224)).astype("float32")
    frame -= mean
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    Queue.append(preds)
    results = np.array(Queue).mean(axis=0)
    i = np.argmax(results)
    label = ib.classes_[i]
    text = "They are Playing {}".format(label)
    cv2.putText(output, text, (45, 68), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 5)
    writer.write(output)
    cv2.imshow("In progress", output)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("a"):
        break

print("Finalizing.")
writer.release()
capture_video.release()
