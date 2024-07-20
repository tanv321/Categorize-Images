""" analyze video with trained model and output to text file """
from keras.models import load_model
from collections import deque
import numpy as np
import pickle
import cv2

# Load the trained model and label binarizer
model = load_model("./model/VideoClassificationModel.keras")
ib = pickle.loads(open("./model/VideoClassificationBinarizer.pickle", "rb").read())

# Initialize some variables
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Queue = deque(maxlen=128)


name_video = "wat.mp4"
# Open the video file
capture_video = cv2.VideoCapture("./model/wat.mp4")

# Open a text file to write the results
output_file = open("output_results.txt", "w")

writer = None
(Width, Height) = (None, None)

# Process each frame of the video
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
    output_file.write(f"Video: {name_video}.mp4, Tag: {label}\n")
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter("outputvideo.mp4", fourcc, 30, (Width, Height), True)
    writer.write(output)

print("Finalizing.")
# Close the text file and release the video capture
output_file.close()
capture_video.release()
