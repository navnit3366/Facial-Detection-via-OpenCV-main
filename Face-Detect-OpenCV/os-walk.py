# NOTE for beginners:
# This file configures the directory of the images to connect to main.py for face identification.
import os
from PIL import Image
import numpy as np
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "resx")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

# recommened value: 1.05; increase value to increase chance of detection (side-effect: decrease accuracy)
scaleFactor = 1.05
# recommended values: 3-6
minNeighbors = 5

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(
                path)).replace(" ", "-").lower()
#                ^
# (os.path.dirname(path)) can be changed to (root)

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            pil_image = Image.open(path).convert("L")
            resized_image = pil_image.resize((550, 550), Image.ANTIALIAS)
            image_array = np.array(resized_image, "uint8")

            faces = face_cascade.detectMultiScale(
                image_array, scaleFactor, minNeighbors)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

# machine learning
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")
