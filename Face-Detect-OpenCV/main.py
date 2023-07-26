import time
import cv2
import pickle
import numpy as np

# Variables:

# select device: 0 -> any; 1 -> external cam 1, -1 -> any
device_cap = 1

font = cv2.FONT_HERSHEY_TRIPLEX
display_face = '\"face\"'
display_eye = '\"eye\"'
color_pink = (127, 127, 255)
color_yellow = (0, 255, 255)
color_green = (0, 255, 0)
stroke = 2
# recommended value: 1.05; increase value to increase chance of detection (side-effect: decrease accuracy)
scaleFactor = 1.25
# recommended values: 3-6
minNeighbors = 5

# Start-up

print("[INFO] Setting up video capture...")

video = cv2.VideoCapture(device_cap, cv2.CAP_DSHOW)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(0.5)

print("[INFO] Activating Haar Cascade classifiers...")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

time.sleep(1)

# End start-up

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name": 1}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

if not video.isOpened():
    video.open(device_cap)

while video.isOpened():
    ret, frame = video.read()

    if ret == False:
        print("Live video feed crashed. Possible cause: faulty driver.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ROI - Region Of Interest
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor, minNeighbors)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color_yellow, 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        id_, confidence = recognizer.predict(roi_gray)

        if confidence >= 50 and confidence <= 99:
            name = (labels[id_])
            cv2.putText(frame, name + " " + str(np.floor(confidence)) + "%", (x, y), font,
                        0.85, color_pink, stroke, cv2.LINE_AA)
        else:
            cv2.putText(frame, display_face, (x, y), font,
                        1, color_pink, stroke, cv2.LINE_AA)

        # eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color_green, 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
