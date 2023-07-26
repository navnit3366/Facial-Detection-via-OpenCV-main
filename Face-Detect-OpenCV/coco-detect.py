import cv2
import numpy as np

# A0. Set-up

print("[INFO] Setting up video capture...")

device_cap = 1
video = cv2.VideoCapture(device_cap, cv2.CAP_DSHOW)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

class_names = []
class_file = 'coco.names'
with open(class_file, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_path = 'frozen_inference_graph.pb'

# A1. Open model
net = cv2.dnn_DetectionModel(weights_path, config_path)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean(127.5)
net.setInputSwapRB(True)

if not video.isOpened():
    video.open(device_cap)

# B0. Loop
while True:
    ret, frame = video.read()
    class_id, conf, bbox = net.detect(frame, confThreshold=0.5)

    if len(class_id) != 0:
        for class_id, confidence, box in zip(class_id.flatten(), conf.flatten(), bbox):

            display_text = class_names[class_id-1] + f" {str(np.floor(confidence * 100))}%"
            cv2.rectangle(frame, box, color=(0, 255, 255),
                          thickness=3)
            cv2.putText(frame, display_text.upper(),
                        (box[0]+10, box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('w'):
        break

video.release()
cv2.destroyAllWindows()
