import random
import cv2
import numpy as np
from ultralytics import YOLO

# Load class names
with open("yolov8/utils/coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# unique color for label
detection_colors = []
for _ in class_list:
    detection_colors.append(
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    )

# Load model
model = YOLO("yolov8/yolov8n.pt")

# Video input
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("yolov8/Input/videos/video1.mp4")

if not cap.isOpened():
    print("Cannot open camera or video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO prediction
    results = model.predict(source=[frame], conf=0.45, save=False)
    boxes = results[0].boxes

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        bb = box.xyxy[0].cpu().numpy()

        # Use class ID to assign color
        color = detection_colors[cls_id]

        cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), color, 2)
        label = f"{class_list[cls_id]} {round(conf*100, 1)}%"
        cv2.putText(frame, label, (int(bb[0]), int(bb[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("ObjectDetection", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
