import random
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort  # <-- Added

# Load class names
with open("yolov8/utils/coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Generate unique color per class label
detection_colors = []
for _ in class_list:
    detection_colors.append(
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    )

# Load YOLOv8 model
model = YOLO("yolov8/yolov8n.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)  # You can adjust age/params as needed

# Video input
cap = cv2.VideoCapture("yolov8/Input/videos/Puppy Benni on German GayPride Karlsruhe 2018 (🟡Human Pupplay🟡) #humanpupplay #shorts #pupplay.mp4")

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

    detections = []

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        bb = box.xyxy[0].cpu().numpy()

        x1, y1, x2, y2 = map(int, bb)
        w, h = x2 - x1, y2 - y1

        detections.append(([x1, y1, w, h], conf, cls_id))  # bbox format: [x, y, w, h]

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        cls_id = track.get_det_class()  # Get class ID from detection
        color = detection_colors[cls_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_list[cls_id]} ID:{track_id}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("ObjectDetection+Tracking", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
