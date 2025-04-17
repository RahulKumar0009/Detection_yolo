from ultralytics import YOLO
import numpy


model = YOLO("yolov8/yolov8n.pt", "v8")

# predict on an image
detection_output = model.predict(source="yolov8/Input/images/img0.JPG", conf=0.25, save=True)
print(detection_output)
print(detection_output[0].numpy())
