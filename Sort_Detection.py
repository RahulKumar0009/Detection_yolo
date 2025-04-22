import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
from ultralytics.utils.plotting import Annotator

track_history = defaultdict(lambda: [])

model = YOLO("yolov8/yolov8n.pt")
names = model.model.names
video_path = "yolov8/Input/videos/video1.mp4"

if not Path(video_path).exists():
    raise FileNotFoundError(f"Source path "
                            f"'{video_path}' "
                            f"does not exist.")

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True)

        boxes = results[0].boxes.xywh.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        ids = results[0].boxes.id
        if ids is None:
            continue
        track_ids = ids.int().cpu().tolist()

        annotator = Annotator(frame, line_width=2,
                              example=str(names))

        for box, track_id, cls in zip(boxes, track_ids, clss):
            x, y, w, h = box
            x1, y1, x2, y2 = (x - w / 2, y - h / 2,
                              x + w / 2, y + h / 2)
            label = str(names[cls]) + " : " + str(track_id)
            annotator.box_label([x1, y1, x2, y2],
                                label, (218, 100, 255))

            # Tracking Lines plot
            track = track_history[track_id]
            track.append((float(box[0]), float(box[1])))
            if len(track) > 30:
                track.pop(0)

            # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            # cv2.polylines(frame, [points], isClosed=False,
            #               color=(37, 255, 225), thickness=2)
            #
            # # Center circle
            # cv2.circle(frame,
            #            (int(track[-1][0]), int(track[-1][1])),
            #            5, (235, 219, 11), -1)

        cv2.imshow("YOLOv8 Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()