import cv2
import numpy as np
from ultralytics import YOLO
from road_mask import build_road_mask
from tracker import CentroidTracker


WEIGHTS = "runs/detect/train/weights/best.pt"
IMG_SZ = 640
CONF = 0.5

LANE_THRESH = 0.08
ROAD_THRESH = 0.15

tracker = CentroidTracker(max_distance=60, max_age=20)
model = YOLO(WEIGHTS)



def mask_ratio(mask, bbox):
    x1, y1, x2, y2 = bbox

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(mask.shape[1]-1, x2)
    y2 = min(mask.shape[0]-1, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    crop = mask[y1:y2, x1:x2]
    return np.count_nonzero(crop) / ((y2-y1) * (x2-x1))


def classify(lane_mask, road_mask, bbox):
    lane_ratio = mask_ratio(lane_mask, bbox)
    road_ratio = mask_ratio(road_mask, bbox)

    if lane_ratio >= LANE_THRESH or road_ratio >= ROAD_THRESH:
        return "IMPASSABLE", lane_ratio, road_ratio
    return "PASSABLE", lane_ratio, road_ratio



cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    road_mask, lane_mask = build_road_mask(frame)

    results = model.predict(frame, imgsz=IMG_SZ, conf=CONF, verbose=False)
    boxes = results[0].boxes

    bboxes = []
    confs = []

    if boxes is not None:
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            bboxes.append((x1, y1, x2, y2))
            confs.append(float(b.conf[0]))

    tracked = tracker.update(bboxes)

    for ((track_id, is_new), bbox, conf) in zip(tracked, bboxes, confs):

        label, lane_r, road_r = classify(lane_mask, road_mask, bbox)
        updated = tracker.set_label(track_id, label)

        if is_new or updated:
            print(
                f"Found {label} id={track_id} conf={conf:.2f} "
                f"(lane={lane_r:.2f}, road={road_r:.2f})"
            )
            print(
                f"Total IMPASSABLE={tracker.counts['IMPASSABLE']} | "
                f"PASSABLE={tracker.counts['PASSABLE']}"
            )

        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,255), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)

    overlay = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
    overlay[:,:,2] = lane_mask

    cv2.imshow("detection", frame)
    cv2.imshow("road_mask", overlay)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
