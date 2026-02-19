import cv2
from ultralytics import YOLO

MODEL_PATH = "/home/mik/PycharmProjects/tours_cheating/runs/detect/train14/weights/best.pt"
IMG_PATH = "/home/mik/PycharmProjects/tours_cheating/dataset_r/images/00034.jpg"

model = YOLO(MODEL_PATH)

img = cv2.imread(IMG_PATH)

results = model(img, conf=0.25, iou=0.45, verbose=False)[0]

if results.boxes is not None:
    boxes = results.boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
