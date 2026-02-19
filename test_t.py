import cv2
from ultralytics import YOLO

MODEL_PATH = "/home/mik/PycharmProjects/tours_cheating/runs/detect/train11/weights/best.pt"
IMAGE_PATH = "/home/mik/PycharmProjects/tours_cheating/dataset_t/images/train/0059.jpg"

model = YOLO(MODEL_PATH)

img = cv2.imread(IMAGE_PATH)

results = model(
    img,
    imgsz=640,
    conf=0.25,
    iou=0.45,
    verbose=False
)[0]

for box, conf in zip(results.boxes.xyxy, results.boxes.conf):
    x1,y1,x2,y2 = map(int, box)
    c = float(conf)

    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.putText(
        img,
        f"{c:.2f}",
        (x1, y1-5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,255,0),
        1
    )



cv2.imshow("cracks_detected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
