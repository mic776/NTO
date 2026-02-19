import cv2
from ultralytics import YOLO

# ===== CONFIG =====
MODEL_PATH = "/home/mik/PycharmProjects/tours_cheating/runs/detect/train9/weights/best.pt"
IMAGE_PATH = "/home/mik/PycharmProjects/tours_cheating/dataset/images/7_p10.png"
CONF = 0.05
# ==================


model = YOLO(MODEL_PATH)

img = cv2.imread(IMAGE_PATH)

results = model(
    img,
    conf=CONF,
    verbose=False
)[0]

if results.boxes is not None:

    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    names = {
        0: "no_bolt",
        1: "t1",
        2: "t2"
    }

    for box, cls, conf in zip(boxes, classes, confs):

        x1, y1, x2, y2 = map(int, box)

        label = f"{names[int(cls)]} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2),
                      (0, 255, 0), 2)

        cv2.putText(img, label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2)

cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
