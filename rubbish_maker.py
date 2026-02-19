import cv2
import os
import random
import numpy as np
from glob import glob

FRAMES_DIR = "frames"
RUBBISH_DIR = "rubbish"

OUT_IMG = "dataset_r/images"
OUT_LBL = "dataset_r/labels"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

IMG_SIZE = 640
CLASS_ID = 0


def extract_object(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # намного шире диапазон жёлтого
    lower = np.array([10, 40, 40])
    upper = np.array([50, 255, 255])

    bg_mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_not(bg_mask)

    obj = cv2.bitwise_and(img, img, mask=mask)

    # cv2.imshow("img", img)
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)

    return obj, mask



def paste(bg, obj, mask, x, y):
    h, w = obj.shape[:2]

    if y+h > bg.shape[0] or x+w > bg.shape[1]:
        print("paste", x, y, w, h)

        return bg

    roi = bg[y:y+h, x:x+w]

    m = mask.astype(float)/255.0
    m = cv2.merge([m, m, m])

    out = (obj*m + roi*(1-m)).astype(np.uint8)
    bg[y:y+h, x:x+w] = out

    return bg


def random_transform(img, mask):
    scale = random.uniform(0.02, 0.15)

    h, w = img.shape[:2]
    nw, nh = int(w*scale), int(h*scale)

    img = cv2.resize(img, (nw, nh))
    mask = cv2.resize(mask, (nw, nh))

    angle = random.uniform(-20, 20)
    M = cv2.getRotationMatrix2D((nw//2, nh//2), angle, 1.0)

    img = cv2.warpAffine(img, M, (nw, nh))
    mask = cv2.warpAffine(mask, M, (nw, nh))

    return img, mask


def generate_one(idx):

    bg_path = random.choice(glob(FRAMES_DIR+"/*"))
    bg = cv2.imread(bg_path)
    bg = cv2.resize(bg, (IMG_SIZE, IMG_SIZE))

    pile_boxes = []

    n_objs = random.randint(2, 5)

    base_x = random.randint(100, 400)
    base_y = random.randint(300, 520)

    for _ in range(n_objs):

        obj_path = random.choice(glob(RUBBISH_DIR+"/*"))
        obj = cv2.imread(obj_path)

        obj, mask = extract_object(obj)

        # print(mask.sum())
        # cv2.imshow("obj", obj)
        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)

        obj, mask = random_transform(obj, mask)

        h, w = obj.shape[:2]

        offset_x = base_x + random.randint(-40, 40)
        offset_y = base_y + random.randint(-20, 20)

        bg = paste(bg, obj, mask, offset_x, offset_y)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue

        x1 = offset_x + xs.min()
        x2 = offset_x + xs.max()
        y1 = offset_y + ys.min()
        y2 = offset_y + ys.max()

        pile_boxes.append([x1, y1, x2, y2])

    if len(pile_boxes) == 0:
        return

    # общий bbox (одна куча)
    x1 = min([b[0] for b in pile_boxes])
    y1 = min([b[1] for b in pile_boxes])
    x2 = max([b[2] for b in pile_boxes])
    y2 = max([b[3] for b in pile_boxes])

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(IMG_SIZE-1, x2)
    y2 = min(IMG_SIZE-1, y2)

    cx = ((x1+x2)/2)/IMG_SIZE
    cy = ((y1+y2)/2)/IMG_SIZE
    w  = (x2-x1)/IMG_SIZE
    h  = (y2-y1)/IMG_SIZE

    img_name = f"{idx:05d}.jpg"
    lbl_name = f"{idx:05d}.txt"

    cv2.imwrite(os.path.join(OUT_IMG, img_name), bg)

    with open(os.path.join(OUT_LBL, lbl_name), "w") as f:
        f.write(f"{CLASS_ID} {cx} {cy} {w} {h}\n")


if __name__ == "__main__":
    for i in range(2000):
        generate_one(i)
