import cv2
import numpy as np
import os
import random
import glob


WHEELS_DIR = "wheels"
OUT_DIR = "out"

T1_DIR = "t1"
T2_DIR = "t2"

PASSES = 40

PROB_EMPTY = 0.3
PROB_T1 = 0.35
PROB_T2 = 0.35


def load_images(folder):
    return glob.glob(os.path.join(folder, "*.*"))


def to_yolo_box(x, y, w, h, img_w, img_h):
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def fill_shaded(img, x, y, w, h):

    roi = img[y:y+h, x:x+w]

    mean_color = img.mean(axis=(0, 1))
    base = (mean_color * 0.45).astype(np.uint8)

    patch = np.full((h, w, 3), base, dtype=np.uint8)

    noise = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
    patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist /= dist.max()

    patch = patch * (0.7 + 0.3 * dist[..., None])
    patch = np.clip(patch, 0, 255).astype(np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(w, h)//2, 255, -1)

    mask_f = np.stack([mask/255.0]*3, axis=-1)

    result = patch * mask_f + roi * (1 - mask_f)
    img[y:y+h, x:x+w] = result.astype(np.uint8)


def paste_bolt_circle(bg, bolt, x, y):

    h, w = bolt.shape[:2]
    roi = bg[y:y+h, x:x+w]

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w//2, h//2), min(w, h)//2, 255, -1)

    mask_f = np.stack([mask/255.0]*3, axis=-1)

    bolt_rgb = bolt[:, :, :3] if bolt.shape[2] == 4 else bolt

    result = bolt_rgb * mask_f + roi * (1 - mask_f)
    bg[y:y+h, x:x+w] = result.astype(np.uint8)


def find_red_holes(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = (
        cv2.inRange(hsv, (0,100,80), (10,255,255))
        | cv2.inRange(hsv, (170,100,80), (180,255,255))
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3,3), np.uint8)
    )

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    for c in cnts:
        if cv2.contourArea(c) < 30:
            continue

        x,y,w,h = cv2.boundingRect(c)

        if 0.6 < w/float(h) < 1.4:
            boxes.append((x,y,w,h))

    return boxes


def random_choice():

    r = random.random()

    if r < PROB_EMPTY:
        return None, 0

    elif r < PROB_EMPTY + PROB_T1:
        files = load_images(T1_DIR)
        cls = 1
    else:
        files = load_images(T2_DIR)
        cls = 2

    if not files:
        return None, 0

    bolt = cv2.imread(random.choice(files), cv2.IMREAD_UNCHANGED)
    return bolt, cls


def process_image(path):

    img = cv2.imread(path)
    H, W = img.shape[:2]

    boxes = find_red_holes(img)
    labels = []

    for (x,y,w,h) in boxes:

        bolt, cls = random_choice()

        if bolt is None:
            fill_shaded(img, x,y,w,h)
        else:
            bolt = cv2.resize(bolt, (w,h))
            paste_bolt_circle(img, bolt, x,y)

        cx,cy,nw,nh = to_yolo_box(x,y,w,h,W,H)
        labels.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return img, labels


def main():

    wheels = load_images(WHEELS_DIR)

    for p in range(PASSES):

        img_dir = os.path.join(OUT_DIR, "images")
        lbl_dir = os.path.join(OUT_DIR, "labels")

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        print("PASS", p)

        for path in wheels:

            img, labels = process_image(path)

            base = os.path.splitext(os.path.basename(path))[0]
            name = f"{base}_p{p}"

            img_path = os.path.join(img_dir, name + ".png")
            lbl_path = os.path.join(lbl_dir, name + ".txt")

            cv2.imwrite(img_path, img)

            with open(lbl_path, "w") as f:
                f.write("\n".join(labels))

            print("done:", name)


if __name__ == "__main__":
    main()
