import os
import cv2
import math
import random
import numpy as np

# ===== CONFIG =====
NUM_IMAGES = 250
IMG_SIZE = 640

BG_FOLDER = "backgrounds"
OUT_IMG = "dataset_t/images/train"
OUT_LBL = "dataset_t/labels/train"

MIN_CRACKS = 2
MAX_CRACKS = 5

MIN_BOX_PX = 12          # фильтр слишком мелких
DILATE_KERNEL = 3        # помогает сделать bbox стабильнее на тонких линиях
MAX_PLACEMENT_TRIES = 50 # чтобы разнести трещины

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)


def grow_crack(img, mask, x, y, angle, energy, thickness):
    h, w = img.shape[:2]

    while energy > 0.05:
        angle += random.uniform(-0.25, 0.25)
        step = random.uniform(3, 7)

        nx = int(x + math.cos(angle) * step)
        ny = int(y + math.sin(angle) * step)
        if nx < 0 or ny < 0 or nx >= w or ny >= h:
            break

        color = random.randint(20, 60)
        t = max(1, int(thickness))

        cv2.line(img, (x, y), (nx, ny), (color, color, color), t)
        cv2.line(mask, (x, y), (nx, ny), 255, t)

        # branching (внутри этой же трещины)
        if random.random() < 0.05 * energy:
            branch_angle = angle + random.uniform(-1.2, 1.2)
            grow_crack(
                img, mask,
                nx, ny,
                branch_angle,
                energy * random.uniform(0.4, 0.7),
                thickness * 0.7
            )

        x, y = nx, ny
        energy -= random.uniform(0.01, 0.03)
        thickness *= 0.995


def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    if (x2 - x1) < MIN_BOX_PX or (y2 - y1) < MIN_BOX_PX:
        return None
    return x1, y1, x2, y2


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def bbox_to_yolo(bbox, w, h):
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return (0, cx, cy, bw, bh)  # class 0


def generate_image(bg):
    img = bg.copy()
    h, w = img.shape[:2]

    labels = []
    existing_boxes = []

    crack_count = random.randint(MIN_CRACKS, MAX_CRACKS)

    for _ in range(crack_count):
        placed = False

        for _try in range(MAX_PLACEMENT_TRIES):
            crack_mask = np.zeros((h, w), dtype=np.uint8)

            sx = random.randint(w // 6, 5 * w // 6)
            sy = random.randint(h // 6, 5 * h // 6)
            angle = random.uniform(0, 2 * math.pi)

            grow_crack(
                img, crack_mask,
                sx, sy, angle,
                energy=1.0,
                thickness=random.randint(2, 6)
            )

            if DILATE_KERNEL > 0:
                k = np.ones((DILATE_KERNEL, DILATE_KERNEL), np.uint8)
                crack_mask = cv2.dilate(crack_mask, k, iterations=1)

            bbox = mask_to_bbox(crack_mask)
            if bbox is None:
                continue

            # не даём трещинам сливаться в один bbox (контроль по IoU)
            ok = True
            for eb in existing_boxes:
                if bbox_iou(bbox, eb) > 0.05:  # порог можно ужесточить: 0.02
                    ok = False
                    break
            if not ok:
                continue

            labels.append(bbox_to_yolo(bbox, w, h))
            existing_boxes.append(bbox)
            placed = True
            break

        if not placed:
            # если не удалось разместить (редко) — просто пропускаем одну трещину
            pass

    # realism
    img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = np.random.normal(0, 4, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img, labels


def main():
    backgrounds = [os.path.join(BG_FOLDER, f) for f in os.listdir(BG_FOLDER)]
    if not backgrounds:
        raise RuntimeError("backgrounds/ пустая")

    saved = 0
    i = 0
    while saved < NUM_IMAGES:
        bg_path = random.choice(backgrounds)
        bg = cv2.imread(bg_path)
        if bg is None:
            continue

        bg = cv2.resize(bg, (IMG_SIZE, IMG_SIZE))

        img, labels = generate_image(bg)

        if len(labels) == 0:
            continue

        img_name = f"{i:04d}.jpg"
        lbl_name = f"{i:04d}.txt"

        cv2.imwrite(os.path.join(OUT_IMG, img_name), img)

        with open(os.path.join(OUT_LBL, lbl_name), "w") as f:
            for cls, cx, cy, bw, bh in labels:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        saved += 1
        i += 1

    print("DONE")


if __name__ == "__main__":
    main()
