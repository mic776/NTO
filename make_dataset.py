import cv2
import numpy as np
import random
from pathlib import Path

SEED = 42
N_IMAGES = 1000

OBSTACLE_DIR = Path("Obstacles")
VIDEO_PATH = "video.MOV"
OUT_DIR = Path("dset")

random.seed(SEED)
rng = np.random.default_rng(SEED)


def list_images(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return [p for p in folder.iterdir() if p.suffix.lower() in exts]


def read_random_frame(cap):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(rng.integers(0, total)))
    _, frame = cap.read()
    return frame


def build_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > 10).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def apply_photometric(img):
    img = img.astype(np.float32)
    img = img * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def alpha_blend(bg, fg, mask, x, y, alpha=1.0):
    h, w = fg.shape[:2]
    bh, bw = bg.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bw, x + w)
    y2 = min(bh, y + h)

    fg = fg[y1 - y:y2 - y, x1 - x:x2 - x]
    mask = mask[y1 - y:y2 - y, x1 - x:x2 - x]

    m = (mask.astype(np.float32) / 255.0)[:, :, None] * alpha
    roi = bg[y1:y2, x1:x2].astype(np.float32)

    bg[y1:y2, x1:x2] = (fg.astype(np.float32) * m + roi * (1 - m)).astype(np.uint8)

    return bg



def bbox_from_masks(masks, positions):
    xs, ys = [], []

    for mask, (x, y) in zip(masks, positions):
        yy, xx = np.where(mask > 0)
        xs.append(xx + x)
        ys.append(yy + y)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def yolo_bbox(bbox, w, h):
    x1, y1, x2, y2 = bbox
    xc = (x1 + x2) / 2 / w
    yc = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def build_rubble(bg, obstacles, ppm):
    h, w = bg.shape[:2]

    cx = int(rng.integers(100, w - 100))
    cy = int(rng.integers(100, h - 100))

    composed = bg.copy()
    masks, positions = [], []

    for _ in range(int(rng.integers(2, 8))):

        part = cv2.imread(str(rng.choice(obstacles)))
        part = apply_photometric(part)

        mask = build_mask(part)

        target_px = int(rng.uniform(0.15, 0.6) * ppm)
        s = target_px / max(part.shape[:2])

        nw, nh = int(part.shape[1] * s), int(part.shape[0] * s)

        part = cv2.resize(part, (nw, nh))
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

        M = cv2.getRotationMatrix2D((nw / 2, nh / 2),
                                    rng.uniform(-30, 30), 1.0)

        part = cv2.warpAffine(part, M, (nw, nh), borderValue=(0, 0, 0))
        mask = cv2.warpAffine(mask, M, (nw, nh), borderValue=0)

        x = cx + int(rng.integers(-80, 80)) - nw // 2
        y = cy + int(rng.integers(-60, 60)) - nh // 2

        composed = alpha_blend(composed, part, mask, x, y,
                               rng.uniform(0.85, 1.0))

        masks.append(mask)
        positions.append((x, y))

    return composed, bbox_from_masks(masks, positions)



obstacles = list_images(OBSTACLE_DIR)

for p in ["images/train", "images/val",
          "labels/train", "labels/val"]:
    (OUT_DIR / p).mkdir(parents=True, exist_ok=True)

(OUT_DIR / "data.yaml").write_text("""path: .
train: images/train
val: images/val
names:
  0: rubble
""")

cap = cv2.VideoCapture(VIDEO_PATH)



for i in range(N_IMAGES):

    frame = read_random_frame(cap)
    frame = cv2.resize(frame, (640, 640))

    h, w = frame.shape[:2]
    ppm = min(w / 4, h / 3)

    composed = frame.copy()
    labels = []

    for _ in range(int(rng.integers(0, 4))):
        composed, bbox = build_rubble(composed, obstacles, ppm)
        labels.append(yolo_bbox(bbox, w, h))

    split = "val" if rng.uniform() < 0.2 else "train"

    img = OUT_DIR / f"images/{split}/img_{i:06d}.jpg"
    lbl = OUT_DIR / f"labels/{split}/img_{i:06d}.txt"

    cv2.imwrite(str(img), composed)
    lbl.write_text("\n".join(labels))

    if (i + 1) % 100 == 0:
        print(i + 1)

cap.release()
print("Done")
