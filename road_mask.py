import cv2
import numpy as np


def _hsv_mask(hsv, lower, upper):
    return cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))


def build_lane_mask(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white = _hsv_mask(hsv, (0, 0, 180), (180, 40, 255))
    yellow = _hsv_mask(hsv, (15, 60, 160), (40, 255, 255))
    mask = cv2.bitwise_or(white, yellow)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def _fit_line(points):
    if len(points) < 2:
        return None
    xs = np.array([p[0] for p in points], dtype=np.float32)
    ys = np.array([p[1] for p in points], dtype=np.float32)
    A = np.vstack([xs, np.ones_like(xs)]).T
    m, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    return m, b


def _line_points(m, b, y1, y2):
    if m is None:
        return None
    x1 = int((y1 - b) / m) if m != 0 else 0
    x2 = int((y2 - b) / m) if m != 0 else 0
    return (x1, int(y1)), (x2, int(y2))


def build_road_mask(frame_bgr, roi_ratio=0.6):
    h, w = frame_bgr.shape[:2]
    lane_mask = build_lane_mask(frame_bgr)

    roi_y = int(h * (1 - roi_ratio))
    roi = lane_mask[roi_y:h, :]
    edges = cv2.Canny(roi, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=40, maxLineGap=80)
    left_pts = []
    right_pts = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:
                continue
            if slope < 0:
                left_pts.append((x1, y1 + roi_y))
                left_pts.append((x2, y2 + roi_y))
            else:
                right_pts.append((x1, y1 + roi_y))
                right_pts.append((x2, y2 + roi_y))

    left_line = _fit_line(left_pts)
    right_line = _fit_line(right_pts)

    road_mask = np.zeros((h, w), dtype=np.uint8)
    if left_line and right_line:
        y_bottom = h - 1
        y_top = roi_y
        left_bottom, left_top = _line_points(left_line[0], left_line[1], y_bottom, y_top)
        right_bottom, right_top = _line_points(right_line[0], right_line[1], y_bottom, y_top)
        if left_bottom and right_bottom:
            poly = np.array([left_bottom, right_bottom, right_top, left_top], dtype=np.int32)
            cv2.fillPoly(road_mask, [poly], 255)

    return road_mask, lane_mask
