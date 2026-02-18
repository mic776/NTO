import math
from dataclasses import dataclass


@dataclass
class Track:
    track_id: int
    centroid: tuple
    last_seen: int
    label: str | None = None


class CentroidTracker:
    def __init__(self, max_distance=60, max_age=20):
        self.max_distance = max_distance
        self.max_age = max_age
        self.next_id = 1
        self.tracks = {}
        self.frame_idx = 0
        self.counts = {"IMPASSABLE": 0, "PASSABLE": 0}

    def _distance(self, c1, c2):
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    def update(self, bboxes):
        self.frame_idx += 1
        detections = []
        for (x1, y1, x2, y2) in bboxes:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            detections.append((cx, cy))

        assigned = {}
        used_tracks = set()
        track_ids = list(self.tracks.keys())

        for det_idx, det in enumerate(detections):
            best_track = None
            best_dist = self.max_distance
            for tid in track_ids:
                if tid in used_tracks:
                    continue
                dist = self._distance(det, self.tracks[tid].centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_track = tid
            if best_track is not None:
                assigned[det_idx] = best_track
                used_tracks.add(best_track)

        results = []
        for i, det in enumerate(detections):
            if i in assigned:
                tid = assigned[i]
                self.tracks[tid].centroid = det
                self.tracks[tid].last_seen = self.frame_idx
                results.append((tid, False))
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = Track(tid, det, self.frame_idx)
                results.append((tid, True))

        self._prune()
        return results

    def _prune(self):
        to_delete = []
        for tid, trk in self.tracks.items():
            if self.frame_idx - trk.last_seen > self.max_age:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

    def set_label(self, track_id, label):
        trk = self.tracks.get(track_id)
        if trk is None:
            return None
        if trk.label is None:
            trk.label = label
            self.counts[label] += 1
            return True
        if trk.label == "PASSABLE" and label == "IMPASSABLE":
            self.counts["PASSABLE"] -= 1
            self.counts["IMPASSABLE"] += 1
            trk.label = "IMPASSABLE"
            return True
        return False
