"""Object tracker (reference): constant-velocity model with greedy
BEV IoU association. Tracks are keyed per class; stale tracks expire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pipeline.sensor_fusion import Detection3D, bev_iou


@dataclass
class Track:
    """A tracked object."""

    track_id: int
    cls: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    last_seen: float = 0.0
    age: int = 0
    hits: int = 1

    def predict(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt

    def update(self, det: Detection3D, t: float) -> None:
        dt = max(t - self.last_seen, 1e-6)
        self.vx = 0.8 * self.vx + 0.2 * (det.x - self.x) / dt
        self.vy = 0.8 * self.vy + 0.2 * (det.y - self.y) / dt
        self.x, self.y = det.x, det.y
        self.last_seen = t
        self.age += 1
        self.hits += 1


class ObjectTracker:
    """Constant-velocity tracker with greedy IoU association."""

    def __init__(self, iou_threshold: float = 0.3,
                 max_age_s: float = 2.0, next_id: int = 1):
        self.iou_threshold = iou_threshold
        self.max_age_s = max_age_s
        self._tracks: List[Track] = []
        self._next_id = next_id

    def update(self, detections: List[Detection3D], timestamp: float) -> List[Track]:
        """Associate detections to tracks and return active tracks."""
        for track in self._tracks:
            track.predict(dt=1.0)

        matched_tracks: set[int] = set()
        for det in detections:
            best, best_iou = None, self.iou_threshold
            for track in self._tracks:
                if track.track_id in matched_tracks:
                    continue
                if track.cls != det.cls:
                    continue
                iou = bev_iou(Detection3D(track.cls, 1.0, track.x, track.y),
                              det)
                if iou > best_iou:
                    best, best_iou = track, iou
            if best is not None:
                best.update(det, timestamp)
                matched_tracks.add(best.track_id)
            else:
                self._tracks.append(
                    Track(track_id=self._next_id, cls=det.cls,
                          x=det.x, y=det.y, last_seen=timestamp)
                )
                self._next_id += 1

        # prune stale tracks
        now = timestamp
        self._tracks = [t for t in self._tracks
                        if now - t.last_seen <= self.max_age_s]
        return list(self._tracks)
