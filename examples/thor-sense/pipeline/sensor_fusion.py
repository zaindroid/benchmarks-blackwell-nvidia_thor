"""Sensor fusion pipeline (reference): cameras + LiDAR -> BEV occupancy
and fused 3D objects.

Projection uses a pinhole camera model; LiDAR points are binned into the
BEV grid; camera 2D detections are lifted to 3D using LiDAR depth and
associated by BEV IoU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from models.bevformer import BEVGrid


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics (pixels)."""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Detection2D:
    """A camera-space 2D detection (pixels)."""

    cls: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Detection3D:
    """A fused 3D object in vehicle coordinates (meters, x forward)."""

    cls: str
    confidence: float
    x: float
    y: float
    z: float = 0.0
    w: float = 1.8
    l: float = 4.5
    h: float = 1.5


def project_to_image(points3d: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Project (N, 3) camera-frame points to (N, 2) pixels (pinhole)."""
    pts = np.asarray(points3d, dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intrinsics.fx * x / z + intrinsics.cx
        v = intrinsics.fy * y / z + intrinsics.cy
    return np.stack([u, v], axis=-1)


def lidar_to_bev(points3d: np.ndarray, grid: BEVGrid) -> np.ndarray:
    """Bin (N, 3) lidar points (x forward, y lateral) into a BEV grid.

    Returns an occupancy grid of shape (gx, gy) with normalized counts.
    """
    pts = np.asarray(points3d, dtype=float)
    occ = np.zeros((grid.gx, grid.gy), dtype=float)
    if pts.size == 0:
        return occ
    ix = ((pts[:, 0] - grid.x_range[0]) / grid.resolution).astype(int)
    iy = ((pts[:, 1] - grid.y_range[0]) / grid.resolution).astype(int)
    valid = (ix >= 0) & (ix < grid.gx) & (iy >= 0) & (iy < grid.gy)
    np.add.at(occ, (ix[valid], iy[valid]), 1.0)
    if occ.max() > 0:
        occ = occ / occ.max()
    return occ


def box_center_2d(det: Detection2D) -> tuple[float, float]:
    return ((det.x1 + det.x2) / 2.0, (det.y1 + det.y2) / 2.0)


def lift_camera_detection(det: Detection2D, intrinsics: CameraIntrinsics,
                          depth: float) -> Detection3D:
    """Lift a 2D detection to 3D vehicle coordinates given a depth (m)."""
    u, v = box_center_2d(det)
    x = depth
    y = (u - intrinsics.cx) * depth / intrinsics.fx
    z = (v - intrinsics.cy) * depth / intrinsics.fy
    return Detection3D(cls=det.cls, confidence=det.confidence,
                       x=float(x), y=float(y), z=float(z))


def bev_iou(a: Detection3D, b: Detection3D) -> float:
    """IoU of two 3D boxes in the BEV plane (x forward, y lateral)."""
    ax1, ax2 = a.x - a.l / 2, a.x + a.l / 2
    ay1, ay2 = a.y - a.w / 2, a.y + a.w / 2
    bx1, bx2 = b.x - b.l / 2, b.x + b.l / 2
    by1, by2 = b.y - b.w / 2, b.y + b.w / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = (a.l * a.w) + (b.l * b.w) - inter
    return inter / union if union > 0 else 0.0


class SensorFusionPipeline:
    """Combine camera detections + lidar points into BEV outputs."""

    def __init__(self, intrinsics: CameraIntrinsics,
                 grid: Optional[BEVGrid] = None,
                 iou_threshold: float = 0.2):
        self.intrinsics = intrinsics
        self.grid = grid or BEVGrid()
        self.iou_threshold = iou_threshold

    def run(self, detections: Sequence[Detection2D],
            lidar_points: np.ndarray,
            lidar_depth_map: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
        """Fuse a frame. Returns BEV occupancy + fused 3D objects."""
        bev = lidar_to_bev(lidar_points, self.grid)

        # Lift camera detections to 3D using per-detection depth when
        # provided (e.g. from a sparse depth map); otherwise use the
        # median lidar depth.
        median_depth = float(np.median(lidar_points[:, 0])) if len(lidar_points) else 25.0
        fused: List[Detection3D] = []
        for det in detections:
            depth = median_depth
            if lidar_depth_map is not None:
                depth = lidar_depth_map.get(id(det), depth)
            obj = lift_camera_detection(det, self.intrinsics, depth)
            # associate with an existing fused object by BEV IoU
            merged = False
            for existing in fused:
                if existing.cls == obj.cls and bev_iou(existing, obj) >= self.iou_threshold:
                    existing.x = (existing.x + obj.x) / 2
                    existing.y = (existing.y + obj.y) / 2
                    existing.confidence = max(existing.confidence, obj.confidence)
                    merged = True
                    break
            if not merged:
                fused.append(obj)

        return {
            "bev_occupancy": bev,
            "objects": [vars(o) for o in fused],
            "grid": self.grid.to_config(),
        }
