"""Tests for the thor-sense sensor-fusion reference implementation."""

import numpy as np
import pytest

from models.bevformer import BEVGrid
from models.encoder import CameraEncoder, LidarEncoder
from pipeline.sensor_fusion import (
    CameraIntrinsics,
    Detection2D,
    Detection3D,
    SensorFusionPipeline,
    lift_camera_detection,
    lidar_to_bev,
    project_to_image,
)
from pipeline.tracker import ObjectTracker

torch = pytest.importorskip("torch")


def test_project_to_image_center_maps_to_principal_point():
    intrinsics = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0)
    # point directly ahead at 10 m depth (x=0, y=0, z=10) -> image center
    pixels = project_to_image(np.array([[0.0, 0.0, 10.0]]), intrinsics)
    np.testing.assert_allclose(pixels[0], [640.0, 360.0], atol=1e-3)


def test_lidar_to_bev_occupancy():
    grid = BEVGrid(x_range=(-10, 10), y_range=(-10, 10), resolution=1.0)
    points = np.array([[2.5, 0.0, 0.0], [3.4, 0.0, 0.0], [-5.0, 5.0, 0.0]])
    bev = lidar_to_bev(points, grid)
    assert bev.shape == (grid.gx, grid.gy)
    assert bev.max() == 1.0
    # two points in the same cell -> cell has the max count
    assert (bev == 1.0).sum() >= 2


def test_lift_camera_detection():
    intrinsics = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0)
    det = Detection2D(cls="car", confidence=0.9, x1=540, y1=260, x2=740, y2=460)
    obj = lift_camera_detection(det, intrinsics, depth=20.0)
    # centered detection at 20 m -> x=20, y≈0
    assert abs(obj.x - 20.0) < 1e-3
    assert abs(obj.y) < 1e-3


def test_sensor_fusion_pipeline():
    intrinsics = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0)
    pipeline = SensorFusionPipeline(intrinsics)
    detections = [
        Detection2D("car", 0.9, 540, 260, 740, 460),
        Detection2D("car", 0.8, 550, 270, 730, 450),
    ]
    lidar = np.array([[20.0, -0.1, 0.0], [20.1, 0.1, 0.0], [15.0, 3.0, 0.0]])
    out = pipeline.run(detections, lidar)
    assert out["bev_occupancy"].shape == (out["grid"]["grid"][0], out["grid"]["grid"][1])
    # overlapping camera detections at the same depth merge into one object
    assert len(out["objects"]) == 1
    assert out["objects"][0]["cls"] == "car"


def test_camera_encoder_forward():
    encoder = CameraEncoder()
    out = encoder(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 64, 16, 16)


def test_lidar_encoder_forward():
    encoder = LidarEncoder(out_features=32)
    out = encoder(torch.randn(4, 100, 3))
    assert out.shape == (4, 32)


def test_tracker_persists_track_across_frames():
    tracker = ObjectTracker()
    det = Detection3D(cls="car", confidence=0.9, x=10.0, y=0.0)
    t1 = tracker.update([det], timestamp=0.0)
    assert len(t1) == 1
    track_id = t1[0].track_id
    moved = Detection3D(cls="car", confidence=0.9, x=10.5, y=0.0)
    t2 = tracker.update([moved], timestamp=1.0)
    assert len(t2) == 1
    assert t2[0].track_id == track_id
    assert t2[0].x == 10.5
