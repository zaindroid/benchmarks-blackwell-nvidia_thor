"""Vision model zoo (detection / segmentation / classification)."""

from __future__ import annotations

from typing import Any, Dict

VISION_MODELS: Dict[str, Dict[str, Any]] = {
    "ultralytics/yolov8n": {
        "name": "YOLOv8-nano", "task": "detection", "architecture": "cnn",
        "parameters": 3150000, "source": "ultralytics", "license": "AGPL-3.0",
        "input_size": [640, 640],
    },
    "ultralytics/yolov8s": {
        "name": "YOLOv8-small", "task": "detection", "architecture": "cnn",
        "parameters": 11100000, "source": "ultralytics", "license": "AGPL-3.0",
        "input_size": [640, 640],
    },
    "ultralytics/yolov8m": {
        "name": "YOLOv8-medium", "task": "detection", "architecture": "cnn",
        "parameters": 25900000, "source": "ultralytics", "license": "AGPL-3.0",
        "input_size": [640, 640],
    },
    "ultralytics/yolov8l": {
        "name": "YOLOv8-large", "task": "detection", "architecture": "cnn",
        "parameters": 43600000, "source": "ultralytics", "license": "AGPL-3.0",
        "input_size": [640, 640],
    },
    "ultralytics/yolov8n-seg": {
        "name": "YOLOv8n-seg", "task": "segmentation", "architecture": "cnn",
        "parameters": 3260000, "source": "ultralytics", "license": "AGPL-3.0",
        "input_size": [640, 640],
    },
    "hustvl/detr-resnet50": {
        "name": "DETR-ResNet50", "task": "detection", "architecture": "transformer",
        "parameters": 41000000, "source": "huggingface", "license": "Apache-2.0",
        "input_size": [800, 800],
    },
    "hustvl/rt-detr": {
        "name": "RT-DETR", "task": "detection", "architecture": "transformer",
        "parameters": 20000000, "source": "huggingface", "license": "Apache-2.0",
        "input_size": [640, 640],
    },
    "nvidia/segformer-b0-finetuned-ade-512-512": {
        "name": "SegFormer-B0", "task": "segmentation", "architecture": "transformer",
        "parameters": 3700000, "source": "huggingface", "license": "Apache-2.0",
        "input_size": [512, 512],
    },
    "timm/resnet50": {
        "name": "ResNet-50", "task": "classification", "architecture": "cnn",
        "parameters": 25600000, "source": "timm", "license": "Apache-2.0",
        "input_size": [224, 224],
    },
    "google/vit-base-patch16-224": {
        "name": "ViT-Base/16", "task": "classification", "architecture": "transformer",
        "parameters": 86000000, "source": "huggingface", "license": "Apache-2.0",
        "input_size": [224, 224],
    },
}
