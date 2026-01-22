# Fight Club

Pose-based spatiotemporal action classification for combat sports using MediaPipe and PyTorch.

## Project Overview

This project implements a machine learning pipeline to classify boxing actions from video footage. It extracts human pose keypoints using MediaPipe and classifies action sequences using a temporal convolutional neural network.

## Module Index

The `src/` directory is organized by pipeline stage:

| Module | Purpose |
|--------|---------|
| `config/` | Video quality thresholds, codec mappings, and configuration dataclasses |
| `data/` | Action taxonomy (13-class boxing enum), dataset classes, and data preparation utilities |
| `pose/` | Pose extraction via MediaPipe, keypoint normalization, and video-level processing |
| `models/` | Neural network architecture (ActionClassifier with temporal convolutions) |
| `training/` | Training loop, loss functions, and optimization utilities |
| `visualization/` | Frame extraction, results plotting, and classification visualization |
| `pipeline/` | End-to-end orchestration combining all modules |

### Data Flow

```
Video Input
    │
    ▼
┌─────────────────┐
│  config         │  ← Video quality validation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  pose           │  ← MediaPipe keypoint extraction
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  data           │  ← Sequence windowing, dataset construction
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  models         │  ← Action classification inference
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  visualization  │  ← Results display
└─────────────────┘
```

## Action Taxonomy

The classifier recognizes 13 boxing actions:

| ID | Action | Category |
|----|--------|----------|
| 0 | Jab (Head) | Straight |
| 1 | Jab (Body) | Straight |
| 2 | Cross (Head) | Straight |
| 3 | Cross (Body) | Straight |
| 4 | Lead Hook (Head) | Hook |
| 5 | Lead Hook (Body) | Hook |
| 6 | Rear Hook (Head) | Hook |
| 7 | Rear Hook (Body) | Hook |
| 8 | Lead Uppercut | Uppercut |
| 9 | Rear Uppercut | Uppercut |
| 10 | Overhand | Other |
| 11 | Defensive Movement | Other |
| 12 | Idle/Stance | Other |

## Installation

```bash
uv sync
```

## Usage

```python
from src.pipeline import run_full_analysis

results = run_full_analysis("path/to/video.mp4")
```

## Requirements

- Python 3.11+
- PyTorch
- MediaPipe
- OpenCV
