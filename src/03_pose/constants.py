"""Keypoint indices and mappings for pose estimation."""

# COCO format defines 17 keypoints
COCO_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Upper body keypoints most relevant for boxing action classification
BOXING_KEYPOINTS = [
    COCO_KEYPOINTS["left_shoulder"],
    COCO_KEYPOINTS["right_shoulder"],
    COCO_KEYPOINTS["left_elbow"],
    COCO_KEYPOINTS["right_elbow"],
    COCO_KEYPOINTS["left_wrist"],
    COCO_KEYPOINTS["right_wrist"],
    COCO_KEYPOINTS["left_hip"],
    COCO_KEYPOINTS["right_hip"],
]

# MediaPipe uses 33 landmarks; map to COCO 17-keypoint format
MEDIAPIPE_TO_COCO = {
    0: 0,  # nose
    2: 1,  # left_eye (MP left_eye_inner -> COCO left_eye)
    5: 2,  # right_eye (MP right_eye_inner -> COCO right_eye)
    7: 3,  # left_ear
    8: 4,  # right_ear
    11: 5,  # left_shoulder
    12: 6,  # right_shoulder
    13: 7,  # left_elbow
    14: 8,  # right_elbow
    15: 9,  # left_wrist
    16: 10,  # right_wrist
    23: 11,  # left_hip
    24: 12,  # right_hip
    25: 13,  # left_knee
    26: 14,  # right_knee
    27: 15,  # left_ankle
    28: 16,  # right_ankle
}

# MediaPipe landmark indices for boxing-relevant keypoints
MEDIAPIPE_LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}
