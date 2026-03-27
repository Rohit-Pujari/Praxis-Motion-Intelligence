from __future__ import annotations

import json
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

from .models import Landmark, PoseFrame, PoseSequence


class PoseEstimator(ABC):
    @abstractmethod
    def estimate(self, video_path: Path) -> Optional[PoseSequence]:
        raise NotImplementedError


class MediaPipePoseEstimator(PoseEstimator):
    def __init__(self) -> None:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore

        self.cv2 = cv2
        self.mp = mp
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def estimate(self, video_path: Path) -> Optional[PoseSequence]:
        cap = self.cv2.VideoCapture(str(video_path))
        fps = cap.get(self.cv2.CAP_PROP_FPS) or 24.0
        frames = []
        names = {
            0: "nose",
            11: "left_shoulder",
            12: "right_shoulder",
            13: "left_elbow",
            14: "right_elbow",
            15: "left_wrist",
            16: "right_wrist",
            23: "left_hip",
            24: "right_hip",
            25: "left_knee",
            26: "right_knee",
            27: "left_ankle",
            28: "right_ankle",
        }
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb)
            if not result.pose_landmarks:
                index += 1
                continue
            landmarks: Dict[str, Landmark] = {}
            for landmark_index, name in names.items():
                point = result.pose_landmarks.landmark[landmark_index]
                landmarks[name] = Landmark(point.x, point.y, point.z, point.visibility)
            frames.append(PoseFrame(timestamp=index / fps, landmarks=landmarks))
            index += 1
        cap.release()
        if not frames:
            return None
        return PoseSequence(label=video_path.stem, fps=fps, frames=frames, source_type="video")


class JsonPoseEstimator(PoseEstimator):
    def estimate(self, video_path: Path) -> Optional[PoseSequence]:
        sidecar = video_path.with_suffix(".pose.json")
        if not sidecar.exists():
            return None
        return load_pose_sequence(sidecar)


def probe_video(video_path: Path) -> Dict[str, str]:
    if not shutil.which("ffprobe"):
        return {"filename": video_path.name}
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {"filename": video_path.name}
    payload = json.loads(result.stdout or "{}")
    stream = (payload.get("streams") or [{}])[0]
    return {
        "filename": video_path.name,
        "width": str(stream.get("width", "")),
        "height": str(stream.get("height", "")),
        "duration_seconds": str(stream.get("duration", "")),
        "frame_rate": str(stream.get("r_frame_rate", "")),
    }


def available_pose_estimator() -> Optional[PoseEstimator]:
    try:
        return MediaPipePoseEstimator()
    except Exception:
        return None


def load_pose_sequence(path: Path) -> PoseSequence:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    frames = []
    for frame in payload["frames"]:
        landmarks = {
            name: Landmark(
                x=value["x"],
                y=value["y"],
                z=value.get("z", 0.0),
                visibility=value.get("visibility", 1.0),
            )
            for name, value in frame["landmarks"].items()
        }
        frames.append(PoseFrame(timestamp=frame["timestamp"], landmarks=landmarks))
    return PoseSequence(
        label=payload.get("label", path.stem),
        fps=float(payload.get("fps", 24.0)),
        frames=frames,
        source_type=payload.get("source_type", "landmarks"),
        metadata=payload.get("metadata", {}),
    )
