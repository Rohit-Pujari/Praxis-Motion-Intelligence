from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

from .models import Landmark, PoseFrame, PoseSequence


class PoseEstimator(ABC):
    @abstractmethod
    def estimate(self, video_path: Path) -> Optional[PoseSequence]:
        raise NotImplementedError


class PoseEstimationError(RuntimeError):
    """Raised when a real video cannot be converted into landmarks."""


def _resolve_pose_model_path() -> Optional[Path]:
    candidates = [
        os.environ.get("PRAXIS_POSE_MODEL_PATH", "").strip(),
        os.environ.get("MEDIAPIPE_POSE_MODEL_PATH", "").strip(),
        "models/pose_landmarker_lite.task",
        "models/pose_landmarker.task",
        "data/models/pose_landmarker_lite.task",
        "data/models/pose_landmarker.task",
        "pose_landmarker_lite.task",
        "pose_landmarker.task",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists():
            return path
    return None


class MediaPipePoseEstimator(PoseEstimator):
    def __init__(self) -> None:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore

        self.cv2 = cv2
        self.mp = mp
        self.backend_name = "unknown"
        self.pose = None
        self.landmarker = None
        self.landmark_names = {
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

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self.backend_name = "mediapipe-solutions"
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return

        model_path = _resolve_pose_model_path()
        if model_path is None:
            raise PoseEstimationError(
                "Installed MediaPipe exposes the Tasks API, which requires a pose model asset file. "
                "Add 'pose_landmarker.task' at 'models/pose_landmarker.task' or set PRAXIS_POSE_MODEL_PATH."
            )

        try:
            tasks = mp.tasks
            vision = tasks.vision
            base_options = tasks.BaseOptions(model_asset_path=str(model_path))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            self.backend_name = f"mediapipe-tasks:{model_path.name}"
        except Exception as exc:
            raise PoseEstimationError(f"MediaPipe Tasks backend failed to initialize: {exc}") from exc

    def estimate(self, video_path: Path) -> Optional[PoseSequence]:
        cap = self.cv2.VideoCapture(str(video_path))
        fps = cap.get(self.cv2.CAP_PROP_FPS) or 24.0
        frames, decoded_frame_count = self._estimate_from_capture(cap, fps)
        cap.release()
        if decoded_frame_count == 0:
            frames = self._estimate_from_ffmpeg(video_path, fps)
        if not frames:
            raise PoseEstimationError(
                "Video was decoded but no human pose landmarks were detected. "
                "Check that the person is clearly visible full-body, the clip is upright, and the subject occupies enough of the frame."
            )
        return PoseSequence(label=video_path.stem, fps=fps, frames=frames, source_type="video")

    def _estimate_from_capture(self, cap, fps: float) -> tuple[list[PoseFrame], int]:
        frames = []
        decoded_frames = 0
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded_frames += 1
            landmarks = self._extract_landmarks(frame, index, fps)
            if not landmarks:
                index += 1
                continue
            frames.append(PoseFrame(timestamp=index / fps, landmarks=landmarks))
            index += 1
        return frames, decoded_frames

    def _estimate_from_ffmpeg(self, video_path: Path, fps: float) -> list[PoseFrame]:
        if not shutil.which("ffmpeg"):
            raise PoseEstimationError(
                "OpenCV could not decode any frames from the uploaded video, and ffmpeg is unavailable for fallback decoding."
            )
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_pattern = str(Path(tmp_dir) / "frame_%05d.png")
            cmd = [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(video_path),
                "-vf",
                "fps=12",
                frame_pattern,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise PoseEstimationError(
                    "Neither OpenCV nor ffmpeg could decode the uploaded video into frames."
                )
            frame_paths = sorted(Path(tmp_dir).glob("frame_*.png"))
            if not frame_paths:
                raise PoseEstimationError(
                    "The uploaded video could not be decoded into image frames."
                )
            frames: list[PoseFrame] = []
            for index, frame_path in enumerate(frame_paths):
                frame = self.cv2.imread(str(frame_path))
                if frame is None:
                    continue
                landmarks = self._extract_landmarks(frame, index, fps)
                if not landmarks:
                    continue
                frames.append(PoseFrame(timestamp=index / max(fps, 1.0), landmarks=landmarks))
            return frames

    def _extract_landmarks(self, frame, index: int, fps: float) -> Dict[str, Landmark]:
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        if self.pose is not None:
            result = self.pose.process(rgb)
            if not getattr(result, "pose_landmarks", None):
                return {}
            return {
                landmark_index_name: Landmark(point.x, point.y, point.z, getattr(point, "visibility", 1.0))
                for landmark_index, landmark_index_name in self.landmark_names.items()
                for point in [result.pose_landmarks.landmark[landmark_index]]
            }

        if self.landmarker is not None:
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((index / max(fps, 1.0)) * 1000)
            result = self.landmarker.detect_for_video(image, timestamp_ms)
            pose_landmarks = getattr(result, "pose_landmarks", None) or []
            if not pose_landmarks:
                return {}
            first_pose = pose_landmarks[0]
            landmarks: Dict[str, Landmark] = {}
            for landmark_index, name in self.landmark_names.items():
                if landmark_index >= len(first_pose):
                    continue
                point = first_pose[landmark_index]
                landmarks[name] = Landmark(
                    x=point.x,
                    y=point.y,
                    z=getattr(point, "z", 0.0),
                    visibility=getattr(point, "visibility", getattr(point, "presence", 1.0)),
                )
            return landmarks

        return {}


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


def pose_backend_status() -> tuple[bool, str]:
    try:
        estimator = MediaPipePoseEstimator()
        return True, f"MediaPipe pose backend is available ({estimator.backend_name})."
    except Exception as exc:
        return False, f"MediaPipe pose backend failed to initialize: {exc}"


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
