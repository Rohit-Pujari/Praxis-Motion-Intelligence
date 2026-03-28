from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
            frames = self._estimate_from_ffmpeg(video_path)
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

    def _estimate_from_ffmpeg(self, video_path: Path) -> list[PoseFrame]:
        if not shutil.which("ffmpeg"):
            raise PoseEstimationError(
                "OpenCV could not decode any frames from the uploaded video, and ffmpeg is unavailable for fallback decoding."
            )
        fallback_fps = 12.0
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_pattern = str(Path(tmp_dir) / "frame_%05d.png")
            cmd = [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(video_path),
                "-vf",
                f"fps={fallback_fps:g}",
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
                landmarks = self._extract_landmarks(frame, index, fallback_fps)
                if not landmarks:
                    continue
                frames.append(PoseFrame(timestamp=index / fallback_fps, landmarks=landmarks))
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


# Skeleton connections for stickman overlay
SKELETON_CONNECTIONS: List[Tuple[str, str]] = [
    # Torso
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # Left arm
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    # Right arm
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # Left leg
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    # Right leg
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def generate_overlay_video(
    video_path: Path,
    sequence: PoseSequence,
    joint_overlay_colors: Optional[Dict[str, str]] = None,
    joint_importance: Optional[Dict[str, float]] = None,
    output_width: int = 640,
    output_fps: float = 15.0,
) -> Optional[Tuple[str, str]]:
    """Generate a browser-safe overlay video and return (mime_type, base64_payload)."""
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    if not sequence.frames:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    # Calculate output height maintaining aspect ratio
    scale = output_width / original_width
    output_height = int(original_height * scale)

    # Create temporary output file
    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_output_path = Path(temp_output.name)
    temp_output.close()

    effective_fps = min(original_fps, output_fps) if output_fps > 0 else original_fps
    effective_fps = max(effective_fps, 1.0)
    frame_stride = max(1, round(original_fps / effective_fps))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(temp_output_path), fourcc, effective_fps, (output_width, output_height))
    if not out.isOpened():
        cap.release()
        return None

    # Build frame index map for pose landmarks
    frame_timestamps = [frame.timestamp for frame in sequence.frames]
    frame_landmarks = [frame.landmarks for frame in sequence.frames]

    frame_index = 0
    pose_index = 0
    total_pose_frames = len(sequence.frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_stride != 0:
            frame_index += 1
            continue

        # Resize frame
        frame = cv2.resize(frame, (output_width, output_height))

        # Calculate current timestamp
        current_time = frame_index / original_fps

        # Find closest pose frame
        while pose_index < total_pose_frames - 1:
            next_time = frame_timestamps[pose_index + 1]
            if abs(current_time - frame_timestamps[pose_index]) <= abs(current_time - next_time):
                break
            pose_index += 1

        # Draw skeleton if we have landmarks
        if pose_index < total_pose_frames:
            landmarks = frame_landmarks[pose_index]
            _draw_skeleton(
                frame,
                landmarks,
                output_width,
                output_height,
                joint_overlay_colors=joint_overlay_colors,
                joint_importance=joint_importance,
            )

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()

    # Read video as base64
    try:
        webm_path = _transcode_overlay_to_webm(temp_output_path)
        if webm_path is not None:
            payload = _encode_video_file(webm_path)
            webm_path.unlink(missing_ok=True)
            temp_output_path.unlink(missing_ok=True)
            return ("video/webm", payload)

        payload = _encode_video_file(temp_output_path)
        temp_output_path.unlink(missing_ok=True)
        return ("video/mp4", payload)
    except Exception:
        temp_output_path.unlink(missing_ok=True)
        return None


def _transcode_overlay_to_webm(source_path: Path) -> Optional[Path]:
    if not shutil.which("ffmpeg"):
        return None

    target_path = source_path.with_suffix(".webm")
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(source_path),
        "-an",
        "-c:v",
        "libvpx-vp9",
        "-crf",
        "36",
        "-b:v",
        "0",
        str(target_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not target_path.exists():
        target_path.unlink(missing_ok=True)
        return None
    return target_path


def _encode_video_file(path: Path) -> str:
    with path.open("rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def _draw_skeleton(
    frame,
    landmarks: Dict[str, Landmark],
    width: int,
    height: int,
    joint_overlay_colors: Optional[Dict[str, str]] = None,
    joint_importance: Optional[Dict[str, float]] = None,
) -> None:
    """Draw stickman skeleton on frame."""
    import cv2  # type: ignore

    # Colors for different body parts (BGR format)
    colors = {
        "torso": (0, 200, 255),      # Cyan
        "left_arm": (0, 255, 100),    # Green
        "right_arm": (255, 100, 0),   # Blue
        "left_leg": (255, 0, 200),    # Magenta
        "right_leg": (0, 165, 255),   # Orange
    }

    connection_colors = {
        # Torso
        ("left_shoulder", "right_shoulder"): colors["torso"],
        ("left_shoulder", "left_hip"): colors["torso"],
        ("right_shoulder", "right_hip"): colors["torso"],
        ("left_hip", "right_hip"): colors["torso"],
        # Left arm
        ("left_shoulder", "left_elbow"): colors["left_arm"],
        ("left_elbow", "left_wrist"): colors["left_arm"],
        # Right arm
        ("right_shoulder", "right_elbow"): colors["right_arm"],
        ("right_elbow", "right_wrist"): colors["right_arm"],
        # Left leg
        ("left_hip", "left_knee"): colors["left_leg"],
        ("left_knee", "left_ankle"): colors["left_leg"],
        # Right leg
        ("right_hip", "right_knee"): colors["right_leg"],
        ("right_knee", "right_ankle"): colors["right_leg"],
    }
    severity_colors = {
        "green": (0, 255, 0),
        "yellow": (0, 215, 255),
        "red": (0, 0, 255),
    }

    # Convert normalized coordinates to pixel coordinates
    points = {}
    for name, lm in landmarks.items():
        x = int(lm.x * width)
        y = int(lm.y * height)
        points[name] = (x, y)

    # Draw connections
    for start_name, end_name in SKELETON_CONNECTIONS:
        if start_name in points and end_name in points:
            color = connection_colors.get((start_name, end_name), (0, 255, 255))
            start_color = severity_colors.get((joint_overlay_colors or {}).get(start_name, ""), color)
            end_color = severity_colors.get((joint_overlay_colors or {}).get(end_name, ""), color)
            if start_color == severity_colors["red"] or end_color == severity_colors["red"]:
                color = severity_colors["red"]
            elif start_color == severity_colors["yellow"] or end_color == severity_colors["yellow"]:
                color = severity_colors["yellow"]
            elif start_color == severity_colors["green"] or end_color == severity_colors["green"]:
                color = severity_colors["green"]
            edge_importance = max(
                float((joint_importance or {}).get(start_name, 0.0)),
                float((joint_importance or {}).get(end_name, 0.0)),
            )
            thickness = 2 + int(round(edge_importance * 5.0))
            cv2.line(frame, points[start_name], points[end_name], color, thickness, cv2.LINE_AA)

    # Draw joint points
    for name, point in points.items():
        joint_color = severity_colors.get((joint_overlay_colors or {}).get(name, ""), (255, 255, 255))
        importance = float((joint_importance or {}).get(name, 0.0))
        outer_radius = 5 + int(round(importance * 8.0))
        inner_radius = max(2, outer_radius - 2)
        cv2.circle(frame, point, outer_radius, joint_color, -1, cv2.LINE_AA)
        cv2.circle(frame, point, inner_radius, (0, 0, 0), -1, cv2.LINE_AA)
