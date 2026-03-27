from __future__ import annotations

import json
import mimetypes
import tempfile
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict
from urllib.parse import parse_qs

from .analysis import active_joint_names, analyze_pose, compute_joint_series
from .pose_estimation import (
    JsonPoseEstimator,
    available_pose_estimator,
    load_pose_sequence,
    pose_backend_status,
    probe_video,
)
from .rehab import detect_limitations, recommend_exercises
from .reporting import serialize_report


BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"


def parse_multipart(headers, body: bytes) -> Dict[str, object]:
    content_type = headers.get("Content-Type", "")
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
    )
    data: Dict[str, object] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            data[name] = {"filename": filename, "content": payload}
        else:
            data[name] = payload.decode(part.get_content_charset() or "utf-8")
    return data


class PraxisHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._send_json({"status": "ok"})
            return
        self._serve_frontend_asset()

    def do_POST(self) -> None:
        if self.path != "/api/analyze":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" in content_type:
            form = parse_multipart(self.headers, body)
        else:
            form = {key: values[0] for key, values in parse_qs(body.decode("utf-8")).items()}
        try:
            report = self._run_analysis(form)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(serialize_report(report))

    def _run_analysis(self, form: Dict[str, object]):
        estimator = available_pose_estimator()
        json_estimator = JsonPoseEstimator()
        sequence = None
        metadata: Dict[str, str] = {}

        landmarks_json = str(form.get("landmarks_json", "")).strip()
        video_file = form.get("video_file")

        if landmarks_json:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
                handle.write(landmarks_json)
                temp_json = Path(handle.name)
            sequence = load_pose_sequence(temp_json)
            temp_json.unlink(missing_ok=True)
        elif isinstance(video_file, dict):
            filename = str(video_file["filename"])
            content = video_file["content"]
            with tempfile.NamedTemporaryFile("wb", suffix=Path(filename).suffix or ".mp4", delete=False) as handle:
                handle.write(content)
                temp_video = Path(handle.name)
            metadata = probe_video(temp_video)
            backend_ok, backend_message = pose_backend_status()
            metadata["pose_backend"] = backend_message
            if not backend_ok:
                temp_video.unlink(missing_ok=True)
                raise ValueError(backend_message)
            sequence = estimator.estimate(temp_video) if estimator else None
            if sequence is None:
                sequence = json_estimator.estimate(temp_video)
            if sequence is None:
                raise ValueError(
                    "Pose landmarks could not be extracted from the uploaded video. "
                    "Install a supported pose backend such as MediaPipe/OpenCV, provide a sidecar '.pose.json' file, "
                    "or paste landmark JSON directly."
                )
            temp_video.unlink(missing_ok=True)
        else:
            raise ValueError("Provide a video file or paste landmark JSON.")

        sequence.metadata.update(metadata)
        joint_series = compute_joint_series(sequence)
        relevant_joints = set(active_joint_names(joint_series))
        limitations = detect_limitations(joint_series, BASE_DIR, relevant_joints=relevant_joints or None)
        exercises = recommend_exercises(limitations, "")
        return analyze_pose(sequence, limitations, exercises)

    def _serve_frontend_asset(self) -> None:
        target = self.path.split("?", 1)[0].lstrip("/") or "index.html"
        asset_path = (FRONTEND_DIST_DIR / target).resolve()
        if FRONTEND_DIST_DIR not in asset_path.parents and asset_path != FRONTEND_DIST_DIR / "index.html":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        if not asset_path.exists() or asset_path.is_dir():
            asset_path = FRONTEND_DIST_DIR / "index.html"
        if not asset_path.exists():
            self._send_json(
                {
                    "error": "Frontend build not found. Run 'npm run build' in the project root."
                },
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return
        content_type = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
        payload = asset_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), PraxisHandler)
    print(f"Praxis Motion Intelligence running on http://{host}:{port}")
    server.serve_forever()
