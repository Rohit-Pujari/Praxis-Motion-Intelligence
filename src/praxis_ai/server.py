from __future__ import annotations

import html
import json
import tempfile
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Tuple
from urllib.parse import parse_qs

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .analysis import analyze_pose, compute_joint_series, match_reference
from .pose_estimation import JsonPoseEstimator, available_pose_estimator, load_pose_sequence, probe_video
from .reference_data import load_reference_patterns
from .rehab import detect_limitations, recommend_exercises
from .reporting import report_context


BASE_DIR = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = BASE_DIR / "src" / "praxis_ai" / "templates"
DEMO_DIR = BASE_DIR / "data" / "demo_landmarks"
ENV = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=select_autoescape())


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
        if self.path not in ("/", "/index.html"):
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        demos = sorted(path.stem for path in DEMO_DIR.glob("*.json"))
        template = ENV.get_template("index.html")
        page = template.render(demos=demos)
        self._send_html(page)

    def do_POST(self) -> None:
        if self.path != "/analyze":
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
            template = ENV.get_template("index.html")
            page = template.render(demos=sorted(path.stem for path in DEMO_DIR.glob("*.json")), error=html.escape(str(exc)))
            self._send_html(page, status=HTTPStatus.BAD_REQUEST)
            return
        page = ENV.get_template("report.html").render(**report_context(report))
        self._send_html(page)

    def _run_analysis(self, form: Dict[str, object]):
        estimator = available_pose_estimator()
        json_estimator = JsonPoseEstimator()
        sequence = None
        metadata: Dict[str, str] = {}

        demo_name = str(form.get("demo_name", "")).strip()
        landmarks_json = str(form.get("landmarks_json", "")).strip()
        video_file = form.get("video_file")

        if demo_name:
            sequence = load_pose_sequence(DEMO_DIR / f"{demo_name}.json")
        elif landmarks_json:
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
            sequence = estimator.estimate(temp_video) if estimator else None
            if sequence is None:
                sequence = json_estimator.estimate(temp_video)
            if sequence is None:
                demo_default = "stroke_gait_asymmetry"
                sequence = load_pose_sequence(DEMO_DIR / f"{demo_default}.json")
                sequence.metadata["fallback_reason"] = (
                    "Vision pose extraction is unavailable in this local environment, so analysis used the closest bundled demo flow."
                )
            temp_video.unlink(missing_ok=True)
        else:
            raise ValueError("Provide a video, a landmark JSON sequence, or choose a demo.")

        sequence.metadata.update(metadata)
        joint_series = compute_joint_series(sequence)
        matched_reference, _, _ = match_reference(joint_series, BASE_DIR)
        relevant_joints = set(load_reference_patterns(BASE_DIR).get(matched_reference, {}).get("joint_patterns", {}).keys())
        limitations = detect_limitations(joint_series, BASE_DIR, relevant_joints=relevant_joints or None)
        exercises = recommend_exercises(limitations, matched_reference)
        return analyze_pose(sequence, limitations, exercises, BASE_DIR)

    def _send_html(self, page: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = page.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), PraxisHandler)
    print(f"Praxis Motion Intelligence running on http://{host}:{port}")
    server.serve_forever()
