# Praxis Motion Intelligence

Praxis Motion Intelligence is a rehab-focused motion analysis prototype with a Python backend and React frontend. It accepts patient video uploads or landmark JSON, extracts pose trajectories, scores movement quality from joint kinematics, flags stroke-oriented movement limitations, and returns exercise guidance with an explainable report.

## What it does

- Accepts either uploaded video or pasted landmark JSON
- Extracts pose landmarks from video with MediaPipe when a backend is available
- Computes joint-angle series for elbows, shoulders, hips, and knees
- Scores movement using mobility, symmetry, and smoothness metrics
- Detects joint ROM deficits and left-right asymmetry using stroke-oriented thresholds
- Returns structured rehab recommendations and feedback through a JSON API
- Serves a React dashboard from the same Python process after a frontend build

## Runtime model

- The backend is implemented in `src/praxis_ai/server.py` and exposes:
  - `GET /api/health`
  - `POST /api/analyze`
- The frontend is a Vite + React app in `frontend/` and is served from `frontend/dist/`
- Video analysis requires a working MediaPipe/OpenCV-compatible environment
- If video landmark extraction fails, the app does not fall back to demo data
- Landmark JSON remains a direct input path for debugging, testing, and integrations

## Setup

```bash
uv sync
npm install
npm run build
UV_CACHE_DIR=.uv-cache uv run praxis
```

Then open `http://127.0.0.1:8000`.

## Input modes

- Upload a video file: metadata is extracted immediately and pose estimation is attempted
- Upload landmark JSON: runs the full analysis pipeline directly

## Landmark JSON format

```json
{
  "label": "demo_walking",
  "fps": 24,
  "frames": [
    {
      "timestamp": 0.0,
      "landmarks": {
        "left_shoulder": {"x": 0.45, "y": 0.30},
        "left_elbow": {"x": 0.42, "y": 0.44},
        "left_wrist": {"x": 0.40, "y": 0.58}
      }
    }
  ]
}
```

Expected landmark names follow common pose-estimation conventions:

`nose`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`.

## Output

Each analysis response includes:

- `overall_score`
- `mobility_score`
- `symmetry_score`
- `smoothness_score`
- joint-level scores and ROM summary rows
- detected limitations with severity and evidence
- suggested exercise guides
- human-readable feedback strings
- metadata such as detected backend status and active joints

## Project layout

- `main.py`: CLI entrypoint exposed as `praxis`
- `app.py`: direct Python launch shim
- `src/praxis_ai/server.py`: HTTP server, multipart parsing, and frontend asset serving
- `src/praxis_ai/pose_estimation.py`: video probing, MediaPipe extraction, and `.pose.json` sidecar loading
- `src/praxis_ai/analysis.py`: joint-angle computation and form scoring
- `src/praxis_ai/rehab.py`: limitation detection and exercise recommendation
- `src/praxis_ai/calibration.py`: ROM calibration loading
- `src/praxis_ai/reporting.py`: API response serialization
- `frontend/`: React application source
- `data/`: checked-in calibration, rules, demo landmarks, and reference assets
- `scripts/`: offline dataset processing and threshold-generation utilities
- `models/`: local MediaPipe task assets when required by the installed backend

## Dataset and calibration scripts

The repository ignores the local `dataset/` directory. That folder is expected to hold large, untracked source datasets used by the offline scripts.

Available helper scripts:

- `scripts/calibrate_dataset.py`: derives target and minimum ROM calibration values from a local video dataset
- `scripts/build_ucf_reference_stats.py`: computes aggregate UCF101-derived joint statistics
- `scripts/extract_stroke_thresholds.py`: extracts ROM thresholds from the local impaired MATLAB datasets

Generated JSON artifacts are written into `data/` and loaded at runtime by the analysis pipeline.

## Pose backend behavior

If video landmarks cannot be extracted, the server fails clearly instead of substituting a demo result. To proceed, provide a supported video, a `.pose.json` sidecar file, or paste landmark JSON directly.
